"""ROS and process adapters; importing this module does not require ROS."""
from collections import deque
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


ROOT = Path(__file__).resolve().parent.parent


def scan_episodes(root):
    """Read only completed manifests; retain errors alongside valid episodes."""
    root = Path(root).expanduser()
    rows, errors = [], []
    if not root.exists():
        return rows, [], errors
    for path in sorted(root.glob('episode_*/manifest.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            frames, fps = int(data['frames']), float(data['fps'])
            if frames < 0 or fps <= 0:
                raise ValueError('frames/fps 값이 잘못되었습니다')
            rows.append(dict(path=str(path.parent), name=path.parent.name,
                             frames=frames, seconds=frames / fps,
                             success=data['success'], task=data['task'],
                             reason=data.get('termination_reason', '')))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f'{path}: {exc}')
    active = sorted(str(p) for p in root.glob('.recording_episode_*') if p.is_dir())
    return rows, active, errors


def collector_command(config, staging, sim):
    if not str(staging).strip():
        raise ValueError('Staging 폴더를 입력하세요')
    args = ['ros2', 'run', 'kmu26_auv_vla_data_collector', 'collector', '--ros-args']
    if config:
        path = Path(config).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f'설정 파일이 없습니다: {path}')
        args += ['--params-file', str(path)]
    return args + ['-p', f'dataset_root:={Path(staging).expanduser().resolve()}',
                   '-p', f'use_sim_time:={str(bool(sim)).lower()}']


def export_command(python, staging, output, repo_id):
    if any(not str(value).strip() for value in (python, staging, output)):
        raise ValueError('Python 실행 파일, 입력 폴더, 출력 폴더를 모두 입력하세요')
    source = Path(staging).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if not source.is_dir():
        raise ValueError('staging 폴더가 없습니다')
    if destination.exists():
        raise ValueError('내보내기 출력은 아직 존재하지 않는 새 폴더여야 합니다')
    if not repo_id.strip() or len(repo_id.split('/')) != 2:
        raise ValueError('데이터셋 ID를 사용자/데이터셋 형식으로 입력하세요')
    return [str(Path(python).expanduser()), '-m',
            'kmu26_auv_vla_data_collector.export_lerobot', str(source),
            str(destination), '--repo-id', repo_id.strip()]


class ManagedProcess:
    """Own a process group and stream its output without blocking Tk."""
    def __init__(self, log):
        self.process = None
        self.log = log
        self.stopping = False

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None

    def start(self, args, cwd=None, isolated_python=False):
        if self.running:
            raise RuntimeError('이미 실행 중입니다')
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        if isolated_python:
            # ROS Humble's Python 3.10 paths must not leak into the 3.12 exporter.
            env.pop('PYTHONPATH', None)
            env.pop('PYTHONHOME', None)
        self.process = subprocess.Popen(args, cwd=cwd, env=env,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True,
                                        errors='replace', start_new_session=True)
        self.stopping = False
        process = self.process

        def read():
            with process.stdout:
                for line in process.stdout:
                    self.log(line.rstrip())
            self.log(f'프로세스 종료 (코드 {process.wait()})')
        threading.Thread(target=read, daemon=True).start()

    def stop(self):
        if self.running and not self.stopping:
            self.stopping = True
            try:
                os.killpg(self.process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass


class RosBridge:
    """Discover the collector's actual parameters, then monitor its inputs."""
    def __init__(self, node_name='/vla_data_collector'):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        self.rclpy = rclpy
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = rclpy.create_node('auv_dataset_dashboard', context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.lock = threading.Lock()
        self.samples = {}
        self.subscriptions = []
        self.name = '/' + node_name.strip('/')
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        try:
            self._configure()
        except Exception:
            self.close()
            raise

    def _wait(self, future, timeout=10):
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout):
            raise TimeoutError('ROS 응답 시간 초과. 요청이 처리되었을 수 있으니 로그/저장 폴더를 확인하세요.')
        return future.result()

    def _configure(self):
        from rcl_interfaces.srv import GetParameters
        from rcl_interfaces.msg import Parameter as ParameterMessage
        from rclpy.parameter import Parameter
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage, Imu
        from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
        from mavros_msgs.msg import OverrideRCIn
        from auv_dvl_a50_msg.msg import DVL
        from std_msgs.msg import String
        from std_srvs.srv import Trigger, SetBool
        keys = ['ego_image_topic', 'buoy_release_image_topic', 'imu_topic',
                'depth_topic', 'dvl_twist_topic', 'dvl_data_topic',
                'rc_override_topic', 'task_description_topic', 'dataset_root',
                'max_sensor_age_sec', 'max_control_age_sec', 'use_sim_time']
        client = self.node.create_client(GetParameters, self.name + '/get_parameters')
        try:
            if not client.wait_for_service(timeout_sec=5):
                raise RuntimeError('수집기를 먼저 실행하고 노드 이름/ROS_DOMAIN_ID를 확인하세요')
            response = self._wait(client.call_async(GetParameters.Request(names=keys)))
            self.params = {key: Parameter.from_parameter_msg(
                ParameterMessage(
                    name=key, value=value)).value
                for key, value in zip(keys, response.values)}
        finally:
            self.node.destroy_client(client)
        if any(self.params.get(key) is None for key in keys):
            raise RuntimeError('수집기에 필요한 파라미터가 없습니다')
        self.node.set_parameters([Parameter('use_sim_time', value=self.params['use_sim_time'])])
        types = [CompressedImage, CompressedImage, Imu, PoseWithCovarianceStamped,
                 TwistWithCovarianceStamped, DVL, OverrideRCIn]
        for key, message_type in zip(keys[:7], types):
            def receive(message, key=key):
                now = time.monotonic()
                with self.lock:
                    previous = self.samples.get(key)
                    times = previous['times'] if previous else deque(maxlen=60)
                    times.append(now)
                    self.samples[key] = dict(message=message, received=now, times=times)
            qos = 20 if key == 'rc_override_topic' else qos_profile_sensor_data
            self.subscriptions.append(self.node.create_subscription(
                message_type, self.params[key], receive, qos))
        self.publisher = self.node.create_publisher(String, self.params['task_description_topic'], 10)
        self.clients = {
            'start': self.node.create_client(Trigger, self.name + '/start_episode'),
            'stop': self.node.create_client(SetBool, self.name + '/stop_episode'),
            'discard': self.node.create_client(Trigger, self.name + '/discard_episode'),
        }

    def publish_task(self, text):
        from std_msgs.msg import String
        if not text.strip():
            raise ValueError('작업 설명을 입력하세요')
        deadline = time.monotonic() + 3
        while self.publisher.get_subscription_count() == 0:
            if time.monotonic() > deadline:
                raise RuntimeError('작업 설명을 받을 수집기가 없습니다')
            time.sleep(0.05)
        self.publisher.publish(String(data=text.strip()))
        return '작업 설명 전송 완료. 시작 시 수집기의 검증 결과를 확인하세요.'

    def call(self, action, success=True):
        from std_srvs.srv import Trigger, SetBool
        client = self.clients[action]
        if not client.wait_for_service(timeout_sec=3):
            raise RuntimeError('수집기 서비스를 찾을 수 없습니다')
        request = SetBool.Request(data=success) if action == 'stop' else Trigger.Request()
        response = self._wait(client.call_async(request), timeout=60)
        if not response.success:
            raise RuntimeError(response.message)
        return response.message

    def snapshot(self):
        now = time.monotonic()
        ros_now = self.node.get_clock().now().nanoseconds / 1e9
        with self.lock:
            result = {}
            for key, sample in self.samples.items():
                times = list(sample['times'])
                age = now - sample['received']
                message = sample['message']
                header = getattr(message, 'header', None)
                source_age = None
                if header is not None:
                    source_age = ros_now - (header.stamp.sec + header.stamp.nanosec / 1e9)
                threshold = self.params['max_control_age_sec' if key == 'rc_override_topic'
                                        else 'max_sensor_age_sec']
                fresh = age <= threshold and (source_age is None or 0 <= source_age <= threshold)
                result[key] = dict(age=age, source_age=source_age, fresh=fresh, message=message,
                                   hz=(len(times)-1)/(times[-1]-times[0])
                                   if len(times)>1 and times[-1]>times[0] else 0)
            return result

    def close(self):
        self.executor.shutdown(timeout_sec=2)
        self.thread.join(timeout=2)
        self.node.destroy_node()
        self.context.try_shutdown()
