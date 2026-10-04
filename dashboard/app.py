"""Tk desktop dashboard. ROS operations run outside the UI thread."""
import argparse
import base64
from datetime import datetime
from pathlib import Path
import queue
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from .backend import ROOT, ManagedProcess, RosBridge, collector_command, export_command, scan_episodes


SENSORS = [('ego_image_topic', '전방 카메라'), ('buoy_release_image_topic', '분리부 카메라'),
           ('imu_topic', 'IMU'), ('depth_topic', '수심'), ('dvl_twist_topic', 'DVL 속도 (선택)'),
           ('dvl_data_topic', 'DVL 고도 (선택)'), ('rc_override_topic', 'RC 명령')]


class Dashboard:
    def __init__(self, root):
        self.root = root
        root.title('KMU26 · AUV Dataset Dashboard')
        root.geometry('1180x850')
        root.minsize(940, 700)
        self.events = queue.Queue()
        self.bridge = None
        self.busy = False
        self.scanning = False
        self.closing = False
        self.ticks = 0
        self.images = {}
        self.camera_stamps = {}
        self.telemetry = tk.StringVar(value='수심 —  ·  DVL 속도 —  ·  RC —')
        self.collector = ManagedProcess(lambda line: self.events.put(('log', '[수집기] ' + line)))
        self.exporter = ManagedProcess(lambda line: self.events.put(('log', '[내보내기] ' + line)))
        self.config = tk.StringVar(value=str(ROOT / 'config' / 'collector.yaml')
                                   if (ROOT / 'config' / 'collector.yaml').exists() else '')
        self.staging = tk.StringVar(value=str(Path.home() / 'vla_data' / 'staging'))
        self.output = tk.StringVar(value=str(Path.home() / 'vla_data' / 'lerobot_v3'))
        export_python = ROOT / '.venv-lerobot-v3' / 'bin' / 'python'
        self.python = tk.StringVar(value=str(export_python) if export_python.exists()
                                   else (shutil.which('python3.12') or ''))
        self.repo = tk.StringVar(value='your-hf-user/kmu26-auv-real')
        self.node = tk.StringVar(value='/vla_data_collector')
        self.sim = tk.BooleanVar(value=False)
        self.summary = tk.StringVar(value='저장 폴더 확인 중…')
        self.connection = tk.StringVar(value='ROS 미연결')
        self.action_status = tk.StringVar(value='수집기 연결 후 작업 설명을 전송하세요')
        self._layout()
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(100, self.tick)

    def _field(self, parent, row, text, variable, browse=None):
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky='ew', padx=6)
        if browse:
            ttk.Button(parent, text='찾기', command=browse).grid(row=row, column=2, padx=6)

    def _browse(self, variable, directory=False):
        value = filedialog.askdirectory() if directory else filedialog.askopenfilename()
        if value:
            variable.set(value)

    def _layout(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('', 20, 'bold'))
        style.configure('Summary.TLabel', font=('', 12, 'bold'))
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill='both', expand=True)
        title = ttk.Frame(outer)
        title.pack(fill='x')
        ttk.Label(title, text='AUV Dataset Dashboard', style='Title.TLabel').pack(side='left')
        ttk.Label(title, textvariable=self.connection).pack(side='right')
        ttk.Label(outer, textvariable=self.summary, style='Summary.TLabel').pack(anchor='w', pady=10)
        notebook = ttk.Notebook(outer)
        notebook.pack(fill='both', expand=True)
        collect = ttk.Frame(notebook, padding=10)
        episodes = ttk.Frame(notebook, padding=10)
        export = ttk.Frame(notebook, padding=10)
        notebook.add(collect, text='  수집 & 모니터링  ')
        notebook.add(episodes, text='  저장된 에피소드  ')
        notebook.add(export, text='  LeRobot v3 내보내기  ')

        settings = ttk.LabelFrame(collect, text='수집기 설정', padding=6)
        settings.pack(fill='x')
        settings.columnconfigure(1, weight=1)
        self._field(settings, 0, '설정 YAML', self.config, lambda: self._browse(self.config))
        self._field(settings, 1, 'Staging 폴더', self.staging, lambda: self._browse(self.staging, True))
        self._field(settings, 2, '수집기 노드', self.node)
        bar = ttk.Frame(settings)
        bar.grid(row=3, column=0, columnspan=3, sticky='ew', pady=5)
        ttk.Checkbutton(bar, text='시뮬레이션 시간', variable=self.sim).pack(side='left')
        self.launch_button = ttk.Button(bar, text='수집기 실행', command=self.launch)
        self.launch_button.pack(side='left', padx=8)
        self.stop_button = ttk.Button(bar, text='수집기 종료', command=self.stop_collector)
        self.stop_button.pack(side='left')
        self.connect_button = ttk.Button(bar, text='ROS 연결 / 다시 연결', command=self.connect)
        self.connect_button.pack(side='right')

        cameras = ttk.Frame(collect)
        cameras.pack(fill='both', expand=True, pady=8)
        self.camera_labels = {}
        for key, name in SENSORS[:2]:
            box = ttk.LabelFrame(cameras, text=name)
            box.pack(side='left', fill='both', expand=True, padx=4)
            label = ttk.Label(box, text='영상 수신 대기', anchor='center')
            label.pack(fill='both', expand=True, padx=4, pady=4)
            self.camera_labels[key] = label
        self.sensor_table = ttk.Treeview(collect, columns=('name', 'topic', 'state', 'age', 'hz'),
                                        show='headings', height=7)
        for key, name, width in [('name', '입력', 140), ('topic', '토픽', 370),
                                 ('state', '수신 상태', 95), ('age', '수신 / 원본 age', 155),
                                 ('hz', 'Hz (최근 수신)', 115)]:
            self.sensor_table.heading(key, text=name)
            self.sensor_table.column(key, width=width, stretch=key == 'topic')
        self.sensor_table.tag_configure('fresh', foreground='#13804a')
        self.sensor_table.tag_configure('stale', foreground='#b54824')
        for key, name in SENSORS:
            self.sensor_table.insert('', 'end', iid=key, values=(name, '—', '미수신', '—', '—'))
        self.sensor_table.pack(fill='x')
        ttk.Label(collect, textvariable=self.telemetry).pack(anchor='w', pady=4)
        ttk.Label(collect, text='수신 상태는 age 기준입니다. 좌표계·RC 소유권 등 최종 시작 조건은 수집기가 검증합니다.').pack(anchor='w', pady=4)
        controls = ttk.LabelFrame(collect, text='에피소드 수집', padding=8)
        controls.pack(fill='x', pady=4)
        ttk.Label(controls, text='영어 작업 설명').pack(side='left')
        self.task = ttk.Entry(controls)
        self.task.insert(0, 'Approach the red buoy.')
        self.task.pack(side='left', fill='x', expand=True, padx=8)
        self.ros_buttons = []
        for text, command in [('전송', self.send_task), ('시작', lambda: self.service('start')),
                              ('성공 저장', lambda: self.service('stop', True)),
                              ('실패 저장', lambda: self.service('stop', False)),
                              ('폐기', self.discard)]:
            button = ttk.Button(controls, text=text, command=command)
            button.pack(side='left', padx=2)
            self.ros_buttons.append(button)
        ttk.Label(collect, textvariable=self.action_status, wraplength=1000).pack(anchor='w')

        ttk.Label(episodes, text='Staging 폴더의 완료된 에피소드를 자동 갱신합니다. 임시 수집 폴더는 상단에 별도 표시됩니다.').pack(anchor='w', pady=6)
        self.episode_table = ttk.Treeview(episodes, columns=('name', 'success', 'frames', 'seconds', 'task', 'reason'), show='headings')
        for key, title, width in [('name', '에피소드', 150), ('success', '결과', 70),
                                   ('frames', '프레임', 75), ('seconds', '길이 (초)', 90),
                                   ('task', '작업 설명', 340), ('reason', '종료 이유', 180)]:
            self.episode_table.heading(key, text=title)
            self.episode_table.column(key, width=width, stretch=key in ('task', 'reason'))
        scroll = ttk.Scrollbar(episodes, orient='vertical', command=self.episode_table.yview)
        self.episode_table.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.episode_table.pack(fill='both', expand=True)

        export.columnconfigure(1, weight=1)
        self._field(export, 0, 'Python 3.12 가상환경 실행 파일', self.python, lambda: self._browse(self.python))
        self._field(export, 1, '입력 staging 폴더 (수집과 공통)', self.staging, lambda: self._browse(self.staging, True))
        self._field(export, 2, '새 출력 폴더', self.output)
        self._field(export, 3, '데이터셋 ID', self.repo)
        ttk.Label(export, text='requirements-export.txt를 설치한 별도 환경의 bin/python을 선택하세요.\nID는 로컬 식별자이며 자동 업로드하지 않습니다. 성공·실패 에피소드를 모두 변환합니다.\n수집을 저장/종료한 뒤 내보내세요. 출력 폴더는 새 경로여야 합니다.', justify='left').grid(row=4, column=0, columnspan=3, sticky='w', padx=6, pady=18)
        self.export_button = ttk.Button(export, text='LeRobot v3 변환 시작', command=self.start_export)
        self.export_button.grid(row=5, column=1, sticky='w')
        self.cancel_button = ttk.Button(export, text='변환 중단', command=self.cancel_export)
        self.cancel_button.grid(row=5, column=2)
        logbox = ttk.LabelFrame(outer, text='실행 로그', padding=4)
        logbox.pack(fill='x', pady=(10, 0))
        self.logs = scrolledtext.ScrolledText(logbox, height=7, state='disabled', font=('monospace', 9))
        self.logs.pack(fill='x')

    def log(self, text):
        self.logs.configure(state='normal')
        self.logs.insert('end', f'{datetime.now():%H:%M:%S}  {text}\n')
        if int(self.logs.index('end-1c').split('.')[0]) > 1200:
            self.logs.delete('1.0', '200.0')
        self.logs.see('end')
        self.logs.configure(state='disabled')

    def work(self, function, callback=None):
        if self.busy or self.closing:
            return
        self.busy = True
        def run():
            try:
                self.events.put(('done', (function(), callback)))
            except Exception as exc:
                self.events.put(('error', str(exc)))
        threading.Thread(target=run, daemon=True).start()

    def launch(self):
        if self.exporter.running:
            messagebox.showerror('변환 중', '변환이 끝난 뒤 수집기를 실행하세요')
            return
        if self.bridge is not None:
            messagebox.showinfo('수집기 연결됨', '이미 연결된 수집기가 있습니다')
            return
        try:
            args = collector_command(self.config.get(), self.staging.get(), self.sim.get())
            self.collector.start(args)
            self.log('수집기 실행 요청. 기동 후 ROS 연결을 누르세요.')
        except Exception as exc:
            messagebox.showerror('실행 실패', str(exc))

    def stop_collector(self):
        if messagebox.askyesno('수집기 종료', '진행 중인 에피소드는 실패(node_shutdown)로 보존됩니다. 수집기를 종료할까요?'):
            self.collector.stop()
            self.log('수집기 종료 요청. 파일 저장과 프로세스 종료를 기다립니다.')

    def connect(self):
        if self.busy or self.closing:
            return
        self.connection.set('ROS 연결 중…')
        self.telemetry.set('수심 —  ·  DVL 속도 —  ·  RC —')
        for key, name in SENSORS:
            self.sensor_table.item(key, values=(name, '—', '연결 대기', '—', '—'), tags=())
        for label in self.camera_labels.values():
            label.configure(image='', text='영상 수신 대기')
        name = self.node.get()
        def operation():
            if self.bridge:
                self.bridge.close()
                self.bridge = None
            return RosBridge(name)
        def connected(bridge):
            self.bridge = bridge
            self.staging.set(str(Path(bridge.params['dataset_root']).expanduser()))
            self.connection.set('ROS 연결됨 · ' + bridge.name)
            self.camera_stamps.clear()
            self.log('수집기의 실제 토픽과 저장 경로를 불러왔습니다.')
        self.work(operation, connected)

    def send_task(self):
        bridge, text = self.bridge, self.task.get()
        if bridge:
            self.work(lambda: bridge.publish_task(text), self.action_done)

    def service(self, action, success=True):
        if action == 'start' and self.exporter.running:
            messagebox.showerror('변환 중', '변환이 끝난 뒤 새 에피소드를 시작하세요')
            return
        bridge = self.bridge
        if bridge:
            self.work(lambda: bridge.call(action, success), self.action_done)

    def action_done(self, message):
        self.action_status.set(message)
        self.log(message)

    def discard(self):
        if messagebox.askyesno('현재 에피소드 폐기', '진행 중인 에피소드를 삭제할까요? 이 데이터는 복구할 수 없습니다.'):
            self.service('discard')

    def start_export(self):
        try:
            _, active, _ = scan_episodes(self.staging.get())
            if active:
                raise ValueError('수집 임시 폴더가 있습니다. 에피소드를 저장/폐기하고, 중단된 임시 폴더는 먼저 확인하세요.')
            args = export_command(self.python.get(), self.staging.get(), self.output.get(), self.repo.get())
            # Use the source root when launched from a checkout; installed packages
            # remain importable when the share directory has no source module.
            cwd = ROOT if (ROOT / 'kmu26_auv_vla_data_collector').is_dir() else None
            self.exporter.start(args, cwd=cwd, isolated_python=True)
            self.log('LeRobot v3 내보내기 시작')
        except Exception as exc:
            messagebox.showerror('내보내기 실패', str(exc))

    def cancel_export(self):
        if messagebox.askyesno('변환 중단', '부분 출력이 남을 수 있습니다. 중단 후에는 새 출력 경로로 다시 변환하세요. 중단할까요?'):
            self.exporter.stop()

    def refresh_sensors(self):
        bridge = self.bridge
        if bridge is None or self.busy:
            return
        alive = all(client.service_is_ready() for client in bridge.clients.values())
        self.connection.set(('ROS 연결됨' if alive else '수집기 서비스 응답 없음') + ' · ' + bridge.name)
        snapshot = bridge.snapshot()
        details = []
        depth = snapshot.get('depth_topic')
        if depth and depth['fresh']:
            details.append(f"수심 pose.z {depth['message'].pose.pose.position.z:.2f} m")
        velocity = snapshot.get('dvl_twist_topic')
        if velocity and velocity['fresh']:
            vector = velocity['message'].twist.twist.linear
            details.append(f'DVL 원본 xyz ({vector.x:.2f}, {vector.y:.2f}, {vector.z:.2f}) m/s')
        control = snapshot.get('rc_override_topic')
        if control and control['fresh']:
            details.append(f"RC 원본 ch3–6 {list(control['message'].channels[2:6])}")
        self.telemetry.set('  ·  '.join(details) or '수심 —  ·  DVL 속도 —  ·  RC —')
        for key, name in SENSORS:
            sample = snapshot.get(key)
            if sample:
                source = sample['source_age']
                ages = f"{sample['age']:.2f}s / " + (f'{source:.2f}s' if source is not None else '—')
                fresh = sample['fresh']
                values = (name, bridge.params[key], '최신' if fresh else '지연 / 시간 오류', ages,
                          f"{sample['hz']:.1f}" if sample['age'] < 1 else '0.0')
                self.sensor_table.item(key, values=values, tags=('fresh' if fresh else 'stale',))
                if key in self.camera_labels and fresh:
                    stamp = sample['message'].header.stamp
                    token = (stamp.sec, stamp.nanosec, id(sample['message']))
                    if self.camera_stamps.get(key) != token:
                        self.show_camera(key, sample['message'].data)
                        self.camera_stamps[key] = token
                elif key in self.camera_labels:
                    self.camera_labels[key].configure(image='', text='영상 지연 / 시간 오류')
                    self.camera_stamps.pop(key, None)
            else:
                self.sensor_table.item(key, values=(name, bridge.params[key], '미수신', '—', '—'), tags=('stale',))
                if key in self.camera_labels:
                    self.camera_labels[key].configure(image='', text='영상 수신 대기')

    def show_camera(self, key, data):
        import cv2
        import numpy as np
        frame = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.camera_labels[key].configure(image='', text='영상 디코딩 실패')
            return
        height, width = frame.shape[:2]
        scale = min(480 / width, 180 / height)
        frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
        ok, encoded = cv2.imencode('.png', frame)
        if ok:
            self.images[key] = tk.PhotoImage(data=base64.b64encode(encoded).decode('ascii'))
            self.camera_labels[key].configure(image=self.images[key], text='')

    def scan(self):
        if self.scanning:
            return
        self.scanning = True
        path = self.staging.get()
        def run():
            try:
                self.events.put(('episodes', (path, scan_episodes(path))))
            except Exception as exc:
                self.events.put(('scan_error', str(exc)))
        threading.Thread(target=run, daemon=True).start()

    def tick(self):
        for _ in range(150):
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == 'log':
                self.log(value)
            elif kind == 'done':
                self.busy = False
                result, callback = value
                if callback:
                    callback(result)
            elif kind == 'error':
                self.busy = False
                if self.bridge is None:
                    self.connection.set('ROS 미연결')
                self.log(value)
                messagebox.showerror('작업 실패', value)
            elif kind == 'scan_error':
                self.scanning = False
                self.log(value)
            elif kind == 'episodes':
                self.scanning = False
                path, (rows, active, errors) = value
                if path != self.staging.get():
                    continue
                total = sum(row['frames'] for row in rows)
                successes = sum(row['success'] is True for row in rows)
                state = f'수집 중 / 임시 폴더 {len(active)}개' if active else '진행 중인 수집 폴더 없음'
                self.summary.set(f'에피소드 {len(rows)}개  ·  성공 {successes}개  ·  {total:,} 프레임  ·  {state}'
                                 + (f'  ·  읽기 오류 {len(errors)}개' if errors else ''))
                selected = self.episode_table.selection()
                self.episode_table.delete(*self.episode_table.get_children())
                for row in rows:
                    self.episode_table.insert('', 'end', iid=row['name'], values=(
                        row['name'], '성공' if row['success'] else '실패', row['frames'],
                        f"{row['seconds']:.1f}", row['task'], row['reason']))
                for iid in selected:
                    if self.episode_table.exists(iid):
                        self.episode_table.selection_add(iid)
        for button in self.ros_buttons:
            button.configure(state='normal' if self.bridge and not self.busy and not self.closing else 'disabled')
        self.connect_button.configure(state='disabled' if self.busy or self.closing else 'normal')
        self.launch_button.configure(state='disabled' if self.collector.running or self.busy or self.closing else 'normal')
        self.stop_button.configure(state='normal' if self.collector.running and not self.collector.stopping and not self.busy and not self.closing else 'disabled')
        self.export_button.configure(state='disabled' if self.exporter.running or self.busy or self.closing else 'normal')
        self.cancel_button.configure(state='normal' if self.exporter.running and not self.exporter.stopping else 'disabled')
        if self.closing:
            if not self.busy and not self.collector.running and not self.exporter.running:
                self.root.destroy()
                return
        else:
            try:
                self.refresh_sensors()
            except Exception as exc:
                self.connection.set('모니터링 오류')
                self.log(str(exc))
            if self.ticks % 10 == 0:
                self.scan()
        self.ticks += 1
        self.root.after(200, self.tick)

    def close(self):
        if self.busy:
            messagebox.showinfo('작업 진행 중', '현재 ROS 요청이 끝난 뒤 창을 닫아주세요')
            return
        if self.collector.running or self.exporter.running:
            if not messagebox.askyesno('대시보드 종료', '대시보드에서 실행한 프로세스를 종료합니다.\n수집 중 데이터는 실패로 저장되며 변환 중 출력은 불완전할 수 있습니다. 종료할까요?'):
                return
        self.collector.stop()
        self.exporter.stop()
        if self.bridge:
            bridge = self.bridge
            self.work(bridge.close)
            self.bridge = None
        self.closing = True
        self.connection.set('프로세스 종료 및 저장을 기다리는 중…')


def main():
    parser = argparse.ArgumentParser(description='KMU26 AUV 데이터 수집 데스크톱 대시보드')
    parser.parse_args()
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        parser.exit(1, f'GUI 화면을 열 수 없습니다: {exc}\n데스크톱 세션에서 실행하세요.\n')
    Dashboard(root)
    root.mainloop()


if __name__ == '__main__':
    main()
