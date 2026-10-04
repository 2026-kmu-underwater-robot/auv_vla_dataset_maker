# KMU26 AUV LeRobotDataset v3 data collector

실제 KMU26 AUV에서 U0 파인튜닝용 episode를 수집하는 ROS 2 패키지입니다. 다음 데이터를
10 Hz로 묶어 저장합니다.

현재 동작은 **ROS 수집 → JPEG/NPZ staging → LeRobotDataset v3 내보내기**입니다.
`stop_episode`만 호출하면 staging이 저장되며, 아래 내보내기 명령까지 실행해야 v3
데이터셋이 완성됩니다. 실시간으로 v3 파일을 직접 기록하는 모드는 없습니다.

- 전방 compressed RGB 카메라
- 부표 분리부 compressed RGB 카메라
- DVL 속도와 고도/validity
- MAVROS IMU 각속도, 선형가속도, 자세 quaternion
- 압력 센서에서 변환된 수심
- `/mavros/rc/override`의 surge, sway, heave, yaw 조작 명령
- 자연어 task description

RC override는 motor PWM이 아니라 다음 순서의 정규화된 action label로 저장됩니다.

```text
[surge, sway, heave, yaw] ∈ [-1, 1]
```

ArduSub 기본 채널은 각각 `[5, 6, 3, 4]`이며 파라미터로 변경할 수 있습니다. `0`
(`CHAN_RELEASE`)는 해당 축의 제어권을 무효화합니다. `65535` (`CHAN_NOCHANGE`)는
직전 명령을 유지하지만 유효 시간을 갱신하지 않습니다.

## 빌드

```bash
cd /home/kuuve/auv_ros2
source /opt/ros/humble/setup.bash
colcon build --base-paths src --symlink-install \
  --packages-ignore mavros_msgs \
  --packages-select auv_dvl_a50_msg kmu26_auv_vla_data_collector
source install/setup.bash
```

이 워크스페이스에는 ROS 1 보관 소스와 ROS 2 MAVROS 소스가 함께 있어, 위 명령은 ROS 1
디렉터리의 동명 `mavros_msgs`가 검색되는 것을 피하고 `/opt/ros/humble`의 메시지를 사용합니다.

## 실행

기본 토픽으로 실행합니다. 기본 저장 위치는 `~/vla_data/staging`입니다.

```bash
ros2 launch kmu26_auv_vla_data_collector collector.launch.py
```

두 번째 카메라 토픽이 다르면 launch argument로 지정합니다.

```bash
ros2 launch kmu26_auv_vla_data_collector collector.launch.py \
  buoy_release_image_topic:=/actual/release/camera/image_raw/compressed
```

전체 설정은 [config/collector.yaml](config/collector.yaml)에 있습니다. 특히 수집 전에 다음을
실제 장비와 대조해야 합니다.

- 두 번째 카메라 토픽
- RC channel 순서 및 각 축 부호
- `/dvl/twist`가 사용하는 좌표축
- `/depth/pose.position.z`가 위쪽 양수인지 여부
- PWM 중립점과 span

원본 master(`4b4e232`) 대비 토픽·메시지 타입·서비스 이름은 변경하지 않았습니다.

| 입력 | 기본 토픽 | 메시지 타입 |
| --- | --- | --- |
| 전방 카메라 | `/imx219/camera0/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` |
| 분리부 카메라 | `/imx219/camera1/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` |
| DVL 속도 | `/dvl/twist` | `geometry_msgs/msg/TwistWithCovarianceStamped` |
| DVL 상태/고도 | `/dvl/data` | `auv_dvl_a50_msg/msg/DVL` |
| IMU | `/mavros/imu/data` | `sensor_msgs/msg/Imu` |
| 수심 pose | `/depth/pose` | `geometry_msgs/msg/PoseWithCovarianceStamped` |
| RC 명령 | `/mavros/rc/override` | `mavros_msgs/msg/OverrideRCIn` |
| 작업 설명 | `/vla/task_description` | `std_msgs/msg/String` |

다른 설정은 YAML을 복사해 수정한 후 `config:=/absolute/path/collector.yaml`로 지정합니다.
분리부 카메라는 launch argument가 YAML보다 우선하므로 변경 시
`buoy_release_image_topic:=...`도 함께 지정해야 합니다.

## Episode 수집

먼저 영어 task instruction을 publish합니다.

```bash
ros2 topic pub --once /vla/task_description std_msgs/msg/String \
  "{data: 'Approach the red buoy.'}"
```

모든 필수 입력이 최신 상태일 때만 episode가 시작됩니다.

```bash
ros2 service call /vla_data_collector/start_episode std_srvs/srv/Trigger "{}"
```

성공한 수행을 저장합니다.

```bash
ros2 service call /vla_data_collector/stop_episode std_srvs/srv/SetBool "{data: true}"
```

실패했지만 recovery 학습에 사용할 수행은 `data: false`로 저장합니다. 센서 설정 오류나
사람이 끼어든 수행처럼 학습에 사용하면 안 되는 episode는 저장 전에 폐기합니다.

```bash
ros2 service call /vla_data_collector/discard_episode std_srvs/srv/Trigger "{}"
```

노드가 recording 도중 종료되면 수집된 frame이 있는 episode는 `success=false`와
`termination_reason=node_shutdown`으로 보존됩니다.

## 저장 구조

수집 단계에서는 ROS 런타임에 pandas/pyarrow를 요구하지 않도록 JPEG와 NPZ로 저장합니다.

```text
vla_data/staging/
└── episode_000000/
    ├── manifest.json
    ├── samples.npz
    └── frames/
        ├── ego/frame_000000.jpg
        └── buoy_release/frame_000000.jpg
```

`samples.npz`에는 23차원 `observation_state`, 4차원 `action`, 원본 RC PWM과 각 채널의
update mask, ROS timestamp, 각 센서의 원본 timestamp와 sample 시점 기준 age가 포함됩니다.

## LeRobotDataset v3/U0 형식으로 변환

수집 후 공식 `LeRobotDataset` writer가 여러 episode를 v3 Parquet/MP4 shard로 묶고,
episode offset과 통계 metadata를 생성합니다. Python 3.12 환경을 ROS Humble의
Python 3.10 환경과 분리합니다. 패키지 버전 `0.6.1`과 데이터 형식 버전 `v3.0`은
서로 다른 버전입니다. `lerobot[dataset]`의 추가 의존성까지 설치해야 합니다.

수집을 중지한 뒤 **새 터미널**에서 다음 명령을 실행합니다. 이 터미널에는 ROS setup을
source할 필요가 없습니다. 저장소 루트에서 실행하면 ROS 패키지 설치 없이도
`python -m`으로 exporter를 실행할 수 있습니다. 필요하면 staging 폴더를 학습 PC로 복사합니다.

```bash
cd /home/kuuve/auv_ros2/src/auv_vla_data_collector  # 실제 클론 경로에 맞게 변경
python3.12 -m venv .venv-lerobot-v3
source .venv-lerobot-v3/bin/activate
python -m pip install -r requirements-export.txt
```

```bash
python -m kmu26_auv_vla_data_collector.export_lerobot \
  "$HOME/vla_data/staging" \
  "$HOME/vla_data/lerobot_v3" \
  --repo-id your-hf-user/kmu26-auv-real
```

`ros2 run export_lerobot`는 colcon 빌드 당시 Python으로 실행될 수 있으므로 위의
가상환경 `python -m` 명령을 사용합니다. 기본 staging 경로와 내보내기 입력 경로를
반드시 일치시키십시오. `--repo-id`는 로컬 데이터셋 식별자이며 자동 업로드하지 않습니다.
`--fps`를 생략하면 manifest의 수집 FPS를 사용합니다. 현재 exporter는 양의 정수 FPS를
지원하며, 다른 FPS로 재샘플링하거나 시간 간격 오류를 보정하지 않습니다.

LeRobot v3 writer가 출력 폴더를 직접 생성하므로 출력 경로가 이미 존재하면 변환기는
덮어쓰지 않고 중단합니다. 변환 결과는
`meta/info.json`, `meta/episodes/`, `meta/tasks.parquet`, `data/`, `videos/`를 갖는
LeRobotDataset v3 형식입니다. v3 writer는 마지막 episode 뒤에 반드시 `finalize()`되어
Parquet footer와 buffered metadata를 기록합니다. 변환기는 이를 자동으로 수행합니다.

데이터셋을 확인할 때는 같은 환경에서 로드합니다.

```bash
python -c "from pathlib import Path; from lerobot.datasets.lerobot_dataset import LeRobotDataset; d = LeRobotDataset('your-hf-user/kmu26-auv-real', root=Path.home()/'vla_data/lerobot_v3', video_backend='pyav'); print(d.num_episodes, d.num_frames); print(d[0]['observation.state'].shape, d[0]['action'].shape)"
```

출력 예시는 다음과 같습니다. shard 개수는 데이터 크기에 따라 달라집니다.

```text
lerobot_v3/
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   ├── modality.json
│   └── episodes/chunk-000/file-000.parquet
├── data/chunk-000/file-000.parquet
└── videos/
    ├── observation.images.ego/chunk-000/file-000.mp4
    └── observation.images.buoy_release/chunk-000/file-000.mp4
```

표준 `timestamp`는 episode 내부 `frame_index / fps`이며, 원본 ROS 시간은
`telemetry.ros_timestamp`에 따로 보존합니다. state는 기존 23차원, action은 기존
`[surge, sway, heave, yaw]` 4차원입니다. OpenCV BGR 영상을 RGB로 변환한 후 저장합니다.
영상은 기존과 같은 H.264 / yuv420p / CRF 18 / medium으로 인코딩하며, 랜덤 프레임
접근을 위한 GOP는 LeRobot 기본값 2를 사용합니다. 영상 크기는 짝수여야 합니다.
`episode_success`, 마지막 프레임의 `next.done=true`, `next.reward=0`도 유지합니다.
실패 episode도 내보내므로 성공 데이터만 필요하면 입력 staging을 별도로 구성하십시오.
완성된 v3 출력에 추가 기록하는 기능은 없으며, 새 출력 경로로 전체 staging을 다시 내보냅니다.
오류로 생성된 부분 출력은 학습에 사용하지 말고 원본을 확인한 뒤 새 경로로 재시도합니다.

DVL이 없으면 staging의 `source_age`와 `source_timestamp` 해당 칸은 NaN입니다.
v3에서는 통계에 NaN이 전파되지 않도록 해당 칸을 0으로 저장하고
`telemetry.source_age_present`, `telemetry.source_timestamp_present`에 `false`를 기록합니다.
각 mask는 7차원 bool이며 순서는 `[ego_camera, buoy_release_camera, imu, depth,
dvl_velocity, dvl_altitude, rc_control]`입니다. `present`는 값의 존재 여부일 뿐
센서의 신선도·유효성을 의미하지 않습니다. 원본 NPZ는 변경하지 않습니다.
필요하면 `np.where(present, value, np.nan)`으로 누락을 복원할 수 있으며, telemetry를
분석할 때는 mask를 적용해야 합니다(공식 통계에는 대체한 0도 포함됩니다).

episode 번호는 staging 폴더 순서대로 0부터 다시 부여됩니다. task 번호는 공식 writer의
첫 등장 순서이므로 기존 v2의 문자열 정렬 순서와 다를 수 있습니다. 숫자 ID를 고정해서
해석하지 말고 `meta/tasks.parquet`의 설명과 연결하십시오.

`modality.json`의 U0 필드 매핑을 유지했지만, 기존 U0 학습기의 **v3 reader 지원 여부는
별도 확인**해야 합니다. v2 전용 로더는 파일명만 바꿔서 이 출력을 읽을 수 없습니다.

테스트는 exporter 환경에서 실행합니다.

```bash
python -m pip install pytest
python -m pytest -q
```

`test_v3_integration.py`는 모의 writer 없이 실제 두 episode를 저장하고 Parquet 경계,
영상 시간 offset, RGB 채널 및 LeRobot 재로딩을 검사합니다. ROS 통합 테스트는
ROS와 `auv_dvl_a50_msg`가 설치된 별도 환경이 필요합니다.

## 중요한 제한

현재 ROV 기본 DVL 축 변환과 시각/RC 유효성 정책은 아래 Current ROV 절을 따릅니다.
다른 장착 방향은 앞단에서 body frame으로 변환해야 합니다.

원본 수집 알고리즘은 최신 센서 값을 10 Hz timer에서 묶습니다. 모든 센서가 동일 시각에
하드웨어 동기화된다는 뜻은 아닙니다. 수집 시작에는 최신 카메라 두 개, IMU, 수심,
4축 RC 명령과 작업 설명이 필요하며 DVL은 선택적입니다. DVL 무효 시 값은 0과 validity로
표현합니다. 수집 도중 오래된 카메라 프레임은 기존 정책대로 validity=0으로 저장될 수
있지만 `policy_observation()`은 오래된 카메라를 거부합니다. 학습 시 validity를 반영해야 합니다.
IMU·수심·RC가 오래되면 프레임을 건너뛰며, 다음 timer에서 시간 간격이 허용 범위를 넘으면
episode가 종료됩니다. 정상 종료는 데이터를 보존하지만 강제 종료·정전까지 복구를 보장하지는
않습니다(NPZ는 episode 종료 때 저장됩니다).

## Current ROV / MuJoCo integration

Install/build the workspace's `auv_dvl_a50_msg` dependency, not legacy `dvl_msgs`.
The defaults now use `/imx219/camera0/image_raw/compressed` and
`/imx219/camera1/image_raw/compressed`. Camera 1 must actually depict the release
area for this embodiment; a stereo partner is not automatically a release camera.
The dataset directory defaults to `~/vla_data/staging`.

```bash
colcon build --packages-select auv_dvl_a50_msg kmu26_auv_vla_data_collector
source install/setup.bash
# Simulation only; omit use_sim_time for the vehicle.
ros2 launch kmu26_auv_vla_data_collector collector.launch.py use_sim_time:=true
```

`dvl_link` FRD vectors are converted once to body FLU (`x, -y, -z`). This matches
the current ROV's X-pi DVL mounting transform. For a different mounting, provide
an upstream body-frame velocity topic and set `dvl_input_frame:=base_link` and
`dvl_convention:=FLU` in a custom YAML. Unknown DVL/IMU frames are rejected;
IMU must already be in `body_frame` (default `base_link`). The collector does
not infer arbitrary mounting transforms. Depth remains positive down in metres.

Capture and receipt ages must both be valid. Clock resets and missed sampling
intervals end the current recording with `sampling_discontinuity`; start a new
episode after recovery. Export rejects gaps/rate changes instead of silently
compressing time. Normal timer jitter up to 25% of a period is accepted.
RC RELEASE invalidates the affected action immediately; NOCHANGE cannot renew
an expired action. All four primary RC channels must have recent explicit
commands. The prior helper remains available, but the recorder uses per-axis
ownership tracking. Keep the same ArduSub mode, axis signs, neutral and PWM span
across demonstrations and policy execution; default span 300 matches joy2mavros.

Use `requirements-export.txt` in a separate export environment. The v3 exporter
uses PyAV; it no longer invokes the system `ffmpeg` command. Keep failed/recovery episodes separate unless deliberately
including them in training; export does not silently filter `success=false`.
The public `policy_observation(previous_command)` method exposes the same sensor
contract to an inference adapter without publishing RC or requiring an RC source.
