"""Convert staged KMU26 episodes into a LeRobotDataset v3 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .contract import ACTION_NAMES, STATE_NAMES, validate_sample_times

VIDEO_KEYS = ("observation.images.ego", "observation.images.buoy_release")
CAMERA_NAMES = ("ego", "buoy_release")
SOURCE_COUNT = 7
SOURCE_NAMES = (
    "ego_camera",
    "buoy_release_camera",
    "imu",
    "depth",
    "dvl_velocity",
    "dvl_altitude",
    "rc_control",
)
DEFAULT_REPO_ID = "kmu26_auv/real"


def _load_lerobot_dataset_class():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise RuntimeError(
            "LeRobotDataset v3 export requires lerobot[dataset]==0.6.1; install "
            "requirements-export.txt in the export environment"
        ) from error
    return LeRobotDataset


def _rgb_encoder():
    from lerobot.configs.video import RGBEncoderConfig

    # Preserve the original codec, pixel format and quality; use short GOPs
    # for LeRobot random frame access instead of the library's AV1 default.
    return RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", crf=18, preset="medium")


def _axis_names(names: tuple[str, ...] | list[str]) -> dict[str, list[str]]:
    return {"axes": list(names)}


def _features(resolutions: dict[str, tuple[int, int]]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for video_key in VIDEO_KEYS:
        height, width = resolutions[video_key]
        features[video_key] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    features.update(
        {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(STATE_NAMES),),
                "names": _axis_names(STATE_NAMES),
            },
            "action": {
                "dtype": "float32",
                "shape": (len(ACTION_NAMES),),
                "names": _axis_names(ACTION_NAMES),
            },
            "telemetry.rc_pwm": {
                "dtype": "int32",
                "shape": (len(ACTION_NAMES),),
                "names": _axis_names(ACTION_NAMES),
            },
            "telemetry.rc_update_mask": {
                "dtype": "float32",
                "shape": (len(ACTION_NAMES),),
                "names": _axis_names(ACTION_NAMES),
            },
            "telemetry.ros_timestamp": {
                "dtype": "float64",
                "shape": (1,),
                "names": _axis_names(["seconds"]),
            },
            "telemetry.source_age": {
                "dtype": "float32",
                "shape": (SOURCE_COUNT,),
                "names": _axis_names(SOURCE_NAMES),
            },
            "telemetry.source_timestamp": {
                "dtype": "float64",
                "shape": (SOURCE_COUNT,),
                "names": _axis_names(SOURCE_NAMES),
            },
            "episode_success": {
                "dtype": "bool",
                "shape": (1,),
                "names": _axis_names(["success"]),
            },
            "next.done": {
                "dtype": "bool",
                "shape": (1,),
                "names": _axis_names(["done"]),
            },
            "next.reward": {
                "dtype": "float32",
                "shape": (1,),
                "names": _axis_names(["reward"]),
            },
        }
    )
    for name in ("source_age", "source_timestamp"):
        features[f"telemetry.{name}_present"] = {
            "dtype": "bool", "shape": (SOURCE_COUNT,),
            "names": _axis_names(SOURCE_NAMES),
        }
    return features


def _modality() -> dict:
    state_dimensions = (4, 3, 3, 3, 4, 1, 1, 4)
    state_keys = (
        "prev_command",
        "dvl_velocity",
        "angular_velocity",
        "linear_acceleration",
        "attitude",
        "depth",
        "altitude",
        "validity",
    )
    start = 0
    state = {}
    for key, dimension in zip(state_keys, state_dimensions):
        state[key] = {"start": start, "end": start + dimension}
        start += dimension
    return {
        "state": state,
        "action": {"motion": {"start": 0, "end": 4}},
        "video": {
            "ego": {"original_key": VIDEO_KEYS[0]},
            "buoy_release": {"original_key": VIDEO_KEYS[1]},
        },
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }


def _read_rgb_frame(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable camera frame: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _episode_arrays(episode_dir: Path, frame_count: int) -> dict[str, np.ndarray]:
    if frame_count <= 0:
        raise ValueError(f"Episode must contain at least one frame: {episode_dir}")
    with np.load(episode_dir / "samples.npz") as samples:
        arrays = {
            "state": samples["observation_state"].astype(np.float32),
            "action": samples["action"].astype(np.float32),
            "rc_pwm": samples["rc_pwm"].astype(np.int32),
            "rc_update_mask": samples["rc_update_mask"].astype(np.float32),
            "ros_timestamp": samples["ros_timestamp"].astype(np.float64),
            "source_age": samples["source_age"].astype(np.float32),
            "source_timestamp": samples["source_timestamp"].astype(np.float64),
        }

    expected_shapes = {
        "state": (frame_count, len(STATE_NAMES)),
        "action": (frame_count, len(ACTION_NAMES)),
        "rc_pwm": (frame_count, len(ACTION_NAMES)),
        "rc_update_mask": (frame_count, len(ACTION_NAMES)),
        "ros_timestamp": (frame_count,),
        "source_age": (frame_count, SOURCE_COUNT),
        "source_timestamp": (frame_count, SOURCE_COUNT),
    }
    for key, expected in expected_shapes.items():
        if arrays[key].shape != expected:
            raise ValueError(
                f"Invalid {key} shape in {episode_dir}: {arrays[key].shape}; expected {expected}"
            )
    if not np.all(np.isfinite(arrays["state"])):
        raise ValueError(f"Non-finite observation state in {episode_dir}")
    if not np.all(np.isfinite(arrays["action"])) or np.any(
        np.abs(arrays["action"]) > 1
    ):
        raise ValueError(f"Invalid normalized action in {episode_dir}")
    if not np.all(np.isin(arrays["rc_update_mask"], [0, 1])):
        raise ValueError(f"Invalid RC update mask in {episode_dir}")
    for key in ("source_age", "source_timestamp"):
        values = arrays[key]
        # Optional DVL inputs may be absent. NaN poisons official v3 stats,
        # so represent absence explicitly without changing raw staging data.
        if np.isinf(values).any() or np.isnan(values[:, [0, 1, 2, 3, 6]]).any():
            raise ValueError(f"Invalid {key} in {episode_dir}")
        arrays[f"{key}_present"] = np.isfinite(values)
        arrays[key] = np.where(arrays[f"{key}_present"], values, 0)
    return arrays


def _inspect_cameras(
    episode_dirs: list[Path], manifests: list[dict]
) -> dict[str, tuple[int, int]]:
    resolutions: dict[str, tuple[int, int]] = {}
    for episode_dir, manifest in zip(episode_dirs, manifests):
        frame_count = int(manifest["frames"])
        for camera_name, video_key in zip(CAMERA_NAMES, VIDEO_KEYS):
            frame_dir = episode_dir / "frames" / camera_name
            frame_files = sorted(frame_dir.glob("frame_*.jpg"))
            if len(frame_files) != frame_count:
                raise ValueError(
                    f"{episode_dir} {camera_name} has {len(frame_files)} frames; "
                    f"expected {frame_count}"
                )
            for index, path in enumerate(frame_files):
                if path.name != f"frame_{index:06d}.jpg":
                    raise ValueError(f"Non-contiguous camera frame sequence: {path}")
                resolution = _read_rgb_frame(path).shape[:2]
                if any(size % 2 for size in resolution):
                    raise ValueError(f"H.264 yuv420p requires even image dimensions: {path}")
                if video_key not in resolutions:
                    resolutions[video_key] = resolution
                elif resolutions[video_key] != resolution:
                    raise ValueError(f"Camera resolution changed: {path}")
    return resolutions


def export_dataset(
    staging_root: Path,
    output_root: Path,
    requested_fps: float | None,
    repo_id: str = DEFAULT_REPO_ID,
) -> None:
    episode_dirs = sorted(
        path
        for path in staging_root.glob("episode_*")
        if (path / "manifest.json").is_file()
    )
    if not episode_dirs:
        raise ValueError(f"No complete episode directories found in {staging_root}")
    if output_root.exists():
        raise FileExistsError(
            f"Output directory already exists (LeRobot v3 creates it): {output_root}"
        )

    manifests = [
        json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        for path in episode_dirs
    ]
    recorded_fps = {float(manifest["fps"]) for manifest in manifests}
    if requested_fps is None:
        if len(recorded_fps) != 1:
            raise ValueError(f"Episodes contain multiple recording rates: {recorded_fps}")
        fps_value = next(iter(recorded_fps))
    else:
        fps_value = float(requested_fps)
        if any(not np.isclose(rate, fps_value) for rate in recorded_fps):
            raise ValueError("--fps cannot retime demonstrations; resample them explicitly")
    if not np.isfinite(fps_value) or fps_value <= 0.0:
        raise ValueError("FPS must be positive")
    if not np.isclose(fps_value, round(fps_value)):
        raise ValueError("LeRobotDataset v3 requires an integer FPS")
    fps = int(round(fps_value))

    for episode_dir, manifest in zip(episode_dirs, manifests):
        if not isinstance(manifest.get("task"), str) or not manifest["task"].strip():
            raise ValueError(f"Missing task description: {episode_dir}")
        if not isinstance(manifest.get("success"), bool):
            raise ValueError(f"Episode success must be boolean: {episode_dir}")
        if not isinstance(manifest.get("frames"), int):
            raise ValueError(f"Episode frames must be an integer: {episode_dir}")
        for key, expected in (("state_names", STATE_NAMES), ("action_names", ACTION_NAMES)):
            if key in manifest and manifest[key] != list(expected):
                raise ValueError(f"Incompatible {key}: {episode_dir}")
        frame_count = int(manifest["frames"])
        arrays = _episode_arrays(episode_dir, frame_count)
        validate_sample_times(arrays["ros_timestamp"], fps)

    resolutions = _inspect_cameras(episode_dirs, manifests)
    LeRobotDataset = _load_lerobot_dataset_class()
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=_features(resolutions),
        root=output_root,
        robot_type="kmu26_auv_real",
        use_videos=True,
        batch_encoding_size=1,
        video_backend="pyav",
        rgb_encoder=_rgb_encoder(),
    )

    try:
        for episode_dir, manifest in zip(episode_dirs, manifests):
            frame_count = int(manifest["frames"])
            arrays = _episode_arrays(episode_dir, frame_count)
            frame_paths = {
                camera_name: sorted(
                    (episode_dir / "frames" / camera_name).glob("frame_*.jpg")
                )
                for camera_name in CAMERA_NAMES
            }
            for frame_index in range(frame_count):
                frame = {
                    "observation.state": arrays["state"][frame_index],
                    "action": arrays["action"][frame_index],
                    "telemetry.rc_pwm": arrays["rc_pwm"][frame_index],
                    "telemetry.rc_update_mask": arrays["rc_update_mask"][frame_index],
                    "telemetry.ros_timestamp": np.asarray(
                        [arrays["ros_timestamp"][frame_index]], dtype=np.float64
                    ),
                    "telemetry.source_age": arrays["source_age"][frame_index],
                    "telemetry.source_timestamp": arrays["source_timestamp"][frame_index],
                    "episode_success": np.asarray(
                        [bool(manifest["success"])], dtype=np.bool_
                    ),
                    "next.done": np.asarray(
                        [frame_index == frame_count - 1], dtype=np.bool_
                    ),
                    "next.reward": np.asarray([0.0], dtype=np.float32),
                    "task": manifest["task"],
                }
                for camera_name, video_key in zip(CAMERA_NAMES, VIDEO_KEYS):
                    frame[video_key] = _read_rgb_frame(
                        frame_paths[camera_name][frame_index]
                    )
                for name in ("source_age", "source_timestamp"):
                    frame[f"telemetry.{name}_present"] = arrays[f"{name}_present"][frame_index]
                dataset.add_frame(frame)
            dataset.save_episode()
    finally:
        # v3 writes multi-episode parquet/video shards incrementally. finalize()
        # is mandatory so writers flush metadata and parquet footers.
        dataset.finalize()

    metadata_dir = output_root / "meta"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "modality.json").write_text(
        json.dumps(_modality(), indent=2) + "\n", encoding="utf-8"
    )
    total_frames = sum(int(manifest["frames"]) for manifest in manifests)
    print(
        f"Exported {len(manifests)} episodes and {total_frames} frames "
        f"as LeRobotDataset v3 to {output_root}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staging_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Dataset identity stored in LeRobot metadata (default: %(default)s)",
    )
    args = parser.parse_args()
    export_dataset(args.staging_root, args.output_root, args.fps, args.repo_id)


if __name__ == "__main__":
    main()
