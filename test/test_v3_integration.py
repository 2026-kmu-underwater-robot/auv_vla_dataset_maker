"""Real writer/reader test; run in the requirements-export.txt environment."""
import json

import cv2
import numpy as np
import pytest

from kmu26_auv_vla_data_collector.export_lerobot import export_dataset, VIDEO_KEYS


def make_episode(staging, index, count=4, missing_dvl=False):
    episode = staging / f"episode_{index:06d}"
    for camera in ("ego", "buoy_release"):
        directory = episode / "frames" / camera
        directory.mkdir(parents=True)
        for frame in range(count):
            image = np.full((32, 48, 3), (10, 30, 200), np.uint8)
            assert cv2.imwrite(str(directory / f"frame_{frame:06d}.jpg"), image)
    age = np.zeros((count, 7), np.float32)
    source_time = np.full((count, 7), 100.0, np.float64)
    if missing_dvl:
        age[:, 4:6] = np.nan
        source_time[:, 4:6] = np.nan
    arrays = dict(
        observation_state=np.full((count, 23), index / 100, np.float32),
        action=np.full((count, 4), index / 100, np.float32),
        rc_pwm=np.full((count, 4), 1500, np.int32),
        rc_update_mask=np.ones((count, 4), np.float32),
        ros_timestamp=100 + np.arange(count) / 10,
        source_age=age,
        source_timestamp=source_time,
    )
    np.savez_compressed(episode / "samples.npz", **arrays)
    (episode / "manifest.json").write_text(json.dumps(dict(
        episode_index=index, task=f"Task {index}", success=index == 7,
        frames=count, fps=10.0, termination_reason="operator_stop",
    )), encoding="utf-8")
    return episode


@pytest.mark.parametrize("missing_dvl", [False, True])
def test_real_v3_roundtrip(tmp_path, missing_dvl):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import pandas as pd

    staging = tmp_path / "staging"
    make_episode(staging, 7, missing_dvl=missing_dvl)
    make_episode(staging, 9, missing_dvl=missing_dvl)
    output = tmp_path / "v3"
    export_dataset(staging, output, None, "local/auv-test")
    info = json.loads((output / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 8
    assert info["splits"]["train"] == "0:2"
    data = pd.concat([pd.read_parquet(p) for p in sorted(output.glob("data/*/*.parquet"))])
    assert list(data["episode_index"]) == [0] * 4 + [1] * 4
    np.testing.assert_allclose(data["timestamp"], [0, .1, .2, .3] * 2)
    np.testing.assert_allclose(np.stack(data["action"]), [[.07] * 4] * 4 + [[.09] * 4] * 4)
    np.testing.assert_allclose(np.stack(data["observation.state"]), [[.07] * 23] * 4 + [[.09] * 23] * 4)
    np.testing.assert_allclose(data["telemetry.ros_timestamp"], [100, 100.1, 100.2, 100.3] * 2)
    np.testing.assert_array_equal(np.stack(data["telemetry.rc_pwm"]), np.full((8, 4), 1500))
    np.testing.assert_array_equal(np.stack(data["telemetry.rc_update_mask"]), np.ones((8, 4)))
    for name in ("source_age", "source_timestamp"):
        present = np.stack(data[f"telemetry.{name}_present"])
        assert present[:, [0, 1, 2, 3, 6]].all()
        assert (present[:, 4:6] == (not missing_dvl)).all()
        if missing_dvl:
            assert (np.stack(data[f"telemetry.{name}"])[:, 4:6] == 0).all()
    assert list(data["next.done"]) == [False, False, False, True] * 2
    assert list(data["episode_success"]) == [True] * 4 + [False] * 4
    episodes = pd.concat([pd.read_parquet(p) for p in output.glob("meta/episodes/*/*.parquet")])
    assert list(episodes["dataset_from_index"]) == [0, 4]
    assert list(episodes["dataset_to_index"]) == [4, 8]
    assert len(list(output.glob("data/*/*.parquet"))) == 1
    for key in VIDEO_KEYS:
        assert info["features"][key]["info"]["video.codec"] == "h264"
        assert info["features"][key]["info"]["video.crf"] == 18
        assert len(list((output / "videos" / key).glob("*/*.mp4"))) == 1
        assert episodes.iloc[1][f"videos/{key}/from_timestamp"] > 0
    dataset = LeRobotDataset("local/auv-test", root=output, video_backend="pyav")
    for index in (0, 3, 4, 7):
        sample = dataset[index]
        assert sample["task"] == ("Task 7" if index < 4 else "Task 9")
        rgb = sample[VIDEO_KEYS[0]]
        assert tuple(rgb.shape) == (3, 32, 48)
        assert float(rgb[0].mean()) > float(rgb[2].mean()) + .4
    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        return np.isfinite(np.asarray(value)).all()
    assert finite(json.loads((output / "meta/stats.json").read_text()))


@pytest.mark.parametrize("damage", ["gap", "nan_state", "bad_action", "missing_frame", "renamed_frame", "resolution", "empty_task", "empty_episode", "infinite_dvl", "missing_imu_time", "bad_mask", "odd_resolution"])
def test_invalid_source_rejected_before_output_creation(tmp_path, damage):
    staging = tmp_path / "staging"
    episode = make_episode(staging, 7)
    if damage in ("gap", "nan_state", "bad_action", "infinite_dvl", "missing_imu_time", "bad_mask"):
        with np.load(episode / "samples.npz") as source:
            arrays = dict(source)
        if damage == "gap":
            arrays["ros_timestamp"][2:] += 1
        elif damage == "nan_state":
            arrays["observation_state"][1, 0] = np.nan
        elif damage == "infinite_dvl":
            arrays["source_age"][1, 4] = np.inf
        elif damage == "missing_imu_time":
            arrays["source_timestamp"][1, 2] = np.nan
        elif damage == "bad_mask":
            arrays["rc_update_mask"][1, 0] = np.nan
        else:
            arrays["action"][1, 0] = 1.5
        np.savez_compressed(episode / "samples.npz", **arrays)
    elif damage in ("empty_task", "empty_episode"):
        manifest = json.loads((episode / "manifest.json").read_text())
        manifest["task" if damage == "empty_task" else "frames"] = "" if damage == "empty_task" else 0
        (episode / "manifest.json").write_text(json.dumps(manifest))
    else:
        path = episode / "frames/ego/frame_000001.jpg"
        if damage == "missing_frame":
            path.unlink()
        elif damage == "renamed_frame":
            path.rename(path.with_name("frame_000008.jpg"))
        elif damage == "odd_resolution":
            cv2.imwrite(str(path), np.zeros((33, 48, 3), np.uint8))
        else:
            cv2.imwrite(str(path), np.zeros((64, 48, 3), np.uint8))
    output = tmp_path / "v3"
    with pytest.raises(ValueError):
        export_dataset(staging, output, None)
    assert not output.exists()
