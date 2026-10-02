import json
from pathlib import Path

import cv2
import numpy as np

import kmu26_auv_vla_data_collector.export_lerobot as exporter


def _write_frames(directory: Path, count: int, shape: tuple[int, int, int]) -> None:
    directory.mkdir(parents=True)
    for index in range(count):
        image = np.full(shape, index * 20, dtype=np.uint8)
        assert cv2.imwrite(str(directory / f"frame_{index:06d}.jpg"), image)


class _FakeV3Dataset:
    created = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.episodes = []
        self.current_episode = []
        self.finalized = False
        type(self).created = self

    @classmethod
    def create(cls, **kwargs):
        return cls(**kwargs)

    def add_frame(self, frame):
        self.current_episode.append(frame)

    def save_episode(self):
        self.episodes.append(self.current_episode)
        self.current_episode = []

    def finalize(self):
        self.finalized = True
        metadata_dir = Path(self.kwargs["root"]) / "meta"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / "info.json").write_text(
            json.dumps(
                {
                    "codebase_version": "v3.0",
                    "features": self.kwargs["features"],
                }
            ),
            encoding="utf-8",
        )


def test_export_dataset_uses_lerobot_v3_writer(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    episode = staging / "episode_000007"
    _write_frames(episode / "frames" / "ego", 3, (8, 12, 3))
    _write_frames(episode / "frames" / "buoy_release", 3, (6, 10, 3))
    np.savez_compressed(
        episode / "samples.npz",
        observation_state=np.zeros((3, 23), dtype=np.float32),
        action=np.zeros((3, 4), dtype=np.float32),
        rc_pwm=np.full((3, 4), 1500, dtype=np.int32),
        rc_update_mask=np.ones((3, 4), dtype=np.float32),
        ros_timestamp=np.arange(3, dtype=np.float64) / 10.0,
        source_age=np.zeros((3, 7), dtype=np.float32),
        source_timestamp=np.zeros((3, 7), dtype=np.float64),
    )
    (episode / "manifest.json").write_text(
        json.dumps(
            {
                "episode_index": 7,
                "task": "Approach the red buoy.",
                "success": True,
                "frames": 3,
                "fps": 10.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        exporter, "_load_lerobot_dataset_class", lambda: _FakeV3Dataset
    )
    monkeypatch.setattr(exporter, "_rgb_encoder", lambda: None)

    output = tmp_path / "lerobot_v3"
    exporter.export_dataset(
        staging, output, requested_fps=None, repo_id="test/kmu26-auv"
    )

    dataset = _FakeV3Dataset.created
    assert dataset.finalized
    assert dataset.kwargs["repo_id"] == "test/kmu26-auv"
    assert dataset.kwargs["fps"] == 10
    assert dataset.kwargs["batch_encoding_size"] == 1
    assert len(dataset.episodes) == 1
    assert len(dataset.episodes[0]) == 3
    assert dataset.episodes[0][0]["task"] == "Approach the red buoy."
    assert dataset.episodes[0][-1]["next.done"].item()
    assert dataset.episodes[0][0]["observation.images.ego"].shape == (8, 12, 3)

    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["features"]["observation.images.ego"]["shape"] == [8, 12, 3]
    assert (output / "meta" / "modality.json").is_file()
