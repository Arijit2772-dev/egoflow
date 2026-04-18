from pathlib import Path

import pytest
import yaml

from src.egoflow.config import load_config
from src.egoflow.pipeline import run_pipeline


def test_smoke_pipeline(tmp_path):
    cv2 = pytest.importorskip("cv2")
    video_path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180))
    assert writer.isOpened()
    for idx in range(40):
        frame = _frame(cv2, idx)
        writer.write(frame)
    writer.release()

    config = load_config("config.yaml")
    config["paths"]["output_root"] = str(tmp_path / "output")
    config["input"]["normalize_resolution"] = [320, 180]
    config["input"]["normalize_fps"] = 10
    config["segmentation"]["max_clip_duration_sec"] = 4.0
    config_path = tmp_path / "config.yaml"
    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)

    uid = run_pipeline(video_path, config_path=config_path)
    out_dir = Path(config["paths"]["output_root"]) / uid
    assert (out_dir / "dataset.json").exists()
    assert (out_dir / "validation_report.json").exists()


def _frame(cv2, idx):
    import numpy as np

    frame = np.full((180, 320, 3), 245, dtype=np.uint8)
    cv2.rectangle(frame, (110, 70), (210, 135), (80, 160, 110), -1)
    cv2.circle(frame, (95 + idx % 12, 120), 20, (180, 120, 80), -1)
    cv2.circle(frame, (225 - idx % 12, 120), 20, (180, 120, 80), -1)
    return frame
