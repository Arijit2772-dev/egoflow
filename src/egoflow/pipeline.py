from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterable

from src.egoflow.config import load_config
from src.egoflow.utils.logging import get_logger
from src.egoflow.utils.paths import output_root, safe_uid, video_dir
from src.egoflow.utils.progress import emit

logger = get_logger(__name__)

PHASE_OUTPUTS = {
    1: "meta.json",
    2: "segments.json",
    3: "annotations",
    4: "narrations",
    5: "dataset.json",
    6: "validation_report.json",
}

PHASE_LABELS = {
    1: ("Ingest", "FFmpeg + OpenCV"),
    2: ("Segment", "PySceneDetect + motion boundaries"),
    3: ("Annotate", "MediaPipe/100DOH + YOLO-World"),
    4: ("Describe", "Gemini narration fallback-safe"),
    5: ("Assemble", "Ego4D-style JSON builder"),
    6: ("Validate", "Rule checks + CLIP fallback-safe QA"),
}


def run_pipeline(
    input_path: str | Path,
    phases: Iterable[int] | None = None,
    resume: bool = False,
    config_path: str | Path = "config.yaml",
) -> str:
    config = load_config(config_path)
    source = Path(input_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input video not found: {source}")
    video_uid = safe_uid(source)
    root = output_root(config)
    out_dir = video_dir(video_uid, config)
    out_dir.mkdir(parents=True, exist_ok=True)
    config["_runtime"] = {"input_path": str(source), "video_uid": video_uid, "output_dir": str(out_dir)}

    selected = list(phases or [1, 2, 3, 4, 5, 6])
    for phase in selected:
        if phase not in PHASE_OUTPUTS:
            raise ValueError(f"Unknown phase: {phase}")
        expected = out_dir / PHASE_OUTPUTS[phase]
        if resume and _exists(expected):
            logger.info("Skipping phase %s, output already exists.", phase)
            continue
        phase_name, backed_by = PHASE_LABELS[phase]
        emit(
            video_uid,
            phase,
            "started",
            f"{phase_name} started",
            root,
            phase_name=phase_name,
            backed_by=backed_by,
            progress=0,
        )
        module = importlib.import_module(f"src.egoflow.phases.phase{phase}_{_phase_name(phase)}")
        module.run(video_uid, config)
        emit(
            video_uid,
            phase,
            "completed",
            f"{phase_name} completed",
            root,
            phase_name=phase_name,
            backed_by=backed_by,
            progress=100,
        )
    return video_uid


def _exists(path: Path) -> bool:
    if path.is_dir():
        return any(path.iterdir())
    return path.exists()


def _phase_name(phase: int) -> str:
    return {
        1: "ingest",
        2: "segment",
        3: "annotate",
        4: "describe",
        5: "assemble",
        6: "validate",
    }[phase]
