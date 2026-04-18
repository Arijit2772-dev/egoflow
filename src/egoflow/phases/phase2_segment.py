from __future__ import annotations

from pathlib import Path

from src.egoflow.schema import Segment, VideoMeta
from src.egoflow.utils.io import read_dataclass, write_json
from src.egoflow.utils.logging import get_logger
from src.egoflow.utils.paths import output_root, video_dir
from src.egoflow.utils.progress import emit
from src.egoflow.utils.video_io import split_clip

logger = get_logger(__name__)


def run(video_uid: str, config: dict) -> None:
    root = output_root(config)
    out_dir = video_dir(video_uid, config)
    normalized_path = out_dir / "normalized.mp4"
    clips_dir = out_dir / "clips"
    meta = read_dataclass(out_dir / "meta.json", VideoMeta)

    emit(
        video_uid,
        2,
        "working",
        "Detecting scene boundaries with PySceneDetect",
        root,
        phase_name="Segment",
        backed_by="PySceneDetect content detector",
        progress=15,
    )
    scene_boundaries = _scene_boundaries(normalized_path, config)
    emit(
        video_uid,
        2,
        "working",
        f"Merging {len(scene_boundaries)} scene cuts into action-length clips",
        root,
        phase_name="Segment",
        backed_by="Scene cuts + duration constraints",
        progress=45,
    )
    boundaries = _merge_boundaries(scene_boundaries, meta.duration_sec, config)
    segments: list[Segment] = []
    for idx, (start, end, confidence) in enumerate(boundaries, start=1):
        emit(
            video_uid,
            2,
            "working",
            f"Writing clip {idx}/{len(boundaries)}: {start:.1f}s to {end:.1f}s",
            root,
            phase_name="Segment",
            backed_by="FFmpeg clip split",
            progress=45 + int((idx / max(1, len(boundaries))) * 45),
        )
        clip_path = clips_dir / f"clip_{idx:03d}.mp4"
        split_clip(normalized_path, clip_path, start, end)
        segments.append(
            Segment(
                segment_id=f"seg_{idx:03d}",
                clip_path=str(clip_path.resolve()),
                start_time=round(float(start), 3),
                end_time=round(float(end), 3),
                boundary_confidence=confidence,
            )
        )
    write_json(out_dir / "segments.json", segments)
    emit(
        video_uid,
        2,
        "working",
        f"Wrote segments.json with {len(segments)} clips",
        root,
        phase_name="Segment",
        backed_by="File-based handoff",
        progress=95,
    )


def _scene_boundaries(video_path: Path, config: dict) -> list[float]:
    try:
        from scenedetect import ContentDetector, detect

        scenes = detect(str(video_path), ContentDetector(threshold=float(config["segmentation"]["scene_threshold"])))
        cuts = [scene[0].get_seconds() for scene in scenes[1:]]
        return sorted({float(cut) for cut in cuts if cut > 0.0})
    except Exception as exc:
        logger.warning("PySceneDetect unavailable, using fixed-duration segments: %s", exc)
        return []


def _merge_boundaries(cuts: list[float], duration: float, config: dict) -> list[tuple[float, float, float]]:
    min_dur = float(config["segmentation"]["min_clip_duration_sec"])
    max_dur = float(config["segmentation"]["max_clip_duration_sec"])
    if duration <= 0:
        return [(0.0, max(min_dur, 1.0), 0.5)]

    points = [0.0]
    for cut in cuts:
        if cut - points[-1] >= min_dur:
            points.append(cut)
    points.append(duration)

    windows: list[tuple[float, float, float]] = []
    for start, end in zip(points, points[1:]):
        cursor = start
        while end - cursor > max_dur:
            windows.append((cursor, cursor + max_dur, 0.72))
            cursor += max_dur
        if end - cursor >= min_dur or not windows:
            windows.append((cursor, end, 0.82 if cuts else 0.65))
        elif windows:
            prev_start, _, prev_conf = windows[-1]
            windows[-1] = (prev_start, end, prev_conf)
    return windows
