from __future__ import annotations

from pathlib import Path

from src.egoflow.utils.io import write_json
from src.egoflow.utils.paths import output_root, video_dir
from src.egoflow.utils.progress import emit
from src.egoflow.utils.video_io import extract_frames, normalize_video, probe_video


def run(video_uid: str, config: dict) -> None:
    root = output_root(config)
    input_path = Path(config["_runtime"]["input_path"]).resolve()
    out_dir = video_dir(video_uid, config)
    frames_dir = out_dir / "frames"
    normalized_path = out_dir / "normalized.mp4"
    out_dir.mkdir(parents=True, exist_ok=True)

    sampled_fps = float(config["input"]["sampled_fps"])
    emit(
        video_uid,
        1,
        "working",
        "Reading video metadata with ffprobe/OpenCV",
        root,
        phase_name="Ingest",
        backed_by="FFmpeg + OpenCV",
        progress=15,
    )
    meta = probe_video(input_path, video_uid, sampled_fps)
    resolution = tuple(config["input"]["normalize_resolution"])
    emit(
        video_uid,
        1,
        "working",
        f"Normalizing to {resolution[0]}x{resolution[1]} at {config['input']['normalize_fps']} fps",
        root,
        phase_name="Ingest",
        backed_by="FFmpeg + OpenCV",
        progress=35,
    )
    normalize_video(
        input_path,
        normalized_path,
        resolution=(int(resolution[0]), int(resolution[1])),
        fps=int(config["input"]["normalize_fps"]),
        duration_sec=meta.duration_sec,
        progress_callback=lambda seconds, percent: emit(
            video_uid,
            1,
            "working",
            f"Normalizing video: {percent:.0f}% ({seconds:.0f}s of {meta.duration_sec:.0f}s)",
            root,
            phase_name="Ingest",
            backed_by="FFmpeg live progress",
            progress=35 + int(min(100, percent) * 0.35),
        ),
    )
    emit(
        video_uid,
        1,
        "working",
        f"Extracting keyframes at {sampled_fps:g} fps for downstream annotation",
        root,
        phase_name="Ingest",
        backed_by="OpenCV frame sampling",
        progress=72,
    )
    frame_count = extract_frames(
        normalized_path,
        frames_dir,
        sampled_fps,
        progress_callback=lambda seconds, percent: emit(
            video_uid,
            1,
            "working",
            f"Extracting keyframes: {percent:.0f}% ({seconds:.0f}s scanned)",
            root,
            phase_name="Ingest",
            backed_by="OpenCV frame sampling",
            progress=72 + int(min(100, percent) * 0.20),
        ),
    )
    write_json(out_dir / "meta.json", meta)
    emit(
        video_uid,
        1,
        "working",
        f"Wrote normalized video, metadata, and {frame_count} sampled frames",
        root,
        phase_name="Ingest",
        backed_by="File-based handoff",
        progress=95,
    )
