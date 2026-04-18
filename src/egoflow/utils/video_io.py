from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np

from src.egoflow.schema import VideoMeta
from src.egoflow.utils.logging import get_logger

logger = get_logger(__name__)


def _cv2():
    import cv2

    return cv2


def probe_video(path: str | Path, video_uid: str, sampled_fps: float) -> VideoMeta:
    source = Path(path)
    try:
        import ffmpeg

        probe = ffmpeg.probe(str(source))
        stream = next(item for item in probe["streams"] if item.get("codec_type") == "video")
        duration = float(stream.get("duration") or probe.get("format", {}).get("duration") or 0.0)
        fps_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
        num, den = fps_text.split("/")
        fps = float(num) / float(den or 1)
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        frame_count = int(stream.get("nb_frames") or round(duration * fps))
        codec = str(stream.get("codec_name") or "unknown")
        return VideoMeta(
            video_uid=video_uid,
            source_path=str(source.resolve()),
            duration_sec=duration,
            fps=fps,
            resolution=(width, height),
            codec=codec,
            frame_count=frame_count,
            sampled_fps=sampled_fps,
        )
    except Exception as exc:
        logger.warning("ffprobe failed, falling back to OpenCV probe: %s", exc)

    cv2 = _cv2()
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps else 0.0
    cap.release()
    return VideoMeta(
        video_uid=video_uid,
        source_path=str(source.resolve()),
        duration_sec=duration,
        fps=fps,
        resolution=(width, height),
        codec="unknown",
        frame_count=frame_count,
        sampled_fps=sampled_fps,
    )


ProgressCallback = Callable[[float, float], None]


def normalize_video(
    source: str | Path,
    target: str | Path,
    resolution: tuple[int, int],
    fps: int,
    duration_sec: float = 0.0,
    progress_callback: ProgressCallback | None = None,
) -> None:
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = resolution
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vf",
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-an",
        "-progress",
        "pipe:1",
        "-nostats",
        str(target_path),
    ]
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        last_emit = 0.0
        last_percent = -1.0
        if process.stdout is not None:
            for raw_line in process.stdout:
                key, _, value = raw_line.strip().partition("=")
                if key not in {"out_time_ms", "out_time_us", "out_time"}:
                    continue
                processed_sec = _parse_ffmpeg_time(key, value)
                if processed_sec <= 0:
                    continue
                percent = min(100.0, (processed_sec / duration_sec) * 100.0) if duration_sec > 0 else 0.0
                now = time.monotonic()
                if progress_callback and (percent - last_percent >= 2.0 or now - last_emit >= 3.0):
                    progress_callback(processed_sec, percent)
                    last_emit = now
                    last_percent = percent
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
    except Exception as exc:
        logger.warning("ffmpeg normalization failed, copying source video instead: %s", exc)
        shutil.copyfile(source, target_path)


def extract_frames(
    video_path: str | Path,
    frames_dir: str | Path,
    sampled_fps: float,
    progress_callback: ProgressCallback | None = None,
) -> int:
    cv2 = _cv2()
    out_dir = Path(frames_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open normalized video: {video_path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(source_fps / sampled_fps)))
    written = 0
    idx = 0
    last_emit = 0.0
    last_percent = -1.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            cv2.imwrite(str(out_dir / f"frame_{written:06d}.jpg"), frame)
            written += 1
        if progress_callback and total_frames > 0:
            percent = min(100.0, (idx / total_frames) * 100.0)
            now = time.monotonic()
            if percent - last_percent >= 5.0 or now - last_emit >= 3.0:
                progress_callback(idx / source_fps if source_fps else 0.0, percent)
                last_emit = now
                last_percent = percent
        idx += 1
    cap.release()
    return written


def _parse_ffmpeg_time(key: str, value: str) -> float:
    if key in {"out_time_ms", "out_time_us"}:
        try:
            return float(value) / 1_000_000.0
        except ValueError:
            return 0.0
    try:
        hours, minutes, seconds = value.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return 0.0


def read_frame_at_time(video_path: str | Path, time_sec: float) -> tuple[np.ndarray, int]:
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_idx = max(0, int(round(time_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame from: {video_path}")
    return frame, frame_idx


def split_clip(source: str | Path, target: str | Path, start_time: float, end_time: float) -> None:
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, end_time - start_time)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        str(target_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as exc:
        logger.warning("ffmpeg clip split failed, copying normalized video for %s: %s", target_path.name, exc)
        shutil.copyfile(source, target_path)
