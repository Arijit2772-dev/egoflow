from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse

from src.egoflow.config import load_config
from src.egoflow.pipeline import run_pipeline
from src.egoflow.utils.io import read_json
from src.egoflow.utils.paths import output_root, safe_uid, video_dir
from src.egoflow.utils.progress import emit

router = APIRouter()


@router.get("/api/video/{uid}/meta")
def get_meta(uid: str):
    path = video_dir(uid, load_config()) / "meta.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="meta.json not found")
    return read_json(path)


@router.get("/api/video/{uid}/segments")
def get_segments(uid: str):
    path = video_dir(uid, load_config()) / "segments.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="segments.json not found")
    return read_json(path)


@router.get("/api/video/{uid}/tracks/{sid}.json")
def get_track(uid: str, sid: str):
    clip_name = sid if sid.startswith("clip_") else sid.replace("seg_", "clip_")
    path = video_dir(uid, load_config()) / "tracks" / f"{clip_name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="track not found")
    return read_json(path)


@router.get("/api/video/{uid}/progress")
def get_progress(uid: str):
    out_dir = video_dir(uid, load_config())
    current_path = out_dir / "progress.json"
    history_path = out_dir / "progress_history.json"
    if not current_path.exists():
        current = {"video_uid": uid, "phase": 0, "status": "queued", "message": "Waiting to start"}
    else:
        current = read_json(current_path)
    history = read_json(history_path) if history_path.exists() else [current]
    return {"current": current, "history": history[-200:]}


@router.get("/stream/{uid}/normalized.mp4")
def stream_normalized(uid: str):
    path = video_dir(uid, load_config()) / "normalized.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="normalized video not found")
    return FileResponse(path, media_type="video/mp4")


@router.get("/stream/{uid}/clip/{sid}.mp4")
def stream_clip(uid: str, sid: str):
    clip_name = sid if sid.startswith("clip_") else sid.replace("seg_", "clip_")
    path = video_dir(uid, load_config()) / "clips" / f"{clip_name}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="clip not found")
    return FileResponse(path, media_type="video/mp4")


@router.post("/api/upload")
async def upload_video(request: Request, background_tasks: BackgroundTasks, filename: str = "upload.mp4"):
    config = load_config()
    upload_dir = output_root(config) / "_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    target = _unique_upload_path(upload_dir, filename)
    bytes_written = 0
    with target.open("wb") as handle:
        async for chunk in request.stream():
            if chunk:
                handle.write(chunk)
                bytes_written += len(chunk)

    if bytes_written == 0:
        raise HTTPException(status_code=400, detail="Uploaded file was empty")

    uid = safe_uid(target)
    emit(uid, 0, "queued", f"Uploaded {target.name}; waiting to process", output_root(config))
    background_tasks.add_task(_run_uploaded_pipeline, str(target))
    return {
        "accepted": True,
        "filename": target.name,
        "video_uid": uid,
        "bytes": bytes_written,
        "message": "Upload complete. Processing started.",
    }


def _unique_upload_path(upload_dir: Path, filename: str) -> Path:
    raw = Path(filename or "upload.mp4").name
    suffix = Path(raw).suffix or ".mp4"
    stem = safe_uid(Path(raw).stem)
    timestamp = int(time.time())
    return upload_dir / f"{stem}_{timestamp}{suffix}"


def _run_uploaded_pipeline(path: str) -> None:
    config = load_config()
    uid = safe_uid(path)
    try:
        run_pipeline(path, None, False, "config.yaml")
    except Exception as exc:
        emit(uid, 0, "failed", f"Pipeline failed: {exc}", output_root(config))
