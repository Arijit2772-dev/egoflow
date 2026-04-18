from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.egoflow.config import load_config
from src.egoflow.utils.paths import output_root, video_dir

router = APIRouter()


@router.get("/api/videos")
def list_videos() -> list[str]:
    root = output_root(load_config())
    if not root.exists():
        return []
    processed = [path for path in root.iterdir() if path.is_dir() and (path / "dataset.json").exists()]
    return [path.name for path in sorted(processed, key=lambda item: (item / "dataset.json").stat().st_mtime)]


@router.get("/api/video/{uid}/dataset.json")
def get_dataset(uid: str):
    path = video_dir(uid, load_config()) / "dataset.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="dataset.json not found")
    return FileResponse(path, media_type="application/json", filename="dataset.json")


@router.get("/api/video/{uid}/validation")
def get_validation(uid: str):
    path = video_dir(uid, load_config()) / "validation_report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="validation_report.json not found")
    return FileResponse(path, media_type="application/json")
