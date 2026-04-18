from __future__ import annotations

import re
from pathlib import Path


def project_root(config: dict) -> Path:
    return Path(config.get("_paths", {}).get("project_root", Path.cwd())).resolve()


def resolve_project_path(config: dict, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root(config) / path).resolve()


def safe_uid(input_path: str | Path) -> str:
    stem = Path(input_path).stem
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    return cleaned or "video"


def output_root(config: dict) -> Path:
    return resolve_project_path(config, config["paths"]["output_root"])


def weights_root(config: dict) -> Path:
    return resolve_project_path(config, config["paths"]["weights_root"])


def video_dir(video_uid: str, config: dict) -> Path:
    return output_root(config) / video_uid
