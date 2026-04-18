from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.egoflow.utils.io import read_json, write_json


_queues: dict[str, list[asyncio.Queue]] = defaultdict(list)


def emit(
    video_uid: str,
    phase: int,
    status: str,
    message: str,
    root: str | Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    event = {
        "video_uid": video_uid,
        "phase": phase,
        "status": status,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    if root is not None:
        progress_dir = Path(root) / video_uid
        write_json(progress_dir / "progress.json", event)
        history_path = progress_dir / "progress_history.json"
        try:
            history = read_json(history_path) if history_path.exists() else []
        except Exception:
            history = []
        history.append(event)
        write_json(history_path, history[-500:])
    for queue in list(_queues.get(video_uid, [])):
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass
    return event


async def subscribe(video_uid: str):
    queue: asyncio.Queue = asyncio.Queue(maxsize=32)
    _queues[video_uid].append(queue)
    try:
        while True:
            yield await queue.get()
    finally:
        _queues[video_uid].remove(queue)
