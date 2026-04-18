from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.egoflow.utils.progress import subscribe

router = APIRouter()


@router.websocket("/ws/progress/{uid}")
async def progress_ws(websocket: WebSocket, uid: str):
    await websocket.accept()
    try:
        async for event in subscribe(uid):
            await websocket.send_json(event)
    except WebSocketDisconnect:
        return
