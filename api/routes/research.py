from __future__ import annotations

from fastapi import APIRouter

from src.egoflow.config import load_research
from src.egoflow.utils.io import _jsonable

router = APIRouter()


@router.get("/api/research")
def get_research():
    return _jsonable(load_research("research.yaml"))
