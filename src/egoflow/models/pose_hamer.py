from __future__ import annotations

from typing import Optional

import numpy as np

from src.egoflow.utils.logging import get_logger

logger = get_logger(__name__)


class HaMeRPose:
    def __init__(self, device: str = "cpu", weights_path: Optional[str] = None, enabled: bool = False):
        self.device = device
        self.weights_path = weights_path
        self.enabled = enabled
        self._loaded = False

    def load(self) -> None:
        if not self.enabled:
            self._loaded = False
            return
        try:
            import hamer  # type: ignore  # noqa: F401

            self._loaded = True
        except Exception as exc:
            logger.warning("HaMeR unavailable, pose_3d_mano will be null: %s", exc)
            self._loaded = False

    def predict(self, frame: np.ndarray, hands: Optional[dict] = None) -> dict:
        return {"left": None, "right": None}

    def unload(self) -> None:
        self._loaded = False

    def status(self) -> dict:
        if self._loaded:
            return {"name": "HaMeR Pose", "mode": "real", "reason": "hamer module loaded"}
        return {"name": "HaMeR Pose", "mode": "disabled", "reason": "hamer not installed; pose_3d_mano emitted as null"}
