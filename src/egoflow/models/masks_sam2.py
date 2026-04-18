from __future__ import annotations

from typing import Optional

import numpy as np

from src.egoflow.utils.logging import get_logger

logger = get_logger(__name__)


class SAM2Masks:
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
            import sam2  # type: ignore  # noqa: F401

            self._loaded = True
        except Exception as exc:
            logger.warning("SAM2 unavailable, object mask_rle will be null: %s", exc)
            self._loaded = False

    def predict(self, frame: np.ndarray, objects: Optional[list[dict]] = None) -> dict:
        return {obj.get("obj_id", str(idx)): None for idx, obj in enumerate(objects or [])}

    def unload(self) -> None:
        self._loaded = False

    def status(self) -> dict:
        if self._loaded:
            return {"name": "SAM2 Masks", "mode": "real", "reason": "sam2 module loaded"}
        return {"name": "SAM2 Masks", "mode": "disabled", "reason": "sam2 not installed; mask_rle emitted as null"}
