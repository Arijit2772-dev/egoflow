from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.egoflow.schema import ContactState, GraspType
from src.egoflow.utils.logging import get_logger

logger = get_logger(__name__)


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return (dx * dx + dy * dy) ** 0.5


class Contact100DOH:
    def __init__(self, device: str = "cpu", weights_path: Optional[str] = None, iou_threshold: float = 0.10):
        self.device = device
        self.weights_path = weights_path
        self.iou_threshold = iou_threshold
        self._loaded = False

    def load(self) -> None:
        if self.weights_path and Path(self.weights_path).exists():
            logger.info("100DOH weights directory found, heuristic adapter remains active for demo output.")
        else:
            logger.warning("100DOH weights unavailable, using hand-object IoU contact fallback.")
        self._loaded = True

    def predict(
        self,
        frame: np.ndarray,
        hands: Optional[dict] = None,
        objects: Optional[list[dict]] = None,
    ) -> dict:
        hands = hands or {}
        objects = objects or []
        result = {"left": self._empty(), "right": self._empty()}
        for side in ("left", "right"):
            hand = hands.get(side)
            if not hand:
                continue
            best_obj = None
            best_score = -1.0
            for obj in objects:
                hand_box = tuple(hand["bbox"])
                obj_box = tuple(obj["bbox"])
                iou = bbox_iou(hand_box, obj_box)
                distance = bbox_distance(hand_box, obj_box)
                proximity = max(0.0, 1.0 - distance / 120.0)
                score = iou + proximity * 0.35
                if score > best_score:
                    best_score = score
                    best_obj = obj
            if best_obj and best_score >= self.iou_threshold:
                result[side] = {
                    "contact_state": ContactState.PORTABLE_OBJECT,
                    "grasp_type": self._grasp_for_label(best_obj.get("label", "")),
                    "in_contact_with_bbox": tuple(best_obj["bbox"]),
                    "confidence": min(0.90, max(0.50, best_score)),
                }
        return result

    def unload(self) -> None:
        self._loaded = False

    def _empty(self) -> dict:
        return {
            "contact_state": ContactState.NO_CONTACT,
            "grasp_type": GraspType.NONE,
            "in_contact_with_bbox": None,
            "confidence": 0.0,
        }

    def _grasp_for_label(self, label: str) -> GraspType:
        if label in {"fork", "spoon", "knife", "wine_glass", "water_glass", "cup", "herb"}:
            return GraspType.PRECISION_GRIP
        if label in {"bottle", "tray", "plate", "bowl"}:
            return GraspType.POWER_GRIP
        if label in {"cloth", "napkin"}:
            return GraspType.PINCH
        return GraspType.PRECISION_GRIP
