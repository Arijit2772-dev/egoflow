from __future__ import annotations

import json
import os
import subprocess
import tempfile
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


_CONTACT_STATE_BY_ID = {
    0: ContactState.NO_CONTACT,
    1: ContactState.SELF_CONTACT,
    2: ContactState.PERSON_CONTACT,
    3: ContactState.PORTABLE_OBJECT,
    4: ContactState.STATIONARY_OBJECT,
}

_CONTACT_STATE_BY_NAME = {state.value: state for state in ContactState}

_PRECISION_LABELS = {"fork", "spoon", "knife", "wine_glass", "water_glass", "cup", "herb"}
_POWER_LABELS = {"bottle", "tray", "plate", "bowl"}
_PINCH_LABELS = {"cloth", "napkin"}


def parse_contact_state(raw: object) -> ContactState:
    """Map 100DOH state (either raw_state_id int or contact_state string) to enum."""
    if isinstance(raw, bool):
        raise ValueError(f"invalid contact state: {raw!r}")
    if isinstance(raw, int):
        if raw not in _CONTACT_STATE_BY_ID:
            raise ValueError(f"unknown 100DOH state id: {raw}")
        return _CONTACT_STATE_BY_ID[raw]
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in _CONTACT_STATE_BY_NAME:
            return _CONTACT_STATE_BY_NAME[key]
        raise ValueError(f"unknown contact_state string: {raw!r}")
    raise ValueError(f"invalid contact state type: {type(raw).__name__}")


def grasp_for_label(label: str) -> GraspType:
    label = (label or "").lower()
    if label in _PRECISION_LABELS:
        return GraspType.PRECISION_GRIP
    if label in _POWER_LABELS:
        return GraspType.POWER_GRIP
    if label in _PINCH_LABELS:
        return GraspType.PINCH
    return GraspType.PRECISION_GRIP


def parse_100doh_payload(
    payload: dict,
    input_objects: Optional[list[dict]] = None,
) -> dict:
    """Convert a raw 100DOH JSON payload into EgoFlow per-side contact dicts.

    input_objects (optional): YOLO-World detections for the same frame, used
    to recover an object label for grasp-type inference.
    """
    if not isinstance(payload, dict):
        raise ValueError("100DOH payload must be an object")

    hands = payload.get("hands") or []
    result = {"left": _empty_side(), "right": _empty_side()}
    for hand in hands:
        side = str(hand.get("side", "")).lower()
        if side not in result:
            continue
        raw_state = hand.get("contact_state")
        if raw_state is None:
            raw_state = hand.get("raw_state_id")
        if raw_state is None:
            continue
        state = parse_contact_state(raw_state)
        contact_bbox_raw = hand.get("in_contact_object_bbox")
        contact_bbox = (
            tuple(int(v) for v in contact_bbox_raw)
            if contact_bbox_raw and len(contact_bbox_raw) == 4
            else None
        )
        label = _label_for_bbox(contact_bbox, input_objects or [])
        grasp = GraspType.NONE if state == ContactState.NO_CONTACT else grasp_for_label(label)
        confidence = float(hand.get("confidence", 0.0))
        result[side] = {
            "contact_state": state,
            "grasp_type": grasp,
            "in_contact_with_bbox": contact_bbox,
            "confidence": max(0.0, min(1.0, confidence)),
        }
    return result


def _label_for_bbox(bbox: Optional[tuple[int, int, int, int]], objects: list[dict]) -> str:
    if bbox is None or not objects:
        return ""
    best, best_iou = "", 0.0
    for obj in objects:
        obox = tuple(obj.get("bbox", ()))
        if len(obox) != 4:
            continue
        iou = bbox_iou(bbox, obox)
        if iou > best_iou:
            best_iou, best = iou, str(obj.get("label", ""))
    return best


def _empty_side() -> dict:
    return {
        "contact_state": ContactState.NO_CONTACT,
        "grasp_type": GraspType.NONE,
        "in_contact_with_bbox": None,
        "confidence": 0.0,
    }


class Contact100DOH:
    def __init__(
        self,
        device: str = "cpu",
        weights_path: Optional[str] = None,
        iou_threshold: float = 0.10,
        enabled: bool = False,
        runner_path: Optional[str] = None,
        timeout_sec: int = 30,
    ):
        self.device = device
        self.weights_path = weights_path
        self.iou_threshold = iou_threshold
        self.enabled = enabled
        self.runner_path = runner_path
        self.timeout_sec = int(timeout_sec)
        self._loaded = False
        self._shim_ready = False

    def load(self) -> None:
        self._shim_ready = False
        if not self.enabled:
            logger.info("100DOH disabled, using hand-object IoU contact fallback.")
        elif not self.runner_path or not Path(self.runner_path).is_file():
            logger.warning("100DOH enabled but runner not found at %s, using fallback.", self.runner_path)
        elif not os.access(self.runner_path, os.X_OK):
            logger.warning("100DOH runner %s is not executable, using fallback.", self.runner_path)
        else:
            self._shim_ready = True
            logger.info("100DOH shim ready at %s", self.runner_path)
        self._loaded = True

    def predict(
        self,
        frame: np.ndarray,
        hands: Optional[dict] = None,
        objects: Optional[list[dict]] = None,
    ) -> dict:
        if self._shim_ready:
            shim_result = self._predict_via_shim(frame, objects or [])
            if shim_result is not None:
                return shim_result
        return self._predict_fallback(hands, objects)

    def unload(self) -> None:
        self._loaded = False
        self._shim_ready = False

    def status(self) -> dict:
        if self._shim_ready:
            return {"name": "100DOH Contact", "mode": "real", "reason": f"shim {self.runner_path} active"}
        if not self.enabled:
            return {"name": "100DOH Contact", "mode": "disabled", "reason": "annotation.enable_100doh=false; IoU+proximity fallback"}
        reason = "runner missing or not executable" if self.runner_path else "runner_path unset"
        return {"name": "100DOH Contact", "mode": "fallback", "reason": f"{reason}; IoU+proximity fallback"}

    def _predict_via_shim(
        self,
        frame: np.ndarray,
        objects: list[dict],
    ) -> Optional[dict]:
        import cv2

        with tempfile.TemporaryDirectory(prefix="egoflow_100doh_") as tmpdir:
            image_path = Path(tmpdir) / "frame.jpg"
            output_path = Path(tmpdir) / "result.json"
            if not cv2.imwrite(str(image_path), frame):
                logger.warning("100DOH shim: failed to write temp image, using fallback.")
                return None
            try:
                proc = subprocess.run(
                    [
                        self.runner_path,
                        "--image",
                        str(image_path),
                        "--output",
                        str(output_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                )
            except subprocess.TimeoutExpired:
                logger.warning("100DOH shim timed out after %ds, using fallback.", self.timeout_sec)
                return None
            except OSError as exc:
                logger.warning("100DOH shim failed to launch: %s. Using fallback.", exc)
                return None
            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip().splitlines()
                tail = stderr[-1] if stderr else ""
                logger.warning(
                    "100DOH shim exited %d (%s), using fallback.",
                    proc.returncode,
                    tail,
                )
                return None
            try:
                payload = json.loads(output_path.read_text())
                return parse_100doh_payload(payload, input_objects=objects)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                logger.warning("100DOH shim output invalid (%s), using fallback.", exc)
                return None

    def _predict_fallback(
        self,
        hands: Optional[dict],
        objects: Optional[list[dict]],
    ) -> dict:
        hands = hands or {}
        objects = objects or []
        result = {"left": _empty_side(), "right": _empty_side()}
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
                    "grasp_type": grasp_for_label(best_obj.get("label", "")),
                    "in_contact_with_bbox": tuple(best_obj["bbox"]),
                    "confidence": min(0.90, max(0.50, best_score)),
                }
        return result
