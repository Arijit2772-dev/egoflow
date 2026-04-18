from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from src.egoflow.utils.logging import get_logger

logger = get_logger(__name__)


class YOLOWorldDetector:
    def __init__(
        self,
        device: str = "cpu",
        weights_path: Optional[str] = None,
        vocab: Optional[list[str]] = None,
        min_confidence: float = 0.4,
    ):
        self.device = device
        self.weights_path = weights_path
        self.vocab = vocab or []
        self.min_confidence = min_confidence
        self._model = None

    def load(self) -> None:
        use_real = os.getenv("EGOFLOW_USE_REAL_YOLO", "0") == "1"
        model_path = self.weights_path or "yolov8s-world.pt"
        if self.weights_path and Path(self.weights_path).exists():
            use_real = True
        if not use_real:
            logger.warning("YOLO-World real inference disabled, using object fallback.")
            return
        try:
            from ultralytics import YOLOWorld

            self._model = YOLOWorld(model_path)
            if self.vocab:
                self._model.set_classes(self.vocab)
        except Exception as exc:
            logger.warning("YOLO-World unavailable, using object fallback: %s", exc)
            self._model = None

    def predict(self, frame: np.ndarray) -> dict:
        if self._model is None:
            return {"objects": self._heuristic_objects(frame)}
        try:
            results = self._model.predict(frame, conf=self.min_confidence, device=self.device, verbose=False)
            objects = []
            for result in results:
                names = result.names
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    xyxy = box.xyxy.cpu().numpy().astype(int).tolist()[0]
                    cls_idx = int(box.cls.item()) if box.cls is not None else -1
                    label = str(names.get(cls_idx, self.vocab[0] if self.vocab else "object"))
                    conf = float(box.conf.item()) if box.conf is not None else 0.0
                    objects.append({"label": label, "bbox": tuple(xyxy), "confidence": conf, "source": "yolo_world"})
            if objects:
                return {"objects": objects}
        except Exception as exc:
            logger.warning("YOLO-World prediction failed, using object fallback: %s", exc)
        return {"objects": self._heuristic_objects(frame)}

    def unload(self) -> None:
        self._model = None

    def _heuristic_objects(self, frame: np.ndarray) -> list[dict]:
        detected = self._color_and_contour_objects(frame)
        if detected:
            return detected

        return []

    def _color_and_contour_objects(self, frame: np.ndarray) -> list[dict]:
        import cv2

        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        specs = [
            ("cloth", (18, 100, 70), (45, 255, 255), 0.72),
            ("herb", (45, 55, 25), (95, 255, 140), 0.39),
        ]
        objects: list[dict] = []
        for label, lower, upper, confidence in specs:
            if label not in self.vocab:
                continue
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask[: int(height * 0.24), :] = 0
            mask = cv2.medianBlur(mask, 5)
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < max(500, width * height * 0.0002):
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if w * h > width * height * 0.18:
                    continue
                if x < width * 0.03 or x + w > width * 0.97 or y + h > height * 0.97:
                    continue
                if w < 18 or h < 18:
                    continue
                objects.append(
                    {
                        "label": label,
                        "bbox": self._pad_bbox((x, y, x + w, y + h), width, height, 8),
                        "confidence": confidence,
                        "source": "color_contour",
                    }
                )

        if not objects:
            objects.extend(self._edge_objects(frame))
        return self._nms(objects)

    def _edge_objects(self, frame: np.ndarray) -> list[dict]:
        import cv2

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 140)
        edges[: int(height * 0.28), :] = 0
        edges[:, : int(width * 0.02)] = 0
        edges[:, int(width * 0.98) :] = 0
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects = []
        generic = "plate" if "plate" in self.vocab else (self.vocab[0] if self.vocab else "object")
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < max(900, width * height * 0.00035):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > width * height * 0.14 or w < 30 or h < 30:
                continue
            objects.append(
                {
                    "label": generic,
                    "bbox": self._pad_bbox((x, y, x + w, y + h), width, height, 10),
                    "confidence": 0.45,
                    "source": "edge_contour",
                }
            )
        return objects

    def _pad_bbox(self, bbox: tuple[int, int, int, int], width: int, height: int, pad: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        return (
            max(0, int(x1 - pad)),
            max(0, int(y1 - pad)),
            min(width - 1, int(x2 + pad)),
            min(height - 1, int(y2 + pad)),
        )

    def _nms(self, objects: list[dict]) -> list[dict]:
        if not objects:
            return []
        kept: list[dict] = []
        for obj in sorted(objects, key=lambda item: item["confidence"], reverse=True):
            if all(self._iou(obj["bbox"], other["bbox"]) < 0.35 for other in kept):
                kept.append(obj)
        return kept[:4]

    def _iou(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(area_a + area_b - inter)
