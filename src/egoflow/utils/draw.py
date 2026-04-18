from __future__ import annotations

from typing import Iterable

import numpy as np


def draw_bbox(frame: np.ndarray, bbox: Iterable[int], color: tuple[int, int, int], label: str = "") -> np.ndarray:
    import cv2

    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(frame, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


def draw_keypoints(frame: np.ndarray, keypoints: list[tuple[float, float]], color: tuple[int, int, int]) -> np.ndarray:
    import cv2

    for x, y in keypoints:
        cv2.circle(frame, (int(x), int(y)), 3, color, -1)
    return frame
