from __future__ import annotations

from typing import Optional

import numpy as np

from src.egoflow.utils.logging import get_logger

logger = get_logger(__name__)


class MediaPipeHands:
    def __init__(self, device: str = "cpu", weights_path: Optional[str] = None):
        self.device = device
        self.weights_path = weights_path
        self._hands = None
        self._mp = None

    def load(self) -> None:
        try:
            import mediapipe as mp

            self._mp = mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as exc:
            logger.warning("MediaPipe unavailable, using geometric hand fallback: %s", exc)
            self._hands = None

    def predict(self, frame: np.ndarray) -> dict:
        if self._hands is None:
            return self._heuristic(frame)

        import cv2

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        hands = {"left": None, "right": None}
        if not result.multi_hand_landmarks:
            return self._heuristic(frame)

        for idx, landmarks in enumerate(result.multi_hand_landmarks):
            side = "left"
            score = 0.75
            if result.multi_handedness and idx < len(result.multi_handedness):
                handedness = result.multi_handedness[idx].classification[0]
                side = handedness.label.lower()
                score = float(handedness.score)
            points = [(float(lm.x * width), float(lm.y * height)) for lm in landmarks.landmark]
            hands[side] = {"keypoints": points, "confidence": score, "source": "mediapipe"}
        return hands

    def unload(self) -> None:
        if self._hands is not None:
            self._hands.close()
        self._hands = None

    def status(self) -> dict:
        if self._hands is not None:
            return {"name": "MediaPipe Hands", "mode": "real", "reason": "MediaPipe initialized"}
        return {"name": "MediaPipe Hands", "mode": "fallback", "reason": "MediaPipe unavailable; skin-contour heuristic"}

    def _heuristic(self, frame: np.ndarray) -> dict:
        detected = self._skin_contour_hands(frame)
        if detected["left"] is not None or detected["right"] is not None:
            return detected

        return {"left": None, "right": None}

    def _skin_contour_hands(self, frame: np.ndarray) -> dict:
        import cv2

        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skin_ycrcb = cv2.inRange(ycrcb, np.array((0, 133, 77)), np.array((255, 173, 127)))
        skin_hsv_a = cv2.inRange(hsv, np.array((0, 45, 40)), np.array((25, 210, 245)))
        skin_hsv_b = cv2.inRange(hsv, np.array((160, 45, 40)), np.array((180, 210, 245)))
        mask = cv2.bitwise_and(skin_ycrcb, cv2.bitwise_or(skin_hsv_a, skin_hsv_b))
        mask[: int(height * 0.28), :] = 0
        mask = cv2.medianBlur(mask, 5)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < max(350, width * height * 0.00015):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if y + h < height * 0.45:
                continue
            bbox = self._endpoint_bbox((x, y, x + w, y + h), width, height)
            bx1, by1, bx2, by2 = bbox
            if bx2 - bx1 < 18 or by2 - by1 < 18:
                continue
            candidates.append(
                {
                    "bbox": bbox,
                    "area": float(area),
                    "center_x": (bx1 + bx2) / 2.0,
                    "center_y": (by1 + by2) / 2.0,
                }
            )

        hands = {"left": None, "right": None}
        for side, side_candidates in {
            "left": [item for item in candidates if item["center_x"] < width * 0.5],
            "right": [item for item in candidates if item["center_x"] >= width * 0.5],
        }.items():
            if not side_candidates:
                continue
            best = max(side_candidates, key=lambda item: item["area"] * (1.0 + item["center_y"] / height))
            bbox = best["bbox"]
            hands[side] = {
                "keypoints": self._bbox_keypoints(bbox, side),
                "confidence": 0.58,
                "source": "skin_contour",
            }
        return hands

    def _endpoint_bbox(self, bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        if w > h * 1.9:
            hand_w = max(70, min(int(h * 1.25), int(w * 0.38)))
            if (x1 + x2) / 2 > width / 2:
                x2 = x1 + hand_w
            else:
                x1 = x2 - hand_w
        elif h > w * 1.9:
            hand_h = max(70, min(int(w * 1.35), int(h * 0.42)))
            y2 = y1 + hand_h
        pad = 14
        return (
            max(0, int(x1 - pad)),
            max(0, int(y1 - pad)),
            min(width - 1, int(x2 + pad)),
            min(height - 1, int(y2 + pad)),
        )

    def _bbox_keypoints(self, bbox: tuple[int, int, int, int], side: str) -> list[tuple[float, float]]:
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        palm_x = x1 + w * 0.5
        wrist_y = y2 - h * 0.12
        tip_y = y1 + h * 0.12
        spread = [-0.34, -0.18, 0.0, 0.18, 0.34]
        points: list[tuple[float, float]] = [(palm_x, wrist_y)]
        for idx, offset in enumerate(spread):
            base_x = palm_x + offset * w
            for joint in range(1, 5):
                alpha = joint / 4.0
                x = base_x + (offset * 0.16 * w * alpha)
                y = wrist_y * (1 - alpha) + tip_y * alpha
                if side == "right":
                    x -= 0.04 * w * alpha
                else:
                    x += 0.04 * w * alpha
                points.append((float(x), float(y)))
        return points[:21]
