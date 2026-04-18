from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PIL import Image

from src.egoflow.utils.logging import get_logger

logger = get_logger(__name__)


class CLIPQA:
    def __init__(self, device: str = "cpu", weights_path: Optional[str] = None):
        self.device = device
        self.weights_path = weights_path
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def load(self) -> None:
        if os.getenv("EGOFLOW_ENABLE_CLIP", "0") != "1":
            logger.warning("CLIP real inference disabled, using lexical consistency fallback.")
            return
        try:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=self.device)
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
        except Exception as exc:
            logger.warning("CLIP unavailable, using lexical consistency fallback: %s", exc)
            self._model = None

    def score(self, image: Image.Image, text: str, labels: Optional[list[str]] = None) -> float:
        if self._model is None:
            return self._lexical_score(text, labels or [])
        try:
            import torch

            image_input = self._preprocess(image).unsqueeze(0).to(self.device)
            text_input = self._tokenizer([text]).to(self.device)
            with torch.no_grad():
                image_features = self._model.encode_image(image_input)
                text_features = self._model.encode_text(text_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                score = (image_features @ text_features.T).item()
            return float(max(0.0, min(1.0, (score + 1.0) / 2.0)))
        except Exception as exc:
            logger.warning("CLIP scoring failed, using lexical fallback: %s", exc)
            return self._lexical_score(text, labels or [])

    def unload(self) -> None:
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def status(self) -> dict:
        if self._model is not None:
            return {"name": "CLIP QA", "mode": "real", "reason": "open_clip ViT-B-32 (OpenAI) loaded"}
        reason = "EGOFLOW_ENABLE_CLIP!=1" if os.getenv("EGOFLOW_ENABLE_CLIP", "0") != "1" else "open_clip load failed"
        return {"name": "CLIP QA", "mode": "fallback", "reason": f"{reason}; lexical consistency score"}

    def _lexical_score(self, text: str, labels: list[str]) -> float:
        normalized = text.lower().replace(" ", "_")
        hits = sum(1 for label in labels if label.lower() in normalized)
        return float(min(0.95, 0.35 + hits * 0.25))
