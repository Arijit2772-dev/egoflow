from __future__ import annotations

import json
import os
import re
from typing import Optional

from PIL import Image

from src.egoflow.utils.logging import get_logger

logger = get_logger(__name__)


class GeminiNarrator:
    def __init__(self, model_name: str = "gemini-flash-latest", temperature: float = 0.2, max_tokens: int = 80):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._model = None

    def load(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY missing, using narration template fallback.")
            return
        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(self.model_name)
        except Exception as exc:
            logger.warning("Gemini client unavailable, using narration fallback: %s", exc)
            self._model = None

    def predict(self, image: Image.Image, prompt: str, fallback_context: Optional[dict] = None) -> dict:
        if self._model is None:
            return self._fallback(fallback_context or {})
        try:
            response = self._model.generate_content(
                [prompt, image],
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "response_mime_type": "application/json",
                },
            )
            text = response.text or "{}"
            return self._parse_json(text)
        except Exception as exc:
            logger.warning("Gemini narration failed, using fallback: %s", exc)
            return self._fallback(fallback_context or {})

    def unload(self) -> None:
        self._model = None

    def _parse_json(self, text: str) -> dict:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.S)
            parsed = json.loads(match.group(0)) if match else {}
        return {
            "narration": str(parsed.get("narration", "")).strip(),
            "verb": str(parsed.get("verb", "")).strip(),
            "noun": str(parsed.get("noun", "")).strip(),
            "tool": parsed.get("tool") or None,
        }

    def _fallback(self, context: dict) -> dict:
        objects = context.get("objects") or []
        labels = [obj.get("label", "object") for obj in objects]
        noun = context.get("noun") or (labels[0] if labels else "object")
        tool = "cloth" if "cloth" in labels and noun != "cloth" else None
        contact = context.get("has_contact", False)
        verb = "polish" if tool and noun in {"wine_glass", "water_glass", "cup"} else ("grasp" if contact else "arrange")
        narration = f"The person {_third_person(verb)} a {noun.replace('_', ' ')}"
        if tool:
            narration += f" with a {tool.replace('_', ' ')}"
        narration += "."
        return {"narration": narration, "verb": verb, "noun": noun, "tool": tool}


def _third_person(verb: str) -> str:
    if verb.endswith(("sh", "ch", "s", "x", "z", "o")):
        return f"{verb}es"
    if verb.endswith("y") and len(verb) > 1 and verb[-2] not in "aeiou":
        return f"{verb[:-1]}ies"
    return f"{verb}s"
