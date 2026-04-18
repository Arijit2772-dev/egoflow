"""Model provenance helpers.

Writes ``runtime_models.json`` next to ``dataset.json`` so the viewer (and the
dataset consumer) can tell at a glance which components were real, which fell
back, and which were disabled. Each entry looks like::

    {"name": "YOLO-World", "mode": "real", "reason": "..."}

``mode`` is one of ``real``, ``fallback``, ``disabled``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol

VALID_MODES = {"real", "fallback", "disabled"}
PROVENANCE_FILENAME = "runtime_models.json"


class _HasStatus(Protocol):
    def status(self) -> dict: ...


def _empty() -> dict:
    return {"updated_at": None, "models": {}}


def read(out_dir: Path) -> dict:
    path = out_dir / PROVENANCE_FILENAME
    if not path.exists():
        return _empty()
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return _empty()
    if not isinstance(data, dict) or "models" not in data:
        return _empty()
    return data


def write(out_dir: Path, data: dict) -> Path:
    path = out_dir / PROVENANCE_FILENAME
    data = dict(data)
    data["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.write_text(json.dumps(data, indent=2, sort_keys=False))
    return path


def _normalize(entry: dict) -> dict:
    name = str(entry.get("name") or "").strip() or "Unknown"
    mode = str(entry.get("mode") or "").strip().lower()
    if mode not in VALID_MODES:
        mode = "fallback"
    reason = str(entry.get("reason") or "").strip()
    return {"name": name, "mode": mode, "reason": reason}


def record(out_dir: Path, models: Iterable[_HasStatus], phase: str) -> Path:
    """Merge status() reports from the given models into runtime_models.json.

    ``phase`` is a short tag (e.g. "annotate", "describe", "validate") used as
    the key so each phase updates its own slice without clobbering others.
    """
    data = read(out_dir)
    entries = []
    for model in models:
        if model is None:
            continue
        try:
            raw = model.status()
        except Exception as exc:  # pragma: no cover - defensive
            raw = {"name": type(model).__name__, "mode": "fallback", "reason": f"status() raised {exc}"}
        entries.append(_normalize(raw))
    data["models"][phase] = entries
    return write(out_dir, data)


def summary(data: dict) -> dict:
    counts = {mode: 0 for mode in VALID_MODES}
    flat = []
    for phase, entries in (data.get("models") or {}).items():
        for entry in entries:
            norm = _normalize(entry)
            norm["phase"] = phase
            counts[norm["mode"]] += 1
            flat.append(norm)
    return {"counts": counts, "entries": flat}
