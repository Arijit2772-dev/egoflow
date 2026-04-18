from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.egoflow.schema import ResearchCitation


REQUIRED_CONFIG_KEYS = {
    "input",
    "segmentation",
    "annotation",
    "object_vocabulary",
    "verb_vocabulary",
    "describe",
    "validate",
    "api",
    "paths",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    return _project_root() / candidate


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    config_path = _resolve(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    missing = REQUIRED_CONFIG_KEYS.difference(config)
    if missing:
        raise ValueError(f"Config missing required keys: {sorted(missing)}")
    config["_paths"] = {
        "config_path": str(config_path),
        "project_root": str(config_path.parent),
    }
    return config


def load_research(path: str | Path = "research.yaml") -> list[ResearchCitation]:
    research_path = _resolve(path)
    if not research_path.exists():
        raise FileNotFoundError(f"Research file not found: {research_path}")
    with research_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    citations: list[ResearchCitation] = []
    for paper_id, paper in (raw.get("papers") or {}).items():
        citation = (
            f"{paper.get('short_name', paper_id)} - {paper.get('title', '')}; "
            f"{paper.get('authors', '')}; {paper.get('venue', '')}."
        )
        citations.append(
            ResearchCitation(
                id=str(paper_id),
                citation=citation,
                link=str(paper.get("link", "")),
                contributes=list(paper.get("contributes", [])),
            )
        )
    return citations
