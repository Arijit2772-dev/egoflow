from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.egoflow.models.vlm_gemini import GeminiNarrator
from src.egoflow.schema import ClipAnnotation, ContactState, Narration
from src.egoflow.utils.io import read_dataclass, write_json
from src.egoflow.utils.paths import output_root, video_dir
from src.egoflow.utils.progress import emit
from src.egoflow.utils.provenance import record as record_provenance


def run(video_uid: str, config: dict) -> None:
    root = output_root(config)
    out_dir = video_dir(video_uid, config)
    annotations_dir = out_dir / "annotations"
    narrations_dir = out_dir / "narrations"
    keyframes_dir = out_dir / "keyframes"
    narrations_dir.mkdir(parents=True, exist_ok=True)

    narrator = GeminiNarrator(
        model_name=config["describe"]["vlm_model"],
        temperature=float(config["describe"]["temperature"]),
        max_tokens=int(config["describe"]["max_tokens"]),
    )
    emit(
        video_uid,
        4,
        "working",
        "Loading narration module and preparing robotics prompt",
        root,
        phase_name="Describe",
        backed_by="Gemini 1.5 Flash or deterministic fallback",
        progress=10,
    )
    narrator.load()
    record_provenance(out_dir, [narrator], phase="describe")
    try:
        annotation_files = sorted(annotations_dir.glob("clip_*.json"))
        for idx, annotation_file in enumerate(annotation_files, start=1):
            emit(
                video_uid,
                4,
                "working",
                f"Generating narration {idx}/{len(annotation_files)} from detections and contact state",
                root,
                phase_name="Describe",
                backed_by="VLM narration prompt",
                progress=15 + int((idx - 1) / max(1, len(annotation_files)) * 75),
            )
            annotation = read_dataclass(annotation_file, ClipAnnotation)
            image_path = keyframes_dir / f"{annotation_file.stem}.jpg"
            image = Image.open(image_path).convert("RGB")
            prompt = _prompt(annotation)
            context = _fallback_context(annotation)
            parsed = narrator.predict(image, prompt, context)
            narration = Narration(
                segment_id=annotation.segment_id,
                narration=parsed.get("narration") or context["narration"],
                verb=parsed.get("verb") or context["verb"],
                noun=parsed.get("noun") or context["noun"],
                tool=parsed.get("tool") or context["tool"],
                vlm_model=config["describe"]["vlm_model"],
            )
            write_json(narrations_dir / f"{annotation_file.stem}.json", narration)
            emit(
                video_uid,
                4,
                "working",
                f"{annotation.segment_id}: {narration.verb} / {narration.noun}",
                root,
                phase_name="Describe",
                backed_by="Structured verb-noun-tool output",
                progress=15 + int((idx / max(1, len(annotation_files))) * 75),
            )
    finally:
        narrator.unload()
    emit(
        video_uid,
        4,
        "working",
        "Wrote narration files",
        root,
        phase_name="Describe",
        backed_by="File-based handoff",
        progress=95,
    )


def _prompt(annotation: ClipAnnotation) -> str:
    left = annotation.hands.get("left")
    right = annotation.hands.get("right")
    left_contact = left.contact_state.value if left else "missing"
    right_contact = right.contact_state.value if right else "missing"
    objects = ", ".join(f"{obj.label}({obj.confidence:.2f})" for obj in annotation.objects) or "none"
    return (
        "You are annotating an egocentric video frame for robotics training.\n"
        f"Detected: left hand contact={left_contact}, right hand contact={right_contact}.\n"
        f"Objects in frame: {objects}.\n"
        "In ONE sentence, describe the action using format "
        '"{verb} + {noun} + optional {with tool}".\n'
        'Respond in JSON: {"narration": "...", "verb": "...", "noun": "...", "tool": "..."}'
    )


def _fallback_context(annotation: ClipAnnotation) -> dict:
    objects = [{"label": obj.label, "confidence": obj.confidence} for obj in annotation.objects]
    labels = [obj["label"] for obj in objects]
    noun = labels[0] if labels else "object"
    tool = "cloth" if "cloth" in labels and noun != "cloth" else None
    has_contact = any(
        hand is not None and hand.contact_state != ContactState.NO_CONTACT for hand in annotation.hands.values()
    )
    verb = "polish" if tool and noun in {"wine_glass", "water_glass", "cup"} else ("grasp" if has_contact else "arrange")
    narration = f"The person {_third_person(verb)} a {noun.replace('_', ' ')}"
    if tool:
        narration += f" with a {tool.replace('_', ' ')}"
    narration += "."
    return {"objects": objects, "noun": noun, "tool": tool, "verb": verb, "has_contact": has_contact, "narration": narration}


def _third_person(verb: str) -> str:
    if verb.endswith(("sh", "ch", "s", "x", "z", "o")):
        return f"{verb}es"
    if verb.endswith("y") and len(verb) > 1 and verb[-2] not in "aeiou":
        return f"{verb[:-1]}ies"
    return f"{verb}s"
