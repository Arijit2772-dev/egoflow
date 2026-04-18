from __future__ import annotations

from PIL import Image

from src.egoflow.models.clip_qa import CLIPQA
from src.egoflow.schema import DatasetManifest, ValidationReport
from src.egoflow.utils.io import read_dataclass, write_json
from src.egoflow.utils.paths import output_root, video_dir
from src.egoflow.utils.progress import emit


def run(video_uid: str, config: dict) -> None:
    root = output_root(config)
    out_dir = video_dir(video_uid, config)
    manifest = read_dataclass(out_dir / "dataset.json", DatasetManifest)
    clip_qa = CLIPQA(device=config["annotation"]["device"])
    emit(
        video_uid,
        6,
        "working",
        "Loading QA checks: schema rules, confidence thresholds, CLIP consistency when enabled",
        root,
        phase_name="Validate",
        backed_by="Rule checks + CLIP semantic QA",
        progress=10,
    )
    clip_qa.load()

    failed = []
    total = 0
    passed = 0
    min_conf = float(config["validate"]["min_avg_confidence"])
    min_consistency = float(config["validate"]["min_clip_consistency"])

    try:
        for video in manifest.videos:
            previous_end = -1.0
            for clip_index, clip in enumerate(video.segments, start=1):
                emit(
                    video_uid,
                    6,
                    "working",
                    f"Validating {clip.segment_id} ({clip_index}/{len(video.segments)})",
                    root,
                    phase_name="Validate",
                    backed_by="Rule checks + CLIP fallback",
                    progress=15 + int((clip_index - 1) / max(1, len(video.segments)) * 72),
                )
                total += 1
                reasons = []
                if clip.start_time >= clip.end_time:
                    reasons.append("start_time must be less than end_time")
                if clip.start_time < previous_end:
                    reasons.append("segment overlaps previous segment")
                previous_end = max(previous_end, clip.end_time)
                if not clip.narration or not clip.verb or not clip.noun:
                    reasons.append("narration fields are incomplete")
                if not clip.objects:
                    reasons.append("no objects detected")
                if all(hand is None for hand in clip.hands.values()):
                    reasons.append("no hands detected")

                labels = [obj.label for obj in clip.objects]
                image_path = out_dir / "keyframes" / f"{clip.segment_id.replace('seg_', 'clip_')}.jpg"
                if image_path.exists():
                    consistency = clip_qa.score(Image.open(image_path).convert("RGB"), clip.narration, labels)
                else:
                    consistency = 0.35
                clip.qa_metrics.clip_consistency_score = round(consistency, 3)

                if clip.qa_metrics.avg_detection_confidence < min_conf:
                    reasons.append("average detection confidence below threshold")
                if consistency < min_consistency:
                    reasons.append("caption-frame consistency below threshold")

                clip.qa_metrics.flagged_for_review = bool(reasons)
                clip.qa_metrics.flag_reasons = reasons
                if reasons:
                    failed.append(
                        {
                            "segment_id": clip.segment_id,
                            "clip_consistency_score": round(consistency, 3),
                            "avg_detection_confidence": clip.qa_metrics.avg_detection_confidence,
                            "reasons": reasons,
                        }
                    )
                else:
                    passed += 1
    finally:
        clip_qa.unload()

    report = ValidationReport(
        total_clips=total,
        passed=passed,
        failed=len(failed),
        failed_clips=failed,
        dataset_quality_score=round(passed / total, 3) if total else 0.0,
    )
    write_json(out_dir / "dataset.json", manifest)
    write_json(out_dir / "validation_report.json", report)
    emit(
        video_uid,
        6,
        "working",
        f"Validation complete: {passed}/{total} clips passed, quality score {report.dataset_quality_score:.2f}",
        root,
        phase_name="Validate",
        backed_by="validation_report.json",
        progress=95,
    )
