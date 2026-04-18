from __future__ import annotations

from datetime import date

from src.egoflow.config import load_research
from src.egoflow.schema import (
    ClipAnnotation,
    ClipRecord,
    DatasetInfo,
    DatasetManifest,
    Narration,
    QAMetrics,
    Segment,
    VideoMeta,
    VideoRecord,
)
from src.egoflow.utils.io import read_dataclass, write_json
from src.egoflow.utils.paths import output_root, video_dir
from src.egoflow.utils.progress import emit


def run(video_uid: str, config: dict) -> None:
    root = output_root(config)
    out_dir = video_dir(video_uid, config)
    emit(
        video_uid,
        5,
        "working",
        "Reading meta, segments, annotations, narrations, and research citations",
        root,
        phase_name="Assemble",
        backed_by="Ego4D-style schema builder",
        progress=20,
    )
    meta = read_dataclass(out_dir / "meta.json", VideoMeta)
    segments = read_dataclass(out_dir / "segments.json", list[Segment])
    research = load_research("research.yaml")

    records: list[ClipRecord] = []
    for idx, segment in enumerate(segments, start=1):
        emit(
            video_uid,
            5,
            "working",
            f"Merging {segment.segment_id} into dataset manifest ({idx}/{len(segments)})",
            root,
            phase_name="Assemble",
            backed_by="Ego4D-style segment record",
            progress=25 + int((idx - 1) / max(1, len(segments)) * 55),
        )
        clip_name = segment.segment_id.replace("seg_", "clip_")
        annotation = read_dataclass(out_dir / "annotations" / f"{clip_name}.json", ClipAnnotation)
        narration = read_dataclass(out_dir / "narrations" / f"{clip_name}.json", Narration)
        avg_conf = _avg_confidence(annotation)
        records.append(
            ClipRecord(
                segment_id=segment.segment_id,
                start_time=segment.start_time,
                end_time=segment.end_time,
                narration=narration.narration,
                verb=narration.verb,
                noun=narration.noun,
                tool=narration.tool,
                hands=annotation.hands,
                objects=annotation.objects,
                qa_metrics=QAMetrics(
                    avg_detection_confidence=avg_conf,
                    clip_consistency_score=0.5,
                    flagged_for_review=False,
                    flag_reasons=[],
                ),
            )
        )

    manifest = DatasetManifest(
        dataset_info=DatasetInfo(
            name=f"EgoFlow-{video_uid}-v1",
            version="1.0",
            schema_compat="Ego4D v1",
            created=date.today().isoformat(),
            research_backing=research,
        ),
        videos=[VideoRecord(meta=meta, segments=records)],
    )
    write_json(out_dir / "dataset.json", manifest)
    emit(
        video_uid,
        5,
        "working",
        f"Wrote dataset.json with {len(records)} segment records and research attribution",
        root,
        phase_name="Assemble",
        backed_by="Ego4D + research.yaml",
        progress=95,
    )


def _avg_confidence(annotation: ClipAnnotation) -> float:
    scores = [obj.confidence for obj in annotation.objects]
    scores.extend(hand.detection_confidence for hand in annotation.hands.values() if hand is not None)
    return round(sum(scores) / len(scores), 3) if scores else 0.0
