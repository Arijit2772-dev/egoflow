from src.egoflow.schema import (
    ClipAnnotation,
    ContactState,
    DatasetInfo,
    DatasetManifest,
    GraspType,
    HandAnnotation,
    ObjectAnnotation,
    QAMetrics,
    ResearchCitation,
    Segment,
    VideoMeta,
    VideoRecord,
    ClipRecord,
)
from src.egoflow.utils.io import read_dataclass, write_json


def test_schema_roundtrip(tmp_path):
    hand = HandAnnotation(
        bbox_2d=(1, 2, 3, 4),
        keypoints_2d=[(1.0, 2.0)] * 21,
        pose_3d_mano=None,
        contact_state=ContactState.PORTABLE_OBJECT,
        grasp_type=GraspType.PRECISION_GRIP,
        in_contact_with="obj_001",
        detection_confidence=0.8,
    )
    obj = ObjectAnnotation("obj_001", "wine_glass", (10, 20, 30, 40), None, 0.91)
    clip = ClipRecord(
        segment_id="seg_001",
        start_time=0.0,
        end_time=3.0,
        narration="The person polishes a wine glass.",
        verb="polish",
        noun="wine_glass",
        tool=None,
        hands={"left": hand, "right": None},
        objects=[obj],
        qa_metrics=QAMetrics(0.85, 0.9, False, []),
    )
    manifest = DatasetManifest(
        dataset_info=DatasetInfo(
            name="test",
            version="1.0",
            schema_compat="Ego4D v1",
            created="2026-04-19",
            research_backing=[ResearchCitation("Shan2020", "100DOH", "https://example.com", ["hands.contact_state"])],
        ),
        videos=[
            VideoRecord(
                meta=VideoMeta("uid", "input.mp4", 3.0, 30.0, (1920, 1080), "h264", 90, 5.0),
                segments=[clip],
            )
        ],
    )
    path = tmp_path / "dataset.json"
    write_json(path, manifest)
    loaded = read_dataclass(path, DatasetManifest)
    assert loaded == manifest


def test_phase_dataclasses_roundtrip(tmp_path):
    segment = Segment("seg_001", "clip_001.mp4", 0.0, 3.0, 0.65)
    annotation = ClipAnnotation(
        segment_id="seg_001",
        keyframe_idx=10,
        hands={"left": None, "right": None},
        objects=[ObjectAnnotation("obj_001", "plate", (1, 2, 3, 4), None, 0.7)],
    )
    write_json(tmp_path / "segments.json", [segment])
    write_json(tmp_path / "annotation.json", annotation)
    assert read_dataclass(tmp_path / "segments.json", list[Segment]) == [segment]
    assert read_dataclass(tmp_path / "annotation.json", ClipAnnotation) == annotation
