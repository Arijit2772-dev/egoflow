from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional


class ContactState(str, Enum):
    NO_CONTACT = "no_contact"
    SELF_CONTACT = "self_contact"
    PERSON_CONTACT = "person_contact"
    PORTABLE_OBJECT = "portable_object"
    STATIONARY_OBJECT = "stationary_object"


class GraspType(str, Enum):
    NONE = "none"
    POWER_GRIP = "power_grip"
    PRECISION_GRIP = "precision_grip"
    PINCH = "pinch"
    LATERAL = "lateral"


@dataclass
class VideoMeta:
    video_uid: str
    source_path: str
    duration_sec: float
    fps: float
    resolution: tuple[int, int]
    codec: str
    frame_count: int
    sampled_fps: float
    domain: str = "hospitality_service"


@dataclass
class Segment:
    segment_id: str
    clip_path: str
    start_time: float
    end_time: float
    boundary_confidence: float


@dataclass
class HandAnnotation:
    bbox_2d: tuple[int, int, int, int]
    keypoints_2d: list[tuple[float, float]]
    pose_3d_mano: Optional[list[tuple[float, float, float]]] = None
    contact_state: ContactState = ContactState.NO_CONTACT
    grasp_type: GraspType = GraspType.NONE
    in_contact_with: Optional[str] = None
    detection_confidence: float = 0.0


@dataclass
class ObjectAnnotation:
    obj_id: str
    label: str
    bbox_2d: tuple[int, int, int, int]
    mask_rle: Optional[str] = None
    confidence: float = 0.0


@dataclass
class ClipAnnotation:
    segment_id: str
    keyframe_idx: int
    hands: dict[str, Optional[HandAnnotation]]
    objects: list[ObjectAnnotation]


@dataclass
class TrackFrame:
    segment_id: str
    time_sec: float
    frame_idx: int
    hands: dict[str, Optional[HandAnnotation]]
    objects: list[ObjectAnnotation]


@dataclass
class ClipTrack:
    segment_id: str
    frames: list[TrackFrame]


@dataclass
class Narration:
    segment_id: str
    narration: str
    verb: str
    noun: str
    tool: Optional[str] = None
    vlm_model: str = "gemini-flash-latest"


@dataclass
class QAMetrics:
    avg_detection_confidence: float
    clip_consistency_score: float
    flagged_for_review: bool
    flag_reasons: list[str] = field(default_factory=list)


@dataclass
class ClipRecord:
    segment_id: str
    start_time: float
    end_time: float
    narration: str
    verb: str
    noun: str
    tool: Optional[str]
    hands: dict[str, Optional[HandAnnotation]]
    objects: list[ObjectAnnotation]
    qa_metrics: QAMetrics


@dataclass
class ResearchCitation:
    id: str
    citation: str
    link: str
    contributes: list[str]


@dataclass
class DatasetInfo:
    name: str
    version: str
    schema_compat: str
    created: str
    research_backing: list[ResearchCitation]


@dataclass
class VideoRecord:
    meta: VideoMeta
    segments: list[ClipRecord]


@dataclass
class DatasetManifest:
    dataset_info: DatasetInfo
    videos: list[VideoRecord]


@dataclass
class ValidationReport:
    total_clips: int
    passed: int
    failed: int
    failed_clips: list[dict]
    dataset_quality_score: float
