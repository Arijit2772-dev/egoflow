from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.egoflow.models.contact_100doh import Contact100DOH, bbox_distance, bbox_iou
from src.egoflow.models.detector_yoloworld import YOLOWorldDetector
from src.egoflow.models.hands_mediapipe import MediaPipeHands
from src.egoflow.models.masks_sam2 import SAM2Masks
from src.egoflow.models.pose_hamer import HaMeRPose
from src.egoflow.schema import ClipAnnotation, ClipTrack, HandAnnotation, ObjectAnnotation, Segment, TrackFrame
from src.egoflow.utils.io import read_dataclass, write_json
from src.egoflow.utils.paths import output_root, video_dir, weights_root
from src.egoflow.utils.progress import emit
from src.egoflow.utils.video_io import read_frame_at_time


def run(video_uid: str, config: dict) -> None:
    root = output_root(config)
    out_dir = video_dir(video_uid, config)
    annotations_dir = out_dir / "annotations"
    tracks_dir = out_dir / "tracks"
    keyframes_dir = out_dir / "keyframes"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    segments = read_dataclass(out_dir / "segments.json", list[Segment])
    device = config["annotation"]["device"]
    min_object_confidence = float(config["annotation"]["min_object_confidence"])
    root_weights = weights_root(config)

    hands_model = MediaPipeHands(device=device)
    detector = YOLOWorldDetector(
        device=device,
        weights_path=str(root_weights / "yolo_world" / "yolov8s-world.pt"),
        vocab=list(config["object_vocabulary"]),
        min_confidence=float(config["annotation"]["min_object_confidence"]),
    )
    contact_model = Contact100DOH(device=device, weights_path=str(root_weights / "100doh"))
    pose_model = HaMeRPose(device=device, weights_path=str(root_weights / "hamer")) if config["annotation"]["enable_hamer"] else None
    mask_model = SAM2Masks(device=device, weights_path=str(root_weights / "sam2")) if config["annotation"]["enable_sam2"] else None

    emit(
        video_uid,
        3,
        "working",
        "Loading hand, object, contact, pose, and mask modules once",
        root,
        phase_name="Annotate",
        backed_by="MediaPipe/100DOH + YOLO-World + optional HaMeR/SAM2",
        progress=8,
    )
    for model in [hands_model, detector, contact_model, pose_model, mask_model]:
        if model is not None:
            model.load()

    try:
        for idx, segment in enumerate(segments, start=1):
            emit(
                video_uid,
                3,
                "working",
                f"Annotating {segment.segment_id} ({idx}/{len(segments)}): hands, objects, contact, grasp",
                root,
                phase_name="Annotate",
                backed_by="Hand-object interaction supervision",
                progress=10 + int((idx - 1) / max(1, len(segments)) * 78),
            )
            midpoint = max(0.0, (segment.end_time - segment.start_time) / 2.0)
            frame, keyframe_idx = read_frame_at_time(segment.clip_path, midpoint)
            cv2.imwrite(str(keyframes_dir / f"{_clip_name(segment.segment_id)}.jpg"), frame)

            hands, objects = _annotate_frame(
                frame,
                hands_model,
                detector,
                contact_model,
                pose_model,
                mask_model,
                min_object_confidence,
            )

            annotation = ClipAnnotation(
                segment_id=segment.segment_id,
                keyframe_idx=keyframe_idx,
                hands=hands,
                objects=objects,
            )
            write_json(annotations_dir / f"{_clip_name(segment.segment_id)}.json", annotation)

            track = _annotate_track(
                segment,
                config,
                hands_model,
                detector,
                contact_model,
                pose_model,
                mask_model,
                min_object_confidence,
            )
            write_json(tracks_dir / f"{_clip_name(segment.segment_id)}.json", track)
            emit(
                video_uid,
                3,
                "working",
                f"{segment.segment_id}: wrote keyframe annotation and {len(track.frames)} overlay track frames",
                root,
                phase_name="Annotate",
                backed_by="YOLO-World plus contact fallback when 100DOH weights are absent",
                progress=12 + int((idx / max(1, len(segments))) * 78),
            )
    finally:
        for model in [hands_model, detector, contact_model, pose_model, mask_model]:
            if model is not None:
                model.unload()
    emit(
        video_uid,
        3,
        "working",
        "Unloaded models and completed annotation files",
        root,
        phase_name="Annotate",
        backed_by="File-based handoff",
        progress=95,
    )


def _clip_name(segment_id: str) -> str:
    return segment_id.replace("seg_", "clip_")


def _annotate_track(
    segment: Segment,
    config: dict,
    hands_model,
    detector,
    contact_model,
    pose_model,
    mask_model,
    min_object_confidence: float,
) -> ClipTrack:
    cap = cv2.VideoCapture(segment.clip_path)
    if not cap.isOpened():
        return ClipTrack(segment_id=segment.segment_id, frames=[])
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or config["input"]["normalize_fps"])
    duration = max(0.0, segment.end_time - segment.start_time)
    track_fps = max(0.25, float(config["annotation"].get("track_sample_fps", 2.0)))
    step_frames = max(1, int(round(source_fps / track_fps)))
    frames: list[TrackFrame] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step_frames == 0:
            local_time = min(duration, idx / source_fps)
            hands, objects = _annotate_frame(
                frame,
                hands_model,
                detector,
                contact_model,
                pose_model,
                mask_model,
                min_object_confidence,
            )
            frames.append(
                TrackFrame(
                    segment_id=segment.segment_id,
                    time_sec=round(segment.start_time + local_time, 3),
                    frame_idx=idx,
                    hands=hands,
                    objects=objects,
                )
            )
        idx += 1
    cap.release()
    return ClipTrack(segment_id=segment.segment_id, frames=frames)


def _annotate_frame(
    frame,
    hands_model,
    detector,
    contact_model,
    pose_model,
    mask_model,
    min_object_confidence: float,
) -> tuple[dict, list[ObjectAnnotation]]:
    raw_hands = hands_model.predict(frame)
    hand_payload = _hands_with_bboxes(raw_hands, frame.shape[1], frame.shape[0])

    raw_objects = detector.predict(frame)["objects"]
    objects = _object_annotations(raw_objects)
    objects = [obj for obj in objects if obj.confidence >= min_object_confidence]
    objects = _filter_objects_near_hands(objects, hand_payload, frame.shape[1], frame.shape[0], frame)
    object_payload = [{"obj_id": obj.obj_id, "label": obj.label, "bbox": obj.bbox_2d, "confidence": obj.confidence} for obj in objects]

    contact = contact_model.predict(frame, hand_payload, object_payload)
    pose = pose_model.predict(frame, hand_payload) if pose_model else {"left": None, "right": None}
    masks = mask_model.predict(frame, object_payload) if mask_model else {}

    for obj in objects:
        obj.mask_rle = masks.get(obj.obj_id)

    hands = {}
    for side in ("left", "right"):
        payload = hand_payload.get(side)
        if payload is None:
            hands[side] = None
            continue
        contact_info = contact.get(side, {})
        in_contact_with = _match_contact_object(contact_info.get("in_contact_with_bbox"), objects)
        detection_conf = max(
            float(payload.get("confidence", 0.0)),
            float(contact_info.get("confidence", 0.0)),
        )
        hands[side] = HandAnnotation(
            bbox_2d=tuple(payload["bbox"]),
            keypoints_2d=[tuple(point) for point in payload["keypoints"]],
            pose_3d_mano=pose.get(side),
            contact_state=contact_info.get("contact_state"),
            grasp_type=contact_info.get("grasp_type"),
            in_contact_with=in_contact_with,
            detection_confidence=round(detection_conf, 3),
        )
    return hands, objects


def _hands_with_bboxes(raw_hands: dict, width: int, height: int) -> dict:
    output = {"left": None, "right": None}
    for side in ("left", "right"):
        item = raw_hands.get(side)
        if not item:
            continue
        keypoints = item["keypoints"]
        xs = [point[0] for point in keypoints]
        ys = [point[1] for point in keypoints]
        pad = 16
        bbox = (
            max(0, int(min(xs) - pad)),
            max(0, int(min(ys) - pad)),
            min(width - 1, int(max(xs) + pad)),
            min(height - 1, int(max(ys) + pad)),
        )
        output[side] = {
            "bbox": bbox,
            "keypoints": keypoints,
            "confidence": float(item.get("confidence", 0.0)),
        }
    return output


def _object_annotations(raw_objects: list[dict]) -> list[ObjectAnnotation]:
    objects = []
    for idx, raw in enumerate(raw_objects, start=1):
        objects.append(
            ObjectAnnotation(
                obj_id=f"obj_{idx:03d}",
                label=str(raw.get("label", "object")),
                bbox_2d=tuple(int(v) for v in raw.get("bbox", (0, 0, 1, 1))),
                confidence=round(float(raw.get("confidence", 0.0)), 3),
            )
        )
    return objects


def _filter_objects_near_hands(
    objects: list[ObjectAnnotation],
    hands: dict,
    width: int,
    height: int,
    frame,
) -> list[ObjectAnnotation]:
    hand_boxes = [tuple(hand["bbox"]) for hand in hands.values() if hand is not None]
    if not objects or not hand_boxes:
        return objects

    max_distance = max(120.0, ((width * width + height * height) ** 0.5) * 0.07)
    kept = []
    for obj in objects:
        if not _passes_label_color_check(frame, obj):
            continue
        close = any(
            bbox_iou(obj.bbox_2d, hand_box) > 0.0 or bbox_distance(obj.bbox_2d, hand_box) <= max_distance
            for hand_box in hand_boxes
        )
        if close:
            kept.append(obj)

    if not kept:
        confident = [obj for obj in objects if obj.confidence >= 0.65]
        kept = confident[:1]

    for idx, obj in enumerate(kept, start=1):
        obj.obj_id = f"obj_{idx:03d}"
    return kept


def _passes_label_color_check(frame, obj: ObjectAnnotation) -> bool:
    if obj.label not in {"cloth", "herb"}:
        return True
    x1, y1, x2, y2 = obj.bbox_2d
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    if obj.label == "cloth":
        mask = cv2.inRange(hsv, np.array((18, 80, 60)), np.array((45, 255, 255)))
        return float(mask.mean() / 255.0) >= 0.04
    mask = cv2.inRange(hsv, np.array((45, 45, 25)), np.array((95, 255, 160)))
    return float(mask.mean() / 255.0) >= 0.06


def _match_contact_object(contact_bbox: tuple[int, int, int, int] | None, objects: list[ObjectAnnotation]) -> str | None:
    if contact_bbox is None:
        return None
    best = None
    best_iou = 0.0
    for obj in objects:
        score = bbox_iou(tuple(contact_bbox), obj.bbox_2d)
        if score > best_iou:
            best_iou = score
            best = obj.obj_id
    return best
