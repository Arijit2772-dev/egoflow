"""Unit tests for 100DOH JSON parsing — no external env required."""

import pytest

from src.egoflow.models.contact_100doh import (
    grasp_for_label,
    parse_100doh_payload,
    parse_contact_state,
)
from src.egoflow.schema import ContactState, GraspType


def test_parse_contact_state_from_id():
    assert parse_contact_state(0) == ContactState.NO_CONTACT
    assert parse_contact_state(1) == ContactState.SELF_CONTACT
    assert parse_contact_state(2) == ContactState.PERSON_CONTACT
    assert parse_contact_state(3) == ContactState.PORTABLE_OBJECT
    assert parse_contact_state(4) == ContactState.STATIONARY_OBJECT


def test_parse_contact_state_from_string():
    assert parse_contact_state("portable_object") == ContactState.PORTABLE_OBJECT
    assert parse_contact_state("NO_CONTACT") == ContactState.NO_CONTACT


def test_parse_contact_state_invalid():
    with pytest.raises(ValueError):
        parse_contact_state(99)
    with pytest.raises(ValueError):
        parse_contact_state("floating")
    with pytest.raises(ValueError):
        parse_contact_state(True)


def test_grasp_for_label():
    assert grasp_for_label("fork") == GraspType.PRECISION_GRIP
    assert grasp_for_label("bottle") == GraspType.POWER_GRIP
    assert grasp_for_label("cloth") == GraspType.PINCH
    assert grasp_for_label("unknown") == GraspType.PRECISION_GRIP


def test_parse_payload_portable_with_label_lookup():
    payload = {
        "hands": [
            {
                "side": "right",
                "bbox": [100, 100, 200, 200],
                "confidence": 0.91,
                "contact_state": "portable_object",
                "in_contact_object_bbox": [150, 150, 220, 220],
                "raw_state_id": 3,
            }
        ],
        "objects": [{"bbox": [150, 150, 220, 220], "confidence": 0.7}],
        "source": "100doh",
    }
    yolo_objects = [{"label": "cloth", "bbox": (150, 150, 220, 220)}]
    parsed = parse_100doh_payload(payload, input_objects=yolo_objects)

    assert parsed["left"]["contact_state"] == ContactState.NO_CONTACT
    assert parsed["left"]["grasp_type"] == GraspType.NONE

    right = parsed["right"]
    assert right["contact_state"] == ContactState.PORTABLE_OBJECT
    assert right["grasp_type"] == GraspType.PINCH  # cloth -> pinch
    assert right["in_contact_with_bbox"] == (150, 150, 220, 220)
    assert right["confidence"] == pytest.approx(0.91)


def test_parse_payload_no_contact_clears_grasp():
    payload = {
        "hands": [
            {
                "side": "left",
                "bbox": [0, 0, 10, 10],
                "confidence": 0.5,
                "raw_state_id": 0,
            }
        ]
    }
    parsed = parse_100doh_payload(payload)
    assert parsed["left"]["contact_state"] == ContactState.NO_CONTACT
    assert parsed["left"]["grasp_type"] == GraspType.NONE
    assert parsed["left"]["in_contact_with_bbox"] is None


def test_parse_payload_clamps_confidence():
    payload = {"hands": [{"side": "right", "raw_state_id": 3, "confidence": 1.7}]}
    parsed = parse_100doh_payload(payload)
    assert parsed["right"]["confidence"] == 1.0


def test_parse_payload_rejects_non_dict():
    with pytest.raises(ValueError):
        parse_100doh_payload([])  # type: ignore[arg-type]


def test_parse_payload_ignores_unknown_side():
    payload = {"hands": [{"side": "third", "raw_state_id": 3}]}
    parsed = parse_100doh_payload(payload)
    assert parsed["left"]["contact_state"] == ContactState.NO_CONTACT
    assert parsed["right"]["contact_state"] == ContactState.NO_CONTACT
