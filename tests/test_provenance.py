from pathlib import Path

from src.egoflow.utils import provenance


class _Model:
    def __init__(self, payload):
        self._payload = payload

    def status(self):
        return self._payload


def test_record_writes_valid_json(tmp_path: Path):
    models = [
        _Model({"name": "YOLO-World", "mode": "real", "reason": "ultralytics loaded"}),
        _Model({"name": "100DOH Contact", "mode": "disabled", "reason": "enable_100doh=false"}),
    ]
    provenance.record(tmp_path, models, phase="annotate")
    data = provenance.read(tmp_path)
    assert "annotate" in data["models"]
    entries = data["models"]["annotate"]
    assert entries[0]["name"] == "YOLO-World"
    assert entries[1]["mode"] == "disabled"
    assert data["updated_at"] is not None


def test_record_merges_across_phases(tmp_path: Path):
    provenance.record(tmp_path, [_Model({"name": "A", "mode": "real", "reason": ""})], phase="annotate")
    provenance.record(tmp_path, [_Model({"name": "B", "mode": "fallback", "reason": "x"})], phase="describe")
    data = provenance.read(tmp_path)
    assert set(data["models"].keys()) == {"annotate", "describe"}


def test_record_normalizes_bad_mode(tmp_path: Path):
    provenance.record(tmp_path, [_Model({"name": "X", "mode": "weird", "reason": ""})], phase="annotate")
    data = provenance.read(tmp_path)
    assert data["models"]["annotate"][0]["mode"] == "fallback"


def test_record_survives_status_exceptions(tmp_path: Path):
    class Boom:
        def status(self):
            raise RuntimeError("nope")

    provenance.record(tmp_path, [Boom()], phase="annotate")
    entry = provenance.read(tmp_path)["models"]["annotate"][0]
    assert entry["mode"] == "fallback"
    assert "nope" in entry["reason"]


def test_summary_counts_and_flattens(tmp_path: Path):
    provenance.record(
        tmp_path,
        [
            _Model({"name": "A", "mode": "real", "reason": ""}),
            _Model({"name": "B", "mode": "fallback", "reason": ""}),
        ],
        phase="annotate",
    )
    provenance.record(tmp_path, [_Model({"name": "C", "mode": "disabled", "reason": ""})], phase="validate")
    data = provenance.read(tmp_path)
    summary = provenance.summary(data)
    assert summary["counts"] == {"real": 1, "fallback": 1, "disabled": 1}
    phases = {entry["phase"] for entry in summary["entries"]}
    assert phases == {"annotate", "validate"}


def test_read_missing_returns_empty(tmp_path: Path):
    data = provenance.read(tmp_path)
    assert data["models"] == {}


def test_model_status_methods_shape():
    from src.egoflow.models.contact_100doh import Contact100DOH
    from src.egoflow.models.masks_sam2 import SAM2Masks
    from src.egoflow.models.pose_hamer import HaMeRPose

    m = Contact100DOH(enabled=False)
    m.load()
    s = m.status()
    assert s["name"] == "100DOH Contact" and s["mode"] == "disabled"

    hamer = HaMeRPose()
    hamer.load()
    assert hamer.status()["mode"] == "disabled"

    sam = SAM2Masks()
    sam.load()
    assert sam.status()["mode"] == "disabled"
