"""FacePipeline liveness behavior: reject / flag / disabled / degraded."""
import numpy as np
import pytest

import config
from face_pipeline import FacePipeline
from mock_backend import MockBackend

_FACE = {
    "bbox": {"x": 50, "y": 60, "w": 60, "h": 70},
    "landmarks": [
        (70.0, 85.0), (105.0, 85.0), (88.0, 105.0), (74.0, 122.0), (102.0, 122.0)
    ],
    "confidence": 0.92,
}


@pytest.fixture
def image():
    return np.full((200, 200, 3), 128, dtype=np.uint8)


def _make_pipeline(monkeypatch, backend, *, enabled=True,
                   threshold=0.5, action="reject"):
    monkeypatch.setattr(config, "LIVENESS_ENABLED", enabled)
    monkeypatch.setattr(config, "LIVENESS_THRESHOLD", threshold)
    monkeypatch.setattr(config, "LIVENESS_FAIL_ACTION", action)
    pipeline = FacePipeline(backend=backend)
    monkeypatch.setattr(
        FacePipeline, "detect", lambda self, img, **kw: [dict(_FACE)]
    )
    return pipeline


# --- reject mode ----------------------------------------------------------- #

def test_reject_spoof_face_skips_embedding(monkeypatch, image):
    backend = MockBackend(liveness_score=0.12)
    pipeline = _make_pipeline(monkeypatch, backend, action="reject")
    result = pipeline.process_image(image)
    assert result["success"] is False
    assert result["reason"] == "spoof"
    assert result["live"] is False
    assert result["liveness_score"] == pytest.approx(0.12)
    assert result["embedding"] is None
    assert backend.embed_calls == 0  # spoof never reaches embed/recognition


def test_reject_live_face_recognized_normally(monkeypatch, image):
    backend = MockBackend(liveness_score=0.93)
    pipeline = _make_pipeline(monkeypatch, backend, action="reject")
    result = pipeline.process_image(image)
    assert result["success"] is True
    assert result["live"] is True
    assert result["liveness_score"] == pytest.approx(0.93)
    assert result["embedding"] is not None
    assert backend.embed_calls == 1


def test_backend_receives_80x80_uint8_bgr_crop(monkeypatch, image):
    backend = MockBackend(liveness_score=0.9)
    pipeline = _make_pipeline(monkeypatch, backend)
    pipeline.process_image(image)
    crop = backend.last_liveness_crop
    assert crop is not None
    assert crop.shape == (80, 80, 3)
    assert crop.dtype == np.uint8


# --- flag mode ------------------------------------------------------------- #

def test_flag_spoof_face_still_recognized(monkeypatch, image):
    backend = MockBackend(liveness_score=0.12)
    pipeline = _make_pipeline(monkeypatch, backend, action="flag")
    result = pipeline.process_image(image)
    assert result["success"] is True
    assert result["live"] is False
    assert result["liveness_score"] == pytest.approx(0.12)
    assert result["embedding"] is not None
    assert backend.embed_calls == 1


# --- disabled: behavior identical to pre-liveness pipeline ----------------- #

def test_disabled_no_liveness_calls_and_null_fields(monkeypatch, image):
    backend = MockBackend(liveness_score=0.0)  # would reject if active
    pipeline = _make_pipeline(monkeypatch, backend, enabled=False)
    assert pipeline.liveness_status == "disabled"
    result = pipeline.process_image(image)
    assert result["success"] is True
    assert result["live"] is None
    assert result["liveness_score"] is None
    assert backend.liveness_calls == 0
    # Legacy result contract intact.
    for key in ("success", "embedding", "face", "aligned", "error"):
        assert key in result


def test_enabled_but_backend_without_liveness_degrades(monkeypatch, image):
    backend = MockBackend(liveness_score=0.0, with_liveness=False)
    pipeline = _make_pipeline(monkeypatch, backend, enabled=True)
    assert pipeline.liveness_status == "missing"
    assert pipeline.liveness_active is False
    result = pipeline.process_image(image)
    assert result["success"] is True
    assert backend.liveness_calls == 0


# --- threshold boundary ---------------------------------------------------- #

def test_score_equal_to_threshold_counts_as_live(monkeypatch, image):
    backend = MockBackend(liveness_score=0.5)
    pipeline = _make_pipeline(monkeypatch, backend, threshold=0.5)
    result = pipeline.process_image(image)
    assert result["success"] is True
    assert result["live"] is True


# --- process_all_faces (/infer path) --------------------------------------- #

def test_process_all_faces_reject_keeps_face_without_embedding(
    monkeypatch, image
):
    backend = MockBackend(liveness_score=0.1)
    pipeline = _make_pipeline(monkeypatch, backend, action="reject")
    out = pipeline.process_all_faces(image)
    assert len(out) == 1
    assert out[0]["live"] is False
    assert out[0]["embedding"] is None
    assert out[0]["aligned"] is None
    assert backend.embed_calls == 0


def test_process_all_faces_flag_embeds_spoof_face(monkeypatch, image):
    backend = MockBackend(liveness_score=0.1)
    pipeline = _make_pipeline(monkeypatch, backend, action="flag")
    out = pipeline.process_all_faces(image)
    assert len(out) == 1
    assert out[0]["live"] is False
    assert out[0]["embedding"] is not None
    assert backend.embed_calls == 1


def test_process_all_faces_disabled_matches_legacy_shape(monkeypatch, image):
    backend = MockBackend()
    pipeline = _make_pipeline(monkeypatch, backend, enabled=False)
    out = pipeline.process_all_faces(image)
    assert len(out) == 1
    assert out[0]["embedding"] is not None
    assert out[0]["live"] is None
    assert out[0]["liveness_score"] is None
    assert backend.liveness_calls == 0
