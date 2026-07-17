"""API-level tests: /recognize spoof handling, /health liveness status, and
backward compatibility of response schemas when liveness is disabled.

Endpoints are invoked as plain coroutines with faked module globals so no
hardware backend (and no FastAPI startup event) is needed.
"""
import asyncio

import pytest

import app as app_module
from app import (
    FaceResult,
    HealthResponse,
    RecognizeRequest,
    RecognizeResponse,
)


class _FakeBackend:
    backend_name = "mock"
    model_tag = "mock:test_v1"

    def health_check(self):
        return True


class _FakePipeline:
    def __init__(self, result, liveness_status="disabled"):
        self._result = result
        self.liveness_status = liveness_status
        self.backend = _FakeBackend()

    def process_image_base64(self, image_base64, strategy="largest"):
        return self._result


class _FakeStore:
    vectors = {}
    metadata = {}

    def __init__(self, search_result=None):
        self._search = search_result or {
            "matched": True, "name": "alice", "confidence": 0.87,
        }

    def search(self, embedding, threshold):
        return self._search


def _recognize(pipeline, store=None):
    app_module.face_pipeline = pipeline
    app_module.vector_store = store or _FakeStore()
    req = RecognizeRequest(image_base64="aGVsbG8=")
    return asyncio.run(app_module.recognize(req))


def _health(pipeline):
    app_module.face_pipeline = pipeline
    app_module.vector_store = _FakeStore()
    return asyncio.run(app_module.health())


# --- /recognize ------------------------------------------------------------ #

def test_recognize_spoof_rejected_with_reason():
    pipeline = _FakePipeline(
        {
            "success": False,
            "error": "Spoof detected (liveness_score=0.1200 < 0.5)",
            "reason": "spoof",
            "face": None,
            "embedding": None,
            "live": False,
            "liveness_score": 0.12,
        },
        liveness_status="loaded",
    )
    resp = _recognize(pipeline)
    assert resp.matched is False
    assert resp.reason == "spoof"
    assert resp.live is False
    assert resp.liveness_score == pytest.approx(0.12)


def test_recognize_live_face_carries_liveness_fields():
    pipeline = _FakePipeline(
        {
            "success": True,
            "embedding": [0.0] * 512,
            "face": {},
            "error": None,
            "live": True,
            "liveness_score": 0.97,
        },
        liveness_status="loaded",
    )
    resp = _recognize(pipeline)
    assert resp.matched is True
    assert resp.name == "alice"
    assert resp.live is True
    assert resp.liveness_score == pytest.approx(0.97)
    assert resp.reason is None


def test_recognize_disabled_matches_legacy_response():
    # Old-style pipeline result (no liveness keys at all) must produce a
    # response identical to the pre-liveness API apart from null new fields.
    pipeline = _FakePipeline(
        {
            "success": True,
            "embedding": [0.0] * 512,
            "face": {},
            "aligned": None,
            "error": None,
        },
        liveness_status="disabled",
    )
    resp = _recognize(pipeline)
    dump = resp.model_dump(exclude_none=True)
    dump.pop("processing_time_ms")
    # Exactly the legacy field set once nulls are stripped.
    assert dump == {"matched": True, "name": "alice", "confidence": 0.87}
    assert resp.live is None and resp.liveness_score is None and resp.reason is None


def test_recognize_no_face_failure_keeps_legacy_shape():
    pipeline = _FakePipeline(
        {"success": False, "error": "No face detected",
         "face": None, "embedding": None},
        liveness_status="disabled",
    )
    resp = _recognize(pipeline)
    assert resp.matched is False
    assert resp.reason is None
    assert resp.live is None


# --- /health ---------------------------------------------------------------- #

def test_health_reports_liveness_loaded():
    resp = _health(_FakePipeline({}, liveness_status="loaded"))
    assert resp.liveness == "loaded"
    assert "liveness" in resp.capabilities


@pytest.mark.parametrize("status", ["disabled", "missing"])
def test_health_reports_liveness_inactive(status):
    resp = _health(_FakePipeline({}, liveness_status=status))
    assert resp.liveness == status
    assert "liveness" not in resp.capabilities
    assert resp.capabilities == ["detect", "embed"]


# --- schema back-compat ------------------------------------------------------ #

def test_recognize_response_new_fields_default_none():
    resp = RecognizeResponse(
        matched=True, name="bob", confidence=0.9, processing_time_ms=5
    )
    assert resp.live is None
    assert resp.liveness_score is None
    assert resp.reason is None


def test_face_result_new_fields_default_none_and_embedding_optional():
    fr = FaceResult(
        bbox=[0, 0, 10, 10],
        landmarks=[[0.0, 0.0]] * 5,
        embedding="AAAA",
        det_score=0.9,
    )
    assert fr.live is None and fr.liveness_score is None
    spoof = FaceResult(
        bbox=[0, 0, 10, 10],
        landmarks=[[0.0, 0.0]] * 5,
        embedding=None,
        det_score=0.9,
        live=False,
        liveness_score=0.1,
    )
    assert spoof.embedding is None


def test_health_response_liveness_defaults_disabled():
    resp = HealthResponse(
        status="healthy", backend="mock", model_tag="t",
        capabilities=["detect", "embed"], users_loaded=0,
        embeddings_file="x.json", uptime_ms=1,
    )
    assert resp.liveness == "disabled"
