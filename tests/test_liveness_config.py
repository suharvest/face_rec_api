"""Config defaults / env overrides / graceful degradation for liveness."""
import importlib

import pytest

import config as config_module
import face_pipeline as fp_module


@pytest.fixture
def reload_config(monkeypatch):
    """Reload config with given env vars, restoring the module afterwards."""
    def _reload(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return importlib.reload(config_module)

    yield _reload
    # Restore module state built from the clean environment.
    monkeypatch.undo()
    importlib.reload(config_module)


def test_defaults(reload_config):
    cfg = reload_config()
    assert cfg.LIVENESS_ENABLED is True
    assert cfg.LIVENESS_THRESHOLD == 0.5
    assert cfg.LIVENESS_FAIL_ACTION == "reject"
    assert cfg.LIVENESS_CROP_SCALE == 2.7
    # Default hailo backend -> .hef extension, conventional filename.
    assert cfg.FACE_LIVENESS_MODEL.endswith("liveness_minifasnet.hef")


def test_env_overrides(reload_config):
    cfg = reload_config(
        LIVENESS_ENABLED="false",
        LIVENESS_THRESHOLD="0.8",
        LIVENESS_FAIL_ACTION="flag",
        FACE_LIVENESS_MODEL="/tmp/custom_liveness.hef",
        LIVENESS_CROP_SCALE="4.0",
    )
    assert cfg.LIVENESS_ENABLED is False
    assert cfg.LIVENESS_THRESHOLD == 0.8
    assert cfg.LIVENESS_FAIL_ACTION == "flag"
    assert cfg.FACE_LIVENESS_MODEL == "/tmp/custom_liveness.hef"
    assert cfg.LIVENESS_CROP_SCALE == 4.0


def test_invalid_fail_action_falls_back_to_reject(reload_config):
    cfg = reload_config(LIVENESS_FAIL_ACTION="explode")
    assert cfg.LIVENESS_FAIL_ACTION == "reject"


def test_backend_extension_follows_backend(reload_config):
    cfg = reload_config(FACE_BACKEND="jetson")
    assert cfg.FACE_LIVENESS_MODEL.endswith("liveness_minifasnet.engine")


# --- resolve_liveness_path degradation ------------------------------------ #

def test_resolve_disabled(monkeypatch):
    monkeypatch.setattr(config_module, "LIVENESS_ENABLED", False)
    assert fp_module.resolve_liveness_path() == (None, "disabled")


def test_resolve_missing_model_degrades_without_crash(monkeypatch, caplog):
    monkeypatch.setattr(config_module, "LIVENESS_ENABLED", True)
    monkeypatch.setattr(
        config_module, "FACE_LIVENESS_MODEL", "/nonexistent/liveness.hef"
    )
    with caplog.at_level("WARNING"):
        path, status = fp_module.resolve_liveness_path()
    assert path is None
    assert status == "missing"
    assert any("auto-disabled" in r.message for r in caplog.records)


def test_resolve_loaded(monkeypatch, tmp_path):
    model = tmp_path / "liveness_minifasnet.hef"
    model.write_bytes(b"fake-hef")
    monkeypatch.setattr(config_module, "LIVENESS_ENABLED", True)
    monkeypatch.setattr(config_module, "FACE_LIVENESS_MODEL", str(model))
    assert fp_module.resolve_liveness_path() == (str(model), "loaded")
