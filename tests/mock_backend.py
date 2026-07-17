"""Hardware-free mock FaceBackend for pipeline/API tests."""
from typing import Dict, Optional, Tuple

import numpy as np

from backends.base import FaceBackend


class MockBackend(FaceBackend):
    """Deterministic backend; detection is monkeypatched at pipeline level."""

    def __init__(
        self,
        liveness_score: float = 0.9,
        with_liveness: bool = True,
    ) -> None:
        self._score = liveness_score
        self._with_liveness = with_liveness
        self.embed_calls = 0
        self.liveness_calls = 0
        self.last_liveness_crop: Optional[np.ndarray] = None

    # -- FaceBackend interface ------------------------------------------- #
    def load(
        self,
        detector_path: str,
        embedder_path: str,
        liveness_path: Optional[str] = None,
    ) -> None:
        self._with_liveness = liveness_path is not None

    def detect_raw(self, bgr: np.ndarray) -> Dict:
        raise NotImplementedError("Tests monkeypatch FacePipeline.detect")

    def embed_raw(self, aligned_112x112: np.ndarray) -> np.ndarray:
        self.embed_calls += 1
        return np.ones(512, dtype=np.float32)

    def liveness_raw(self, face_crop_bgr: np.ndarray) -> float:
        self.liveness_calls += 1
        self.last_liveness_crop = face_crop_bgr
        return self._score

    @property
    def liveness_loaded(self) -> bool:
        return self._with_liveness

    @property
    def detector_input_hw(self) -> Tuple[int, int]:
        return 640, 640

    @property
    def model_tag(self) -> str:
        return "mock:test_v1"

    @property
    def backend_name(self) -> str:
        return "mock"
