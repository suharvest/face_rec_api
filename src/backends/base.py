"""
Abstract Face Backend interface.

All hardware-specific backends (Hailo / Jetson TensorRT / Rockchip RKNN / ...)
implement this interface so the upper FacePipeline stays hardware-agnostic.

Contract:
- `detect_raw` returns raw decoder inputs keyed by SCRFD stride (8/16/32),
  fully dequantized to fp32 and reshaped to a canonical layout. The pipeline
  layer never sees vendor-specific layer names.
- `embed_raw` returns a fp32 [512] vector. **No L2 normalization** at this
  layer — normalization happens in the pipeline so it is consistent across
  backends.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np


class FaceBackendError(RuntimeError):
    """Raised by a backend when an inference call fails irrecoverably."""


class FaceBackend(ABC):
    """Hardware-abstract face detection + embedding backend."""

    @abstractmethod
    def load(
        self,
        detector_path: str,
        embedder_path: str,
        liveness_path: Optional[str] = None,
    ) -> None:
        """Load detector and embedder models from disk.

        Args:
            detector_path: SCRFD detector model file.
            embedder_path: ArcFace embedder model file.
            liveness_path: optional MiniFASNet anti-spoofing model file.
                ``None`` means liveness is not requested for this backend.
                Backends that do not (yet) support liveness must accept and
                ignore this argument (with a warning) instead of raising.
        """

    @abstractmethod
    def detect_raw(
        self, bgr: np.ndarray
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Run face detection on a BGR image.

        Returns:
            Dict keyed by SCRFD stride (8, 16, 32). Each value is a tuple
            ``(scores, bboxes, kps)`` where:
              - ``scores``: fp32 array shape ``(N, 1)``
              - ``bboxes``: fp32 array shape ``(N, 4)`` — raw deltas (l,t,r,b)
              - ``kps``:    fp32 array shape ``(N, 10)`` — raw 5pt deltas
            All vendor-specific quantization and layer-name mapping is hidden.

        Additionally, the backend must attach preprocessing info under a
        special key ``-1`` of the returned dict, as a single-element tuple
        ``((scale, offset_x, offset_y, model_h, model_w),)``, so the pipeline
        can map detections back to the original image coordinates.
        """

    @abstractmethod
    def embed_raw(self, aligned_112x112: np.ndarray) -> np.ndarray:
        """
        Run ArcFace embedding on an aligned 112x112 BGR face crop.

        Returns:
            fp32 array shape ``(512,)``. **Not** L2-normalized.
        """

    @abstractmethod
    def liveness_raw(self, face_crop_bgr: np.ndarray) -> float:
        """
        Run passive anti-spoofing (MiniFASNet) on a face crop.

        Args:
            face_crop_bgr: ``(80, 80, 3)`` BGR **uint8** crop produced by
                :func:`liveness.crop_liveness_input` (detection bbox expanded
                by scale=2.7, resized to 80x80, NO normalization — values stay
                0-255, channel order stays BGR; see ``liveness.py`` module
                docstring for the Minivision transform rationale).

        Returns:
            Probability (0-1) that the face is *real* (softmax index 1).

        Raises:
            NotImplementedError: backend has no liveness support yet.
            FaceBackendError: liveness model not loaded or inference failed.
        """

    @property
    def liveness_loaded(self) -> bool:
        """True if a liveness model is loaded and ``liveness_raw`` is usable."""
        return False

    @property
    @abstractmethod
    def detector_input_hw(self) -> Tuple[int, int]:
        """Detector model's expected input (H, W)."""

    @property
    @abstractmethod
    def model_tag(self) -> str:
        """Identifier such as ``hailo:scrfd10g+arcface_mbf_v1``."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short backend identifier such as ``hailo`` / ``jetson`` / ``rknn``."""

    def close(self) -> None:
        """Tear down hardware resources. Idempotent."""
        return None

    def health_check(self) -> bool:
        """
        Liveness probe: run a dummy detection on a small black image.

        Returns True if both detector and embedder responded without raising;
        a zero detection result still counts as healthy.
        """
        try:
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self.detect_raw(dummy)
            # Embedder dummy
            dummy_face = np.zeros((112, 112, 3), dtype=np.uint8)
            vec = self.embed_raw(dummy_face)
            return vec is not None and vec.shape == (512,)
        except Exception:
            return False
