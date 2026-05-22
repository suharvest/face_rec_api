"""Backend registry for face detection + embedding."""
from __future__ import annotations

from .base import FaceBackend, FaceBackendError

__all__ = ["FaceBackend", "FaceBackendError", "create_backend"]


def create_backend(name: str) -> FaceBackend:
    """Instantiate a backend by name.

    Args:
        name: backend identifier (``hailo`` / ``jetson`` / ``rknn``).

    Raises:
        ValueError: unknown backend name.
        NotImplementedError: backend exists in registry but not yet implemented.
    """
    name = (name or "").lower().strip()
    if name == "hailo":
        from .hailo import HailoBackend
        return HailoBackend()
    if name == "jetson":
        from .tensorrt import TensorRTBackend
        return TensorRTBackend()
    if name == "rknn":
        raise NotImplementedError(
            "Rockchip (RKNN) backend is planned for P2 and not yet available."
        )
    raise ValueError(f"Unknown backend: {name!r}")
