"""
Hailo-8 backend for face detection (SCRFD-10g) + embedding (ArcFace-MBF).

Encapsulates:
- Shared VDevice between detector and embedder.
- Async inference threads with sentinel-based shutdown and bounded queues.
- Quantization metadata + Hailo layer-name -> SCRFD stride mapping.
- Dequantization and layout normalization, so the upper pipeline never sees
  vendor-specific names.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .base import FaceBackend, FaceBackendError

logger = logging.getLogger(__name__)

# Hailo imports are deferred to backend instantiation, not module import,
# so other backends (or tests on a Mac) can import this package safely until
# they actually try to instantiate HailoBackend.
try:
    from hailo_platform import VDevice, HailoSchedulingAlgorithm  # type: ignore
    _HAILO_AVAILABLE = True
    _HAILO_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _e:  # pragma: no cover - import gating only
    VDevice = None  # type: ignore
    HailoSchedulingAlgorithm = None  # type: ignore
    _HAILO_AVAILABLE = False
    _HAILO_IMPORT_ERROR = _e


# SCRFD-10g (Hailo HEF) output layer-name -> (stride, kind) mapping.
# `kind` in {"scores", "bboxes", "kps"}.
SCRFD_LAYER_MAP: Dict[str, Tuple[int, str]] = {
    "scrfd_10g/conv41": (8, "scores"),
    "scrfd_10g/conv42": (8, "bboxes"),
    "scrfd_10g/conv43": (8, "kps"),
    "scrfd_10g/conv49": (16, "scores"),
    "scrfd_10g/conv50": (16, "bboxes"),
    "scrfd_10g/conv51": (16, "kps"),
    "scrfd_10g/conv56": (32, "scores"),
    "scrfd_10g/conv57": (32, "bboxes"),
    "scrfd_10g/conv58": (32, "kps"),
}


# Sentinel used to terminate inference loops cleanly on shutdown.
_STOP = object()

# Default queue put/get timeout in seconds.
_QUEUE_TIMEOUT = 15.0
_INFER_TIMEOUT_MS = 10000


class _HailoModelRunner:
    """Single-model async runner sharing a VDevice."""

    def __init__(self, name: str, model_path: str, vdevice):
        self.name = name
        self.model_path = model_path
        self.vdevice = vdevice
        self.input_queue: queue.Queue = queue.Queue(maxsize=20)
        self.output_queue: queue.Queue = queue.Queue(maxsize=20)
        self.infer_model = None
        self.quant_infos: Dict[str, Tuple[float, float]] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"{self.name} model not found: {self.model_path}")

        self.infer_model = self.vdevice.create_infer_model(self.model_path)

        vstream_infos = self.infer_model.hef.get_output_vstream_infos()
        self.quant_infos = {
            info.name: (info.quant_info.qp_scale, info.quant_info.qp_zp)
            for info in vstream_infos
        }

        self._thread = threading.Thread(
            target=self._inference_loop, name=f"hailo-{self.name}", daemon=True
        )
        self._thread.start()
        logger.info("Hailo %s model loaded: %s", self.name, self.model_path)

    def _inference_loop(self) -> None:
        def _cb(completion_info):
            if completion_info.exception:
                logger.error(
                    "%s inference error: %s", self.name, completion_info.exception
                )

        try:
            with self.infer_model.configure() as configured_model:
                while not self._stop_event.is_set():
                    try:
                        item = self.input_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    if item is _STOP:
                        break
                    original_frame, preprocessed = item
                    try:
                        output_buffers = {
                            info.name: np.empty(info.shape, dtype=np.uint8)
                            for info in self.infer_model.outputs
                        }
                        bindings = configured_model.create_bindings(
                            output_buffers=output_buffers
                        )
                        bindings.input().set_buffer(preprocessed)
                        configured_model.wait_for_async_ready(
                            timeout_ms=_INFER_TIMEOUT_MS
                        )
                        job = configured_model.run_async([bindings], _cb)
                        job.wait(_INFER_TIMEOUT_MS)
                        self.output_queue.put((original_frame, output_buffers))
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "%s inference failed: %s", self.name, exc, exc_info=True
                        )
                        self.output_queue.put((original_frame, exc))
        except Exception as exc:  # noqa: BLE001
            logger.error("%s inference loop crashed: %s", self.name, exc, exc_info=True)

    def submit(self, original, preprocessed) -> None:
        try:
            self.input_queue.put((original, preprocessed), timeout=_QUEUE_TIMEOUT)
        except queue.Full as exc:
            raise FaceBackendError(
                f"{self.name} input queue full (timeout)"
            ) from exc

    def fetch(self) -> Tuple[object, object]:
        try:
            return self.output_queue.get(timeout=_QUEUE_TIMEOUT)
        except queue.Empty as exc:
            raise FaceBackendError(f"{self.name} inference timeout") from exc

    def infer_sync(self, original, preprocessed) -> Dict[str, np.ndarray]:
        """Single-shot serialized inference (callers must hold the runner lock)."""
        self.submit(original, preprocessed)
        _, result = self.fetch()
        if isinstance(result, Exception):
            raise FaceBackendError(
                f"{self.name} inference failed"
            ) from result
        return result  # dict[layer_name -> uint8 ndarray]

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self.input_queue.put_nowait(_STOP)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("%s inference thread did not exit cleanly", self.name)
            self._thread = None


class HailoBackend(FaceBackend):
    """Hailo-8 implementation of :class:`FaceBackend`."""

    BACKEND_NAME = "hailo"
    MODEL_TAG = "hailo:scrfd10g+arcface_mbf_v1"

    def __init__(self) -> None:
        if not _HAILO_AVAILABLE:
            raise ImportError(
                "hailo_platform is not installed. Install HailoRT (see README)."
            ) from _HAILO_IMPORT_ERROR
        self.vdevice = None
        self._detector: Optional[_HailoModelRunner] = None
        self._embedder: Optional[_HailoModelRunner] = None
        self._det_lock = threading.Lock()
        self._emb_lock = threading.Lock()
        self._closed = False

    # ---- lifecycle ------------------------------------------------------ #
    def load(self, detector_path: str, embedder_path: str) -> None:
        logger.info("Creating shared Hailo VDevice (ROUND_ROBIN scheduler)...")
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.vdevice = VDevice(params)

        self._detector = _HailoModelRunner("detector", detector_path, self.vdevice)
        self._detector.start()
        self._embedder = _HailoModelRunner("embedder", embedder_path, self.vdevice)
        self._embedder.start()
        logger.info("HailoBackend ready (model_tag=%s)", self.MODEL_TAG)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        logger.info("Shutting down HailoBackend...")
        for runner in (self._detector, self._embedder):
            if runner is not None:
                try:
                    runner.stop()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Runner stop error: %s", exc)
        # Release VDevice
        if self.vdevice is not None:
            try:
                self.vdevice.release()
            except Exception:
                # Some HailoRT versions release on GC; ignore.
                pass
            self.vdevice = None
        logger.info("HailoBackend shutdown complete")

    # ---- properties ----------------------------------------------------- #
    @property
    def backend_name(self) -> str:
        return self.BACKEND_NAME

    @property
    def model_tag(self) -> str:
        return self.MODEL_TAG

    @property
    def detector_input_hw(self) -> Tuple[int, int]:
        assert self._detector is not None and self._detector.infer_model is not None
        shape = self._detector.infer_model.input().shape
        return int(shape[0]), int(shape[1])

    # ---- preprocessing -------------------------------------------------- #
    def _preprocess_detect(
        self, bgr: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int], Tuple[int, int]]:
        model_h, model_w = self.detector_input_hw
        h, w = bgr.shape[:2]
        scale = min(model_w / w, model_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(bgr, (new_w, new_h))
        padded = np.zeros((model_h, model_w, 3), dtype=np.uint8)
        top = (model_h - new_h) // 2
        left = (model_w - new_w) // 2
        padded[top : top + new_h, left : left + new_w] = resized
        return padded, scale, (left, top), (model_h, model_w)

    def _preprocess_embed(self, aligned_112x112: np.ndarray) -> np.ndarray:
        assert self._embedder is not None and self._embedder.infer_model is not None
        model_shape = self._embedder.infer_model.input().shape
        model_h, model_w = int(model_shape[0]), int(model_shape[1])
        h, w = aligned_112x112.shape[:2]
        if h == 0 or w == 0:
            raise ValueError("Invalid face image dimensions")
        if (h, w) == (model_h, model_w):
            return aligned_112x112
        scale = min(model_w / w, model_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(aligned_112x112, (new_w, new_h))
        padded = np.zeros((model_h, model_w, 3), dtype=np.uint8)
        top = (model_h - new_h) // 2
        left = (model_w - new_w) // 2
        padded[top : top + new_h, left : left + new_w] = resized
        return padded

    # ---- inference ------------------------------------------------------ #
    def detect_raw(
        self, bgr: np.ndarray
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self._detector is None:
            raise FaceBackendError("HailoBackend not loaded")

        preprocessed, scale, offset, model_hw = self._preprocess_detect(bgr)

        with self._det_lock:
            raw = self._detector.infer_sync(bgr, preprocessed)
            quant = self._detector.quant_infos

        # Group layers by stride
        by_stride: Dict[int, Dict[str, np.ndarray]] = {8: {}, 16: {}, 32: {}}
        for layer_name, buf in raw.items():
            mapping = SCRFD_LAYER_MAP.get(layer_name)
            if mapping is None:
                continue
            stride, kind = mapping
            qp_scale, qp_zp = quant[layer_name]
            arr = (buf.astype(np.float32) - qp_zp) * qp_scale
            if kind == "scores":
                arr = arr.reshape(-1, 1)
            elif kind == "bboxes":
                arr = arr.reshape(-1, 4)
            else:  # kps
                arr = arr.reshape(-1, 10)
            by_stride[stride][kind] = arr

        out: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for stride, parts in by_stride.items():
            if {"scores", "bboxes", "kps"}.issubset(parts.keys()):
                out[stride] = (parts["scores"], parts["bboxes"], parts["kps"])

        # Stash preprocessing meta under sentinel key -1 so the pipeline can
        # map detections back to the original image without re-running prep.
        out[-1] = (  # type: ignore[assignment]
            np.array(
                [scale, offset[0], offset[1], model_hw[0], model_hw[1]],
                dtype=np.float32,
            ),
        )
        return out

    def embed_raw(self, aligned_112x112: np.ndarray) -> np.ndarray:
        if self._embedder is None:
            raise FaceBackendError("HailoBackend not loaded")

        preprocessed = self._preprocess_embed(aligned_112x112)
        with self._emb_lock:
            raw = self._embedder.infer_sync(aligned_112x112, preprocessed)
            quant = self._embedder.quant_infos

        # ArcFace HEF typically has a single output.
        if len(raw) == 1:
            name = next(iter(raw))
            buf = raw[name]
        else:
            # Defensive fallback: pick the layer that flattens to 512.
            name, buf = next(
                ((n, b) for n, b in raw.items() if int(np.prod(b.shape)) == 512),
                next(iter(raw.items())),
            )

        if name in quant:
            qp_scale, qp_zp = quant[name]
            emb = (buf.astype(np.float32) - qp_zp) * qp_scale
        else:
            emb = buf.astype(np.float32)

        emb = emb.flatten()
        if emb.size != 512:
            if emb.size > 512:
                emb = emb[:512]
            else:
                emb = np.concatenate(
                    [emb, np.zeros(512 - emb.size, dtype=np.float32)]
                )
        return emb.astype(np.float32, copy=False)
