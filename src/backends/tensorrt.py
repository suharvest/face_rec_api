"""
NVIDIA Jetson TensorRT backend for face detection (SCRFD-10g) + embedding
(ArcFace-MobileFaceNet).

Loads pre-compiled `.engine` files produced by ``trtexec`` (or the TensorRT
Python API). Engines are JetPack- and SM-version-specific, so they MUST be
built on a device with the same JetPack release and GPU compute capability
as the runtime device (see ``tools/build_engine.sh``).

This backend produces the same canonical output layout as the Hailo backend
so the upper :class:`face_pipeline.FacePipeline` stays hardware-agnostic.

Runtime dependencies (must be available at import time when this backend is
selected):

- ``tensorrt`` >= 10.0  — usually pre-installed on the L4T JetPack image in
  ``/usr/lib/python3.10/dist-packages``; do NOT ``pip install`` over it.
- ``cuda-python`` (the ``cuda`` namespace package) — installed via pip.

Both imports are deferred so the module can be imported on a Mac or a Hailo
device without crashing — instantiation is what fails.
"""
from __future__ import annotations

import ctypes
import logging
import os
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import FaceBackend, FaceBackendError

logger = logging.getLogger(__name__)

# Deferred TensorRT / CUDA imports — only required when this backend is loaded.
try:
    import tensorrt as trt  # type: ignore

    _TRT_AVAILABLE = True
    _TRT_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _e:  # pragma: no cover - import gating only
    trt = None  # type: ignore
    _TRT_AVAILABLE = False
    _TRT_IMPORT_ERROR = _e

try:
    from cuda import cudart  # type: ignore

    _CUDA_AVAILABLE = True
    _CUDA_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _e:  # pragma: no cover - import gating only
    cudart = None  # type: ignore
    _CUDA_AVAILABLE = False
    _CUDA_IMPORT_ERROR = _e


# ---- CUDA helpers --------------------------------------------------------- #
def _cuda_check(ret):
    """``cuda-python`` returns ``(err, ...)`` tuples. Raise if err != 0."""
    if isinstance(ret, tuple):
        err = ret[0]
        rest = ret[1:]
    else:
        err = ret
        rest = ()
    if int(err) != 0:
        # cudart.cudaError_t is enum-ish; cast to int and fetch name if we can
        try:
            name = cudart.cudaGetErrorName(err)[1].decode("utf-8")
            msg = cudart.cudaGetErrorString(err)[1].decode("utf-8")
            raise FaceBackendError(f"CUDA error {int(err)} ({name}): {msg}")
        except FaceBackendError:
            raise
        except Exception:
            raise FaceBackendError(f"CUDA error {int(err)}")
    if len(rest) == 0:
        return None
    if len(rest) == 1:
        return rest[0]
    return rest


# ---- SCRFD output-name classification ------------------------------------- #
# TensorRT engines built from the InsightFace ``det_10g.onnx`` keep the
# original ONNX output names. Across known variants these include things
# like ``score_8`` / ``bbox_8`` / ``kps_8`` (and 16, 32 strides) as well as
# the older ``448`` / ``451`` / ``454`` integer tensor names. We classify
# by inspecting the per-stride shape rather than the textual name so the
# mapping is robust to ONNX export variations.
#
# Expected per-stride channel counts (with num_anchors=2):
#   scores : 2  (one per anchor)
#   bboxes : 8  (4 deltas × 2 anchors)
#   kps    : 20 (10 deltas × 2 anchors)
_SCRFD_CHANNELS = {2: "scores", 8: "bboxes", 20: "kps"}
_STRIDES = (8, 16, 32)
_DET_INPUT_H = 640
_DET_INPUT_W = 640
_EMB_INPUT_H = 112
_EMB_INPUT_W = 112


# ---- Engine helpers ------------------------------------------------------- #
class _TRTEngine:
    """One TensorRT engine + execution context + pre-allocated buffers."""

    def __init__(self, name: str, engine_path: str, logger_trt) -> None:
        self.name = name
        self.engine_path = engine_path
        self._logger_trt = logger_trt

        self.engine = None
        self.context = None
        self.stream = None

        # tensor_name -> dict(shape, dtype, nbytes, host, device, is_input)
        self.tensors: Dict[str, Dict] = {}
        self.input_name: Optional[str] = None
        self.output_names: List[str] = []

    # -- lifecycle -------------------------------------------------------- #
    def load(self) -> None:
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(
                f"{self.name} engine not found: {self.engine_path}\n"
                "Engines are JetPack + sm-version specific. Build with "
                "`tools/build_engine.sh` on the target Jetson device."
            )

        with open(self.engine_path, "rb") as f:
            engine_bytes = f.read()

        runtime = trt.Runtime(self._logger_trt)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise FaceBackendError(
                f"Failed to deserialize {self.name} engine: {self.engine_path}"
            )

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise FaceBackendError(f"Failed to create execution context for {self.name}")

        # Dedicated stream per engine. Cheap and avoids cross-engine sync hazards.
        self.stream = _cuda_check(cudart.cudaStreamCreate())

        # Discover all tensors
        for i in range(self.engine.num_io_tensors):
            tname = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(tname)
            is_input = mode == trt.TensorIOMode.INPUT
            shape = tuple(self.engine.get_tensor_shape(tname))

            # If shape has a -1 (dynamic), pin it to canonical dims for our model.
            if any(d < 0 for d in shape):
                if is_input:
                    if self.name == "detector":
                        shape = (1, 3, _DET_INPUT_H, _DET_INPUT_W)
                    elif self.name == "embedder":
                        shape = (1, 3, _EMB_INPUT_H, _EMB_INPUT_W)
                    else:
                        raise FaceBackendError(
                            f"Cannot resolve dynamic input shape for {self.name}/{tname}"
                        )
                    self.context.set_input_shape(tname, shape)
                else:
                    # Re-query after setting input shape
                    shape = tuple(self.context.get_tensor_shape(tname))
                    if any(d < 0 for d in shape):
                        raise FaceBackendError(
                            f"Could not resolve output shape for {self.name}/{tname}"
                        )

            dtype = trt.nptype(self.engine.get_tensor_dtype(tname))
            size = int(np.prod(shape))
            nbytes = size * np.dtype(dtype).itemsize

            # CUDA pinned (page-locked) host memory. DMA can transfer directly
            # without an intermediate staging copy, cutting H2D/D2H wall time
            # 30-50% on Jetson where host and device share the same physical
            # RAM but the kernel still distinguishes pageable vs pinned.
            host_ptr = _cuda_check(
                cudart.cudaHostAlloc(nbytes, cudart.cudaHostAllocDefault)
            )
            # Wrap the raw pointer as a numpy array of the right shape/dtype.
            buf_type = ctypes.c_uint8 * nbytes
            raw = buf_type.from_address(int(host_ptr))
            host_buf = (
                np.frombuffer(raw, dtype=np.uint8).view(dtype).reshape(shape)
            )

            device_ptr = _cuda_check(cudart.cudaMalloc(nbytes))

            self.tensors[tname] = {
                "shape": shape,
                "dtype": dtype,
                "size": size,
                "nbytes": nbytes,
                "host": host_buf,
                "host_ptr": int(host_ptr),
                "device": int(device_ptr),
                "is_input": is_input,
            }
            self.context.set_tensor_address(tname, int(device_ptr))

            if is_input:
                self.input_name = tname
            else:
                self.output_names.append(tname)

        if self.input_name is None:
            raise FaceBackendError(f"{self.name} engine has no input tensor")
        if not self.output_names:
            raise FaceBackendError(f"{self.name} engine has no output tensors")

        logger.info(
            "TRT %s engine loaded: %s (inputs=%s outputs=%s)",
            self.name,
            self.engine_path,
            self.input_name,
            self.output_names,
        )

    def close(self) -> None:
        # Free device + pinned-host buffers.
        for t in self.tensors.values():
            try:
                _cuda_check(cudart.cudaFree(t["device"]))
            except Exception:
                pass
            try:
                _cuda_check(cudart.cudaFreeHost(t["host_ptr"]))
            except Exception:
                pass
        self.tensors.clear()
        if self.stream is not None:
            try:
                _cuda_check(cudart.cudaStreamDestroy(self.stream))
            except Exception:
                pass
            self.stream = None
        # Drop context + engine
        self.context = None
        self.engine = None

    # -- inference -------------------------------------------------------- #
    def infer(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Synchronous infer: H2D copy → enqueueV3 → D2H copy → stream sync."""
        if self.context is None:
            raise FaceBackendError(f"{self.name} engine not loaded")

        tin = self.tensors[self.input_name]
        expected_shape = tin["shape"]
        if tuple(input_data.shape) != expected_shape:
            raise FaceBackendError(
                f"{self.name} input shape mismatch: got {input_data.shape}, "
                f"expected {expected_shape}"
            )
        if input_data.dtype != tin["dtype"]:
            input_data = input_data.astype(tin["dtype"], copy=False)
        if not input_data.flags["C_CONTIGUOUS"]:
            input_data = np.ascontiguousarray(input_data)

        # Stage into pinned host buffer so the H2D copy is a true DMA from
        # page-locked memory (otherwise CUDA falls back to a staging copy
        # through an internal pinned ring buffer, doubling the bandwidth bill).
        np.copyto(tin["host"], input_data, casting="no")

        # H2D
        _cuda_check(
            cudart.cudaMemcpyAsync(
                tin["device"],
                tin["host"].ctypes.data,
                tin["nbytes"],
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            )
        )

        # Run
        ok = self.context.execute_async_v3(stream_handle=self.stream)
        if not ok:
            raise FaceBackendError(f"{self.name} enqueueV3 returned false")

        # D2H for all outputs
        for oname in self.output_names:
            t = self.tensors[oname]
            _cuda_check(
                cudart.cudaMemcpyAsync(
                    t["host"].ctypes.data,
                    t["device"],
                    t["nbytes"],
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream,
                )
            )

        _cuda_check(cudart.cudaStreamSynchronize(self.stream))

        # Copy host buffers out so callers can mutate freely without trashing
        # our preallocated host arrays on the next infer.
        return {n: self.tensors[n]["host"].copy() for n in self.output_names}


# ---- Backend -------------------------------------------------------------- #
class TensorRTBackend(FaceBackend):
    """NVIDIA Jetson TensorRT implementation of :class:`FaceBackend`."""

    BACKEND_NAME = "jetson"
    MODEL_TAG = "jetson:scrfd10g+arcface_mbf_v1"

    def __init__(self) -> None:
        if not _TRT_AVAILABLE:
            raise ImportError(
                "tensorrt is not available. On Jetson it ships with JetPack "
                "in /usr/lib/python3.10/dist-packages — make sure your env "
                "can see system site-packages."
            ) from _TRT_IMPORT_ERROR
        if not _CUDA_AVAILABLE:
            raise ImportError(
                "cuda-python is not installed. Install via "
                "`pip install cuda-python` (or `uv sync --extra jetson`)."
            ) from _CUDA_IMPORT_ERROR

        self._logger_trt = trt.Logger(trt.Logger.WARNING)
        self._detector: Optional[_TRTEngine] = None
        self._embedder: Optional[_TRTEngine] = None
        self._det_lock = threading.Lock()
        self._emb_lock = threading.Lock()
        self._det_output_map: Dict[str, Tuple[int, str]] = {}
        self._closed = False

    # -- lifecycle ------------------------------------------------------- #
    def load(self, detector_path: str, embedder_path: str) -> None:
        self._detector = _TRTEngine("detector", detector_path, self._logger_trt)
        self._detector.load()

        self._embedder = _TRTEngine("embedder", embedder_path, self._logger_trt)
        self._embedder.load()

        # Build SCRFD output mapping by inspecting per-tensor shapes.
        self._det_output_map = self._classify_scrfd_outputs(self._detector)

        # Sanity: must cover all 9 (stride, kind) combinations.
        seen = set(self._det_output_map.values())
        expected = {(s, k) for s in _STRIDES for k in ("scores", "bboxes", "kps")}
        missing = expected - seen
        if missing:
            raise FaceBackendError(
                f"SCRFD engine output mapping incomplete; missing: {sorted(missing)}. "
                f"Available outputs: {self._detector.output_names}"
            )

        # Warm up: 3 dummy infers per engine. The first call after engine
        # deserialization pays for lazy CUDA kernel JIT, TRT plan finalisation
        # and cuDNN tactic selection; without warmup the first /infer request
        # eats 100-300 ms of cold-start cost. Three iterations is enough for
        # the cuDNN tactic cache to stabilise.
        try:
            dummy_bgr = np.zeros((640, 640, 3), dtype=np.uint8)
            dummy_face = np.zeros((112, 112, 3), dtype=np.uint8)
            for _ in range(3):
                self.detect_raw(dummy_bgr)
                self.embed_raw(dummy_face)
            logger.info("TensorRTBackend warmup complete (3 iterations)")
        except Exception as exc:  # noqa: BLE001
            logger.warning("TensorRTBackend warmup failed: %s", exc)

        logger.info("TensorRTBackend ready (model_tag=%s)", self.MODEL_TAG)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        logger.info("Shutting down TensorRTBackend...")
        for eng in (self._detector, self._embedder):
            if eng is not None:
                try:
                    eng.close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Engine close error: %s", exc)
        logger.info("TensorRTBackend shutdown complete")

    # -- properties ------------------------------------------------------ #
    @property
    def backend_name(self) -> str:
        return self.BACKEND_NAME

    @property
    def model_tag(self) -> str:
        return self.MODEL_TAG

    @property
    def detector_input_hw(self) -> Tuple[int, int]:
        assert self._detector is not None
        shape = self._detector.tensors[self._detector.input_name]["shape"]
        # NCHW
        return int(shape[2]), int(shape[3])

    # -- classification helpers ----------------------------------------- #
    @staticmethod
    def _classify_scrfd_outputs(eng: _TRTEngine) -> Dict[str, Tuple[int, str]]:
        """Map every detector output tensor name -> (stride, kind).

        Strategy:
        1. Group outputs by spatial size (H*W). SCRFD-640 yields 3 groups:
           80*80 (stride 8), 40*40 (stride 16), 20*20 (stride 32).
        2. Within each group, classify by channel count: 2→scores, 8→bboxes,
           20→kps.

        The engine may return shapes either as (1, C, H, W) or as already-
        flattened (1, H*W*num_anchors, C) depending on the ONNX export.
        We handle both.
        """
        # First pass: figure out spatial extent per tensor.
        # We expect any of:
        #   4D: (1, C, H, W)   — C in {2, 8, 20}; HxW = stride feature map.
        #   3D: (1, N, C)      — N = H*W*num_anchors, per-anchor C in {1, 4, 10}.
        #   2D: (N, C)         — same as 3D but with leading batch elided. This
        #                        is what the InsightFace `det_10g.onnx`
        #                        produces (e.g. (12800, 1) at stride 8).
        # For 3D/2D we divide N by num_anchors=2 to recover the spatial size.
        normalized: List[Tuple[str, int, str]] = []  # (name, spatial, kind)
        kind_for_c_per_anchor = {1: "scores", 4: "bboxes", 10: "kps"}
        for name in eng.output_names:
            shape = eng.tensors[name]["shape"]
            if len(shape) == 4:
                _, c, h, w = shape
                spatial = int(h) * int(w)
                kind = _SCRFD_CHANNELS.get(int(c))
                if kind is None:
                    raise FaceBackendError(
                        f"Unexpected SCRFD output channel count C={c} for tensor "
                        f"{name} shape={shape}"
                    )
            elif len(shape) in (2, 3):
                if len(shape) == 3:
                    _, n, c = shape
                else:
                    n, c = shape
                if int(n) % 2 != 0:
                    raise FaceBackendError(
                        f"SCRFD output {name} shape={shape}: N={n} not divisible by 2"
                    )
                spatial = int(n) // 2
                kind = kind_for_c_per_anchor.get(int(c))
                if kind is None:
                    raise FaceBackendError(
                        f"Unexpected SCRFD per-anchor channel count C={c} for "
                        f"tensor {name} shape={shape}"
                    )
            else:
                raise FaceBackendError(
                    f"Unexpected SCRFD output rank for {name}: shape={shape}"
                )
            normalized.append((name, spatial, kind))

        # Group by spatial. Sort spatials descending → largest first = stride 8.
        spatials = sorted({s for _, s, _ in normalized}, reverse=True)
        if len(spatials) != 3:
            raise FaceBackendError(
                f"Expected exactly 3 distinct spatial sizes from SCRFD, got {spatials}"
            )
        stride_for_spatial = dict(zip(spatials, _STRIDES))

        out: Dict[str, Tuple[int, str]] = {}
        for name, spatial, kind in normalized:
            out[name] = (stride_for_spatial[spatial], kind)
        return out

    # -- preprocessing --------------------------------------------------- #
    def _preprocess_detect(
        self, bgr: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int], Tuple[int, int]]:
        """Letterbox + normalize to NCHW float32 for SCRFD.

        Normalization matches the Hailo/InsightFace convention:
            x = (pixel - 127.5) / 128.0
        """
        model_h, model_w = self.detector_input_hw
        h, w = bgr.shape[:2]
        scale = min(model_w / w, model_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(bgr, (new_w, new_h))
        padded = np.zeros((model_h, model_w, 3), dtype=np.uint8)
        top = (model_h - new_h) // 2
        left = (model_w - new_w) // 2
        padded[top : top + new_h, left : left + new_w] = resized

        # BGR->RGB, HWC->CHW, normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1).astype(np.float32, copy=False)
        chw = (chw - 127.5) / 128.0
        nchw = np.expand_dims(chw, 0).astype(np.float32, copy=False)
        nchw = np.ascontiguousarray(nchw)
        return nchw, scale, (left, top), (model_h, model_w)

    def _preprocess_embed(self, aligned_112x112: np.ndarray) -> np.ndarray:
        img = aligned_112x112
        h, w = img.shape[:2]
        if (h, w) != (_EMB_INPUT_H, _EMB_INPUT_W):
            img = cv2.resize(img, (_EMB_INPUT_W, _EMB_INPUT_H))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1).astype(np.float32, copy=False)
        # ArcFace normalization: x = (pixel / 127.5) - 1.0
        chw = (chw / 127.5) - 1.0
        nchw = np.expand_dims(chw, 0).astype(np.float32, copy=False)
        return np.ascontiguousarray(nchw)

    # -- inference ------------------------------------------------------- #
    def detect_raw(
        self, bgr: np.ndarray
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self._detector is None:
            raise FaceBackendError("TensorRTBackend not loaded")

        nchw, scale, offset, model_hw = self._preprocess_detect(bgr)

        with self._det_lock:
            raw = self._detector.infer(nchw)

        by_stride: Dict[int, Dict[str, np.ndarray]] = {s: {} for s in _STRIDES}
        for name, buf in raw.items():
            mapping = self._det_output_map.get(name)
            if mapping is None:
                continue
            stride, kind = mapping
            # Normalize layout to (N, C_per_anchor) where:
            #   scores  -> (N, 1)
            #   bboxes  -> (N, 4)
            #   kps     -> (N, 10)
            # Input shape can be:
            #   (1, C, H, W) — fold anchors into N: N = H*W*num_anchors, per-anchor C = C/num_anchors
            #   (1, N, C)    — already in per-anchor layout, C in {1,4,10}.
            shape = buf.shape
            if buf.ndim == 4:
                _, c, h, w = shape
                # SCRFD packs num_anchors=2 along channel dimension as
                # [anchor0_kind, anchor1_kind, ...]. To unfold we reshape
                # to (1, num_anchors, C/num_anchors, H, W) → transpose →
                # (H, W, num_anchors, C/num_anchors) → flatten.
                num_anchors = 2
                cpa = c // num_anchors  # channels per anchor
                arr = buf.reshape(1, num_anchors, cpa, h, w)
                arr = np.transpose(arr, (0, 3, 4, 1, 2))  # (1, H, W, A, Cpa)
                arr = arr.reshape(-1, cpa)
            elif buf.ndim in (2, 3):
                # Already in (N, C) or (1, N, C) layout with anchors folded into N.
                arr = buf.reshape(-1, shape[-1])
            else:
                raise FaceBackendError(
                    f"Unexpected SCRFD output rank for {name}: shape={shape}"
                )

            if kind == "scores":
                arr = arr.reshape(-1, 1)
            elif kind == "bboxes":
                arr = arr.reshape(-1, 4)
            else:  # kps
                arr = arr.reshape(-1, 10)
            by_stride[stride][kind] = arr.astype(np.float32, copy=False)

        out: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for stride, parts in by_stride.items():
            if {"scores", "bboxes", "kps"}.issubset(parts.keys()):
                out[stride] = (parts["scores"], parts["bboxes"], parts["kps"])

        out[-1] = (  # type: ignore[assignment]
            np.array(
                [scale, offset[0], offset[1], model_hw[0], model_hw[1]],
                dtype=np.float32,
            ),
        )
        return out

    def embed_raw(self, aligned_112x112: np.ndarray) -> np.ndarray:
        if self._embedder is None:
            raise FaceBackendError("TensorRTBackend not loaded")

        nchw = self._preprocess_embed(aligned_112x112)
        with self._emb_lock:
            raw = self._embedder.infer(nchw)

        # ArcFace ONNX typically emits a single 1x512 output.
        if len(raw) == 1:
            buf = next(iter(raw.values()))
        else:
            buf = next(
                (b for b in raw.values() if int(np.prod(b.shape)) == 512),
                next(iter(raw.values())),
            )
        emb = buf.flatten().astype(np.float32, copy=False)
        if emb.size != 512:
            if emb.size > 512:
                emb = emb[:512]
            else:
                emb = np.concatenate(
                    [emb, np.zeros(512 - emb.size, dtype=np.float32)]
                )
        return emb
