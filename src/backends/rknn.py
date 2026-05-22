"""
Rockchip RKNN backend for face detection (SCRFD-10g) + embedding
(ArcFace-MobileFaceNet).

Targets RK3576 / RK3588 NPUs via the ``rknn-toolkit-lite2`` runtime
(``rknnlite.api.RKNNLite``). Loads pre-compiled ``.rknn`` files produced
by ``tools/build_rknn.py`` on an **x86 Linux** dev host with the full
``rknn-toolkit2`` (which itself pulls torch / onnx / tensorflow ~ 2 GB and
MUST NOT enter the runtime image).

This backend produces the same canonical output layout as the Hailo /
TensorRT backends so the upper :class:`face_pipeline.FacePipeline` stays
hardware-agnostic.

Runtime requirements at the device:
- ``/usr/lib/librknnrt.so`` (Rockchip NPU runtime, version >= 2.3.0). The
  Dockerfile expects this to be bind-mounted from the host.
- ``/dev/dri/renderD129`` (NPU char device). Pass via ``--device`` to docker.
- ``rknn-toolkit-lite2`` (pip, ~10 MB) — the **lite** wheel, not the full
  ``rknn-toolkit2`` conversion toolkit.

Important platform notes:
- RK3576 has a 2-core NPU. RK3588 has a 3-core NPU. ``.rknn`` engines are
  not portable between Rockchip SoC generations — build per target.
- The ``rknn-toolkit2`` conversion tool only supports x86 Linux with
  glibc >= 2.27 and Python 3.8-3.11; it does not run on macOS or aarch64.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import FaceBackend, FaceBackendError

logger = logging.getLogger(__name__)

# Deferred RKNN import — only needed at instantiation. Module is importable on
# macOS / Hailo / Jetson environments so other backends keep working.
try:
    from rknnlite.api import RKNNLite  # type: ignore

    _RKNN_AVAILABLE = True
    _RKNN_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _e:  # pragma: no cover - import gating only
    RKNNLite = None  # type: ignore
    _RKNN_AVAILABLE = False
    _RKNN_IMPORT_ERROR = _e


# SCRFD output classification — mirrors the TensorRT backend. We classify by
# per-tensor shape rather than name because ``rknn-toolkit2`` does not preserve
# the original ONNX output names in a uniform way across versions.
_SCRFD_CHANNELS = {2: "scores", 8: "bboxes", 20: "kps"}
_KIND_FOR_C_PER_ANCHOR = {1: "scores", 4: "bboxes", 10: "kps"}
_STRIDES = (8, 16, 32)
_DET_INPUT_H = 640
_DET_INPUT_W = 640
_EMB_INPUT_H = 112
_EMB_INPUT_W = 112


def _resolve_core_mask() -> int:
    """Pick an NPU core mask via env var, defaulting to AUTO.

    Override via ``RKNN_CORE_MASK`` env var. Accepted values (case-insensitive):
      AUTO, 0, 1, 2, 0_1, 0_1_2, ALL.

    RK3576 has 2 cores; RK3588 has 3. AUTO works on both — the lite runtime
    silently masks unavailable cores.
    """
    val = (os.environ.get("RKNN_CORE_MASK") or "AUTO").upper().strip()
    table = {
        "AUTO": RKNNLite.NPU_CORE_AUTO,
        "0": RKNNLite.NPU_CORE_0,
        "1": RKNNLite.NPU_CORE_1,
        "2": RKNNLite.NPU_CORE_2,
        "0_1": RKNNLite.NPU_CORE_0_1,
        "0_1_2": RKNNLite.NPU_CORE_0_1_2,
        "ALL": RKNNLite.NPU_CORE_ALL,
    }
    if val not in table:
        logger.warning("Unknown RKNN_CORE_MASK=%r, falling back to AUTO", val)
        return RKNNLite.NPU_CORE_AUTO
    return table[val]


class _RKNNModelRunner:
    """One RKNNLite instance + output-shape introspection."""

    def __init__(self, name: str, model_path: str) -> None:
        self.name = name
        self.model_path = model_path
        self.rknn: Optional["RKNNLite"] = None
        # output_shapes is populated lazily on the first inference, since
        # rknn-toolkit-lite2 does not expose a reliable pre-inference shape
        # query API.
        self.output_shapes: List[Tuple[int, ...]] = []

    def load(self, core_mask: int) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"{self.name} RKNN model not found: {self.model_path}\n"
                "Build with tools/build_rknn.py on an x86 Linux dev host."
            )
        self.rknn = RKNNLite(verbose=False)
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise FaceBackendError(
                f"{self.name}: load_rknn failed (ret={ret}) for {self.model_path}"
            )
        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise FaceBackendError(
                f"{self.name}: init_runtime failed (ret={ret}). "
                "Check /usr/lib/librknnrt.so is present and /dev/dri/renderD12{8,9} "
                "is accessible to the container."
            )
        logger.info("RKNN %s model loaded: %s", self.name, self.model_path)

    def infer(self, input_data: np.ndarray, data_format: str = "nhwc") -> List[np.ndarray]:
        """Run inference. Returns list of output arrays in model order."""
        if self.rknn is None:
            raise FaceBackendError(f"{self.name} runner not loaded")
        if not input_data.flags["C_CONTIGUOUS"]:
            input_data = np.ascontiguousarray(input_data)
        outputs = self.rknn.inference(inputs=[input_data], data_format=data_format)
        if outputs is None:
            raise FaceBackendError(f"{self.name}: inference returned None")
        if not self.output_shapes:
            self.output_shapes = [tuple(o.shape) for o in outputs]
            logger.info(
                "RKNN %s output shapes: %s", self.name, self.output_shapes
            )
        return outputs

    def close(self) -> None:
        if self.rknn is not None:
            try:
                self.rknn.release()
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s release error: %s", self.name, exc)
            self.rknn = None


class RKNNBackend(FaceBackend):
    """Rockchip RK3576 / RK3588 NPU implementation of :class:`FaceBackend`."""

    BACKEND_NAME = "rknn"
    MODEL_TAG = "rknn:scrfd10g+arcface_mbf_v1"

    def __init__(self) -> None:
        if not _RKNN_AVAILABLE:
            raise ImportError(
                "rknn-toolkit-lite2 is not installed. Install with "
                "`pip install rknn-toolkit-lite2` (or `uv sync --extra rknn`). "
                "Note: this is the LITE runtime — do not install the full "
                "`rknn-toolkit2` on the device (it pulls torch + tensorflow)."
            ) from _RKNN_IMPORT_ERROR

        self._detector: Optional[_RKNNModelRunner] = None
        self._embedder: Optional[_RKNNModelRunner] = None
        self._det_lock = threading.Lock()
        self._emb_lock = threading.Lock()
        # Maps detector output index -> (stride, kind). Built after the first
        # successful detect_raw call once we know the output shapes.
        self._det_output_map: Dict[int, Tuple[int, str]] = {}
        self._closed = False

    # -- lifecycle ------------------------------------------------------- #
    def load(self, detector_path: str, embedder_path: str) -> None:
        core_mask = _resolve_core_mask()
        logger.info(
            "Initializing RKNN backend (core_mask=%d, AUTO=%d)",
            core_mask, RKNNLite.NPU_CORE_AUTO,
        )

        self._detector = _RKNNModelRunner("detector", detector_path)
        self._detector.load(core_mask)
        self._embedder = _RKNNModelRunner("embedder", embedder_path)
        self._embedder.load(core_mask)

        # Warm up: 3 dummy infers. The first call after init_runtime() pays for
        # NPU kernel binding / weight upload; without warmup the first /infer
        # request can spike 50-150 ms.
        try:
            dummy_bgr = np.zeros((640, 640, 3), dtype=np.uint8)
            dummy_face = np.zeros((112, 112, 3), dtype=np.uint8)
            for _ in range(3):
                self.detect_raw(dummy_bgr)
                self.embed_raw(dummy_face)
            logger.info("RKNNBackend warmup complete (3 iterations)")
        except Exception as exc:  # noqa: BLE001
            logger.warning("RKNNBackend warmup failed: %s", exc)

        logger.info("RKNNBackend ready (model_tag=%s)", self.MODEL_TAG)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        logger.info("Shutting down RKNNBackend...")
        for runner in (self._detector, self._embedder):
            if runner is not None:
                try:
                    runner.close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Runner close error: %s", exc)
        logger.info("RKNNBackend shutdown complete")

    # -- properties ------------------------------------------------------ #
    @property
    def backend_name(self) -> str:
        return self.BACKEND_NAME

    @property
    def model_tag(self) -> str:
        return self.MODEL_TAG

    @property
    def detector_input_hw(self) -> Tuple[int, int]:
        # We pin SCRFD-10g to 640x640 at conversion time (see build_rknn.py).
        return _DET_INPUT_H, _DET_INPUT_W

    # -- preprocessing -------------------------------------------------- #
    def _preprocess_detect(
        self, bgr: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int], Tuple[int, int]]:
        """Letterbox to 640x640. Returns (NHWC uint8, scale, (left,top), (H,W)).

        The RKNN graph is built with ``mean_values`` and ``std_values`` baked
        into the model (see ``tools/build_rknn.py``), so we feed raw uint8
        in RGB NHWC and let the NPU do the normalization.
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
        # BGR -> RGB to match the ONNX export convention. We rely on
        # `quant_img_RGB2BGR=False` in build_rknn.py so the NPU expects RGB.
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        nhwc = np.expand_dims(rgb, 0)  # (1, H, W, 3) uint8
        return np.ascontiguousarray(nhwc), scale, (left, top), (model_h, model_w)

    def _preprocess_embed(self, aligned_112x112: np.ndarray) -> np.ndarray:
        img = aligned_112x112
        h, w = img.shape[:2]
        if (h, w) != (_EMB_INPUT_H, _EMB_INPUT_W):
            img = cv2.resize(img, (_EMB_INPUT_W, _EMB_INPUT_H))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        nhwc = np.expand_dims(rgb, 0)  # (1,112,112,3) uint8
        return np.ascontiguousarray(nhwc)

    # -- classification ------------------------------------------------- #
    def _classify_scrfd_outputs(self, outputs: List[np.ndarray]) -> None:
        """Build {output_index -> (stride, kind)} from one set of outputs.

        Strategy mirrors TensorRTBackend._classify_scrfd_outputs:
          - 4D (1, C, H, W): C in {2, 8, 20} => kind; spatial = H*W
          - 4D (1, H, W, C): NHWC fallback; same mapping
          - 3D (1, N, C):    per-anchor; spatial = N // 2
          - 2D (N, C):       per-anchor; spatial = N // 2
        Then group by spatial size; sort desc => strides 8/16/32.
        """
        normalized: List[Tuple[int, int, str]] = []  # (idx, spatial, kind)
        for idx, buf in enumerate(outputs):
            shape = buf.shape
            kind: Optional[str] = None
            spatial: Optional[int] = None
            if buf.ndim == 4:
                # Try both NCHW and NHWC layouts.
                _, d1, d2, d3 = shape
                # NCHW: d1=C
                if int(d1) in _SCRFD_CHANNELS:
                    kind = _SCRFD_CHANNELS[int(d1)]
                    spatial = int(d2) * int(d3)
                # NHWC: d3=C
                elif int(d3) in _SCRFD_CHANNELS:
                    kind = _SCRFD_CHANNELS[int(d3)]
                    spatial = int(d1) * int(d2)
            elif buf.ndim in (2, 3):
                if buf.ndim == 3:
                    _, n, c = shape
                else:
                    n, c = shape
                if int(n) % 2 == 0 and int(c) in _KIND_FOR_C_PER_ANCHOR:
                    kind = _KIND_FOR_C_PER_ANCHOR[int(c)]
                    spatial = int(n) // 2
            if kind is None or spatial is None:
                raise FaceBackendError(
                    f"Unrecognized SCRFD RKNN output #{idx} shape={shape}"
                )
            normalized.append((idx, spatial, kind))

        spatials = sorted({s for _, s, _ in normalized}, reverse=True)
        if len(spatials) != 3:
            raise FaceBackendError(
                f"Expected 3 distinct spatial sizes from SCRFD, got {spatials}"
            )
        stride_for_spatial = dict(zip(spatials, _STRIDES))

        out: Dict[int, Tuple[int, str]] = {}
        for idx, spatial, kind in normalized:
            out[idx] = (stride_for_spatial[spatial], kind)

        seen = set(out.values())
        expected = {(s, k) for s in _STRIDES for k in ("scores", "bboxes", "kps")}
        missing = expected - seen
        if missing:
            raise FaceBackendError(
                f"SCRFD RKNN output mapping incomplete; missing: {sorted(missing)}"
            )
        self._det_output_map = out
        logger.info("RKNN SCRFD output map: %s", self._det_output_map)

    # -- inference ------------------------------------------------------ #
    def detect_raw(
        self, bgr: np.ndarray
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self._detector is None:
            raise FaceBackendError("RKNNBackend not loaded")

        nhwc, scale, offset, model_hw = self._preprocess_detect(bgr)

        with self._det_lock:
            outputs = self._detector.infer(nhwc, data_format="nhwc")
            if not self._det_output_map:
                self._classify_scrfd_outputs(outputs)

        by_stride: Dict[int, Dict[str, np.ndarray]] = {s: {} for s in _STRIDES}
        for idx, buf in enumerate(outputs):
            mapping = self._det_output_map.get(idx)
            if mapping is None:
                continue
            stride, kind = mapping
            shape = buf.shape
            # Normalize to (N, Cpa)
            if buf.ndim == 4:
                _, d1, d2, d3 = shape
                if int(d1) in _SCRFD_CHANNELS:
                    # NCHW: (1, C, H, W), C contains num_anchors * Cpa
                    c, h, w = int(d1), int(d2), int(d3)
                    num_anchors = 2
                    cpa = c // num_anchors
                    arr = buf.reshape(1, num_anchors, cpa, h, w)
                    arr = np.transpose(arr, (0, 3, 4, 1, 2))  # (1,H,W,A,Cpa)
                    arr = arr.reshape(-1, cpa)
                else:
                    # NHWC: (1, H, W, C)
                    h, w, c = int(d1), int(d2), int(d3)
                    num_anchors = 2
                    cpa = c // num_anchors
                    arr = buf.reshape(1, h, w, num_anchors, cpa)
                    arr = arr.reshape(-1, cpa)
            else:
                arr = buf.reshape(-1, shape[-1])

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
            raise FaceBackendError("RKNNBackend not loaded")

        nhwc = self._preprocess_embed(aligned_112x112)
        with self._emb_lock:
            outputs = self._embedder.infer(nhwc, data_format="nhwc")

        if len(outputs) == 1:
            buf = outputs[0]
        else:
            buf = next(
                (b for b in outputs if int(np.prod(b.shape)) == 512),
                outputs[0],
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
