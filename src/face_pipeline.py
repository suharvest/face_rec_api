"""
FacePipeline — hardware-agnostic face recognition pipeline.

Orchestrates: BGR image -> backend.detect_raw -> SCRFD decode + NMS ->
5-point landmark alignment + 112x112 crop -> backend.embed_raw ->
L2 normalization -> result dict.

All hardware-specific bits live behind :class:`backends.base.FaceBackend`.
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.transform import SimilarityTransform

import config
import liveness as liveness_util
from backends import FaceBackend, FaceBackendError, create_backend

logger = logging.getLogger(__name__)

# Standard ArcFace landmark positions for 112x112 aligned face
ARCFACE_DEST_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],  # Left eye
        [73.5318, 51.5014],  # Right eye
        [56.0252, 71.7366],  # Nose
        [41.5493, 92.3655],  # Left mouth
        [70.7299, 92.2041],  # Right mouth
    ],
    dtype=np.float32,
)

_STRIDES = (8, 16, 32)
_NUM_ANCHORS = 2


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _generate_anchors(model_h: int, model_w: int) -> Dict[int, np.ndarray]:
    """SCRFD anchor centers per stride (shape: ``(N, 2)`` of ``(cx, cy)``)."""
    anchors: Dict[int, np.ndarray] = {}
    for stride in _STRIDES:
        fh = model_h // stride
        fw = model_w // stride
        x_centers = (np.arange(fw) + 0.5) * stride
        y_centers = (np.arange(fh) + 0.5) * stride
        xv, yv = np.meshgrid(x_centers, y_centers)
        centers = np.stack([xv, yv], axis=-1).reshape(-1, 2)
        anchors[stride] = np.repeat(centers, _NUM_ANCHORS, axis=0)
    return anchors


def _nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> List[int]:
    """Plain greedy NMS, returns indices to keep."""
    if boxes.shape[0] == 0:
        return []
    idxs = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    keep: List[int] = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = area[i] + area[idxs[1:]] - inter
        iou = inter / np.maximum(union, 1e-9)
        remaining = np.where(iou <= thresh)[0]
        idxs = idxs[remaining + 1]
    return keep


def _align(image: np.ndarray, landmarks: List[Tuple[float, float]],
           output_size: int = 112) -> np.ndarray:
    """5-point similarity-transform alignment to 112x112 ArcFace canonical pose."""
    src = np.array(landmarks, dtype=np.float32)
    tform = SimilarityTransform()
    tform.estimate(src, ARCFACE_DEST_LANDMARKS)
    M = tform.params[0:2, :]
    return cv2.warpAffine(image, M, (output_size, output_size), borderValue=0.0)


def resolve_liveness_path() -> Tuple[Optional[str], str]:
    """Resolve the liveness model path from config, with graceful degradation.

    Returns:
        ``(path, status)`` — ``path`` is the model file to load (or None),
        ``status`` is one of:
          - ``"loaded"``   liveness requested and model file exists;
          - ``"disabled"`` liveness turned off via ``LIVENESS_ENABLED``;
          - ``"missing"``  liveness requested but model file absent —
            auto-degraded to off (warning, no crash).
    """
    if not config.LIVENESS_ENABLED:
        return None, "disabled"
    path = config.FACE_LIVENESS_MODEL
    if not os.path.exists(path):
        logger.warning(
            "LIVENESS_ENABLED=true but liveness model not found at %s — "
            "liveness auto-disabled (recognition continues without "
            "anti-spoofing)", path,
        )
        return None, "missing"
    return path, "loaded"


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class FacePipeline:
    """Hardware-agnostic face recognition orchestration."""

    def __init__(
        self,
        backend: Optional[FaceBackend] = None,
        *,
        detection_model_path: Optional[str] = None,
        recognition_model_path: Optional[str] = None,
        backend_name: Optional[str] = None,
    ):
        """
        Args:
            backend: Pre-instantiated backend (preferred). If None, one is
                created from ``backend_name`` (or ``config.FACE_BACKEND``).
            detection_model_path: override for detector path.
            recognition_model_path: override for embedder path.
            backend_name: override for backend selection.
        """
        if backend is None:
            liveness_path, liveness_status = resolve_liveness_path()
            backend = create_backend(backend_name or config.FACE_BACKEND)
            backend.load(
                detector_path=detection_model_path or config.FACE_DETECTION_MODEL,
                embedder_path=recognition_model_path or config.FACE_RECOGNITION_MODEL,
                liveness_path=liveness_path,
            )
            if liveness_path is not None and not backend.liveness_loaded:
                # Backend accepted but ignored the model (no liveness support).
                logger.warning(
                    "Backend %s has no liveness support — liveness disabled",
                    backend.backend_name,
                )
                liveness_status = "disabled"
        else:
            # Pre-instantiated (and pre-loaded) backend, e.g. in tests.
            if not config.LIVENESS_ENABLED:
                liveness_status = "disabled"
            elif backend.liveness_loaded:
                liveness_status = "loaded"
            else:
                logger.warning(
                    "LIVENESS_ENABLED=true but provided backend has no "
                    "liveness model loaded — liveness disabled"
                )
                liveness_status = "missing"
        self.backend: FaceBackend = backend
        # "loaded" | "disabled" | "missing" — surfaced by /health.
        self.liveness_status: str = liveness_status
        self._anchor_cache: Dict[Tuple[int, int], Dict[int, np.ndarray]] = {}
        logger.info(
            "FacePipeline ready (backend=%s, model_tag=%s, liveness=%s)",
            self.backend.backend_name,
            self.backend.model_tag,
            self.liveness_status,
        )

    @property
    def liveness_active(self) -> bool:
        """True when the anti-spoofing step actually runs."""
        return self.liveness_status == "loaded"

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        try:
            self.backend.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Backend close error: %s", exc)

    # ------------------------------------------------------------------ #
    def _decode_detections(
        self,
        raw: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        original_shape: Tuple[int, int],
        conf_thresh: float,
        nms_thresh: float,
        min_size: int,
    ) -> List[Dict]:
        """Decode SCRFD raw outputs to image-space detections."""
        # Pop preprocessing meta written by backend.detect_raw under key -1.
        meta = raw.pop(-1, None)  # type: ignore[arg-type]
        if meta is None:
            raise FaceBackendError("Backend did not provide preprocessing meta")
        scale, offset_x, offset_y, model_h, model_w = meta[0].tolist()
        model_h, model_w = int(model_h), int(model_w)

        cache_key = (model_h, model_w)
        if cache_key not in self._anchor_cache:
            self._anchor_cache[cache_key] = _generate_anchors(model_h, model_w)
        anchors = self._anchor_cache[cache_key]

        all_props: List[np.ndarray] = []
        for stride in _STRIDES:
            if stride not in raw:
                continue
            scores, bbox_deltas, kps_deltas = raw[stride]
            keep_idx = np.where(scores >= conf_thresh)[0]
            if keep_idx.size == 0:
                continue
            scores = scores[keep_idx]
            bbox_deltas = bbox_deltas[keep_idx]
            kps_deltas = kps_deltas[keep_idx]
            cur_anchors = anchors[stride][keep_idx]

            ax = cur_anchors[:, 0]
            ay = cur_anchors[:, 1]
            x1 = ax - bbox_deltas[:, 0] * stride
            y1 = ay - bbox_deltas[:, 1] * stride
            x2 = ax + bbox_deltas[:, 2] * stride
            y2 = ay + bbox_deltas[:, 3] * stride
            boxes = np.stack([x1, y1, x2, y2], axis=-1)

            kps = np.zeros_like(kps_deltas)
            for i in range(5):
                kps[:, i * 2] = ax + kps_deltas[:, i * 2] * stride
                kps[:, i * 2 + 1] = ay + kps_deltas[:, i * 2 + 1] * stride

            all_props.append(np.concatenate([boxes, scores, kps], axis=1))

        if not all_props:
            return []
        props = np.concatenate(all_props, axis=0)

        keep = _nms(props[:, :4], props[:, 4], nms_thresh)
        props = props[keep]

        h_orig, w_orig = original_shape
        faces: List[Dict] = []
        for p in props:
            x1 = max(0.0, min(float(w_orig), (p[0] - offset_x) / scale))
            y1 = max(0.0, min(float(h_orig), (p[1] - offset_y) / scale))
            x2 = max(0.0, min(float(w_orig), (p[2] - offset_x) / scale))
            y2 = max(0.0, min(float(h_orig), (p[3] - offset_y) / scale))
            bw = x2 - x1
            bh = y2 - y1
            if min(bw, bh) < min_size:
                continue
            lm: List[Tuple[float, float]] = []
            for i in range(5):
                kx = (p[5 + i * 2] - offset_x) / scale
                ky = (p[5 + i * 2 + 1] - offset_y) / scale
                lm.append((float(kx), float(ky)))
            faces.append(
                {
                    "bbox": {
                        "x": int(x1),
                        "y": int(y1),
                        "w": int(bw),
                        "h": int(bh),
                    },
                    "landmarks": lm,
                    "confidence": float(p[4]),
                }
            )
        return faces

    # ------------------------------------------------------------------ #
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = None,
        nms_threshold: float = None,
        min_face_size: int = None,
    ) -> List[Dict]:
        """Detect faces. Returns list of dicts with bbox/landmarks/confidence."""
        conf = confidence_threshold if confidence_threshold is not None else config.CONFIDENCE_THRESHOLD
        nms = nms_threshold if nms_threshold is not None else config.NMS_THRESHOLD
        mins = min_face_size if min_face_size is not None else config.MIN_FACE_SIZE
        h, w = image.shape[:2]
        raw = self.backend.detect_raw(image)
        return self._decode_detections(raw, (h, w), conf, nms, mins)

    def embed(self, aligned_112x112: np.ndarray) -> np.ndarray:
        """Return L2-normalized 512-D fp32 embedding."""
        emb = self.backend.embed_raw(aligned_112x112)
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32, copy=False)

    def align(self, image: np.ndarray, landmarks: List[Tuple[float, float]]) -> np.ndarray:
        return _align(image, landmarks)

    # ------------------------------------------------------------------ #
    def liveness_score(self, image: np.ndarray, bbox: Dict) -> float:
        """Passive anti-spoofing score (real probability 0-1) for one face.

        The bbox-expansion crop (Minivision CropImage, scale=2.7, clamped to
        image borders) + 80x80 resize live in the shared ``liveness`` util so
        every backend consumes the identical input; only the inference call
        itself (``backend.liveness_raw``) is backend-specific.
        """
        crop = liveness_util.crop_liveness_input(
            image,
            (bbox["x"], bbox["y"], bbox["w"], bbox["h"]),
            scale=config.LIVENESS_CROP_SCALE,
        )
        return float(self.backend.liveness_raw(crop))

    def _annotate_liveness(
        self, image: np.ndarray, face: Dict
    ) -> Tuple[Optional[float], Optional[bool]]:
        """Run liveness for one detected face and attach score/flag to it.

        Returns ``(liveness_score, live)`` — both None when liveness is
        inactive.
        """
        if not self.liveness_active:
            return None, None
        score = self.liveness_score(image, face["bbox"])
        live = score >= config.LIVENESS_THRESHOLD
        face["liveness_score"] = score
        face["live"] = live
        return score, live

    # ------------------------------------------------------------------ #
    def process_image(self, image: np.ndarray, strategy: str = "largest") -> Dict:
        """Run full pipeline; returns dict with success / embedding / face / error."""
        try:
            faces = self.detect(image)

            if len(faces) == 0:
                return {
                    "success": False,
                    "error": "No face detected",
                    "face": None,
                    "embedding": None,
                }

            if len(faces) > 1:
                if strategy == "error":
                    return {
                        "success": False,
                        "error": f"Multiple faces detected ({len(faces)})",
                        "face": None,
                        "embedding": None,
                    }
                if strategy == "largest":
                    face = max(faces, key=lambda f: f["bbox"]["w"] * f["bbox"]["h"])
                else:
                    face = faces[0]
            else:
                face = faces[0]

            # Passive anti-spoofing — runs on the raw detection bbox BEFORE
            # align/embed so spoof faces never reach recognition in reject
            # mode. No-op (None/None) when liveness is disabled/missing, so
            # behavior is identical to the pre-liveness pipeline.
            liveness_score, live = self._annotate_liveness(image, face)
            if live is False and config.LIVENESS_FAIL_ACTION == "reject":
                return {
                    "success": False,
                    "error": (
                        f"Spoof detected (liveness_score="
                        f"{liveness_score:.4f} < {config.LIVENESS_THRESHOLD})"
                    ),
                    "reason": "spoof",
                    "face": face,
                    "embedding": None,
                    "live": live,
                    "liveness_score": liveness_score,
                }

            aligned = self.align(image, face["landmarks"])
            embedding = self.embed(aligned)

            return {
                "success": True,
                "embedding": embedding.tolist(),
                "face": face,
                "aligned": aligned,
                "error": None,
                "live": live,
                "liveness_score": liveness_score,
            }

        except FaceBackendError as exc:
            logger.error("Backend error in pipeline: %s", exc, exc_info=True)
            return {
                "success": False,
                "error": f"backend: {exc}",
                "face": None,
                "embedding": None,
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("Pipeline processing failed: %s", exc, exc_info=True)
            return {
                "success": False,
                "error": str(exc),
                "face": None,
                "embedding": None,
            }

    def process_image_base64(self, image_base64: str, strategy: str = "largest") -> Dict:
        """Decode base64 image then run full pipeline."""
        try:
            image_data = base64.b64decode(image_base64)
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                return {
                    "success": False,
                    "error": "Failed to decode image",
                    "face": None,
                    "embedding": None,
                }
            return self.process_image(image, strategy)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to process base64 image: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "face": None,
                "embedding": None,
            }

    def process_all_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect and embed *every* face. Used by /infer endpoint."""
        faces = self.detect(image)
        out: List[Dict] = []
        for face in faces:
            liveness_score, live = self._annotate_liveness(image, face)
            if live is False and config.LIVENESS_FAIL_ACTION == "reject":
                # Spoof face: keep detection info but never embed/recognize.
                out.append(
                    {
                        "bbox": face["bbox"],
                        "landmarks": face["landmarks"],
                        "confidence": face["confidence"],
                        "embedding": None,
                        "aligned": None,
                        "live": live,
                        "liveness_score": liveness_score,
                    }
                )
                continue
            try:
                aligned = self.align(image, face["landmarks"])
                emb = self.embed(aligned)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Embedding failed for one face: %s", exc)
                continue
            out.append(
                {
                    "bbox": face["bbox"],
                    "landmarks": face["landmarks"],
                    "confidence": face["confidence"],
                    "embedding": emb,
                    "aligned": aligned,
                    "live": live,
                    "liveness_score": liveness_score,
                }
            )
        return out
