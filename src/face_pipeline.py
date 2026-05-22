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
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.transform import SimilarityTransform

import config
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
            backend = create_backend(backend_name or config.FACE_BACKEND)
            backend.load(
                detector_path=detection_model_path or config.FACE_DETECTION_MODEL,
                embedder_path=recognition_model_path or config.FACE_RECOGNITION_MODEL,
            )
        self.backend: FaceBackend = backend
        self._anchor_cache: Dict[Tuple[int, int], Dict[int, np.ndarray]] = {}
        logger.info(
            "FacePipeline ready (backend=%s, model_tag=%s)",
            self.backend.backend_name,
            self.backend.model_tag,
        )

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

            aligned = self.align(image, face["landmarks"])
            embedding = self.embed(aligned)

            return {
                "success": True,
                "embedding": embedding.tolist(),
                "face": face,
                "aligned": aligned,
                "error": None,
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
                }
            )
        return out
