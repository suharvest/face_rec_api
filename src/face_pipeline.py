"""
Face Pipeline - Simplified face detection, alignment, and embedding extraction
Extracted from face_embed_api for standalone service use
"""
import base64
import logging
import os
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.transform import SimilarityTransform

# Hailo imports
try:
    from hailo_platform import VDevice, HailoSchedulingAlgorithm
except (ImportError, ModuleNotFoundError):
    raise ImportError("HailoRT is not installed. Please install it following the setup guide.")

import config

logger = logging.getLogger(__name__)

# Standard ArcFace landmark positions for 112x112 aligned face
ARCFACE_DEST_LANDMARKS = np.array([
    [38.2946, 51.6963],  # Left eye
    [73.5318, 51.5014],  # Right eye
    [56.0252, 71.7366],  # Nose
    [41.5493, 92.3655],  # Left mouth
    [70.7299, 92.2041]   # Right mouth
], dtype=np.float32)


class FaceDetector:
    """Face detection using SCRFD model on Hailo-8"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.infer_model = None
        self.input_queue = queue.Queue(maxsize=20)
        self.output_queue = queue.Queue(maxsize=20)
        self.quant_infos = {}
        self.inference_thread = None
        self._initialize()

    def _initialize(self):
        """Initialize Hailo device and load detection model"""
        logger.info(f"Loading face detection model: {self.model_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Detection model not found: {self.model_path}")

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.target = VDevice(params)

        self.infer_model = self.target.create_infer_model(self.model_path)

        # Extract quantization parameters
        vstream_infos = self.infer_model.hef.get_output_vstream_infos()
        self.quant_infos = {
            info.name: (info.quant_info.qp_scale, info.quant_info.qp_zp)
            for info in vstream_infos
        }

        # Start inference thread
        self.inference_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True
        )
        self.inference_thread.start()

        logger.info("Face detection model loaded successfully")

    def _inference_loop(self):
        """Run inference loop in dedicated thread"""
        def callback(completion_info):
            if completion_info.exception:
                logger.error(f"Detection inference error: {completion_info.exception}")

        with self.infer_model.configure() as configured_model:
            while True:
                batch_data = self.input_queue.get()
                if batch_data is None:
                    break

                original_frame, preprocessed_frame = batch_data

                try:
                    output_buffers = {
                        info.name: np.empty(info.shape, dtype=np.uint8)
                        for info in self.infer_model.outputs
                    }
                    bindings = configured_model.create_bindings(output_buffers=output_buffers)
                    bindings.input().set_buffer(preprocessed_frame)

                    configured_model.wait_for_async_ready(timeout_ms=10000)
                    job = configured_model.run_async([bindings], callback)
                    job.wait(10000)

                    self.output_queue.put((original_frame, output_buffers))

                except Exception as e:
                    logger.error(f"Detection inference failed: {e}", exc_info=True)
                    self.output_queue.put((original_frame, e))

    def detect(self, image: np.ndarray, confidence_threshold: float = 0.55,
               nms_threshold: float = 0.45, min_face_size: int = 8) -> List[Dict]:
        """
        Detect faces in image

        Returns:
            List of dicts with keys: bbox (x,y,w,h), landmarks (5 points), confidence
        """
        h, w = image.shape[:2]

        # Preprocess image
        preprocessed, scale, offset = self._preprocess_image(image)

        # Run inference
        self.input_queue.put((image, preprocessed))

        try:
            original_frame, results = self.output_queue.get(timeout=15.0)
            if isinstance(results, Exception):
                raise RuntimeError("Detection inference failed") from results

            # Parse results
            input_shape = self.infer_model.input().shape
            faces = self._parse_results(
                results, (h, w), (input_shape[0], input_shape[1]),
                scale, offset, confidence_threshold, nms_threshold, min_face_size
            )

            return faces

        except queue.Empty:
            raise RuntimeError("Detection inference timeout")

    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess image for detection model - resize and pad"""
        input_shape = self.infer_model.input().shape
        model_h, model_w = int(input_shape[0]), int(input_shape[1])
        h, w = image.shape[:2]

        # Calculate scale to maintain aspect ratio
        scale = min(model_w / w, model_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize and pad
        resized = cv2.resize(image, (new_w, new_h))
        padded = np.zeros((model_h, model_w, 3), dtype=np.uint8)

        # Center the image
        top = (model_h - new_h) // 2
        left = (model_w - new_w) // 2
        padded[top:top + new_h, left:left + new_w] = resized

        return padded, scale, (left, top)

    def _parse_results(self, raw_outputs: Dict[str, np.ndarray],
                      original_shape: Tuple[int, int],
                      model_shape: Tuple[int, int],
                      scale: float, offset: Tuple[int, int],
                      conf_thresh: float, nms_thresh: float,
                      min_size: int) -> List[Dict]:
        """Parse SCRFD detection results"""
        strides = [8, 16, 32]
        num_anchors = 2

        # Generate anchors
        anchors = self._generate_anchors(model_shape, strides, num_anchors)

        all_proposals = []

        # Decode each stride
        for stride in strides:
            # Get layer names
            if stride == 8:
                score_layer = 'scrfd_10g/conv41'
                bbox_layer = 'scrfd_10g/conv42'
                kps_layer = 'scrfd_10g/conv43'
            elif stride == 16:
                score_layer = 'scrfd_10g/conv49'
                bbox_layer = 'scrfd_10g/conv50'
                kps_layer = 'scrfd_10g/conv51'
            else:  # stride == 32
                score_layer = 'scrfd_10g/conv56'
                bbox_layer = 'scrfd_10g/conv57'
                kps_layer = 'scrfd_10g/conv58'

            scores_raw = raw_outputs.get(score_layer)
            bbox_raw = raw_outputs.get(bbox_layer)
            kps_raw = raw_outputs.get(kps_layer)

            if scores_raw is None or bbox_raw is None or kps_raw is None:
                continue

            # Dequantize
            score_scale, score_zp = self.quant_infos[score_layer]
            bbox_scale, bbox_zp = self.quant_infos[bbox_layer]
            kps_scale, kps_zp = self.quant_infos[kps_layer]

            scores = (scores_raw.astype(np.float32) - score_zp) * score_scale
            scores = scores.reshape(-1, 1)

            bbox_deltas = (bbox_raw.astype(np.float32) - bbox_zp) * bbox_scale
            bbox_deltas = bbox_deltas.reshape(-1, 4)

            kps_deltas = (kps_raw.astype(np.float32) - kps_zp) * kps_scale
            kps_deltas = kps_deltas.reshape(-1, 10)

            # Filter by confidence
            current_anchors = anchors[stride]
            keep_idx = np.where(scores >= conf_thresh)[0]
            if keep_idx.shape[0] == 0:
                continue

            scores = scores[keep_idx]
            bbox_deltas = bbox_deltas[keep_idx]
            kps_deltas = kps_deltas[keep_idx]
            current_anchors = current_anchors[keep_idx]

            # Decode boxes
            anchor_cx = current_anchors[:, 0]
            anchor_cy = current_anchors[:, 1]

            x1 = anchor_cx - bbox_deltas[:, 0] * stride
            y1 = anchor_cy - bbox_deltas[:, 1] * stride
            x2 = anchor_cx + bbox_deltas[:, 2] * stride
            y2 = anchor_cy + bbox_deltas[:, 3] * stride
            boxes = np.stack([x1, y1, x2, y2], axis=-1)

            # Decode landmarks
            kps = np.zeros_like(kps_deltas)
            for i in range(5):
                kps[:, i * 2] = anchor_cx + kps_deltas[:, i * 2] * stride
                kps[:, i * 2 + 1] = anchor_cy + kps_deltas[:, i * 2 + 1] * stride

            proposals = np.concatenate([boxes, scores, kps], axis=1)
            all_proposals.append(proposals)

        if not all_proposals:
            return []

        # NMS
        all_proposals = np.concatenate(all_proposals, axis=0)
        boxes_nms = all_proposals[:, :4]
        scores_nms = all_proposals[:, 4]

        keep = self._nms(boxes_nms, scores_nms, nms_thresh)
        final_proposals = all_proposals[keep]

        # Scale back to original image
        h_orig, w_orig = original_shape
        offset_x, offset_y = offset

        faces = []
        for prop in final_proposals:
            x1 = max(0, min(w_orig, (prop[0] - offset_x) / scale))
            y1 = max(0, min(h_orig, (prop[1] - offset_y) / scale))
            x2 = max(0, min(w_orig, (prop[2] - offset_x) / scale))
            y2 = max(0, min(h_orig, (prop[3] - offset_y) / scale))

            bbox_w = x2 - x1
            bbox_h = y2 - y1

            if min(bbox_w, bbox_h) < min_size:
                continue

            landmarks = []
            for i in range(5):
                kx = (prop[5 + i * 2] - offset_x) / scale
                ky = (prop[5 + i * 2 + 1] - offset_y) / scale
                landmarks.append((float(kx), float(ky)))

            faces.append({
                'bbox': {'x': int(x1), 'y': int(y1), 'w': int(bbox_w), 'h': int(bbox_h)},
                'landmarks': landmarks,
                'confidence': float(prop[4])
            })

        return faces

    def _generate_anchors(self, model_shape: Tuple[int, int],
                         strides: List[int], num_anchors: int) -> Dict[int, np.ndarray]:
        """Generate anchor boxes for SCRFD"""
        all_anchors = {}
        for stride in strides:
            fh = model_shape[0] // stride
            fw = model_shape[1] // stride

            x_centers = (np.arange(fw) + 0.5) * stride
            y_centers = (np.arange(fh) + 0.5) * stride

            xv, yv = np.meshgrid(x_centers, y_centers)
            centers = np.stack([xv, yv], axis=-1).reshape(-1, 2)

            repeated = np.repeat(centers, num_anchors, axis=0)
            stride_col = np.full((repeated.shape[0], 1), stride)

            all_anchors[stride] = np.concatenate([repeated, stride_col], axis=1)

        return all_anchors

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, thresh: float) -> List[int]:
        """Non-maximum suppression"""
        if boxes.shape[0] == 0:
            return []

        idxs = scores.argsort()[::-1]

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        area = (x2 - x1) * (y2 - y1)

        keep = []
        while idxs.size > 0:
            i = idxs[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)

            intersection = w * h
            union = area[i] + area[idxs[1:]] - intersection
            iou = intersection / union

            remaining = np.where(iou <= thresh)[0]
            idxs = idxs[remaining + 1]

        return keep


class FaceAligner:
    """Align faces using landmark-based similarity transformation"""

    @staticmethod
    def align(image: np.ndarray, landmarks: List[Tuple[float, float]],
              output_size: int = 112) -> np.ndarray:
        """
        Align face using 5-point landmarks

        Args:
            image: Input image
            landmarks: List of 5 (x, y) tuples
            output_size: Output image size (default 112x112)

        Returns:
            Aligned face image
        """
        src_landmarks = np.array(landmarks, dtype=np.float32)

        # Estimate transformation
        tform = SimilarityTransform()
        tform.estimate(src_landmarks, ARCFACE_DEST_LANDMARKS)
        M = tform.params[0:2, :]

        # Apply warp
        aligned = cv2.warpAffine(image, M, (output_size, output_size), borderValue=0.0)

        return aligned


class FaceEmbedder:
    """Extract face embeddings using ArcFace model on Hailo-8"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.infer_model = None
        self.input_queue = queue.Queue(maxsize=20)
        self.output_queue = queue.Queue(maxsize=20)
        self.quant_infos = {}
        self.inference_thread = None
        self._initialize()

    def _initialize(self):
        """Initialize Hailo device and load recognition model"""
        logger.info(f"Loading face recognition model: {self.model_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Recognition model not found: {self.model_path}")

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.target = VDevice(params)

        self.infer_model = self.target.create_infer_model(self.model_path)

        # Extract quantization parameters
        vstream_infos = self.infer_model.hef.get_output_vstream_infos()
        self.quant_infos = {
            info.name: (info.quant_info.qp_scale, info.quant_info.qp_zp)
            for info in vstream_infos
        }

        # Start inference thread
        self.inference_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True
        )
        self.inference_thread.start()

        logger.info("Face recognition model loaded successfully")

    def _inference_loop(self):
        """Run inference loop in dedicated thread"""
        def callback(completion_info):
            if completion_info.exception:
                logger.error(f"Recognition inference error: {completion_info.exception}")

        with self.infer_model.configure() as configured_model:
            while True:
                batch_data = self.input_queue.get()
                if batch_data is None:
                    break

                original_frame, preprocessed_frame = batch_data

                try:
                    output_buffers = {
                        info.name: np.empty(info.shape, dtype=np.uint8)
                        for info in self.infer_model.outputs
                    }
                    bindings = configured_model.create_bindings(output_buffers=output_buffers)
                    bindings.input().set_buffer(preprocessed_frame)

                    configured_model.wait_for_async_ready(timeout_ms=10000)
                    job = configured_model.run_async([bindings], callback)
                    job.wait(10000)

                    result = list(output_buffers.values())[0] if len(output_buffers) == 1 else output_buffers
                    self.output_queue.put((original_frame, result))

                except Exception as e:
                    logger.error(f"Recognition inference failed: {e}", exc_info=True)
                    self.output_queue.put((original_frame, e))

    def embed(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract 512-D embedding from aligned face image

        Args:
            face_image: Aligned face image (should be 112x112 from aligner)

        Returns:
            512-D L2-normalized embedding vector
        """
        # Preprocess for model
        preprocessed = self._preprocess(face_image)

        # Run inference
        self.input_queue.put((face_image, preprocessed))

        try:
            original_frame, result = self.output_queue.get(timeout=15.0)
            if isinstance(result, Exception):
                raise RuntimeError("Recognition inference failed") from result

            # Get output
            if isinstance(result, dict):
                output_name = list(result.keys())[0]
                embedding_raw = result[output_name]
            else:
                output_name = self.infer_model.hef.get_output_vstream_infos()[0].name
                embedding_raw = result

            # Dequantize
            if output_name in self.quant_infos:
                scale, zp = self.quant_infos[output_name]
                embedding = (embedding_raw.astype(np.float32) - zp) * scale
            else:
                embedding = embedding_raw.astype(np.float32)

            # Flatten and ensure 512-D
            embedding = embedding.flatten()
            if len(embedding) != 512:
                if len(embedding) > 512:
                    embedding = embedding[:512]
                else:
                    padding = np.zeros(512 - len(embedding))
                    embedding = np.concatenate([embedding, padding])

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding.astype(np.float32)

        except queue.Empty:
            raise RuntimeError("Recognition inference timeout")

    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face for recognition model"""
        model_shape = self.infer_model.input().shape
        model_h, model_w = int(model_shape[0]), int(model_shape[1])
        h, w = face_image.shape[:2]

        if h == 0 or w == 0:
            raise ValueError("Invalid face image dimensions")

        # Scale to fit model input
        scale = min(model_w / w, model_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(face_image, (new_w, new_h))

        # Pad with black
        padded = np.zeros((model_h, model_w, 3), dtype=np.uint8)
        top = (model_h - new_h) // 2
        left = (model_w - new_w) // 2
        padded[top:top + new_h, left:left + new_w] = resized

        return padded


class FacePipeline:
    """Complete face recognition pipeline: detect -> align -> embed"""

    def __init__(self, detection_model_path: str = None, recognition_model_path: str = None):
        self.detection_model_path = detection_model_path or config.FACE_DETECTION_MODEL
        self.recognition_model_path = recognition_model_path or config.FACE_RECOGNITION_MODEL

        logger.info("Initializing Face Pipeline...")
        self.detector = FaceDetector(self.detection_model_path)
        self.aligner = FaceAligner()
        self.embedder = FaceEmbedder(self.recognition_model_path)
        logger.info("Face Pipeline initialized successfully")

    def process_image(self, image: np.ndarray, strategy: str = "largest") -> Dict:
        """
        Process image through full pipeline

        Args:
            image: Input image (BGR format)
            strategy: How to handle multiple faces ("largest", "first", "error")

        Returns:
            Dict with keys:
                - success: bool
                - embedding: List[float] (512-D) if success
                - face: Dict with bbox, landmarks, confidence
                - error: str if not success
        """
        try:
            # Detect faces
            faces = self.detector.detect(image,
                                        confidence_threshold=config.CONFIDENCE_THRESHOLD,
                                        nms_threshold=config.NMS_THRESHOLD,
                                        min_face_size=config.MIN_FACE_SIZE)

            if len(faces) == 0:
                return {
                    'success': False,
                    'error': 'No face detected',
                    'face': None,
                    'embedding': None
                }

            # Handle multiple faces
            if len(faces) > 1:
                if strategy == "error":
                    return {
                        'success': False,
                        'error': f'Multiple faces detected ({len(faces)})',
                        'face': None,
                        'embedding': None
                    }
                elif strategy == "largest":
                    # Take face with largest area
                    face = max(faces, key=lambda f: f['bbox']['w'] * f['bbox']['h'])
                else:  # "first"
                    face = faces[0]
            else:
                face = faces[0]

            # Align face
            aligned = self.aligner.align(image, face['landmarks'])

            # Extract embedding
            embedding = self.embedder.embed(aligned)

            return {
                'success': True,
                'embedding': embedding.tolist(),
                'face': face,
                'error': None
            }

        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'face': None,
                'embedding': None
            }

    def process_image_base64(self, image_base64: str, strategy: str = "largest") -> Dict:
        """Process base64-encoded image"""
        try:
            image_data = base64.b64decode(image_base64)
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                return {
                    'success': False,
                    'error': 'Failed to decode image',
                    'face': None,
                    'embedding': None
                }

            return self.process_image(image, strategy)

        except Exception as e:
            logger.error(f"Failed to process base64 image: {e}")
            return {
                'success': False,
                'error': str(e),
                'face': None,
                'embedding': None
            }
