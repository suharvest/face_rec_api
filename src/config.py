"""
Configuration for Standalone Face Recognition Service
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
PHOTOS_FOLDER = os.getenv("PHOTOS_FOLDER", str(BASE_DIR / "photos"))
EMBEDDINGS_JSON = os.getenv("EMBEDDINGS_JSON", str(BASE_DIR / "data" / "embeddings.json"))

# Backend selection (hailo | jetson | rknn). Default: hailo (P0).
FACE_BACKEND = os.getenv("FACE_BACKEND", "hailo").lower()

# Models live under models/<backend>/ by default.
MODELS_PATH = os.getenv("MODELS_PATH", str(BASE_DIR / "models" / FACE_BACKEND))

# Per-backend model file conventions
_BACKEND_EXT = {"hailo": "hef", "jetson": "engine", "rknn": "rknn"}
_EXT = _BACKEND_EXT.get(FACE_BACKEND, "bin")

# Model paths — prefer new env names, fall back to legacy ones for back-compat.
FACE_DETECTION_MODEL = os.getenv(
    "FACE_DETECTION_MODEL",
    os.getenv(
        "FACE_DETECTION_HEF",
        str(Path(MODELS_PATH) / f"scrfd_10g.{_EXT}"),
    ),
)
FACE_RECOGNITION_MODEL = os.getenv(
    "FACE_RECOGNITION_MODEL",
    os.getenv(
        "FACE_RECOGNITION_HEF",
        str(Path(MODELS_PATH) / f"arcface_mobilefacenet.{_EXT}"),
    ),
)

# Model tag (overridable). Backends also expose their own default tag.
_DEFAULT_MODEL_TAGS = {
    "hailo": "hailo:scrfd10g+arcface_mbf_v1",
    "jetson": "jetson:scrfd10g+arcface_mbf_v1",
    "rknn": "rknn:scrfd10g+arcface_mbf_v1",
}
MODEL_TAG = os.getenv("MODEL_TAG", _DEFAULT_MODEL_TAGS.get(FACE_BACKEND, "unknown"))

# Recognition settings
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.4"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
NMS_THRESHOLD = float(os.getenv("NMS_THRESHOLD", "0.45"))
MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "8"))

# Multiple faces handling strategy
# Options: "largest" (take largest bbox), "first" (take first detected), "error" (reject if multiple)
MULTIPLE_FACES_STRATEGY = os.getenv("MULTIPLE_FACES_STRATEGY", "largest")

# Service settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))

# Debug settings
DEBUG_SAVE_IMAGES = os.getenv("DEBUG_SAVE_IMAGES", "false").lower() in ("true", "1", "t")
DEBUG_SAVE_INTERVAL_S = int(os.getenv("DEBUG_SAVE_INTERVAL_S", "10"))
DEBUG_IMAGE_DIR = os.getenv("DEBUG_IMAGE_DIR", str(BASE_DIR / "debug_images"))
