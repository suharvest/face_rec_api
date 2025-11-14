"""
Configuration for Standalone Face Recognition Service
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
PHOTOS_FOLDER = os.getenv("PHOTOS_FOLDER", str(BASE_DIR / "photos"))
EMBEDDINGS_JSON = os.getenv("EMBEDDINGS_JSON", str(BASE_DIR / "data" / "embeddings.json"))
MODELS_PATH = os.getenv("MODELS_PATH", str(BASE_DIR / "models"))

# Model paths
FACE_DETECTION_MODEL = os.getenv(
    "FACE_DETECTION_HEF",
    str(Path(MODELS_PATH) / "scrfd_10g.hef")
)
FACE_RECOGNITION_MODEL = os.getenv(
    "FACE_RECOGNITION_HEF",
    str(Path(MODELS_PATH) / "arcface_mobilefacenet.hef")
)

# Recognition settings
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
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
