#!/bin/bash
set -e

echo "=========================================="
echo "Standalone Face Recognition Service"
echo "=========================================="

# Change to script directory
cd "$(dirname "$0")"

# Backend selection (default: hailo). Override via env: FACE_BACKEND=jetson ./start_standalone.sh
FACE_BACKEND="${FACE_BACKEND:-hailo}"
export FACE_BACKEND

# Per-backend model extension mapping
case "$FACE_BACKEND" in
    hailo)  MODEL_EXT="hef"   ;;
    jetson) MODEL_EXT="engine";;
    rknn)   MODEL_EXT="rknn"  ;;
    *)
        echo "Error: unknown FACE_BACKEND=$FACE_BACKEND"
        echo "Supported: hailo | jetson | rknn"
        exit 1
        ;;
esac

MODELS_DIR="${MODELS_PATH:-models/$FACE_BACKEND}"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run: uv sync --extra $FACE_BACKEND"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if embeddings.json exists
if [ ! -f "data/embeddings.json" ]; then
    echo ""
    echo "[WARN] No embeddings.json found!"
    echo ""
    echo "Please run batch processing first:"
    echo "  python scripts/batch_process.py"
    echo ""
    echo "Or the service will start with an empty database."
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check model files for the selected backend
DET_MODEL="${FACE_DETECTION_MODEL:-$MODELS_DIR/scrfd_10g.$MODEL_EXT}"
EMB_MODEL="${FACE_RECOGNITION_MODEL:-$MODELS_DIR/arcface_mobilefacenet.$MODEL_EXT}"

if [ ! -f "$DET_MODEL" ] || [ ! -f "$EMB_MODEL" ]; then
    echo "Error: model files not found for backend '$FACE_BACKEND':"
    echo "  detector: $DET_MODEL"
    echo "  embedder: $EMB_MODEL"
    exit 1
fi

# Show configuration
echo ""
echo "Configuration:"
echo "  Backend:             $FACE_BACKEND"
echo "  Models dir:          $MODELS_DIR"
echo "  Detector model:      $DET_MODEL"
echo "  Embedder model:      $EMB_MODEL"
echo "  Photos folder:       ${PHOTOS_FOLDER:-./photos}"
echo "  Embeddings file:     ${EMBEDDINGS_JSON:-./data/embeddings.json}"
echo "  Similarity threshold: ${SIMILARITY_THRESHOLD:-0.4}"
echo "  Host: ${HOST:-0.0.0.0}"
echo "  Port: ${PORT:-8001}"
echo ""

echo "Starting service..."
echo "=========================================="
echo ""

PYTHONPATH=src python src/app.py
