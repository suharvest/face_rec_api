#!/bin/bash
set -e

echo "=========================================="
echo "Standalone Face Recognition Service"
echo "=========================================="

# Change to script directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run: uv venv && uv pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if embeddings.json exists
if [ ! -f "data/embeddings.json" ]; then
    echo ""
    echo "⚠️  No embeddings.json found!"
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

# Check if models exist
if [ ! -f "models/scrfd_10g.hef" ] || [ ! -f "models/arcface_mobilefacenet.hef" ]; then
    echo "Error: Hailo model files not found in models/"
    echo "Please ensure the following files exist:"
    echo "  - models/scrfd_10g.hef"
    echo "  - models/arcface_mobilefacenet.hef"
    exit 1
fi

# Show configuration
echo ""
echo "Configuration:"
echo "  Photos folder: ${PHOTOS_FOLDER:-./photos}"
echo "  Embeddings file: ${EMBEDDINGS_JSON:-./data/embeddings.json}"
echo "  Similarity threshold: ${SIMILARITY_THRESHOLD:-0.5}"
echo "  Host: ${HOST:-0.0.0.0}"
echo "  Port: ${PORT:-8001}"
echo ""

# Start the service
echo "Starting service..."
echo "=========================================="
echo ""

PYTHONPATH=src python src/app.py

# Or use uvicorn directly:
# PYTHONPATH=src uvicorn src.app:app --host ${HOST:-0.0.0.0} --port ${PORT:-8001}
