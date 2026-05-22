#!/usr/bin/env bash
# Download InsightFace `buffalo_l` model bundle and extract the ONNX files
# needed to build SCRFD-10g + ArcFace-MobileFaceNet TensorRT engines.
#
# Outputs (under <out_dir>/buffalo_l):
#   det_10g.onnx     — SCRFD-10g detector
#   w600k_mbf.onnx   — ArcFace MobileFaceNet embedder
#
# Usage:
#   ./tools/download_insightface.sh [out_dir]
#
# Default out_dir: ./models/onnx

set -euo pipefail

OUT_DIR="${1:-./models/onnx}"
URL="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
ZIP_PATH="${OUT_DIR}/buffalo_l.zip"

mkdir -p "${OUT_DIR}"

if [ ! -f "${ZIP_PATH}" ]; then
    echo "[download] ${URL}"
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail -o "${ZIP_PATH}" "${URL}"
    else
        wget -O "${ZIP_PATH}" "${URL}"
    fi
else
    echo "[skip] ${ZIP_PATH} already exists"
fi

echo "[extract] ${ZIP_PATH} -> ${OUT_DIR}/buffalo_l"
mkdir -p "${OUT_DIR}/buffalo_l"
unzip -o "${ZIP_PATH}" -d "${OUT_DIR}/buffalo_l" >/dev/null

# Some releases nest the ONNX files inside a buffalo_l/ subdir already.
# Flatten if needed.
if [ -d "${OUT_DIR}/buffalo_l/buffalo_l" ]; then
    mv "${OUT_DIR}/buffalo_l/buffalo_l/"* "${OUT_DIR}/buffalo_l/" || true
    rmdir "${OUT_DIR}/buffalo_l/buffalo_l" || true
fi

echo
echo "ONNX files available under ${OUT_DIR}/buffalo_l:"
ls -lh "${OUT_DIR}/buffalo_l"/*.onnx 2>/dev/null || {
    echo "ERROR: no .onnx files found after extraction." >&2
    exit 1
}
