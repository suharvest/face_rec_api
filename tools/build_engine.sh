#!/usr/bin/env bash
# Build TensorRT engines for the Jetson face_rec_api backend.
#
# WHY this exists and not in the runtime Dockerfile:
#   TensorRT engines are JetPack release + GPU SM specific. Building them on
#   a different device produces an engine that fails to load with `Engine
#   deserialization failed` or `getPluginCreator could not find Plugin`.
#   Therefore engines MUST be built on (or with identical specs to) the
#   target inference device, and we keep all build dependencies — trtexec,
#   onnx graphsurgeon, etc. — out of the runtime image.
#
# Inputs:
#   buffalo_l_dir  — directory containing det_10g.onnx and w600k_mbf.onnx
#                    (use `tools/download_insightface.sh` to fetch).
#   out_dir        — where to write the .engine files
#                    (default: ./models/jetson; consumed by the service).
#
# Usage on the Jetson:
#   ./tools/download_insightface.sh ./models/onnx
#   ./tools/build_engine.sh ./models/onnx/buffalo_l ./models/jetson

set -euo pipefail

BUFFALO_DIR="${1:-./models/onnx/buffalo_l}"
OUT_DIR="${2:-./models/jetson}"

if [ ! -d "${BUFFALO_DIR}" ]; then
    echo "ERROR: buffalo_l dir not found: ${BUFFALO_DIR}" >&2
    echo "Hint: run ./tools/download_insightface.sh first." >&2
    exit 1
fi

DET_ONNX="${BUFFALO_DIR}/det_10g.onnx"
EMB_ONNX="${BUFFALO_DIR}/w600k_mbf.onnx"

for f in "${DET_ONNX}" "${EMB_ONNX}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: missing ONNX file: ${f}" >&2
        exit 1
    fi
done

# Locate trtexec. JetPack ships it at /usr/src/tensorrt/bin/trtexec.
TRTEXEC="${TRTEXEC:-}"
if [ -z "${TRTEXEC}" ]; then
    if command -v trtexec >/dev/null 2>&1; then
        TRTEXEC="$(command -v trtexec)"
    elif [ -x /usr/src/tensorrt/bin/trtexec ]; then
        TRTEXEC=/usr/src/tensorrt/bin/trtexec
    else
        echo "ERROR: trtexec not found. Set TRTEXEC=<path> or install TensorRT." >&2
        exit 1
    fi
fi

mkdir -p "${OUT_DIR}"

echo "[build] SCRFD-10g detector -> ${OUT_DIR}/scrfd_10g.engine"
# The InsightFace `det_10g.onnx` has dynamic H/W on input.1; pin to 640x640.
"${TRTEXEC}" \
    --onnx="${DET_ONNX}" \
    --saveEngine="${OUT_DIR}/scrfd_10g.engine" \
    --fp16 \
    --memPoolSize=workspace:1024 \
    --minShapes=input.1:1x3x640x640 \
    --optShapes=input.1:1x3x640x640 \
    --maxShapes=input.1:1x3x640x640 \
    --skipInference

echo "[build] ArcFace MobileFaceNet embedder -> ${OUT_DIR}/arcface_mobilefacenet.engine"
"${TRTEXEC}" \
    --onnx="${EMB_ONNX}" \
    --saveEngine="${OUT_DIR}/arcface_mobilefacenet.engine" \
    --fp16 \
    --memPoolSize=workspace:512 \
    --skipInference

echo
echo "Engines built:"
ls -lh "${OUT_DIR}"/*.engine
echo
echo "NOTE: these engines are tied to this device's JetPack version and"
echo "      compute capability. Rebuild when moving to a different Jetson."
