#!/usr/bin/env bash
# Build a calibration image list for RKNN INT8 quantization.
#
# Usage:
#   ./tools/build_rknn_calib.sh [photos_dir] [output_file] [max_count]
#
# Defaults:
#   photos_dir  = ./photos
#   output_file = ./calib.txt
#   max_count   = 100
#
# Picks up to <max_count> .jpg/.jpeg/.png files, shuffled. Writes absolute
# paths (one per line) to <output_file>. RKNN needs 50-100 representative
# images for good INT8 calibration; fewer than 50 will print a warning.

set -euo pipefail

PHOTOS_DIR="${1:-./photos}"
OUT_FILE="${2:-./calib.txt}"
MAX_COUNT="${3:-100}"

if [ ! -d "$PHOTOS_DIR" ]; then
    echo "ERROR: photos directory not found: $PHOTOS_DIR" >&2
    exit 1
fi

# Resolve to absolute path (portable: avoid `realpath` not on all hosts).
ABS_PHOTOS_DIR="$(cd "$PHOTOS_DIR" && pwd)"

# Collect images. find is POSIX-portable, sort+shuf gives reproducibility
# alternative across BSD/GNU; we use head as the cap.
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

find "$ABS_PHOTOS_DIR" -type f \( \
    -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
\) > "$TMP"

TOTAL_FOUND=$(wc -l < "$TMP" | tr -d ' ')
if [ "$TOTAL_FOUND" -eq 0 ]; then
    echo "ERROR: no .jpg/.jpeg/.png images found under $ABS_PHOTOS_DIR" >&2
    exit 1
fi

# Shuffle if possible (gshuf on macOS via coreutils, shuf on Linux).
if command -v shuf >/dev/null 2>&1; then
    shuf "$TMP" | head -n "$MAX_COUNT" > "$OUT_FILE"
elif command -v gshuf >/dev/null 2>&1; then
    gshuf "$TMP" | head -n "$MAX_COUNT" > "$OUT_FILE"
else
    # Fallback: deterministic but at least bounded.
    head -n "$MAX_COUNT" "$TMP" > "$OUT_FILE"
fi

KEPT=$(wc -l < "$OUT_FILE" | tr -d ' ')
echo "[ok] wrote $KEPT calibration paths to $OUT_FILE (from $TOTAL_FOUND candidates)"

if [ "$KEPT" -lt 50 ]; then
    echo "[warn] only $KEPT images. RKNN INT8 calibration recommends 50-100." >&2
    echo "       Either add more face images to $PHOTOS_DIR or accept lower" >&2
    echo "       quantization quality." >&2
fi
