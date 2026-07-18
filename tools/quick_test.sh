#!/bin/sh
# Quick manual test against a running face_rec_api instance.
#
#   tools/quick_test.sh <image.jpg|png> [host]
#
# host defaults to harvest-pi (100.116.230.60). Prints /health once, then
# the /recognize verdict for the image (liveness + match).
set -e

IMG="$1"
HOST="${2:-100.116.230.60}"
[ -f "$IMG" ] || { echo "usage: $0 <image> [host]"; exit 1; }

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
python3 - "$IMG" "$TMP" <<'EOF'
import base64, json, sys
b64 = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
json.dump({"image_base64": b64}, open(sys.argv[2], "w"))
EOF

echo "== /health @ $HOST =="
curl -s -m 8 "http://$HOST:8001/health"
echo
echo "== /recognize: $IMG =="
curl -s -m 20 -X POST "http://$HOST:8001/recognize" \
  -H 'Content-Type: application/json' -d @"$TMP" | python3 -m json.tool
