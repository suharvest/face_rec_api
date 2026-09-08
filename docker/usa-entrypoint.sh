#!/bin/sh
# Entrypoint for face-rec-api:*-usa-jetson.
#
# Turns the mounted ONNX weights into TensorRT engines before uvicorn binds the
# port, then execs the server. Two reasons it is here and not left to the
# runtime fallback in src/backends/tensorrt.py:
#
#   * A missing weight becomes a start-time failure with a message naming the
#     download step, instead of an error inside the first /match. The compose
#     healthcheck then keeps access-node from starting against a service that
#     can never answer.
#   * The build takes minutes on an Orin. Doing it before the port is bound
#     means the healthcheck stays red until the service is genuinely ready,
#     rather than green-then-timeout on the first request.
#
# ENGINE_BOOTSTRAP:
#   build (default) — build what is missing/stale, fail if required weights are absent
#   skip            — try to build, but start anyway when weights are absent
#                     (the runtime fallback still applies if they appear later)
#   off             — do not build at all; straight to the server
set -e

BOOTSTRAP="${ENGINE_BOOTSTRAP:-build}"

case "$BOOTSTRAP" in
    off)
        echo "[entrypoint] ENGINE_BOOTSTRAP=off, skipping engine build"
        ;;
    skip)
        python3 /app/scripts/build_engine.py --skip-if-no-onnx
        ;;
    build)
        python3 /app/scripts/build_engine.py
        ;;
    *)
        echo "[entrypoint] unknown ENGINE_BOOTSTRAP=$BOOTSTRAP (build|skip|off)" >&2
        exit 2
        ;;
esac

exec "$@"
