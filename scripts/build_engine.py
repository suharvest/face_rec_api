#!/usr/bin/env python3
"""Build the Jetson TensorRT engines from mounted ONNX weights.

Why this file exists
--------------------
``Dockerfile.jetson-usa`` ships code only: the ONNX weights are derived from
InsightFace releases whose licence is non-commercial (``MODEL_LICENSE.md``),
and a TensorRT engine is bound to the JetPack release *and* the GPU compute
capability of the machine that built it — an engine built elsewhere fails to
deserialize. So the image cannot carry either artefact, and the container was
left unable to bootstrap itself: ``/models/onnx`` empty, no builder inside.

This script closes that gap. The operator mounts the ONNX weights (host
``/opt/usa/models`` → container ``/models``), and the first start turns them
into engines next to themselves.

``tools/build_engine.sh`` does the same job with ``trtexec``. That binary is
part of the *host* JetPack install (``/usr/src/tensorrt/bin/trtexec``) and is
not bind-mounted into the runtime container, whereas the TensorRT Python
bindings already are — they are what the backend itself imports. So the
container-side builder has to go through the Python API. Build parameters are
kept identical to the shell script on purpose: same fp16 flag, same workspace
pool per role, same min=opt=max optimization profile pinning dynamic inputs to
the canonical shape. Engines from the two paths are interchangeable.

Relation to the runtime fallback
--------------------------------
``src/backends/tensorrt.py`` already rebuilds a missing or undeserializable
engine from ``<models>/onnx/<stem>.onnx`` on first load. That fallback stays;
this script is the same work moved ahead of the first request, so that

* a missing weight is reported at container start with a message that names
  the download step, instead of surfacing as a failed ``/match`` minutes later;
* the several-minute build does not land inside a request timeout;
* the engine cache is validated against the ONNX digest, so replacing a weight
  file actually rebuilds instead of silently reusing a stale engine.

Output location
---------------
Default output dir is ``MODELS_PATH`` (``/models`` in the image), because that
is where ``src/config.py`` looks for ``scrfd_10g.engine`` and
``arcface_mobilefacenet.engine``. ``--out-dir`` moves them; pointing it
somewhere else (e.g. ``/models/engines``) means also setting
``FACE_DETECTION_MODEL`` / ``FACE_RECOGNITION_MODEL`` / ``FACE_LIVENESS_MODEL``
to the new paths, otherwise the service will not find what was just built.

Usage
-----
    # inside the container, weights mounted at /models/onnx
    python3 /app/scripts/build_engine.py

    # on a host checkout
    MODELS_PATH=./models/jetson python3 scripts/build_engine.py \
        --onnx-dir ./models/onnx

    python3 scripts/build_engine.py --force        # ignore the cache
    python3 scripts/build_engine.py --only detector
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

#: role -> (onnx basename, engine basename, workspace MiB, canonical CHW input)
#: Names and workspace sizes mirror ``tools/build_engine.sh`` and
#: ``src/backends/tensorrt.py``'s ``_BUILD_WORKSPACE_MB`` /
#: ``_CANONICAL_INPUT_HW``. Changing one without the other means the runtime
#: fallback and this script produce different engines from the same weights.
ROLES = {
    "detector": ("scrfd_10g.onnx", "scrfd_10g.engine", 1024, (1, 3, 640, 640)),
    "embedder": ("arcface_mobilefacenet.onnx", "arcface_mobilefacenet.engine",
                 512, (1, 3, 112, 112)),
    "liveness": ("liveness_minifasnet.onnx", "liveness_minifasnet.engine",
                 256, (1, 3, 80, 80)),
}

#: Roles without which the service cannot answer a single request. Liveness is
#: optional: ``LIVENESS_ENABLED=false`` is a supported configuration, so a
#: missing liveness ONNX is a warning, not a failure.
REQUIRED_ROLES = ("detector", "embedder")

#: Sidecar next to each engine. Holds the inputs that determine the engine
#: bytes; any mismatch means the cached engine does not correspond to the
#: weights on disk and must be rebuilt.
STAMP_SUFFIX = ".build.json"

WEIGHTS_HINT = """\
ONNX weights are not shipped in this image (InsightFace-derived, non-commercial
licence). Fetch them onto the host and mount that directory at /models:

  # on the host
  mkdir -p /opt/usa/models/onnx
  #   ...download scrfd_10g.onnx / arcface_mobilefacenet.onnx into it
  #   (face_rec_api tools/download_insightface.sh fetches the InsightFace
  #    buffalo_l bundle: det_10g.onnx -> scrfd_10g.onnx,
  #    w600k_mbf.onnx -> arcface_mobilefacenet.onnx)

  # then run the container with
  docker run ... -v /opt/usa/models:/models ...

The solution deploy step 'download face models' does this for you."""


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def device_key(trt) -> str:
    """What the engine is bound to: TensorRT version + GPU compute capability.

    An engine built under a different TensorRT or on a different SM silently
    fails to deserialize later. Recording both here turns that into a rebuild
    at start instead of a runtime error — the same image moved from an Orin NX
    to an Orin Nano, or across a JetPack upgrade, rebuilds on first start.
    """
    parts = ["trt=%s" % trt.__version__]
    try:
        import ctypes

        class _Prop(ctypes.Structure):
            _fields_ = [("major", ctypes.c_int), ("minor", ctypes.c_int)]

        cudart = ctypes.CDLL("libcudart.so")
        major = ctypes.c_int()
        minor = ctypes.c_int()
        # cudaDevAttrComputeCapabilityMajor = 75, Minor = 76
        if (cudart.cudaDeviceGetAttribute(ctypes.byref(major), 75, 0) == 0
                and cudart.cudaDeviceGetAttribute(ctypes.byref(minor), 76, 0) == 0):
            parts.append("sm=%d%d" % (major.value, minor.value))
    except Exception:                                   # noqa: BLE001
        # Not fatal: without the SM the stamp is weaker (a TensorRT-compatible
        # move between two different Jetsons would not be caught here), but the
        # runtime fallback in tensorrt.py still rebuilds on a deserialization
        # failure. Losing the whole build over a missing libcudart is worse.
        parts.append("sm=unknown")
    return " ".join(parts)


def stamp_for(onnx_path: str, workspace_mb: int, shape, dev_key: str) -> dict:
    return {
        "onnx": os.path.basename(onnx_path),
        "onnx_sha256": sha256_file(onnx_path),
        "onnx_bytes": os.path.getsize(onnx_path),
        "fp16": True,
        "workspace_mb": workspace_mb,
        "static_shape": list(shape),
        "device": dev_key,
        "builder": "scripts/build_engine.py",
    }


def cache_is_valid(engine_path: str, want: dict) -> tuple[bool, str]:
    if not os.path.exists(engine_path):
        return False, "no engine at %s" % engine_path
    stamp_path = engine_path + STAMP_SUFFIX
    if not os.path.exists(stamp_path):
        return False, "engine present but unstamped (built by another path)"
    try:
        with open(stamp_path, "r", encoding="utf-8") as fh:
            have = json.load(fh)
    except (OSError, ValueError) as exc:
        return False, "unreadable stamp: %s" % exc
    for key in ("onnx_sha256", "fp16", "workspace_mb", "static_shape", "device"):
        if have.get(key) != want.get(key):
            return False, "%s changed (%r -> %r)" % (key, have.get(key), want.get(key))
    return True, "cached"


def build_one(trt, role: str, onnx_path: str, engine_path: str,
              workspace_mb: int, shape) -> None:
    """Parse the ONNX and serialize an fp16 engine with the input pinned.

    The optimization profile uses min == opt == max on purpose: the service
    always feeds one image at the canonical resolution, and a static profile
    lets TensorRT specialize instead of keeping a shape-generic plan.
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    try:  # TensorRT < 10 requires EXPLICIT_BATCH; 10+ dropped the flag
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    except AttributeError:
        flags = 0
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as fh:
        if not parser.parse(fh.read()):
            errs = "; ".join(str(parser.get_error(i))
                             for i in range(parser.num_errors))
            raise SystemExit("[build_engine] %s: cannot parse %s: %s"
                             % (role, onnx_path, errs))

    config = builder.create_builder_config()
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        print("[build_engine] %s: no fast fp16 on this device, building fp32"
              % role)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 workspace_mb << 20)

    profile = builder.create_optimization_profile()
    pinned = False
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        if any(d < 0 for d in tuple(inp.shape)):
            profile.set_shape(inp.name, shape, shape, shape)
            pinned = True
    if pinned:
        config.add_optimization_profile(profile)

    started = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("[build_engine] %s: TensorRT returned no engine for %s"
                         % (role, onnx_path))
    data = bytes(serialized)

    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    tmp = "%s.tmp.%d" % (engine_path, os.getpid())
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, engine_path)
    print("[build_engine] %s: %s (%.1f MB, %.0f s)"
          % (role, engine_path, len(data) / 1e6, time.time() - started))


def main(argv=None) -> int:
    models_path = os.environ.get("MODELS_PATH", "/models")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-dir", default=models_path,
                    help="where the service looks for engines (default: "
                         "$MODELS_PATH, else /models)")
    ap.add_argument("--onnx-dir", default=None,
                    help="ONNX weights (default: <models-dir>/onnx)")
    ap.add_argument("--out-dir", default=None,
                    help="engine output (default: <models-dir>; anywhere else "
                         "needs FACE_*_MODEL env vars pointed at it)")
    ap.add_argument("--only", action="append", choices=sorted(ROLES),
                    help="build just this role, repeatable")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even when the stamp matches")
    ap.add_argument("--skip-if-no-onnx", action="store_true",
                    help="exit 0 when required weights are absent instead of "
                         "failing; for entrypoints that must not block a "
                         "container whose models dir is filled in later")
    args = ap.parse_args(argv)

    onnx_dir = args.onnx_dir or os.path.join(args.models_dir, "onnx")
    out_dir = args.out_dir or args.models_dir
    roles = args.only or list(ROLES)

    try:
        import tensorrt as trt                          # noqa: WPS433
    except ImportError:
        print("[build_engine] the tensorrt python module is not importable.\n"
              "It is part of JetPack and cannot be pip-installed; the container "
              "gets it by bind-mounting the host copy:\n"
              "  -v /usr/lib/python3.10/dist-packages/tensorrt:"
              "/usr/local/lib/python3.10/dist-packages/tensorrt:ro\n"
              "plus the host CUDA/TensorRT shared libraries on LD_LIBRARY_PATH.",
              file=sys.stderr)
        return 2

    dev_key = device_key(trt)
    print("[build_engine] onnx=%s out=%s device=%s" % (onnx_dir, out_dir, dev_key))

    missing_required = []
    built = skipped = 0
    for role in roles:
        onnx_name, engine_name, workspace_mb, shape = ROLES[role]
        onnx_path = os.path.join(onnx_dir, onnx_name)
        engine_path = os.path.join(out_dir, engine_name)
        if not os.path.exists(onnx_path):
            if role in REQUIRED_ROLES:
                missing_required.append(onnx_path)
            else:
                print("[build_engine] %s: no %s, skipping (liveness is optional; "
                      "run with LIVENESS_ENABLED=false)" % (role, onnx_path))
            continue
        want = stamp_for(onnx_path, workspace_mb, shape, dev_key)
        ok, why = cache_is_valid(engine_path, want)
        if ok and not args.force:
            print("[build_engine] %s: up to date (%s)" % (role, engine_path))
            skipped += 1
            continue
        print("[build_engine] %s: building — %s" % (role, "forced" if args.force else why))
        build_one(trt, role, onnx_path, engine_path, workspace_mb, shape)
        with open(engine_path + STAMP_SUFFIX, "w", encoding="utf-8") as fh:
            json.dump(want, fh, indent=2, sort_keys=True)
        built += 1

    if missing_required:
        for path in missing_required:
            print("[build_engine] missing required weight: %s" % path,
                  file=sys.stderr)
        print(WEIGHTS_HINT, file=sys.stderr)
        if args.skip_if_no_onnx:
            print("[build_engine] --skip-if-no-onnx: continuing without engines",
                  file=sys.stderr)
            return 0
        return 3

    print("[build_engine] done: %d built, %d cached" % (built, skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
