#!/usr/bin/env python3
"""
Build RKNN models from InsightFace ONNX exports.

This is a DEV-MACHINE tool. It requires the full ``rknn-toolkit2`` package,
which pulls torch + tensorflow (~2 GB) and only supports:
- Linux x86_64
- glibc >= 2.27 (Ubuntu 18.04+ / Debian 10+)
- Python 3.8 to 3.11

It does NOT run on:
- macOS (any chip)
- aarch64 (Jetson, Raspberry Pi, RK3576/3588 itself)
- CentOS 7 / RHEL 7 (glibc too old)

Do NOT install rknn-toolkit2 inside the runtime image — that's what
rknn-toolkit-lite2 is for. See Dockerfile.rknn.

Usage:
    pip install rknn-toolkit2  # on the x86 Linux dev host
    ./tools/download_insightface.sh ./models/onnx
    ./tools/build_rknn_calib.sh ./photos ./calib.txt
    python tools/build_rknn.py \\
        --onnx-dir ./models/onnx/buffalo_l \\
        --out-dir  ./models/rknn \\
        --calib-list ./calib.txt \\
        --target rk3576

Notes:
- The output ``.rknn`` engines are SoC-specific. RK3576 != RK3588.
- INT8 quantization needs 50-100 representative face images. With <50,
  quality may degrade noticeably; the script warns but does not fail.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple


def _build_one(
    rknn_cls,
    onnx_path: str,
    rknn_path: str,
    mean: List[List[float]],
    std: List[List[float]],
    input_size: Tuple[int, int],
    calib_list: str,
    target_platform: str,
    quantize: bool,
) -> None:
    """Convert a single ONNX model to .rknn with optional INT8 quantization."""
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    if quantize and not os.path.exists(calib_list):
        raise FileNotFoundError(
            f"Calibration list not found: {calib_list}\n"
            "Run tools/build_rknn_calib.sh to generate one from your photos."
        )

    rknn = rknn_cls(verbose=True)
    try:
        # Mean/std baked into the graph -> runtime can feed raw uint8 NHWC.
        # quant_img_RGB2BGR=False: we feed RGB (matches our runtime preprocessing).
        rknn.config(
            mean_values=mean,
            std_values=std,
            target_platform=target_platform,
            quant_img_RGB2BGR=False,
            dynamic_input=[[[1, 3, input_size[0], input_size[1]]]],
        )

        ret = rknn.load_onnx(
            model=onnx_path,
        )
        if ret != 0:
            raise RuntimeError(f"load_onnx failed: ret={ret}")

        ret = rknn.build(
            do_quantization=quantize,
            dataset=calib_list if quantize else None,
        )
        if ret != 0:
            raise RuntimeError(f"build failed: ret={ret}")

        ret = rknn.export_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"export_rknn failed: ret={ret}")

        size_mb = os.path.getsize(rknn_path) / (1024 * 1024)
        print(f"[ok] {onnx_path} -> {rknn_path} ({size_mb:.2f} MB)")
    finally:
        rknn.release()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Convert InsightFace ONNX models to Rockchip RKNN."
    )
    p.add_argument(
        "--onnx-dir",
        default="./models/onnx/buffalo_l",
        help="Directory containing det_10g.onnx and w600k_mbf.onnx",
    )
    p.add_argument(
        "--out-dir",
        default="./models/rknn",
        help="Output directory for .rknn files",
    )
    p.add_argument(
        "--calib-list",
        default="./calib.txt",
        help="Text file with one image path per line (50-100 face images)",
    )
    p.add_argument(
        "--target",
        default="rk3576",
        choices=["rk3576", "rk3588", "rk3568", "rk3562", "rv1103", "rv1106"],
        help="Target Rockchip SoC (engines are NOT portable across SoCs)",
    )
    p.add_argument(
        "--no-quantize",
        action="store_true",
        help="Build fp16 instead of int8 (larger, slower, no calib needed)",
    )
    args = p.parse_args()

    try:
        from rknn.api import RKNN  # type: ignore
    except ImportError:
        print(
            "ERROR: rknn-toolkit2 not installed. Install on an x86 Linux host:\n"
            "  pip install rknn-toolkit2\n"
            "Do NOT install this on the target device.",
            file=sys.stderr,
        )
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    quantize = not args.no_quantize

    if quantize and os.path.exists(args.calib_list):
        with open(args.calib_list) as f:
            n_calib = sum(1 for line in f if line.strip())
        if n_calib < 50:
            print(
                f"[warn] only {n_calib} calibration images "
                f"(recommended: 50-100). Quantization quality may suffer.",
                file=sys.stderr,
            )

    # SCRFD-10g: input 1x3x640x640, normalization (pixel-127.5)/128 -> mean=127.5 std=128.
    _build_one(
        RKNN,
        onnx_path=os.path.join(args.onnx_dir, "det_10g.onnx"),
        rknn_path=os.path.join(args.out_dir, "scrfd_10g.rknn"),
        mean=[[127.5, 127.5, 127.5]],
        std=[[128.0, 128.0, 128.0]],
        input_size=(640, 640),
        calib_list=args.calib_list,
        target_platform=args.target,
        quantize=quantize,
    )

    # ArcFace recognition: input 1x3x112x112, normalization (pixel-127.5)/127.5 -> mean=127.5 std=127.5.
    # Prefer MobileFaceNet (w600k_mbf), fall back to ResNet-50 (w600k_r50) if not found.
    embedder_onnx = os.path.join(args.onnx_dir, "w600k_mbf.onnx")
    if not os.path.exists(embedder_onnx):
        fallback = os.path.join(args.onnx_dir, "w600k_r50.onnx")
        if os.path.exists(fallback):
            print(f"[note] w600k_mbf.onnx not found, using {fallback}", file=sys.stderr)
            embedder_onnx = fallback
    _build_one(
        RKNN,
        onnx_path=embedder_onnx,
        rknn_path=os.path.join(args.out_dir, "arcface_mobilefacenet.rknn"),
        mean=[[127.5, 127.5, 127.5]],
        std=[[127.5, 127.5, 127.5]],
        input_size=(112, 112),
        calib_list=args.calib_list,
        target_platform=args.target,
        quantize=quantize,
    )

    print(f"\nDone. Engines written to {args.out_dir} for target={args.target}.")
    print("Copy them to the device under <project>/models/rknn/ and run:")
    print("  FACE_BACKEND=rknn ./start_standalone.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
