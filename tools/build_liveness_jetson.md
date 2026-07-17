# Build: MiniFASNet liveness TensorRT engine (Jetson)

Passive single-frame anti-spoofing model for the liveness step in the face
pipeline, Jetson/TensorRT flavor. Produces
`models/jetson/liveness_minifasnet.engine`.

Engines are JetPack + GPU SM specific (same rule as
`tools/build_engine.sh`): build **on the target Jetson device**, never
commit the `.engine` to git.

## Model source / ONNX

Identical to the Hailo build — see `tools/build_liveness_hailo.md` for the
full preprocessing contract and the FC-tail folding story.

- Weight: Minivision Silent-Face-Anti-Spoofing
  `resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth`
  (md5 `a699bb3d549b6fb9c3a2caa1c058fb99`).
- Export: `tools/export_liveness_onnx.py` (run on a dev machine with torch —
  Mac or wsl2-local; do NOT install torch on the Jetson). The folded ONNX
  (md5 `6bafac19f5af68946eb234fdd7e5b365`) has a static `1x3x80x80` float32
  input named `input` and a `1x3` logits output named `logits`.
- Input convention (must match `src/liveness.py`): 80x80 **BGR**, float32,
  **raw 0-255** — no /255, no mean/std. Output is raw logits
  `[2D-spoof, real, 3D-spoof]`; softmax on host, `real = softmax(logits)[1]`.

## trtexec compile (on the Jetson)

FP16 is enough for MiniFASNet (~1.8 MB model; no INT8 calibration needed),
and `trtexec` is the lightest path — no Python TRT build stack required.

```bash
# copy tools/liveness_minifasnet.onnx to the device first
/usr/src/tensorrt/bin/trtexec \
    --onnx=tools/liveness_minifasnet.onnx \
    --saveEngine=models/jetson/liveness_minifasnet.engine \
    --fp16 \
    --memPoolSize=workspace:256 \
    --skipInference
```

The input is static, so no `--minShapes/--optShapes/--maxShapes` are needed
(unlike the SCRFD detector in `build_engine.sh`).

## Runtime

`src/backends/tensorrt.py` loads the engine as a third `_TRTEngine`
("liveness") when `FACE_LIVENESS_MODEL` exists and `LIVENESS_ENABLED=true`
(default path: `models/jetson/liveness_minifasnet.engine`, see
`src/config.py`). `liveness_raw` feeds NCHW float32 0-255 BGR and applies
softmax on host — bit-identical convention to the Hailo backend, so
live/spoof decisions must match the Hailo deployment on the same samples.

## Reference build (2026-07-17, orin-nano)

- Device: Jetson Orin Nano, JetPack 6.2.1, TensorRT 10.3.0
  (`trtexec` from `/usr/src/tensorrt/bin`).
- Consistency vs Hailo baseline (threshold 0.5): identical live/false
  decisions on the 3 Silent-Face repo samples + 4 real-face photos; scores
  differ only by quantization (Hailo INT8 vs TRT FP16).
