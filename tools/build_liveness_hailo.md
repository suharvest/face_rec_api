# Build: MiniFASNet liveness HEF (Hailo-8)

Passive single-frame anti-spoofing model for the liveness step in the face
pipeline. Produces `models/hailo/liveness_minifasnet.hef`.

## Model source

- Repo: [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) (Apache-2.0)
- Weight: `resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth`
  (md5 `a699bb3d549b6fb9c3a2caa1c058fb99`)
- Architecture: `MiniFASNetV2`, `conv6_kernel=get_kernel(80,80)=(5,5)`, 3-class head.

## Preprocessing convention (CRITICAL — must match at inference time)

Verified against repo source (`src/generate_patches.py`, `src/data_io/functional.py`,
`src/anti_spoof_predict.py`, `test.py`):

1. **Crop**: take the raw detector bbox `[x, y, w, h]` (NOT the aligned 112x112
   crop). Expand around bbox center by `scale = 2.7`
   (`scale = min((src_h-1)/h, (src_w-1)/w, 2.7)` — clamped so the expanded box
   fits the image; shifted back inside borders, see `CropImage._get_new_box`).
2. **Resize**: crop -> `cv2.resize(img, (80, 80))` (default bilinear).
3. **Color**: **BGR** (repo uses `cv2.imread` and never converts). Do NOT swap to RGB.
4. **Scaling**: float32, **raw 0-255. NO /255, NO mean/std normalization**
   (repo's custom `to_tensor` deliberately dropped `.div(255)`).
5. **Layout**: ONNX input is NCHW `1x3x80x80`. The compiled HEF takes native
   Hailo NHWC `80x80x3` (uint8 works: quantization calib range is 0-255) and
   outputs `1x1x3` logits (flatten on host).

## Output convention

- Output `logits`: shape `[1, 3]`, **raw logits — the HEF does NOT contain a
  softmax layer** (the PyTorch model's `forward` returns logits; the repo applies
  `F.softmax` outside the model). Downstream MUST apply softmax itself.
- Index meaning: `[0] = 2D spoof (print/photo), [1] = REAL, [2] = 3D spoof (screen replay etc.)`.
- `real_prob = softmax(logits)[1]`. Verified on repo samples (ONNX CPU):
  - `image_T1.jpg` (real): real = **0.9902**
  - `image_F1.jpg` (fake): real = 0.0013
  - `image_F2.jpg` (fake): real = 0.0006
  - ONNX vs PyTorch max prob diff < 1.2e-7.

## ONNX export (folded tail — required for Hailo)

Done on macOS with uv (torch 2.x, opset 13, legacy exporter).

**A naive export of the full model does NOT compile on DFC 3.31.0**: the FC tail
`conv_6_dw -> Flatten -> Linear(512,128) -> BatchNorm1d -> Linear(128,3)` fails
allocation with `BackendAllocatorException: No format for fc1 -> conv33`.

Fix: at `conv_6_dw` output the tensor is `1x512x1x1`, so the whole tail folds
analytically into ONE 1x1 Conv (verified maxdiff 6e-7 vs original):

```python
a = bn.weight / sqrt(bn.running_var + bn.eps)           # (128,)
W = prob.weight @ diag(a) @ linear.weight               # (3, 512)
b = prob.weight @ (bn.bias - a * bn.running_mean)       # (3,)
# backbone ... conv_6_dw -> Conv2d(512, 3, 1, bias=True)[W, b] -> flatten
```

Script: `tools/export_liveness_onnx.py` (run next to a checkout of the
Silent-Face repo; uses uv with torch/onnx/opencv-python==4.10.0.84).

- ONNX (folded) md5: `6bafac19f5af68946eb234fdd7e5b365`
- Smoke test needs `opencv-python==4.10.0.84` (4.13 dropped `cv2.dnn.readNetFromCaffe`
  used by the repo's bundled RetinaFace detector).

## Calibration data

- 259 face-region crops, `(259, 80, 80, 3)` float32 NHWC BGR 0-255
  (md5 `9bbec3daf4ba09ee2d0cdfcffb25ba97`).
- Source: 256 public face photos (LFW/CFP subset from
  `wsl2-local:~/gv2_face_train/gv2_face_data.tar`, one image per identity)
  + the 3 Silent-Face repo samples (1 real, 2 spoof).
- Each crop produced with the EXACT inference preprocessing above
  (RetinaFace bbox -> scale 2.7 -> 80x80 BGR, no normalization).

## DFC compile (x86_64 only)

Environment: `wsl2-local`, Hailo AI SW Suite 2025-04 docker image
(`hailo_ai_sw_suite_2025-04:1`), **DFC 3.31.0**, target `hailo8`.

```python
# runs inside the suite docker, /work = dir with onnx + calib npy
from hailo_sdk_client import ClientRunner
import numpy as np
runner = ClientRunner(hw_arch="hailo8")
runner.translate_onnx_model("/work/liveness_minifasnet.onnx", "liveness_minifasnet",
                            start_node_names=["input"], end_node_names=["/head/Conv"],
                            net_input_shapes={"input": [1, 3, 80, 80]})
runner.optimize(np.load("/work/calib_liveness_80x80.npy"))
open("/work/liveness_minifasnet.hef", "wb").write(runner.compile())
```

Gotchas hit during the build (do not repeat):

1. `chmod 777` the mounted work dir — the suite container runs as user `hailo`
   and `runner.optimize` writes cache dirs into cwd
   (`PermissionError: /work/bias_correction_cache_*` otherwise).
2. End node must be `/head/Conv`, not the graph output: the trailing `Flatten`
   raises `UnsupportedShuffleLayerError` (parser itself recommends `/head/Conv`).
   HEF output is therefore `1x1x3` logits — flatten + softmax on host.
3. FC-tail mapping failure (see ONNX section) — fold FCs into a 1x1 conv.

Launcher:

```bash
docker run --rm -v $PWD:/work --entrypoint /bin/bash \
  hailo_ai_sw_suite_2025-04:1 -lc 'cd /work && python3 dfc_compile.py'
```

Compile script: `tools/dfc_compile_liveness.py` (this exact file was used).

Result (2026-07-17):

- HEF: `models/hailo/liveness_minifasnet.hef`, 2,755,399 bytes,
  md5 `6d5e92c790aca515eeefba2d3cc04331`.
- Single-context flow failed (recoverable) -> compiled as **2 contexts**;
  `Successful Compilation (compilation time: 5s)`; context_0 total utilization:
  control 60.2%, compute 15%, memory 21.3%.
- Runtime pairing: HEF from DFC 3.31.0 runs on HailoRT 4.21.0 (the version
  pinned on the RPi deployment, see device gotchas).
