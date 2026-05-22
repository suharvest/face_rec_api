# Jetson INT8 Calibration Guide

Build an INT8-quantised SCRFD-10g detection engine with proper entropy
calibration. Requires a Jetson device with JetPack 6.x / TensorRT 10.x.

## Prerequisites

- 100+ representative face images at 640×640 (or larger, will be resized)
- `polygraphy` installed: `pip install polygraphy`
- `opencv-python` and `numpy`

## Step 1 — Prepare calibration images

Resize face images to 640×640 and place them in a single directory:

```bash
mkdir -p /tmp/calib_640
# For each source image:
python3 -c "
import cv2, glob
for f in glob.glob('/path/to/faces/*.jpg'):
    img = cv2.imread(f)
    img = cv2.resize(img, (640, 640))
    cv2.imwrite(f'/tmp/calib_640/{f.split(\"/\")[-1]}', img)
"
```

## Step 2 — Create the data-loader

`calib_loader.py`:

```python
import glob, os, numpy as np, cv2

INPUT_NAME = "input.1"
CALIB_DIR = "/tmp/calib_640"

def load_data():
    paths = sorted(glob.glob(os.path.join(CALIB_DIR, "*.jpg")))
    for p in paths:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        yield {INPUT_NAME: img}
```

## Step 3 — Build the INT8 engine

```bash
polygraphy convert /path/to/det_10g.onnx \
    --int8 \
    --data-loader-script calib_loader.py \
    --calibration-cache scrfd_int8.cache \
    --trt-min-shapes input.1:[1,3,640,640] \
    --trt-opt-shapes input.1:[1,3,640,640] \
    --trt-max-shapes input.1:[1,3,640,640] \
    -o scrfd_10g.int8.engine
```

The calibration cache is reusable — keep it around for faster rebuilds.

## Step 4 — Deploy

Replace the active detector engine:

```bash
ln -f scrfd_10g.int8.engine models/jetson/scrfd_10g.engine
docker restart frc-jetson
```

## Expected results

| Precision | Engine size | Latency (p50) | Accuracy |
|---|---|---|---|
| FP16 | ~8.7 MB | ~40 ms | baseline |
| INT8 (calibrated) | ~5.3 MB | ~35 ms | det_score within 0.5% of FP16 |
