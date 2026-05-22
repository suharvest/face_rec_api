# Standalone Face Recognition Service

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
![Backends](https://img.shields.io/badge/backends-Hailo%20|%20Jetson%20|%20RKNN-green)

**One face recognition API. Three edge AI platforms.**

Deploy the same REST API on a Raspberry Pi 5 (Hailo-8), NVIDIA Jetson Orin
(TensorRT), or Rockchip RK3576/3588 (RKNN) — pick the hardware that fits
your project, not the other way around.  Stateless design, zero external
dependencies, and per-backend Docker images under 2 GB.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Deployment](#deployment)
  - [Jetson (TensorRT)](#jetson-deployment-tensorrt)
  - [RKNN (Rockchip)](#rknn-deployment-rockchip)
  - [Hailo (Raspberry Pi)](#hailo-deployment-raspberry-pi)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## Features

- ⚡ **Multi-backend**: Hailo-8 · TensorRT · RKNN — same API, one config switch
- 🚀 **Fast Recognition**: 30–53 ms p50, depending on hardware
- 📁 **Simple Storage**: JSON-based embedding store, no database required
- 🔄 **Hot Reload**: Update the face database while the service is running
- 🔌 **Self-contained**: Zero external services — just Python + numpy + OpenCV
- 📡 **RESTful API**: `/infer`, `/recognize`, `/enroll`, `/health`, and more

## Quick Start

<!-- TODO: Add a ~15s terminal GIF showing curl /infer + JSON response -->

### 1. Prepare Photos

Add photos to the `photos/` folder. Each photo should contain one face and be named as the person's name:

```bash
photos/
├── john_doe.jpg
├── jane_smith.png
├── alice_wang.jpg
└── bob_lee.png
```

**Supported formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`

### 2. Process Photos (Generate Embeddings)

```bash
# Activate virtual environment
cd services/standalone_face_api
source .venv/bin/activate

# Run batch processing
uv run scripts/batch_process.py
```

This creates `data/embeddings.json` with face embeddings for all photos.

### 3. Start API Service

```bash
# Start the service
PYTHONPATH=src python src/app.py

# Or use the convenience script
./start_standalone.sh
```

The service will:
- Load Hailo models (2-3s)
- Load `embeddings.json` into memory
- Start API server on `http://localhost:8001`

### 4. Recognize Faces

```bash
# Encode an image to base64
IMAGE_B64=$(base64 -i test_image.jpg)

# Call recognition API
curl -X POST http://localhost:8001/recognize \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$IMAGE_B64\"}"
```

Response:
```json
{
  "matched": true,
  "name": "john_doe",
  "confidence": 0.85,
  "processing_time_ms": 45
}
```

## API Endpoints

Full interactive docs at `http://localhost:8001/docs` (Swagger UI).

| Method | Endpoint | Description | Key response field |
|--------|----------|-------------|-------------------|
| `GET` | `/health` | Backend probe + service status | `status`, `backend`, `model_tag` |
| `POST` | `/infer` | Stateless inference — detect + embed | `faces[].embedding` (base64), `face_count` |
| `POST` | `/recognize` | Identify a face against stored users | `matched`, `name`, `confidence` |
| `POST` | `/enroll` | Register a new person | `embedding_saved` |
| `POST` | `/reload` | Hot-reload `photos/` → embeddings | `loaded`, `reload_time_ms` |
| `DELETE` | `/remove/{name}` | Delete a user | `embeddings_saved` |
| `GET` | `/list` | List enrolled users | `users`, `count` |
| `POST` | `/detect_and_embed` | Debug: raw faces + embeddings | `faces[].bbox`, `faces[].embedding` |

**Quick examples:**

```bash
# Health check
curl http://localhost:8001/health

# Stateless inference (returns embedding + bbox + landmarks)
curl -X POST http://localhost:8001/infer \
  -H 'Content-Type: application/json' \
  -d '{"image_b64":"'$(base64 -w0 photo.jpg)'"}'

# Recognize a face
curl -X POST http://localhost:8001/recognize \
  -H 'Content-Type: application/json' \
  -d '{"image_base64":"'$(base64 -w0 photo.jpg)'"}'

# Register a new user
curl -X POST http://localhost:8001/enroll \
  -H 'Content-Type: application/json' \
  -d '{"name":"alice","image_base64":"'$(base64 -w0 alice.jpg)'"}'
```

## Configuration

All configuration is via environment variables. See `src/config.py` for details.

### Key Settings

```bash
# Folders
export PHOTOS_FOLDER=./photos
export EMBEDDINGS_JSON=./data/embeddings.json

# Recognition settings
export SIMILARITY_THRESHOLD=0.5  # 0-1, higher = stricter matching
export CONFIDENCE_THRESHOLD=0.55  # Face detection confidence
export MULTIPLE_FACES_STRATEGY=largest  # "largest", "first", "error"

# Service settings
export HOST=0.0.0.0
export PORT=8001

# Debug
export DEBUG_SAVE_IMAGES=false
export DEBUG_SAVE_INTERVAL_S=10
```

### Multiple Faces Handling

When a photo contains multiple faces:

- **`largest`** (default): Take the face with the largest bounding box
- **`first`**: Take the first detected face
- **`error`**: Reject the photo and return an error

## Development Workflow

### Adding New People

**Option 1: Using photos folder (Recommended)**

```bash
# 1. Add photo to folder
cp new_person.jpg photos/

# 2. Trigger reload (while API is running)
curl -X POST http://localhost:8001/reload
```

**Option 2: Using /enroll endpoint**

```bash
# Encode image
IMAGE_B64=$(base64 -i new_person.jpg)

# Call enroll API
curl -X POST http://localhost:8001/enroll \
  -H "Content-Type: application/json" \
  -d "{\"name\": \"new_person\", \"image_base64\": \"$IMAGE_B64\"}"
```

### Removing People

```bash
# Option 1: Via API
curl -X DELETE http://localhost:8001/remove/john_doe

# Option 2: Delete photo + reload
rm photos/john_doe.jpg
curl -X POST http://localhost:8001/reload
```

### Updating Embeddings

After modifying photos in the `photos/` folder:

```bash
# If API is running
curl -X POST http://localhost:8001/reload

# If API is not running
python scripts/batch_process.py
```

## Architecture

```
┌──────────────────┐
│   photos/        │  Source of truth
│   ├── john.jpg   │
│   └── jane.png   │
└────────┬─────────┘
         │
         │ [Batch Process or /reload]
         ▼
┌──────────────────┐
│ embeddings.json  │  Persistent storage
│ {                │
│   "john": [...], │
│   "jane": [...]  │
│ }                │
└────────┬─────────┘
         │
         │ [Auto-load on startup]
         ▼
┌──────────────────┐
│  In-Memory Dict  │  Fast recognition
│  {'john': arr,   │
│   'jane': arr}   │
└──────────────────┘
```

## Performance

All measurements on real hardware with MobileFaceNet embedder and 640×640 input.
Steady-state p50 over 10 consecutive `/infer` calls.

| Backend | Device | Precision | Latency (p50) | Det+Emb Engine |
|---------|--------|-----------|---------------|----------------|
| Hailo-8 | Raspberry Pi 5 | INT8 | **30 ms** | 3.4 MB `.hef` |
| TensorRT | Jetson Orin Nano 8GB | INT8 | **35 ms** | 5.3 + 7.2 MB `.engine` |
| RKNN | RK3576 LubanCat-3 | INT8 | **53 ms** | 4.1 + 5.2 MB `.rknn` |

**Startup**: < 5 s (100 users from JSON).  **Memory**: ~200 KB per user (512-D fp32).

## Troubleshooting

### No faces detected

- Check image quality (min face size: 8px)
- Adjust `CONFIDENCE_THRESHOLD` (default 0.55)
- Ensure face is clearly visible

### Wrong person recognized

- **Too loose**: Decrease `SIMILARITY_THRESHOLD` (e.g., 0.5 → 0.4)
- **Too strict**: Increase `SIMILARITY_THRESHOLD` (e.g., 0.5 → 0.6)

### Multiple faces error

Change strategy to `largest` or `first`:

```bash
export MULTIPLE_FACES_STRATEGY=largest
```

### Service won't start

Check Hailo models exist:

```bash
ls -la models/
# Should show:
# - scrfd_10g.hef
# - arcface_mobilefacenet.hef
```

## Requirements

### Hardware

- Raspberry Pi 5 + Hailo-8 AI Kit
- 4GB+ RAM recommended
- 8GB+ storage

### Software

- Python 3.11+
- HailoRT 4.21.0 (installed separately)
- See `requirements.txt` for Python packages

### Setup

1. Install HailoRT following [Hailo Setup Guide](../face_embed_api/docs/HAILO_SETUP.md)
2. Place model files in `models/` directory
3. Install Python dependencies: `uv sync`

## Jetson Deployment (TensorRT)

The service also runs on NVIDIA Jetson devices (Orin Nano / Orin NX / AGX
Orin) using a TensorRT backend. Select via `FACE_BACKEND=jetson`.

### System Requirements

- JetPack 6.2 / L4T R36.4.x (TensorRT 10.3, CUDA 12.5)
- Python 3.10 (system) — the JetPack-bundled `tensorrt` module lives in
  `/usr/lib/python3.10/dist-packages` and must not be replaced by pip.
- Docker with `nvidia` runtime configured.

### Step 1 — Build Engines on the Target Device

**Engines are JetPack release + GPU compute-capability specific.** An
engine built for Orin Nano (sm_87) is not portable to AGX Orin (sm_87
but different JetPack) or to a different host. Always rebuild when you
change devices.

```bash
# On the target Jetson, in this repo:
./tools/download_insightface.sh ./models/onnx
./tools/build_engine.sh ./models/onnx/buffalo_l ./models/jetson
# Produces:
#   models/jetson/scrfd_10g.engine
#   models/jetson/arcface_mobilefacenet.engine
```

### Step 2 — Build the Runtime Image

```bash
docker build -t face_rec_api:jetson -f Dockerfile.jetson .
```

The image is based on `nvcr.io/nvidia/l4t-base:r36.2.0` (~1 GB) and
deliberately ships **without** CUDA toolkit or TensorRT inside. The
JetPack 6.2 host already provides them in `/usr/lib/...`, and they are
bind-mounted into the container at run time (see Step 3). This keeps
the image around **1 GB instead of ~16 GB** of the previous
`l4t-jetpack`-based build, and avoids shipping two copies of the same
~5 GB TRT/CUDA stack on every Jetson.

> **JetPack 6.2.1 is required on the host** — TensorRT 10.3.0 +
> CUDA 12.6 + cuDNN 9.3 + libcuda from the host are what the container
> dlopen()s. Older JetPack hosts will fail at engine deserialisation.

### Step 3 — Lock Jetson clocks (one-time, per boot)

For deterministic latency, lock CPU/GPU/EMC to their maximum frequency
**before** starting the container:

```bash
sudo nvpmodel -m 0      # MAXN power profile (Orin Nano)
sudo jetson_clocks      # lock everything to max
```

Without this the first few inferences will be slower while the
governor ramps clocks up, and steady-state latency can drift up to
~2x.

**Performance** (Orin Nano 8GB, MobileFaceNet embedder, `jetson_clocks` off):

| Precision | Detector | p50 | Engine |
|---|---|---|---|
| FP16 | scrfd_10g.engine (8.7 MB) | ~40 ms | Default |
| INT8† | scrfd_10g.int8.engine (5.3 MB) | **~35 ms** | Requires calibration |

With `sudo jetson_clocks` expect another 5–10 ms improvement. Orin NX / AGX
Orin are faster due to higher GPU clocks and more memory bandwidth.

† INT8 calibration uses `polygraphy convert --int8 --data-loader-script`
with 100 representative 640×640 face images. See `tools/build_int8_calib.md`
for the workflow.

### Step 4 — Run

The container mounts the host's TensorRT Python bindings + the
TRT/CUDA/cuDNN shared libs read-only. NVIDIA's `nvidia-container-runtime`
CSV files on JetPack 6.2 only auto-mount the driver (`libcuda.so`), so
the TRT/CUDA runtime libs are passed explicitly here.

```bash
docker run --runtime nvidia -d \
    --name frc-jetson \
    -v $(pwd)/models/jetson:/models:ro \
    -v $(pwd)/photos:/photos \
    -v $(pwd)/data:/data \
    -p 8001:8001 \
    \
    -v /usr/lib/python3.10/dist-packages/tensorrt:/usr/lib/python3.10/dist-packages/tensorrt:ro \
    -v /usr/lib/python3.10/dist-packages/tensorrt-10.3.0.dist-info:/usr/lib/python3.10/dist-packages/tensorrt-10.3.0.dist-info:ro \
    -v /usr/lib/python3.10/dist-packages/tensorrt_dispatch:/usr/lib/python3.10/dist-packages/tensorrt_dispatch:ro \
    -v /usr/lib/python3.10/dist-packages/tensorrt_lean:/usr/lib/python3.10/dist-packages/tensorrt_lean:ro \
    -v /usr/lib/aarch64-linux-gnu/libnvinfer.so.10:/usr/lib/aarch64-linux-gnu/libnvinfer.so.10:ro \
    -v /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.10:/usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.10:ro \
    -v /usr/lib/aarch64-linux-gnu/libnvonnxparser.so.10:/usr/lib/aarch64-linux-gnu/libnvonnxparser.so.10:ro \
    -v /usr/lib/aarch64-linux-gnu/libcudnn.so.9:/usr/lib/aarch64-linux-gnu/libcudnn.so.9:ro \
    -v /usr/local/cuda:/usr/local/cuda:ro \
    \
    face_rec_api:jetson

curl http://localhost:8001/health
```

The service expects engine files mounted at `/models` (or override via
`MODELS_PATH`). If they are missing, startup fails with an explicit
"engine not found" error — there is no silent fallback.

### Image-size before/after

| Build                  | Image size | Notes                              |
|------------------------|------------|------------------------------------|
| `l4t-jetpack` (P1)     | 16.1 GB    | Ships full CUDA toolkit + TRT      |
| `l4t-base` + mounts    | ~1 GB      | TRT/CUDA from host bind-mounts     |

### Running Without Docker

```bash
pip3 install --break-system-packages cuda-python opencv-python-headless \
    fastapi 'uvicorn[standard]' pydantic scikit-image
FACE_BACKEND=jetson ./start_standalone.sh
```

Make sure the system `tensorrt` module is on `PYTHONPATH` (it is by
default on JetPack-provisioned devices).

## RKNN Deployment (Rockchip RK3576 / RK3588)

The service also runs on Rockchip NPU devices (RK3576 dev boards, RK3588
boards such as Radxa Rock 5 / Orange Pi 5) using a `.rknn`-engine backend.
Select via `FACE_BACKEND=rknn`.

### System Requirements

- Debian 12 (bookworm) aarch64 host (Python 3.11 inside the container)
- Rockchip NPU userland with **`librknnrt.so` >= 2.3.0** present at
  `/usr/lib/librknnrt.so` (ships with the official Rockchip BSP).
- NPU character device: `/dev/dri/renderD129` (RK3576) or
  `/dev/dri/renderD128` depending on board. Check with `ls /dev/dri/`.
- Docker (no special runtime required — the NPU is passed via `--device`).

> **Engines are SoC-specific.** A `.rknn` built for RK3576 (2-core NPU)
> will not run on RK3588 (3-core NPU) and vice versa. Always set
> `--target` to the SoC you intend to deploy on.

### Step 1 — Convert ONNX → RKNN on an x86 Linux dev host

`rknn-toolkit2` (the conversion tool) **requires** Linux x86_64 with
glibc >= 2.27 (Ubuntu 18.04+ / Debian 10+) and Python 3.8-3.11. It
does **not** run on macOS, on aarch64 Jetson/Pi, or on the RK3576
itself. Do this step on a dev workstation.

```bash
# On your x86 Linux dev host:
pip install rknn-toolkit2
./tools/download_insightface.sh ./models/onnx
./tools/build_rknn_calib.sh ./photos ./calib.txt   # builds calib list from your faces
python tools/build_rknn.py \
    --onnx-dir ./models/onnx/buffalo_l \
    --out-dir  ./models/rknn \
    --calib-list ./calib.txt \
    --target rk3576
# Produces:
#   models/rknn/scrfd_10g.rknn               (~9 MB, INT8)
#   models/rknn/arcface_mobilefacenet.rknn   (~3 MB, INT8)
```

For best INT8 quality use **50-100 representative face images** as the
calibration set. With fewer the script warns; with significantly fewer
detection recall and embedding cosine similarity will degrade.

Copy the generated `.rknn` files to the device:

```bash
scp models/rknn/*.rknn user@<rk3576-host>:/path/to/face_rec_api/models/rknn/
```

### Step 2 — Build the Runtime Image (on the RK3576 device)

```bash
docker build -t face_rec_api:rknn -f Dockerfile.rknn .
```

The image is based on `python:3.11-slim-bookworm` (~150 MB) and only
ships `rknn-toolkit-lite2` (the ~10 MB runtime binding). The full
`rknn-toolkit2` (which pulls torch + tensorflow, ~2 GB) is **never**
installed in the runtime image — it is a dev-machine-only tool.

`librknnrt.so` itself is **bind-mounted from the host** at run time, so
the image stays under ~250 MB and is automatically kept in sync with
the host BSP.

### Step 3 — Run

**RK3576 (dual-core NPU):**

```bash
docker run -d --name frc-rknn \
    --privileged \
    -v /usr/lib/librknnrt.so:/usr/lib/librknnrt.so:ro \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
    -v $(pwd)/models/rknn:/models:ro \
    -v $(pwd)/photos:/photos \
    -v $(pwd)/data:/data \
    -p 8001:8001 \
    face_rec_api:rknn
```

`--privileged` is required: the RKNN Lite runtime performs container-integrity
checks that need access to `/sys/kernel/debug/rknpu` and the NPU devfreq
node — neither is reachable through a regular `--device` mapping.

`/proc/device-tree/compatible` must be bind-mounted so the runtime can detect
the Rockchip SoC model (RK3576 / RK3588 / etc.) at start-up.

**Performance** (RK3576 2-core NPU, INT8-quantized MobileFaceNet, 640x640 input):
~53 ms p50 / ~56 ms mean over 10 consecutive `/infer` calls. RK3588 (3-core NPU)
is expected to be ~30-40 % faster.

**RK3588 (triple-core NPU):**

Same command; optionally add `-e RKNN_CORE_MASK=0_1_2` to use all three cores
(default `AUTO` already picks the right core layout per SoC).

### Running Without Docker

```bash
uv sync --extra rknn
FACE_BACKEND=rknn ./start_standalone.sh
```

Make sure `/usr/lib/librknnrt.so` is present on the host and the
current user has read/write access to `/dev/dri/renderD12*`.

## Acknowledgements

Built on [InsightFace](https://github.com/deepinsight/insightface) (SCRFD detection + ArcFace recognition),
with per-backend acceleration via [HailoRT](https://hailo.ai/),
[TensorRT](https://developer.nvidia.com/tensorrt), and
[RKNN-Toolkit-Lite2](https://github.com/airockchip/rknn-toolkit2).

## License

MIT — see [LICENSE](LICENSE).

## Support

For issues or questions, please open an issue in the repository.
