# Standalone Face Recognition Service

A lightweight, standalone face recognition service with JSON-based vector storage. No external dependencies like Node-RED or Qdrant required.

## Features

- 🚀 **Fast Recognition**: < 50ms recognition latency
- 📁 **Simple Storage**: JSON-based embedding storage
- 🔄 **Hot Reload**: Update face database while service is running
- 🎯 **High Accuracy**: Hailo-8 accelerated ArcFace embeddings
- 🔌 **Standalone**: No external services required
- 📡 **RESTful API**: Easy integration

## Quick Start

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

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "loaded_users": 42,
  "embeddings_file": "data/embeddings.json",
  "last_reload": "2025-06-14T10:30:00",
  "uptime_ms": 12345
}
```

### POST `/recognize`

**Primary endpoint** - Recognize face in image.

**Request:**
```json
{
  "image_base64": "base64_encoded_image"
}
```

**Response:**
```json
{
  "matched": true,
  "name": "john_doe",
  "confidence": 0.85,
  "processing_time_ms": 45
}
```

### POST `/enroll`

Manually enroll a single person.

**Request:**
```json
{
  "name": "new_person",
  "image_base64": "base64_encoded_image"
}
```

**Response:**
```json
{
  "success": true,
  "name": "new_person",
  "embedding_saved": true,
  "error": null
}
```

### POST `/reload`

**Key feature** - Reload embeddings from photos folder.

Use this after adding new photos to the `photos/` folder.

**Request:**
```json
{
  "force": false
}
```

**Response:**
```json
{
  "success": true,
  "loaded": 42,
  "failed": [],
  "embeddings_saved": true,
  "reload_time_ms": 2500
}
```

**Process:**
1. Scans `photos/` folder
2. Processes all images → embeddings
3. Saves to `data/embeddings.json`
4. Reloads into memory

### DELETE `/remove/{name}`

Remove a user from the database.

**Response:**
```json
{
  "success": true,
  "removed": "john_doe",
  "embeddings_saved": true
}
```

### GET `/list`

List all enrolled users.

**Response:**
```json
{
  "users": ["alice_wang", "bob_lee", "jane_smith", "john_doe"],
  "count": 4
}
```

### POST `/detect_and_embed`

Debug endpoint - detect faces and return embeddings.

**Request:**
```json
{
  "image_base64": "base64_encoded_image"
}
```

**Response:**
```json
{
  "success": true,
  "faces": [
    {
      "bbox": {"x": 100, "y": 50, "w": 200, "h": 250},
      "landmarks": [[150, 120], [220, 118], ...],
      "confidence": 0.98,
      "embedding": [0.123, -0.456, ...]
    }
  ],
  "error": null,
  "processing_time_ms": 45
}
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

- **Startup**: < 5s with 100 users (loading JSON)
- **Recognition**: < 50ms (detect + embed + search)
- **Reload**: ~300ms per photo (includes processing)
- **Memory**: ~200KB per user (512-D float32 vector)

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

## License

MIT License

## Support

For issues or questions, please create an issue in the main repository.
