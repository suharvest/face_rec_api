# Deployment — Containerized Inference Service

Both production devices run the face recognition + liveness service as a
single long-lived Docker container, one process per container, port 8001,
`--restart unless-stopped`, bounded json-file logs. Models and embeddings
live on the host and are bind-mounted; `src/` is bind-mounted read-only so a
`git pull` + `docker restart` updates code without rebuilding the image.

| Device     | Host              | Image               | Container    | Backend |
|------------|-------------------|---------------------|--------------|---------|
| harvest-pi | RPi5 + Hailo-8    | `face_rec_api:hailo`  | `frc-hailo`  | hailo   |
| orin-nano  | Jetson Orin Nano  | `face_rec_api:jetson` | `frc-jetson` | jetson  |

## harvest-pi (Raspberry Pi 5 + Hailo-8)

### HailoRT version alignment (critical)

The kernel driver (`hailo_pci`), userland `libhailort.so` and the Python
binding `_pyhailort` must all match **major.minor** — the 4.x ioctl protocol
is not forward compatible. Current pinned version: **4.21.0**.

- Host: `hailort` + `hailort-pcie-driver` 4.21.0, both **apt-mark hold**
  (verify: `apt-mark showhold | grep hailo` must show both).
- Image: the repo-bundled `hailort-4.21.0-cp311-cp311-linux_aarch64.whl`
  installed against the image's Python 3.11 (host Python is 3.13 — never
  mount the host `hailo_platform` package into the container).
- `libhailort.so.4.21.0` is bind-mounted from the host (the 4.21.0 arm64
  .deb is not in any reachable apt repo; RPi archive carries 4.20/4.23
  only). `_pyhailort` links against the exact soname `libhailort.so.4.21.0`,
  so any host drift fails loudly at startup.

Upgrading HailoRT means: new host deb pair + re-hold, new cp311 wheel in the
repo, rebuild image — all three in one step.

### Build (on the Mac, Apple Silicon)

```bash
cd ~/project/face_rec_api
docker buildx build --builder multiarch --platform linux/arm64 --load \
  -t face_rec_api:hailo -f Dockerfile.hailo .
docker save face_rec_api:hailo | gzip > /tmp/frc-hailo.tar.gz
fleet push harvest-pi /tmp/frc-hailo.tar.gz /tmp/frc-hailo.tar.gz   # /tmp is tmpfs on the Pi
fleet exec --sudo --timeout 600 harvest-pi -- 'docker load < /tmp/frc-hailo.tar.gz && rm /tmp/frc-hailo.tar.gz'
```

Disk note: the Pi root FS is small (~29 G, chronically >90%). Before loading
a new image, clean caches (`~/.cache/uv`, `~/.cache/pip`,
`docker builder prune -af`) and remove the superseded `face_rec_api:hailo`
image after the new one is verified.

### Run

```bash
docker run -d --name frc-hailo \
  --restart unless-stopped \
  --device /dev/hailo0 \
  -p 8001:8001 \
  -v /usr/lib/libhailort.so.4.21.0:/usr/lib/libhailort.so.4.21.0:ro \
  -v /home/harvest/face_rec_api/src:/app/src:ro \
  -v /home/harvest/face_rec_api/models/hailo:/models:ro \
  -v /home/harvest/face_rec_api/data:/data \
  -v /home/harvest/face_rec_api/photos:/photos \
  -e LIVENESS_ENABLED=true \
  --log-opt max-size=10m --log-opt max-file=3 \
  face_rec_api:hailo
```

The container runs as non-root uid 1000 (`frc`), which matches `harvest` on
the Pi so `/data` stays writable. `/dev/hailo0` is world-rw on Pi OS.

The old bare-metal deployment (`~/face_rec_api` venv + `src/app.py` under
nohup) is retired but the directory is kept as-is — it provides the
bind-mounted `src/`, `models/hailo/` and `data/`, and is the rollback path
(`PYTHONPATH=src .venv/bin/python src/app.py`).

### Code update

```bash
cd ~/face_rec_api && git pull   # branch: liveness-sync
docker restart frc-hailo
```

Dependency or HailoRT changes require an image rebuild (see Build).

## orin-nano (Jetson Orin Nano, JetPack 6.2)

Image `face_rec_api:jetson` (built from `Dockerfile.jetson` on the device;
TensorRT engines are device-built via `tools/build_engine.sh`). TRT/CUDA
libraries are bind-mounted from the host JetPack install — the image itself
carries no CUDA.

### Run

```bash
docker run -d --name frc-jetson \
  --restart unless-stopped \
  --runtime nvidia \
  -p 8001:8001 \
  --log-opt max-size=10m --log-opt max-file=3 \
  -v /usr/lib/python3.10/dist-packages/tensorrt:/usr/lib/python3.10/dist-packages/tensorrt:ro \
  -v /usr/lib/python3.10/dist-packages/tensorrt_dispatch:/usr/lib/python3.10/dist-packages/tensorrt_dispatch:ro \
  -v /usr/lib/python3.10/dist-packages/tensorrt_lean:/usr/lib/python3.10/dist-packages/tensorrt_lean:ro \
  -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro \
  -v /usr/local/cuda:/usr/local/cuda:ro \
  -v /home/harvest/face_rec_api/models/jetson:/models:ro \
  -v /home/harvest/face_rec_api/src:/app/src:ro \
  -v /home/harvest/face_rec_api/data:/data \
  -v /home/harvest/face_rec_api/photos:/photos \
  face_rec_api:jetson
```

(`/data` was added 2026-07 — before that, embeddings lived only inside the
container FS and were lost on recreation.)

### Code update

```bash
cd ~/face_rec_api && git pull   # branch: main-liveness
docker restart frc-jetson
```

Rollback container `frc-jetson-prep2` and
`~/face_rec_api_untracked_backup_p2/` still exist on the device.

## Smoke test (both devices, mandatory after any deploy)

```bash
curl -s http://<device>:8001/health          # expect "liveness":"loaded"
curl -s -X POST http://<device>:8001/recognize \
  -H 'Content-Type: application/json' -d @req_real.json   # live photo → liveness passes
curl -s -X POST http://<device>:8001/recognize \
  -H 'Content-Type: application/json' -d @req_fake.json   # spoof photo → reason "spoof"
docker restart frc-hailo && sleep 20 && curl -s http://<device>:8001/health
```

`req_*.json` shape: `{"image_base64": "<base64 jpeg/png>"}`.
