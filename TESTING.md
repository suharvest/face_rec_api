# Testing Guide

## Bug Fix: VDevice Sharing

### Issue Fixed
**Error**: `HAILO_OUT_OF_PHYSICAL_DEVICES` when initializing FacePipeline

**Root Cause**: FaceDetector and FaceEmbedder were each creating separate VDevice instances, but Hailo-8 hardware only supports one VDevice at a time.

**Solution**: FacePipeline now creates a single shared VDevice and passes it to both detector and embedder.

### Testing on Hailo Device

```bash
# 1. Pull latest changes
cd ~/AI_Presence_Control
git checkout standalone-face-service
git pull

# 2. Navigate to standalone service
cd services/standalone_face_api

# 3. Activate virtual environment
source .venv/bin/activate

# 4. Test batch processing
python scripts/batch_process.py

# Expected output:
# ============================================================
# Face Recognition Batch Processor
# ============================================================
# Photos folder: ./photos
# Output file: ./data/embeddings.json
# Multiple faces strategy: largest
# Detection threshold: 0.55
# ============================================================
# Step 1/3: Loading Hailo models...
# INFO - Creating shared Hailo VDevice...
# INFO - ✓ Shared VDevice created
# INFO - Loading face detection model: models/scrfd_10g.hef
# INFO - Face detection model loaded successfully
# INFO - Loading face recognition model: models/arcface_mobilefacenet.hef
# INFO - Face recognition model loaded successfully
# ✓ Models loaded
# ...
```

## Test Cases

### 1. Basic Batch Processing

```bash
# Add test photos to photos/ folder
cp /path/to/test_photos/*.jpg photos/

# Run batch processing
python scripts/batch_process.py
```

**Expected**:
- No HAILO_OUT_OF_PHYSICAL_DEVICES error
- Successfully processes all photos
- Creates data/embeddings.json

### 2. Multiple Faces in Photo

Test with a photo containing multiple faces (like the example with person holding their photo):

```bash
# Add the test photo
cp person_with_photo.jpg photos/allen_pan.jpg

# Process
python scripts/batch_process.py

# Check result
cat data/embeddings.json | grep allen_pan
```

**Expected**:
- Detects both faces (real person and photo)
- Automatically selects the larger face (real person)
- Successfully generates embedding

### 3. API Service

```bash
# Start the service
./start_standalone.sh

# In another terminal, test recognition
IMAGE_B64=$(base64 -i test_face.jpg | tr -d '\n')

curl -X POST http://localhost:8001/recognize \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$IMAGE_B64\"}"
```

**Expected**:
- Service starts without errors
- Recognition works correctly
- Returns match result with confidence

### 4. Hot Reload

```bash
# With service running, add new photo
cp new_person.jpg photos/

# Trigger reload
curl -X POST http://localhost:8001/reload

# Verify new person is loaded
curl http://localhost:8001/list
```

**Expected**:
- Reload succeeds
- New person appears in list
- Can recognize new person

## Verification Checklist

- [ ] Batch processing completes without VDevice errors
- [ ] Both detection and recognition models load successfully
- [ ] Multi-face photos are handled correctly (largest face selected)
- [ ] embeddings.json is created with correct format
- [ ] API service starts and runs
- [ ] Recognition endpoint works
- [ ] Hot reload updates the database
- [ ] Performance is acceptable (<50ms recognition)

## Troubleshooting

### Still getting VDevice errors?

Check if another process is using Hailo:
```bash
# Check for other processes using Hailo
ps aux | grep -i hailo
ps aux | grep python

# Kill any old processes
pkill -f "python.*face"
```

### Models not found?

```bash
# Verify model symlink
ls -la models/
# Should show symlink to ../face_embed_api/models

# Check actual model files exist
ls -la ../face_embed_api/models/*.hef
```

### Permission errors?

```bash
# Check Hailo device permissions
ls -la /dev/hailo*

# If needed, add user to hailo group
sudo usermod -a -G hailo $USER
# Then logout and login again
```

## Performance Benchmarks

After successful testing, record performance metrics:

```bash
# Test with 10 photos
time python scripts/batch_process.py

# Test recognition latency
# (Use the /detect_and_embed endpoint for detailed timing)
```

Expected benchmarks:
- Batch processing: ~300ms per photo
- Recognition latency: <50ms
- Startup time: <5s with 100 users

## Known Limitations

1. **Single VDevice**: Only one FacePipeline instance can exist per process
2. **Sequential processing**: Detection and embedding run sequentially (not in parallel)
3. **Memory**: Each user consumes ~2KB (512 floats × 4 bytes)

## Next Steps After Testing

1. Test with real production photos
2. Tune similarity threshold based on false positive/negative rates
3. Measure performance with larger databases (100+ users)
4. Consider adding metrics/monitoring
5. Document optimal threshold values for your use case
