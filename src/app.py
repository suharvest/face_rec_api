"""
Standalone Face Recognition API.

Stateless face recognition service. Detects faces, extracts embeddings, and
optionally matches them against a JSON-backed vector store. Backend selection
(Hailo / Jetson / RKNN) is controlled via the ``FACE_BACKEND`` env var.
"""
import base64
import logging
import time
from typing import List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

import config
from face_pipeline import FacePipeline
from photo_processor import PhotoProcessor
from vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Standalone Face Recognition API",
    description="Face recognition service with pluggable backends (Hailo / Jetson / RKNN)",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
face_pipeline: Optional[FacePipeline] = None
vector_store: Optional[VectorStore] = None
photo_processor: Optional[PhotoProcessor] = None
service_start_time: float = 0


# --- Pydantic Models ---

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str = Field(..., description="'healthy' or 'degraded'")
    backend: str = Field(..., description="Active backend name")
    model_tag: str = Field(..., description="Model identifier")
    capabilities: List[str] = Field(..., description="Backend capabilities")
    users_loaded: int = Field(..., description="Number of loaded users in vector store")
    embeddings_file: str = Field(..., description="Path to embeddings JSON file")
    last_reload: Optional[str] = Field(None, description="Last reload timestamp")
    uptime_ms: int = Field(..., description="Service uptime in milliseconds")


class RecognizeRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image")


class RecognizeResponse(BaseModel):
    matched: bool
    name: Optional[str] = None
    confidence: float
    processing_time_ms: int


class EnrollRequest(BaseModel):
    name: str
    image_base64: str


class EnrollResponse(BaseModel):
    success: bool
    name: str
    embedding_saved: bool
    error: Optional[str] = None


class ReloadRequest(BaseModel):
    force: bool = False


class ReloadResponse(BaseModel):
    success: bool
    loaded: int
    failed: list
    embeddings_saved: bool
    reload_time_ms: int


class RemoveResponse(BaseModel):
    success: bool
    removed: str
    embeddings_saved: bool


class ListResponse(BaseModel):
    users: list
    count: int


class DetectAndEmbedRequest(BaseModel):
    image_base64: str


class DetectAndEmbedResponse(BaseModel):
    success: bool
    faces: list
    error: Optional[str] = None
    processing_time_ms: int


class InferRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded image")
    return_aligned: bool = Field(False, description="If True, return aligned 112x112 crops (debug)")


class FaceResult(BaseModel):
    bbox: List[float] = Field(..., description="[x, y, w, h]")
    landmarks: List[List[float]] = Field(..., description="5 x [x, y]")
    embedding: str = Field(..., description="base64(float32[512])")
    det_score: float
    aligned_b64: Optional[str] = None


class InferResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    faces: List[FaceResult]
    face_count: int
    model_tag: str
    backend: str
    processing_time_ms: float


# --- Startup/Shutdown Events ---

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global face_pipeline, vector_store, photo_processor, service_start_time

    service_start_time = time.time()

    logger.info("=" * 60)
    logger.info("Starting Standalone Face Recognition Service")
    logger.info("Backend: %s", config.FACE_BACKEND)
    logger.info("=" * 60)

    try:
        # 1. Initialize Face Pipeline (loads backend + models)
        logger.info("Step 1/3: Initializing Face Pipeline (backend=%s)...", config.FACE_BACKEND)
        face_pipeline = FacePipeline(
            detection_model_path=config.FACE_DETECTION_MODEL,
            recognition_model_path=config.FACE_RECOGNITION_MODEL,
            backend_name=config.FACE_BACKEND,
        )
        logger.info("Face Pipeline initialized (model_tag=%s)", face_pipeline.backend.model_tag)

        # 2. Vector store
        logger.info("Step 2/3: Loading embeddings from JSON...")
        vector_store = VectorStore(config.EMBEDDINGS_JSON)
        loaded = vector_store.load_from_json()
        if loaded:
            logger.info("Loaded %d users from %s", len(vector_store.vectors), config.EMBEDDINGS_JSON)
        else:
            logger.warning("No embeddings file found at %s", config.EMBEDDINGS_JSON)

        # 3. Photo processor
        logger.info("Step 3/3: Initializing Photo Processor...")
        photo_processor = PhotoProcessor(
            photos_folder=config.PHOTOS_FOLDER,
            pipeline=face_pipeline,
            vector_store=vector_store
        )
        stats = photo_processor.get_photo_stats()
        logger.info("Photos in folder: %d", stats['total_photos'])

        logger.info("=" * 60)
        logger.info("Service Ready! users=%d threshold=%s",
                    len(vector_store.vectors), config.SIMILARITY_THRESHOLD)
        logger.info("=" * 60)

    except Exception as e:
        logger.error("Failed to initialize service: %s", e, exc_info=True)
        raise RuntimeError("Service initialization failed") from e


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown — stop inference threads, release VDevice."""
    logger.info("Shutting down Standalone Face Recognition Service...")
    if face_pipeline is not None:
        try:
            face_pipeline.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pipeline shutdown error: %s", exc)
    logger.info("Shutdown complete")


# --- helpers ---

def _decode_b64_image(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")
    return img


def _encode_image_jpeg_b64(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


# --- API Endpoints ---

@app.get("/", response_model=dict)
async def root():
    return {
        "service": "Standalone Face Recognition API",
        "version": "1.1.0",
        "status": "running",
        "backend": face_pipeline.backend.backend_name if face_pipeline else None,
        "endpoints": {
            "health": "/health",
            "infer": "/infer",
            "recognize": "/recognize",
            "enroll": "/enroll",
            "reload": "/reload",
            "remove": "/remove/{name}",
            "list": "/list",
            "detect_and_embed": "/detect_and_embed"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check with active backend probe."""
    uptime_ms = int((time.time() - service_start_time) * 1000)

    backend_name = face_pipeline.backend.backend_name if face_pipeline else "none"
    model_tag = face_pipeline.backend.model_tag if face_pipeline else "n/a"

    try:
        ok = bool(face_pipeline and face_pipeline.backend.health_check())
    except Exception as exc:  # noqa: BLE001
        logger.warning("health_check threw: %s", exc)
        ok = False

    return HealthResponse(
        status="healthy" if ok else "degraded",
        backend=backend_name,
        model_tag=model_tag,
        capabilities=["detect", "embed"],
        users_loaded=len(vector_store.vectors) if vector_store else 0,
        embeddings_file=config.EMBEDDINGS_JSON,
        last_reload=vector_store.metadata.get('last_updated') if vector_store else None,
        uptime_ms=uptime_ms,
    )


@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    """
    Stateless face inference: returns all detected faces + embeddings.

    Does NOT query the vector store and does NOT identify by name. Upstream
    services own embedding storage and matching.
    """
    if face_pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    t0 = time.time()
    image = _decode_b64_image(request.image_b64)

    try:
        results = face_pipeline.process_all_faces(image)
    except Exception as exc:
        logger.error("infer failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail=f"backend error: {exc}")

    faces_out: List[FaceResult] = []
    for r in results:
        bb = r["bbox"]
        emb: np.ndarray = r["embedding"]
        emb_b64 = base64.b64encode(emb.astype(np.float32).tobytes()).decode("ascii")
        aligned_b64: Optional[str] = None
        if request.return_aligned and r.get("aligned") is not None:
            aligned_b64 = _encode_image_jpeg_b64(r["aligned"])
        faces_out.append(
            FaceResult(
                bbox=[float(bb["x"]), float(bb["y"]), float(bb["w"]), float(bb["h"])],
                landmarks=[[float(x), float(y)] for (x, y) in r["landmarks"]],
                embedding=emb_b64,
                det_score=float(r["confidence"]),
                aligned_b64=aligned_b64,
            )
        )

    return InferResponse(
        faces=faces_out,
        face_count=len(faces_out),
        model_tag=face_pipeline.backend.model_tag,
        backend=face_pipeline.backend.backend_name,
        processing_time_ms=(time.time() - t0) * 1000.0,
    )


@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(request: RecognizeRequest):
    """Recognize face against the in-memory vector store."""
    start_time = time.time()
    try:
        result = face_pipeline.process_image_base64(
            request.image_base64,
            strategy=config.MULTIPLE_FACES_STRATEGY
        )
        if not result['success']:
            logger.warning("Recognition failed: %s", result['error'])
            return RecognizeResponse(
                matched=False, name=None, confidence=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000),
            )
        embedding = result['embedding']
        search_result = vector_store.search(embedding, threshold=config.SIMILARITY_THRESHOLD)
        return RecognizeResponse(
            matched=search_result['matched'],
            name=search_result['name'],
            confidence=search_result['confidence'],
            processing_time_ms=int((time.time() - start_time) * 1000),
        )
    except Exception as e:
        logger.error("Recognition error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enroll", response_model=EnrollResponse)
async def enroll(request: EnrollRequest):
    try:
        result = face_pipeline.process_image_base64(
            request.image_base64,
            strategy=config.MULTIPLE_FACES_STRATEGY
        )
        if not result['success']:
            return EnrollResponse(
                success=False, name=request.name,
                embedding_saved=False, error=result['error'],
            )
        embedding = result['embedding']
        vector_store.add(request.name, embedding)
        saved = vector_store.save_to_json()
        logger.info("Enrolled '%s' successfully", request.name)
        return EnrollResponse(
            success=True, name=request.name, embedding_saved=saved, error=None,
        )
    except Exception as e:
        logger.error("Enrollment error: %s", e, exc_info=True)
        return EnrollResponse(
            success=False, name=request.name, embedding_saved=False, error=str(e)
        )


@app.post("/reload", response_model=ReloadResponse)
async def reload(request: ReloadRequest = ReloadRequest(force=False)):
    start_time = time.time()
    try:
        logger.info("Starting photo folder reload...")
        vector_store.clear()
        result = photo_processor.process_all_photos()
        saved = vector_store.save_to_json()
        reload_time_ms = int((time.time() - start_time) * 1000)
        logger.info("Reload complete: %d success, %d failed, %dms",
                    result['success'], len(result['failed']), reload_time_ms)
        return ReloadResponse(
            success=True, loaded=result['success'], failed=result['failed'],
            embeddings_saved=saved, reload_time_ms=reload_time_ms,
        )
    except Exception as e:
        logger.error("Reload error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/remove/{name}", response_model=RemoveResponse)
async def remove(name: str):
    try:
        removed = vector_store.remove(name)
        if not removed:
            raise HTTPException(status_code=404, detail=f"User '{name}' not found")
        saved = vector_store.save_to_json()
        logger.info("Removed '%s' from vector store", name)
        return RemoveResponse(success=True, removed=name, embeddings_saved=saved)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Remove error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list", response_model=ListResponse)
async def list_users():
    try:
        users = vector_store.list_all()
        return ListResponse(users=sorted(users), count=len(users))
    except Exception as e:
        logger.error("List error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect_and_embed", response_model=DetectAndEmbedResponse)
async def detect_and_embed(request: DetectAndEmbedRequest):
    """Debug endpoint: detect + embed first (largest) face. Kept for compatibility."""
    start_time = time.time()
    try:
        result = face_pipeline.process_image_base64(request.image_base64)
        processing_time_ms = int((time.time() - start_time) * 1000)
        if not result['success']:
            return DetectAndEmbedResponse(
                success=False, faces=[], error=result['error'],
                processing_time_ms=processing_time_ms,
            )
        face_data = {
            'bbox': result['face']['bbox'],
            'landmarks': result['face']['landmarks'],
            'confidence': result['face']['confidence'],
            'embedding': result['embedding'],
        }
        return DetectAndEmbedResponse(
            success=True, faces=[face_data], error=None,
            processing_time_ms=processing_time_ms,
        )
    except Exception as e:
        logger.error("Detect and embed error: %s", e, exc_info=True)
        return DetectAndEmbedResponse(
            success=False, faces=[], error=str(e),
            processing_time_ms=int((time.time() - start_time) * 1000),
        )


# --- Main Entry Point ---

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        log_level="info",
        reload=False,
    )
