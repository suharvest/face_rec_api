"""
Standalone Face Recognition API
Provides face recognition service with JSON-based vector storage
"""
import asyncio
import logging
import time
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
    description="Face recognition service with JSON-based vector storage",
    version="1.0.0"
)

# Add CORS middleware
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
    status: str = Field(..., description="Service status")
    loaded_users: int = Field(..., description="Number of loaded users")
    embeddings_file: str = Field(..., description="Path to embeddings JSON file")
    last_reload: Optional[str] = Field(None, description="Last reload timestamp")
    uptime_ms: int = Field(..., description="Service uptime in milliseconds")


class RecognizeRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image")


class RecognizeResponse(BaseModel):
    matched: bool = Field(..., description="Whether a match was found")
    name: Optional[str] = Field(None, description="Matched person name")
    confidence: float = Field(..., description="Similarity confidence (0-1)")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class EnrollRequest(BaseModel):
    name: str = Field(..., description="Person name")
    image_base64: str = Field(..., description="Base64-encoded image")


class EnrollResponse(BaseModel):
    success: bool = Field(..., description="Whether enrollment succeeded")
    name: str = Field(..., description="Person name")
    embedding_saved: bool = Field(..., description="Whether embedding was saved to JSON")
    error: Optional[str] = Field(None, description="Error message if failed")


class ReloadRequest(BaseModel):
    force: bool = Field(False, description="Force reload even if no changes detected")


class ReloadResponse(BaseModel):
    success: bool = Field(..., description="Whether reload succeeded")
    loaded: int = Field(..., description="Number of users loaded")
    failed: list = Field(..., description="List of failed photos")
    embeddings_saved: bool = Field(..., description="Whether embeddings were saved to JSON")
    reload_time_ms: int = Field(..., description="Total reload time in milliseconds")


class RemoveResponse(BaseModel):
    success: bool = Field(..., description="Whether removal succeeded")
    removed: str = Field(..., description="Name of removed user")
    embeddings_saved: bool = Field(..., description="Whether changes were saved to JSON")


class ListResponse(BaseModel):
    users: list = Field(..., description="List of user names")
    count: int = Field(..., description="Total count")


class DetectAndEmbedRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image")


class DetectAndEmbedResponse(BaseModel):
    success: bool = Field(..., description="Whether processing succeeded")
    faces: list = Field(..., description="List of detected faces with embeddings")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


# --- Startup/Shutdown Events ---

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global face_pipeline, vector_store, photo_processor, service_start_time

    service_start_time = time.time()

    logger.info("=" * 60)
    logger.info("Starting Standalone Face Recognition Service")
    logger.info("=" * 60)

    try:
        # 1. Initialize Face Pipeline (load Hailo models)
        logger.info("Step 1/3: Initializing Face Pipeline...")
        face_pipeline = FacePipeline(
            detection_model_path=config.FACE_DETECTION_MODEL,
            recognition_model_path=config.FACE_RECOGNITION_MODEL
        )
        logger.info("✓ Face Pipeline initialized")

        # 2. Initialize Vector Store and load embeddings from JSON
        logger.info("Step 2/3: Loading embeddings from JSON...")
        vector_store = VectorStore(config.EMBEDDINGS_JSON)

        loaded = vector_store.load_from_json()
        if loaded:
            logger.info(f"✓ Loaded {len(vector_store.vectors)} users from {config.EMBEDDINGS_JSON}")
        else:
            logger.warning(f"No embeddings file found at {config.EMBEDDINGS_JSON}")
            logger.info("Will create new embeddings file on first reload/enroll")

        # 3. Initialize Photo Processor
        logger.info("Step 3/3: Initializing Photo Processor...")
        photo_processor = PhotoProcessor(
            photos_folder=config.PHOTOS_FOLDER,
            pipeline=face_pipeline,
            vector_store=vector_store
        )
        logger.info(f"✓ Photo Processor initialized (folder: {config.PHOTOS_FOLDER})")

        # Show stats
        stats = photo_processor.get_photo_stats()
        logger.info(f"Photos in folder: {stats['total_photos']}")

        logger.info("=" * 60)
        logger.info("✓ Service Ready!")
        logger.info(f"Users loaded: {len(vector_store.vectors)}")
        logger.info(f"Similarity threshold: {config.SIMILARITY_THRESHOLD}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to initialize service: {e}", exc_info=True)
        raise RuntimeError("Service initialization failed") from e


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Standalone Face Recognition Service...")


# --- API Endpoints ---

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "service": "Standalone Face Recognition API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
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
    """Health check endpoint"""
    uptime_ms = int((time.time() - service_start_time) * 1000)

    return HealthResponse(
        status="ok",
        loaded_users=len(vector_store.vectors) if vector_store else 0,
        embeddings_file=config.EMBEDDINGS_JSON,
        last_reload=vector_store.metadata.get('last_updated') if vector_store else None,
        uptime_ms=uptime_ms
    )


@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(request: RecognizeRequest):
    """
    Recognize face in image

    Primary endpoint for face recognition. Detects face, extracts embedding,
    and searches for best match in loaded embeddings.
    """
    start_time = time.time()

    try:
        # Process image through pipeline
        result = face_pipeline.process_image_base64(
            request.image_base64,
            strategy=config.MULTIPLE_FACES_STRATEGY
        )

        if not result['success']:
            logger.warning(f"Recognition failed: {result['error']}")
            return RecognizeResponse(
                matched=False,
                name=None,
                confidence=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )

        # Search in vector store
        embedding = result['embedding']
        search_result = vector_store.search(embedding, threshold=config.SIMILARITY_THRESHOLD)

        processing_time_ms = int((time.time() - start_time) * 1000)

        return RecognizeResponse(
            matched=search_result['matched'],
            name=search_result['name'],
            confidence=search_result['confidence'],
            processing_time_ms=processing_time_ms
        )

    except Exception as e:
        logger.error(f"Recognition error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enroll", response_model=EnrollResponse)
async def enroll(request: EnrollRequest):
    """
    Manually enroll a single person

    Processes the image, extracts embedding, adds to vector store,
    and saves to JSON file.
    """
    try:
        # Process image
        result = face_pipeline.process_image_base64(
            request.image_base64,
            strategy=config.MULTIPLE_FACES_STRATEGY
        )

        if not result['success']:
            return EnrollResponse(
                success=False,
                name=request.name,
                embedding_saved=False,
                error=result['error']
            )

        # Add to vector store
        embedding = result['embedding']
        vector_store.add(request.name, embedding)

        # Save to JSON
        saved = vector_store.save_to_json()

        logger.info(f"Enrolled '{request.name}' successfully")

        return EnrollResponse(
            success=True,
            name=request.name,
            embedding_saved=saved,
            error=None
        )

    except Exception as e:
        logger.error(f"Enrollment error: {e}", exc_info=True)
        return EnrollResponse(
            success=False,
            name=request.name,
            embedding_saved=False,
            error=str(e)
        )


@app.post("/reload", response_model=ReloadResponse)
async def reload(request: ReloadRequest = ReloadRequest(force=False)):
    """
    Reload embeddings from photos folder

    Scans the photos folder, processes all images, updates embeddings in memory,
    and saves to JSON file. This is the primary way to update the recognition database.
    """
    start_time = time.time()

    try:
        logger.info("Starting photo folder reload...")

        # Clear current vectors
        vector_store.clear()

        # Process all photos
        result = photo_processor.process_all_photos()

        # Save to JSON
        saved = vector_store.save_to_json()

        reload_time_ms = int((time.time() - start_time) * 1000)

        logger.info(f"Reload complete: {result['success']} success, "
                   f"{len(result['failed'])} failed, {reload_time_ms}ms")

        return ReloadResponse(
            success=True,
            loaded=result['success'],
            failed=result['failed'],
            embeddings_saved=saved,
            reload_time_ms=reload_time_ms
        )

    except Exception as e:
        logger.error(f"Reload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/remove/{name}", response_model=RemoveResponse)
async def remove(name: str):
    """
    Remove a user from the vector store

    Removes the user from memory and updates the JSON file.
    """
    try:
        removed = vector_store.remove(name)

        if not removed:
            raise HTTPException(status_code=404, detail=f"User '{name}' not found")

        # Save to JSON
        saved = vector_store.save_to_json()

        logger.info(f"Removed '{name}' from vector store")

        return RemoveResponse(
            success=True,
            removed=name,
            embeddings_saved=saved
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Remove error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list", response_model=ListResponse)
async def list_users():
    """
    List all users in vector store

    Returns list of all enrolled user names.
    """
    try:
        users = vector_store.list_all()
        return ListResponse(
            users=sorted(users),
            count=len(users)
        )

    except Exception as e:
        logger.error(f"List error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect_and_embed", response_model=DetectAndEmbedResponse)
async def detect_and_embed(request: DetectAndEmbedRequest):
    """
    Debug endpoint: Detect faces and return embeddings

    Useful for testing the face detection and embedding pipeline.
    Returns all detected faces with their embeddings.
    """
    start_time = time.time()

    try:
        result = face_pipeline.process_image_base64(request.image_base64)

        processing_time_ms = int((time.time() - start_time) * 1000)

        if not result['success']:
            return DetectAndEmbedResponse(
                success=False,
                faces=[],
                error=result['error'],
                processing_time_ms=processing_time_ms
            )

        face_data = {
            'bbox': result['face']['bbox'],
            'landmarks': result['face']['landmarks'],
            'confidence': result['face']['confidence'],
            'embedding': result['embedding']
        }

        return DetectAndEmbedResponse(
            success=True,
            faces=[face_data],
            error=None,
            processing_time_ms=processing_time_ms
        )

    except Exception as e:
        logger.error(f"Detect and embed error: {e}", exc_info=True)
        return DetectAndEmbedResponse(
            success=False,
            faces=[],
            error=str(e),
            processing_time_ms=int((time.time() - start_time) * 1000)
        )


# --- Main Entry Point ---

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        log_level="info",
        reload=False
    )
