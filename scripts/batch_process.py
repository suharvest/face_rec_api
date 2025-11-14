#!/usr/bin/env python3
"""
Batch Process Photos - CLI tool for processing photos folder

This script can be run independently of the API service to process
all photos in the photos folder and generate embeddings.json.

Usage:
    python scripts/batch_process.py
    python scripts/batch_process.py --photos ./photos --output ./data/embeddings.json
"""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import config
from face_pipeline import FacePipeline
from photo_processor import PhotoProcessor
from vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Batch process photos to generate face embeddings"
    )
    parser.add_argument(
        "--photos",
        type=str,
        default=config.PHOTOS_FOLDER,
        help=f"Path to photos folder (default: {config.PHOTOS_FOLDER})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=config.EMBEDDINGS_JSON,
        help=f"Path to output JSON file (default: {config.EMBEDDINGS_JSON})"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["largest", "first", "error"],
        default=config.MULTIPLE_FACES_STRATEGY,
        help="Strategy for handling multiple faces in one photo"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help="Face detection confidence threshold"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Face Recognition Batch Processor")
    logger.info("=" * 60)
    logger.info(f"Photos folder: {args.photos}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Multiple faces strategy: {args.strategy}")
    logger.info(f"Detection threshold: {args.threshold}")
    logger.info("=" * 60)

    try:
        # 1. Initialize Face Pipeline
        logger.info("Step 1/3: Loading Hailo models...")
        pipeline = FacePipeline(
            detection_model_path=config.FACE_DETECTION_MODEL,
            recognition_model_path=config.FACE_RECOGNITION_MODEL
        )
        logger.info("✓ Models loaded")

        # 2. Initialize Vector Store
        logger.info("Step 2/3: Initializing vector store...")
        vector_store = VectorStore(args.output)
        logger.info(f"✓ Vector store initialized (output: {args.output})")

        # 3. Process photos
        logger.info("Step 3/3: Processing photos...")
        processor = PhotoProcessor(args.photos, pipeline, vector_store)

        # Get photo stats first
        stats = processor.get_photo_stats()
        logger.info(f"Found {stats['total_photos']} photos")

        if stats['total_photos'] == 0:
            logger.warning("No photos found! Please add photos to the folder.")
            logger.info("Exiting...")
            return 0

        # Process all photos
        result = processor.process_all_photos(strategy=args.strategy)

        logger.info("=" * 60)
        logger.info("Processing Results:")
        logger.info(f"  Total photos: {result['total']}")
        logger.info(f"  Successful: {result['success']}")
        logger.info(f"  Failed: {len(result['failed'])}")
        logger.info(f"  Duration: {result['duration_ms']}ms")

        if result['failed']:
            logger.warning("Failed photos:")
            for failed_item in result['failed'][:10]:  # Show first 10
                logger.warning(f"  - {failed_item['file']}: {failed_item['error']}")
            if len(result['failed']) > 10:
                logger.warning(f"  ... and {len(result['failed']) - 10} more")

        # Save to JSON
        logger.info(f"Saving embeddings to {args.output}...")
        saved = vector_store.save_to_json()

        if saved:
            logger.info(f"✓ Embeddings saved successfully")
            logger.info(f"  Users: {len(vector_store.vectors)}")
            logger.info(f"  File: {args.output}")
        else:
            logger.error("Failed to save embeddings!")
            return 1

        logger.info("=" * 60)
        logger.info("✓ Batch processing complete!")
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
