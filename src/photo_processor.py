"""
Photo Processor - Batch process photos from folder to embeddings
Scans photo directory and extracts face embeddings for each person
"""
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

import config
from face_pipeline import FacePipeline
from vector_store import VectorStore

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


class PhotoProcessor:
    """Batch process photos from folder to generate face embeddings"""

    def __init__(self, photos_folder: str, pipeline: FacePipeline, vector_store: VectorStore):
        """
        Initialize photo processor

        Args:
            photos_folder: Path to folder containing photos
            pipeline: FacePipeline instance for processing
            vector_store: VectorStore instance for storing embeddings
        """
        self.photos_folder = Path(photos_folder)
        self.pipeline = pipeline
        self.vector_store = vector_store

        if not self.photos_folder.exists():
            logger.warning(f"Photos folder does not exist: {self.photos_folder}")
            self.photos_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created photos folder: {self.photos_folder}")

    def process_all_photos(self, strategy: str = None) -> Dict[str, any]:
        """
        Process all photos in folder

        Args:
            strategy: How to handle multiple faces ("largest", "first", "error")
                     Defaults to config.MULTIPLE_FACES_STRATEGY

        Returns:
            Dict with keys:
                - success: int (count of successfully processed photos)
                - failed: List[Dict] (list of failed photos with errors)
                - total: int (total photos found)
                - duration_ms: int (processing time)
        """
        import time
        start_time = time.time()

        strategy = strategy or config.MULTIPLE_FACES_STRATEGY

        # Find all image files
        photo_files = self._find_photos()

        if not photo_files:
            logger.warning(f"No photos found in {self.photos_folder}")
            return {
                'success': 0,
                'failed': [],
                'total': 0,
                'duration_ms': 0
            }

        logger.info(f"Found {len(photo_files)} photos to process")

        success_count = 0
        failed = []

        for photo_path in photo_files:
            result = self.process_single_photo(photo_path, strategy=strategy)

            if result['success']:
                success_count += 1
            else:
                failed.append({
                    'file': photo_path.name,
                    'error': result['error']
                })

            # Progress logging every 10 photos
            if (success_count + len(failed)) % 10 == 0:
                logger.info(f"Progress: {success_count + len(failed)}/{len(photo_files)}")

        duration_ms = int((time.time() - start_time) * 1000)

        logger.info(f"Batch processing complete: {success_count} success, {len(failed)} failed, {duration_ms}ms")

        return {
            'success': success_count,
            'failed': failed,
            'total': len(photo_files),
            'duration_ms': duration_ms
        }

    def process_single_photo(self, file_path: Path, strategy: str = None) -> Dict[str, any]:
        """
        Process a single photo file

        Args:
            file_path: Path to photo file
            strategy: How to handle multiple faces

        Returns:
            Dict with keys:
                - success: bool
                - name: str (extracted from filename)
                - embedding: List[float] if success
                - error: str if failed
        """
        strategy = strategy or config.MULTIPLE_FACES_STRATEGY

        # Extract name from filename (remove extension)
        name = file_path.stem

        try:
            # Read image
            image = cv2.imread(str(file_path))

            if image is None:
                error_msg = f"Failed to read image file"
                logger.error(f"{error_msg}: {file_path}")
                return {
                    'success': False,
                    'name': name,
                    'embedding': None,
                    'error': error_msg
                }

            # Process through pipeline
            result = self.pipeline.process_image(image, strategy=strategy)

            if not result['success']:
                logger.warning(f"Failed to process '{name}': {result['error']}")
                return {
                    'success': False,
                    'name': name,
                    'embedding': None,
                    'error': result['error']
                }

            # Add to vector store (in-memory, doesn't save yet)
            embedding = result['embedding']
            self.vector_store.add(name, embedding)

            logger.debug(f"Processed '{name}' successfully")
            return {
                'success': True,
                'name': name,
                'embedding': embedding,
                'error': None
            }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Exception processing '{name}': {error_msg}", exc_info=True)
            return {
                'success': False,
                'name': name,
                'embedding': None,
                'error': error_msg
            }

    def _find_photos(self) -> List[Path]:
        """
        Find all image files in photos folder

        Returns:
            List of Path objects for image files
        """
        if not self.photos_folder.exists():
            return []

        photos = []
        for file_path in self.photos_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                photos.append(file_path)

        # Sort by name for consistent processing order
        photos.sort(key=lambda p: p.name)

        return photos

    def get_photo_stats(self) -> Dict[str, any]:
        """
        Get statistics about photos in folder

        Returns:
            Dict with stats
        """
        photos = self._find_photos()

        # Group by extension
        by_extension = {}
        for photo in photos:
            ext = photo.suffix.lower()
            by_extension[ext] = by_extension.get(ext, 0) + 1

        return {
            'total_photos': len(photos),
            'photos_folder': str(self.photos_folder),
            'by_extension': by_extension
        }
