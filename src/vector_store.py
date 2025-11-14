"""
Vector Store - In-memory vector storage with JSON persistence
Supports efficient cosine similarity search for face embeddings
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """In-memory vector store with JSON persistence for face embeddings"""

    def __init__(self, json_path: str):
        """
        Initialize vector store

        Args:
            json_path: Path to JSON file for persistence
        """
        self.json_path = Path(json_path)
        self.vectors: Dict[str, np.ndarray] = {}  # {name: 512-D vector}
        self.metadata: Dict[str, any] = {
            'version': '1.0',
            'last_updated': None,
            'total_users': 0
        }

        # Ensure directory exists
        self.json_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"VectorStore initialized with JSON path: {self.json_path}")

    def load_from_json(self) -> bool:
        """
        Load embeddings from JSON file

        Returns:
            True if successfully loaded, False if file doesn't exist or error
        """
        if not self.json_path.exists():
            logger.warning(f"JSON file not found: {self.json_path}")
            return False

        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract metadata
            if '_metadata' in data:
                self.metadata = data['_metadata']
                del data['_metadata']

            # Load vectors (convert lists back to numpy arrays)
            self.vectors = {
                name: np.array(vector, dtype=np.float32)
                for name, vector in data.items()
            }

            logger.info(f"Loaded {len(self.vectors)} vectors from {self.json_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load JSON: {e}", exc_info=True)
            return False

    def save_to_json(self) -> bool:
        """
        Save embeddings to JSON file

        Returns:
            True if successfully saved, False on error
        """
        try:
            # Update metadata
            self.metadata['last_updated'] = datetime.now().isoformat()
            self.metadata['total_users'] = len(self.vectors)

            # Convert numpy arrays to lists for JSON serialization
            data = {
                name: vector.tolist()
                for name, vector in self.vectors.items()
            }

            # Add metadata
            data['_metadata'] = self.metadata

            # Write to file (with pretty formatting)
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(self.vectors)} vectors to {self.json_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save JSON: {e}", exc_info=True)
            return False

    def add(self, name: str, vector: np.ndarray) -> bool:
        """
        Add vector to store (in-memory only, doesn't auto-save)

        Args:
            name: User name
            vector: 512-D embedding vector

        Returns:
            True if added successfully
        """
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)

        if vector.shape[0] != 512:
            logger.error(f"Invalid vector dimension: {vector.shape[0]}, expected 512")
            return False

        # Ensure L2 normalized
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        self.vectors[name] = vector
        logger.debug(f"Added vector for '{name}'")
        return True

    def remove(self, name: str) -> bool:
        """
        Remove vector from store (in-memory only, doesn't auto-save)

        Args:
            name: User name

        Returns:
            True if removed successfully, False if not found
        """
        if name in self.vectors:
            del self.vectors[name]
            logger.debug(f"Removed vector for '{name}'")
            return True
        else:
            logger.warning(f"Vector for '{name}' not found")
            return False

    def search(self, query_vector: np.ndarray,
              threshold: float = 0.5) -> Dict[str, any]:
        """
        Search for most similar vector using cosine similarity

        Args:
            query_vector: Query embedding vector (512-D)
            threshold: Minimum similarity threshold (0-1)

        Returns:
            Dict with keys:
                - matched: bool
                - name: str or None
                - confidence: float (similarity score 0-1)
        """
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)

        if query_vector.shape[0] != 512:
            logger.error(f"Invalid query vector dimension: {query_vector.shape[0]}")
            return {'matched': False, 'name': None, 'confidence': 0.0}

        if len(self.vectors) == 0:
            logger.warning("Vector store is empty")
            return {'matched': False, 'name': None, 'confidence': 0.0}

        # L2 normalize query vector
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        # Compute cosine similarity with all vectors
        # (Dot product of L2-normalized vectors = cosine similarity)
        best_name = None
        best_similarity = -1.0

        for name, vector in self.vectors.items():
            similarity = float(np.dot(vector, query_vector))
            if similarity > best_similarity:
                best_similarity = similarity
                best_name = name

        # Check threshold
        if best_similarity >= threshold:
            logger.info(f"Match found: '{best_name}' with confidence {best_similarity:.3f}")
            return {
                'matched': True,
                'name': best_name,
                'confidence': best_similarity
            }
        else:
            logger.info(f"No match found (best: {best_similarity:.3f} < threshold: {threshold})")
            return {
                'matched': False,
                'name': None,
                'confidence': best_similarity
            }

    def list_all(self) -> List[str]:
        """
        Get list of all stored user names

        Returns:
            List of user names
        """
        return list(self.vectors.keys())

    def clear(self) -> None:
        """Clear all vectors from memory"""
        self.vectors.clear()
        logger.info("Cleared all vectors from memory")

    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the vector store

        Returns:
            Dict with stats
        """
        return {
            'total_users': len(self.vectors),
            'json_path': str(self.json_path),
            'json_exists': self.json_path.exists(),
            'last_updated': self.metadata.get('last_updated'),
            'version': self.metadata.get('version')
        }

    def batch_add(self, vectors_dict: Dict[str, np.ndarray]) -> Tuple[int, List[str]]:
        """
        Add multiple vectors at once (in-memory only, doesn't auto-save)

        Args:
            vectors_dict: Dictionary of {name: vector}

        Returns:
            Tuple of (success_count, failed_names)
        """
        success_count = 0
        failed_names = []

        for name, vector in vectors_dict.items():
            if self.add(name, vector):
                success_count += 1
            else:
                failed_names.append(name)

        logger.info(f"Batch add: {success_count} success, {len(failed_names)} failed")
        return success_count, failed_names

    def export_to_dict(self) -> Dict[str, List[float]]:
        """
        Export all vectors as a dictionary (for debugging)

        Returns:
            Dict of {name: vector_list}
        """
        return {
            name: vector.tolist()
            for name, vector in self.vectors.items()
        }
