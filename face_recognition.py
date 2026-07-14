import numpy as np
import logging

logger = logging.getLogger(__name__)

class FaceRecognizer:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        logger.info("Initializing ArcFace Recognizer (Mock)")

    def extract_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extracts 512-dim embedding from an aligned 112x112 face crop.
        Returns: normalized embedding array (512,)
        """
        embedding = np.random.randn(512).astype(np.float32)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding
