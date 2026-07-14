import numpy as np
import logging

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        logger.info("Initializing SCRFD Face Detector (Mock)")

    def detect(self, image: np.ndarray):
        """
        Detect faces in the image.
        Returns:
            boxes: [[x1, y1, x2, y2], ...]
            landmarks: [[[x,y]*5], ...]
            scores: [float, ...]
        """
        return [], [], []
