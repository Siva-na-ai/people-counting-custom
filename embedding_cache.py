import time
import numpy as np
from typing import Dict, Optional, Tuple, Any
import config

class CacheEntry:
    def __init__(
        self, 
        face_emb: Optional[np.ndarray], 
        body_emb: Optional[np.ndarray], 
        box: np.ndarray, 
        quality_score: float, 
        score: float
    ):
        self.face_emb = face_emb
        self.body_emb = body_emb
        self.box = box.copy()
        self.quality_score = quality_score
        self.score = score
        self.timestamp = time.time()
        self.frames_active = 0

class EmbeddingCache:
    """
    Caches face and body embeddings per Track ID to minimize Hailo-8L inference load.
    Only re-runs inference if significant motion occurs, quality improves, or age expires.
    """
    def __init__(self, max_cache_age_frames: int = 15):
        self.max_cache_age_frames = max_cache_age_frames
        self.cache: Dict[int, CacheEntry] = {}

    def get(self, track_id: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Retrieves cached embeddings if available.
        """
        if track_id in self.cache:
            entry = self.cache[track_id]
            return entry.face_emb, entry.body_emb
        return None, None

    def update_and_check_should_infer(
        self, 
        track_id: int, 
        new_box: np.ndarray, 
        new_score: float, 
        new_quality: float
    ) -> bool:
        """
        Checks if we must recompute embeddings for the given track_id.
        Updates cache tracking metrics.
        Returns True if we must run Hailo inference, False if we can reuse cache.
        """
        if track_id not in self.cache:
            return True
            
        entry = self.cache[track_id]
        entry.frames_active += 1
        
        # 1. Force refresh if cache is too old (exceeds 15 frames)
        if entry.frames_active >= self.max_cache_age_frames:
            return True
            
        # 2. Recompute if resolution/quality improved significantly
        if new_quality > entry.quality_score + 0.08:
            return True
            
        # 3. Recompute if significant motion occurred
        # Calculate overlap (IoU) between previous box and new box
        x1 = max(entry.box[0], new_box[0])
        y1 = max(entry.box[1], new_box[1])
        x2 = min(entry.box[2], new_box[2])
        y2 = min(entry.box[3], new_box[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (entry.box[2] - entry.box[0]) * (entry.box[3] - entry.box[1])
        box2_area = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0.0
        
        # If IoU is high (e.g. > 0.85), target is stationary, reuse embedding
        if iou < 0.80:
            return True
            
        return False

    def put(
        self, 
        track_id: int, 
        face_emb: Optional[np.ndarray], 
        body_emb: Optional[np.ndarray], 
        box: np.ndarray, 
        quality_score: float, 
        score: float
    ):
        """
        Stores computed embeddings into the cache.
        """
        self.cache[track_id] = CacheEntry(face_emb, body_emb, box, quality_score, score)

    def invalidate(self, track_id: int):
        if track_id in self.cache:
            del self.cache[track_id]
            
    def tick(self, active_track_ids: list):
        """
        Performs periodic cleanup for tracks no longer active.
        """
        inactive_ids = [tid for tid in self.cache.keys() if tid not in active_track_ids]
        for tid in inactive_ids:
            self.invalidate(tid)
