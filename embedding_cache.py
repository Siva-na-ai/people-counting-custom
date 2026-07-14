import logging
import time

logger = logging.getLogger(__name__)

class EmbeddingCache:
    def __init__(self, cache_timeout_seconds=5.0):
        self.cache = {}
        self.timeout = cache_timeout_seconds

    def get_embedding(self, track_id: int, quality: float):
        """Returns cached embedding if quality isn't significantly better and hasn't timed out."""
        if track_id not in self.cache:
            return None
            
        cached_time, cached_quality, cached_emb = self.cache[track_id]
        if time.time() - cached_time > self.timeout:
            return None
            
        # Only return cached if the new quality isn't much better
        if quality > cached_quality + 0.1:
            return None
            
        return cached_emb

    def set_embedding(self, track_id: int, embedding, quality: float):
        self.cache[track_id] = (time.time(), quality, embedding)

    def clear(self, track_id: int):
        if track_id in self.cache:
            del self.cache[track_id]
