from collections import defaultdict

class EmbeddingCache:
    def __init__(self, max_cache_size=1000):
        self.cache = {}
        self.max_cache_size = max_cache_size
        
    def get(self, track_id):
        """Retrieve cached embedding for a track ID."""
        return self.cache.get(track_id, None)
        
    def put(self, track_id, embedding, quality):
        """
        Store embedding for a track ID. 
        Only updates if the new embedding has a higher quality score, 
        or if it's the first time seeing this track_id.
        """
        if len(self.cache) >= self.max_cache_size and track_id not in self.cache:
            # Simple eviction: remove a random key (can be optimized to LRU)
            self.cache.pop(next(iter(self.cache)))
            
        if track_id not in self.cache:
            self.cache[track_id] = {'embedding': embedding, 'quality': quality}
        else:
            if quality > self.cache[track_id]['quality']:
                self.cache[track_id] = {'embedding': embedding, 'quality': quality}
                
    def clear(self, track_id):
        if track_id in self.cache:
            del self.cache[track_id]
