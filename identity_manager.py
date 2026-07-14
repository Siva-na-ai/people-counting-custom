from person_registry import PersonRegistry
from identity_matcher import IdentityMatcher
from fusion_engine import FusionEngine
from gallery_manager import GalleryManager
from embedding_cache import EmbeddingCache
from temporal_validator import TemporalValidator
from worker_pool import WorkerPool
from event_logger import EventLogger
import config

class IdentityManager:
    def __init__(self, qdrant_client):
        self.qdrant_client = qdrant_client
        self.registry = PersonRegistry(timeout_sec=config.IDENTITY_TIMEOUT_SEC)
        self.matcher = IdentityMatcher(self.qdrant_client, face_threshold=config.FACE_MATCH_THRESHOLD, body_threshold=config.BODY_MATCH_THRESHOLD)
        self.fusion = FusionEngine(face_weight=0.7, body_weight=0.3)
        self.gallery = GalleryManager()
        self.cache = EmbeddingCache()
        self.temporal = TemporalValidator()
        self.worker = WorkerPool()
        self.logger = EventLogger()

    def process_track(self, track_id, camera_id, face_emb, body_emb, face_quality, body_quality):
        """
        Main logic for identity lifecycle processing per track.
        """
        # 1. Check cache first
        cached = self.cache.get(track_id)
        if cached:
            # For simplicity, if we have a high quality cache, we might skip DB lookup.
            pass

        # 2. Match with Qdrant
        candidates = self.matcher.find_match(face_emb, body_emb)
        
        best_person_id = None
        best_score = 0.0

        if candidates:
            # Fuse similarities
            for pid, scores in candidates.items():
                score = self.fusion.compute_similarity(scores['face_score'], scores['body_score'])
                if score > best_score and score >= config.FUSION_MATCH_THRESHOLD:
                    best_score = score
                    best_person_id = pid
                    
        # 3. Temporal Validation (prevent flickering)
        self.temporal.update(track_id, best_person_id)
        stable_id = self.temporal.get_stable_identity(track_id)
        
        final_person_id = stable_id if stable_id else best_person_id

        # 4. Identity lifecycle
        if final_person_id is None:
            # Create new identity
            final_person_id = self.registry.create_person(track_id, camera_id)
            self.logger.log_creation(final_person_id)
            
            # Write to Qdrant asynchronously
            if face_emb is not None and face_quality >= config.MIN_FACE_QUALITY:
                self.worker.submit_task(self.qdrant_client.insert_face_embedding, final_person_id, face_emb, face_quality, camera_id)
            if body_emb is not None and body_quality >= config.MIN_BODY_QUALITY:
                self.worker.submit_task(self.qdrant_client.insert_body_embedding, final_person_id, body_emb, body_quality, camera_id)
        else:
            # Update existing identity
            self.registry.update_person(final_person_id, track_id, camera_id)
            
            # Maintain galleries
            if face_emb is not None and face_quality >= config.MIN_FACE_QUALITY:
                self.gallery.add_face(final_person_id, face_emb, face_quality)
            if body_emb is not None and body_quality >= config.MIN_BODY_QUALITY:
                self.gallery.add_body(final_person_id, body_emb, body_quality)

        return final_person_id, best_score
