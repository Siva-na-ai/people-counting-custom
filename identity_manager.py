import logging
import uuid
from typing import Optional, Tuple
import numpy as np
from person_registry import PersonRegistry
from identity_matcher import IdentityMatcher
from qdrant_client import QdrantClientWrapper

logger = logging.getLogger(__name__)

class IdentityManager:
    def __init__(self, db: QdrantClientWrapper):
        self.db = db
        self.registry = PersonRegistry()
        self.matcher = IdentityMatcher(self.db)
        # map track_id to person_id for active tracks
        self.active_tracks = {}
        logger.info("Initialized IdentityManager")

    def process_detection(self, track_id: int, camera_id: str, 
                          face_emb: Optional[np.ndarray], body_emb: Optional[np.ndarray], 
                          face_quality: float, body_quality: float) -> str:
        """
        Coordinates identity workflow for a single track.
        Returns the assigned person_id.
        """
        # If we already linked this track to a person, just update
        if track_id in self.active_tracks:
            person_id = self.active_tracks[track_id]
            self.registry.update_person(
                person_id, track_id, camera_id, 
                has_face=(face_emb is not None), 
                has_body=(body_emb is not None)
            )
        else:
            # Try to find a match
            person_id, score = self.matcher.find_match(face_emb, body_emb)
            
            if person_id:
                # Matched existing identity
                self.active_tracks[track_id] = person_id
                self.registry.update_person(
                    person_id, track_id, camera_id, 
                    has_face=(face_emb is not None), 
                    has_body=(body_emb is not None)
                )
            else:
                # Create new identity
                person_id = str(uuid.uuid4())
                self.active_tracks[track_id] = person_id
                self.registry.create_person(person_id, camera_id, track_id)
                
        # Insert embeddings to DB if they meet quality thresholds
        if face_emb is not None:
            self.db.insert_face_embedding(person_id, face_emb, face_quality, camera_id)
        if body_emb is not None:
            self.db.insert_body_embedding(person_id, body_emb, body_quality, camera_id)
            
        return person_id
        
    def end_track(self, track_id: int):
        if track_id in self.active_tracks:
            del self.active_tracks[track_id]
