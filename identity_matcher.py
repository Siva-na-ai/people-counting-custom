import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import config

logger = logging.getLogger("identity_matcher")

class IdentityMatcher:
    """
    Performs similarity search in Qdrant, merges face and body match vectors,
    resolves priority (face > body), and runs spatiotemporal movement gates.
    """
    def __init__(self, qdrant_client, movement_validator, registry):
        self.qdrant = qdrant_client
        self.movement_validator = movement_validator
        self.registry = registry

    def match_identity(
        self, 
        face_emb: Optional[np.ndarray], 
        body_emb: Optional[np.ndarray], 
        det_box: np.ndarray,
        time_since_update: int,
        img_w: int, 
        img_h: int
    ) -> Tuple[Optional[int], float]:
        """
        Retrieves Top-5 matches from Qdrant, fuses scores, applies movement validation,
        and returns the selected person_id and match confidence.
        """
        face_hits = []
        body_hits = []
        
        # 1. Search vector DB (Top-5 Face & Top-5 Body)
        if face_emb is not None:
            face_hits = self.qdrant.search_similar("face_embeddings", face_emb.tolist(), limit=5)
            logger.info(f"[IdentityMatcher] Face search hits: {[(hit['payload']['person_id'], round(hit['score'], 4)) for hit in face_hits]}")
            
        if body_emb is not None:
            body_hits = self.qdrant.search_similar("body_embeddings", body_emb.tolist(), limit=5)
            logger.info(f"[IdentityMatcher] Body search hits: {[(hit['payload']['person_id'], round(hit['score'], 4)) for hit in body_hits]}")
            
        # 2. Priority check: High confidence face override (similarity >= 0.90)
        best_face_pid = None
        best_face_score = 0.0
        for hit in face_hits:
            score = hit["score"]
            pid = hit["payload"]["person_id"]
            if score > best_face_score:
                best_face_score = score
                best_face_pid = pid
                
        if best_face_pid is not None and best_face_score >= 0.90:
            # Check spatiotemporal validity
            if self._validate_motion_for_pid(best_face_pid, det_box, time_since_update, img_w, img_h):
                return best_face_pid, best_face_score

        # 3. Fuse similarities (Face weight: 0.6, Body weight: 0.4)
        candidates = {}  # person_id -> combined_score
        
        # Parse face scores
        for hit in face_hits:
            pid = hit["payload"]["person_id"]
            score = hit["score"]
            if score >= config.REID_THRESHOLD_FACE:
                candidates[pid] = candidates.get(pid, 0.0) + 0.60 * score
                
        # Parse body scores
        for hit in body_hits:
            pid = hit["payload"]["person_id"]
            score = hit["score"]
            if score >= config.REID_THRESHOLD_BODY:
                weight = 0.40 if len(face_hits) > 0 else 1.0
                candidates[pid] = candidates.get(pid, 0.0) + weight * score

        # 4. Filter candidates using MovementValidator spatiotemporal checks
        valid_candidates = {}
        for pid, score in candidates.items():
            if self._validate_motion_for_pid(pid, det_box, time_since_update, img_w, img_h):
                valid_candidates[pid] = score
                
        # 5. Return highest scoring candidate
        if len(valid_candidates) > 0:
            best_pid = max(valid_candidates, key=valid_candidates.get)
            best_score = valid_candidates[best_pid]
            
            # Normalize confidence back to [0.0, 1.0] range
            norm_confidence = min(1.0, best_score)
            return best_pid, norm_confidence
            
        return None, 0.0

    def _validate_motion_for_pid(
        self, 
        person_id: int, 
        det_box: np.ndarray, 
        time_since_update: int, 
        img_w: int, 
        img_h: int
    ) -> bool:
        """
        Helper method to retrieve person's last location and validate physical motion constraints.
        """
        if person_id not in self.registry.persons:
            # Fallback if registry profile not fully loaded yet in memory
            return True
            
        # Get metadata
        p = self.registry.persons[person_id]
        
        # Check active track coordinates first (if they are currently mapped to a track)
        last_box = None
        if self.qdrant:
            # Look up last_box in the registry/global_gallery
            # We can read 'last_box' payload from person_metadata collection
            # Retrieve from registry memory (fast cache)
            # (We will store 'last_box' inside the registry persons metadata dict)
            last_box = p.get("last_box", None)
            
        if last_box is None:
            return True
            
        # Get last seen timestamp
        last_seen_str = p.get("last_seen", None)
        dt_seconds = 0.0
        if last_seen_str is not None:
            try:
                from datetime import datetime
                last_seen_dt = datetime.fromisoformat(last_seen_str)
                dt_seconds = (datetime.now() - last_seen_dt).total_seconds()
            except Exception:
                pass
                
        is_lost = p.get("identity_state", None) == "LOST"
        
        if is_lost or dt_seconds > 2.0:
            # Use long-term camera transition validation
            return self.movement_validator.validate_camera_transition(
                last_box, det_box, dt_seconds, img_w, img_h
            )
        else:
            # Validate same-camera short-term walking speed limit
            return self.movement_validator.validate_track_motion(
                last_box, det_box, time_since_update, img_w, img_h
            )
