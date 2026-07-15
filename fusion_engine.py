import numpy as np
from typing import Optional, Tuple
import config

class FusionEngine:
    """
    Orchestrates the decision engine, combining Top-K similarity searches,
    temporal stability validation, visual quality scores, and movement constraints
    to output a final consolidated Person ID.
    """
    def __init__(self, matcher, temporal_validator, registry):
        self.matcher = matcher
        self.temporal_validator = temporal_validator
        self.registry = registry

    def resolve_identity(
        self,
        track_id: int,
        face_emb: Optional[np.ndarray],
        body_emb: Optional[np.ndarray],
        quality_score: float,
        det_box: np.ndarray,
        time_since_update: int,
        img_w: int,
        img_h: int,
        next_person_id_callback: callable
    ) -> Tuple[int, float]:
        """
        Runs multi-modal fusion checks.
        Returns: (final_person_id, confidence_score)
        """
        # 1. Search vector DB and fuse similarity metrics
        candidate_pid, confidence = self.matcher.match_identity(
            face_emb, body_emb, det_box, time_since_update, img_w, img_h
        )
        
        # 2. If no candidate matched, allocate a new Person ID only if a face is detected
        if candidate_pid is None:
            if face_emb is not None:
                candidate_pid = next_person_id_callback()
                confidence = 1.0
            else:
                return None, 0.0
            
        # 3. Apply TemporalValidator to filter flickering identity transitions
        final_person_id = self.temporal_validator.validate_identity(track_id, candidate_pid)
        
        return final_person_id, confidence
