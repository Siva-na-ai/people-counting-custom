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
        next_person_id_callback: callable,
        face_detected: bool = False
    ) -> Tuple[int, float]:
        """
        Runs multi-modal fusion checks.
        Returns: (final_person_id, confidence_score)

        face_detected: True if a face was visible in the crop, even if quality
                       check prevented embedding extraction. Used to assign a
                       new Person ID even when face_emb is None.
        """
        # 1. Search vector DB and fuse similarity metrics
        candidate_pid, confidence = self.matcher.match_identity(
            face_emb, body_emb, det_box, time_since_update, img_w, img_h
        )
        
        # 2. If no candidate matched, allocate a new Person ID ONLY if a quality
        #    face embedding was extracted. A detected face without a good embedding
        #    (face_detected=True but face_emb=None) must NOT get a new ID because
        #    nothing would be stored in Qdrant → the person can never be re-matched
        #    on re-entry → duplicate IDs every visit.
        if candidate_pid is None:
            if face_emb is not None:
                # Full quality face embedding — assign with full confidence
                candidate_pid = next_person_id_callback()
                confidence = 1.0
            else:
                # Face may be visible but quality check blocked embedding extraction.
                # Do NOT assign a new ID — wait until a good embedding frame arrives.
                return None, 0.0
            
        # 3. Apply TemporalValidator to filter flickering identity transitions
        final_person_id = self.temporal_validator.validate_identity(track_id, candidate_pid)
        
        return final_person_id, confidence
