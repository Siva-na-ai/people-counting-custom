import numpy as np
from typing import Optional, Tuple, Dict, Any
import config

class FusionEngine:
    """
    Orchestrates the decision engine, combining Top-K similarity searches,
    temporal stability validation, visual quality scores, and body verification
    to resolve a stable consolidated Person ID.
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
        face_quality_passed: bool = False
    ) -> Tuple[Optional[int], float, Dict[str, Any]]:
        """
        Runs multi-modal fusion checks.
        Returns: (final_person_id, confidence_score, match_details)
        """
        # 1. Search vector DB and fuse similarity metrics
        candidate_pid, confidence, match_details = self.matcher.match_identity(
            face_emb, body_emb, det_box, time_since_update, img_w, img_h
        )

        state = self.temporal_validator.get_state(track_id)
        if state is None:
            from temporal_validator import TrackIdentityState
            state = TrackIdentityState(track_id)
            self.temporal_validator.track_states[track_id] = state

        # Initialize unmatched count if not present
        if not hasattr(state, "unmatched_face_observations"):
            state.unmatched_face_observations = 0

        # 2. Handle unmatched observations (New Person Creation Delay)
        if candidate_pid is None:
            if face_emb is not None and face_quality_passed:
                state.unmatched_face_observations += 1
                match_details["reason"] = f"Unmatched high-quality face ({state.unmatched_face_observations}/{config.MIN_NEW_PERSON_OBSERVATIONS} frames)"
                
                # Check if we have collected enough stable, unmatched observations
                if state.unmatched_face_observations >= config.MIN_NEW_PERSON_OBSERVATIONS:
                    # Allocate a brand new PID
                    candidate_pid = next_person_id_callback()
                    confidence = 1.0
                    state.unmatched_face_observations = 0
                    match_details["decision"] = "Accepted"
                    match_details["reason"] = "Created new persistent Person ID"
                else:
                    # Keep tentative
                    confidence = 0.0
            else:
                # Poor quality / no face -> reset unmatched count
                state.unmatched_face_observations = 0
                confidence = 0.0
        else:
            # We matched someone -> reset unmatched counter
            state.unmatched_face_observations = 0

        # 3. Apply TemporalValidator to filter flickering identity transitions
        final_person_id, p_state, final_conf = self.temporal_validator.validate_identity(
            track_id, candidate_pid, confidence, face_quality_passed
        )

        return final_person_id, final_conf, match_details
