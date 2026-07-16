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
        state = self.temporal_validator.get_state(track_id)
        if state is None:
            from temporal_validator import TrackIdentityState
            state = TrackIdentityState(track_id)
            self.temporal_validator.track_states[track_id] = state

        # If already verified/confirmed, return immediately without searching
        if state.state in ("CONFIRMED", "TRACK_LOCKED", "REIDENTIFIED") and state.confirmed_id is not None:
            final_person_id, p_state, final_conf = self.temporal_validator.validate_identity(
                track_id, None, 0.0, face_quality_passed
            )
            match_details = {
                "top1_pid": state.confirmed_id,
                "top1_score": 1.0,
                "top2_pid": None,
                "top2_score": 0.0,
                "gap": 0.0,
                "body_sim": 1.0,
                "fusion_score": 1.0,
                "decision": "Accepted",
                "reason": "Track locked"
            }
            return final_person_id, final_conf, match_details

        # Run search only after enough consecutive high-quality faces are observed
        if state.good_face_confirmations >= config.GOOD_FACE_CONFIRMATIONS:
            # 1. Search vector DB and fuse similarity metrics
            candidate_pid, confidence, match_details = self.matcher.match_identity(
                face_emb, body_emb, det_box, time_since_update, img_w, img_h
            )
            
            # 2. If unmatched, immediately create new PID without delay
            if candidate_pid is None:
                candidate_pid = next_person_id_callback()
                confidence = 1.0
                match_details["decision"] = "Accepted"
                match_details["reason"] = f"Created new Person ID (immediate creation after {config.GOOD_FACE_CONFIRMATIONS} good faces)"
        else:
            # Not enough good faces yet -> stay TENTATIVE/TEMPORARY_TRACK
            candidate_pid = None
            confidence = 0.0
            match_details = {
                "top1_pid": None,
                "top1_score": 0.0,
                "top2_pid": None,
                "top2_score": 0.0,
                "gap": 0.0,
                "body_sim": 0.0,
                "fusion_score": 0.0,
                "decision": "Rejected",
                "reason": f"Waiting for good face ({state.good_face_confirmations}/{config.GOOD_FACE_CONFIRMATIONS} frames)"
            }

        # 3. Apply TemporalValidator to update state
        final_person_id, p_state, final_conf = self.temporal_validator.validate_identity(
            track_id, candidate_pid, confidence, face_quality_passed
        )

        return final_person_id, final_conf, match_details
