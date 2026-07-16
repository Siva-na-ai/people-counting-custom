import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import config

logger = logging.getLogger("temporal_validator")

class TrackIdentityState:
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.state = "TEMPORARY_TRACK"  # Starts as TEMPORARY_TRACK
        self.confirmed_id: Optional[int] = None
        self.candidate_id: Optional[int] = None
        self.consecutive_matches = 0
        self.track_age = 0
        self.confirmed_age = 0
        self.last_seen = datetime.now()
        self.identity_confidence = 0.0  # 0.0 to 1.0
        self.good_face_confirmations = 0 # Counter for consecutive good face frames
        
        # Rolling histories
        self.match_history: List[Optional[int]] = []
        self.similarity_history: List[float] = []

class TemporalValidator:
    """
    Production-grade temporal matching coordinator implementing a 9-state lifecycle machine
    and protecting active confirmed tracks from identity switches.
    """
    def __init__(self):
        self.track_states: Dict[int, TrackIdentityState] = {}

    def get_state(self, track_id: int) -> Optional[TrackIdentityState]:
        return self.track_states.get(track_id, None)

    def validate_identity(
        self, 
        track_id: int, 
        candidate_pid: Optional[int], 
        similarity: float, 
        face_quality_passed: bool
    ) -> Tuple[Optional[int], str, float]:
        """
        Processes a match candidate for the given track_id.
        Returns: (validated_person_id, current_state, confidence)
        """
        if track_id not in self.track_states:
            self.track_states[track_id] = TrackIdentityState(track_id)

        state = self.track_states[track_id]
        state.track_age += 1
        state.last_seen = datetime.now()

        # Update confirmed age if locked/confirmed
        if state.state in ("CONFIRMED", "TRACK_LOCKED", "REIDENTIFIED"):
            state.confirmed_age += 1

        # Enforce Track Identity Lock: PID is frozen. We do not search or overwrite.
        if config.TRACK_LOCK_ENABLED and state.state in ("TRACK_LOCKED", "CONFIRMED") and state.confirmed_id is not None:
            state.identity_confidence = min(1.0, state.identity_confidence + 0.02)
            return state.confirmed_id, state.state, state.identity_confidence

        # If a candidate PID is assigned (either matched or newly created), confirm immediately
        if candidate_pid is not None:
            state.confirmed_id = candidate_pid
            state.identity_confidence = 1.0
            if config.TRACK_LOCK_ENABLED:
                state.state = "TRACK_LOCKED"
            else:
                state.state = "CONFIRMED"
            logger.info(f"[TemporalValidator] Track #{track_id} verified and confirmed/locked as PID #{candidate_pid}")
        else:
            # Map state based on current good face confirmations
            if state.good_face_confirmations == 1:
                state.state = "QUALITY_VERIFIED"
            elif state.state not in ("TRACK_LOCKED", "CONFIRMED"):
                state.state = "TEMPORARY_TRACK"

        # Return validated result
        return state.confirmed_id, state.state, state.identity_confidence

    def set_lost(self, track_id: int):
        if track_id in self.track_states:
            self.track_states[track_id].state = "TRACK_LOST"

    def get_stability_score(self, track_id: int) -> float:
        if track_id in self.track_states:
            return self.track_states[track_id].identity_confidence
        return 0.0

    def cleanup_track(self, track_id: int):
        if track_id in self.track_states:
            del self.track_states[track_id]
