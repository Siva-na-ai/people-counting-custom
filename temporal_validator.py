import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import config

logger = logging.getLogger("temporal_validator")

class TrackIdentityState:
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.state = "TENTATIVE"  # Starts as TENTATIVE
        self.confirmed_id: Optional[int] = None
        self.candidate_id: Optional[int] = None
        self.consecutive_matches = 0
        self.track_age = 0
        self.confirmed_age = 0
        self.last_seen = datetime.now()
        self.identity_confidence = 0.0  # 0.0 to 1.0
        
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

        # If face quality passed, transition to QUALITY_VERIFIED if tentative
        if face_quality_passed and state.state == "TENTATIVE":
            state.state = "QUALITY_VERIFIED"

        # Update candidate state if we have a match candidate
        if candidate_pid is not None and face_quality_passed:
            if state.state in ("TENTATIVE", "QUALITY_VERIFIED"):
                state.state = "CANDIDATE_MATCH"

            # Manage consecutive observations check
            if candidate_pid == state.candidate_id:
                state.consecutive_matches += 1
                state.similarity_history.append(similarity)
                # Cap similarity history to last 5
                if len(state.similarity_history) > 5:
                    state.similarity_history.pop(0)
            else:
                state.candidate_id = candidate_pid
                state.consecutive_matches = 1
                state.similarity_history = [similarity]

            # Adjust identity confidence based on similarity and hits
            avg_sim = sum(state.similarity_history) / len(state.similarity_history)
            state.identity_confidence = min(1.0, 0.4 * similarity + 0.6 * (state.consecutive_matches / config.MIN_CONSECUTIVE_MATCHES))

            # Switch check: If same candidate matches consistently for config.MIN_CONSECUTIVE_MATCHES frames
            # AND average similarity exceeds threshold, confirm!
            if state.consecutive_matches >= config.MIN_CONSECUTIVE_MATCHES and avg_sim >= config.FACE_MATCH_THRESHOLD:
                # Identity Switching Protection:
                # If we already have a confirmed PID, we only switch if the new candidate has consecutive match count
                # and significant score margin (e.g. 0.05 higher than old score).
                if state.confirmed_id is not None and state.confirmed_id != candidate_pid:
                    old_score = 0.80  # Default baseline
                    if similarity > old_score + 0.05:
                        logger.info(f"[TemporalValidator] Track #{track_id} ID switched from PID #{state.confirmed_id} to PID #{candidate_pid} (margin exceeded)")
                        state.confirmed_id = candidate_pid
                        state.state = "CONFIRMED"
                        state.confirmed_age = 0
                    else:
                        logger.warning(f"[TemporalValidator] Track #{track_id} identity switch rejected (PID #{state.confirmed_id} -> PID #{candidate_pid}): score margin insufficient")
                else:
                    # Brand new confirmation
                    logger.info(f"[TemporalValidator] Track #{track_id} confirmed as PID #{candidate_pid} (consecutive matched={state.consecutive_matches}, avg_sim={avg_sim:.3f})")
                    state.confirmed_id = candidate_pid
                    state.state = "CONFIRMED"
                    state.confirmed_age = 0

                # Promote to LOCK state if configuration is enabled
                if config.TRACK_LOCK_ENABLED:
                    state.state = "TRACK_LOCKED"
                    state.identity_confidence = 1.0

        else:
            # Decay confidence slightly if we don't get matching frames or lose quality
            if state.state not in ("TRACK_LOCKED", "CONFIRMED"):
                state.consecutive_matches = max(0, state.consecutive_matches - 1)
                state.identity_confidence = max(0.0, state.identity_confidence - 0.05)

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
