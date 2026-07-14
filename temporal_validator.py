import logging
import config

logger = logging.getLogger("temporal_validator")

class TrackIdentityState:
    def __init__(self, confirmed_id: int):
        self.confirmed_id = confirmed_id
        self.candidate_id = None
        self.confirmation_count = 0
        self.stability_score = 1.0

class TemporalValidator:
    """
    Prevents identity flickering by requiring confirmation over multiple consecutive frames
    before confirming identity changes.
    """
    def __init__(self, required_confirm_frames: int = config.STABILITY_CONFIRM_FRAMES):
        self.required_confirm_frames = required_confirm_frames
        self.track_states = {}  # track_id -> TrackIdentityState

    def validate_identity(self, track_id: int, matched_person_id: int) -> int:
        """
        Validates the proposed matched_person_id for the given track_id.
        Returns the validated (stable) person_id.
        """
        if track_id not in self.track_states:
            # First observation: establish confirmed identity
            self.track_states[track_id] = TrackIdentityState(matched_person_id)
            return matched_person_id
            
        state = self.track_states[track_id]
        
        if matched_person_id == state.confirmed_id:
            # Stable hit: decay candidate counts and increase stability score
            state.stability_score = 0.9 * state.stability_score + 0.1 * 1.0
            state.candidate_id = None
            state.confirmation_count = 0
            return state.confirmed_id
            
        # Proposed identity change detected:
        state.stability_score = 0.9 * state.stability_score + 0.1 * 0.0
        
        if matched_person_id == state.candidate_id:
            state.confirmation_count += 1
            if state.confirmation_count >= self.required_confirm_frames:
                # Identity transition confirmed
                logger.info(
                    f"[TemporalValidator] Track #{track_id} identity changed "
                    f"from Person #{state.confirmed_id} to Person #{matched_person_id}"
                )
                state.confirmed_id = matched_person_id
                state.candidate_id = None
                state.confirmation_count = 0
                return matched_person_id
            else:
                # Gated: return previously confirmed identity
                return state.confirmed_id
        else:
            # Set new candidate, reset confirmation counter
            state.candidate_id = matched_person_id
            state.confirmation_count = 1
            return state.confirmed_id

    def get_stability_score(self, track_id: int) -> float:
        if track_id in self.track_states:
            return self.track_states[track_id].stability_score
        return 0.0
        
    def cleanup_track(self, track_id: int):
        if track_id in self.track_states:
            del self.track_states[track_id]
