import logging

logger = logging.getLogger(__name__)

class TemporalValidator:
    def __init__(self, required_frames=3):
        self.required_frames = required_frames
        self.track_history = {}

    def is_valid_candidate(self, track_id: int, identity_id: str) -> bool:
        """
        Dampens identity flickering by requiring multiple consecutive frames 
        matching the same identity before transitioning state.
        """
        if track_id not in self.track_history:
            self.track_history[track_id] = []
            
        history = self.track_history[track_id]
        history.append(identity_id)
        
        # Keep only the latest frames
        if len(history) > self.required_frames:
            history.pop(0)
            
        # Check if all recent frames matched the same identity
        return len(history) == self.required_frames and all(x == identity_id for x in history)

    def clear_track(self, track_id: int):
        if track_id in self.track_history:
            del self.track_history[track_id]
