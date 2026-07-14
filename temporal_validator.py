# temporal_validator.py
# Temporal Validator to dampen identity flickering over consecutive frames

class TemporalValidator:
    def __init__(self, required_stable_frames=3):
        self.required_stable_frames = required_stable_frames
        self.history = {}
        
    def update(self, track_id, candidate_id):
        if track_id not in self.history:
            self.history[track_id] = []
            
        self.history[track_id].append(candidate_id)
        
        # Keep only recent frames
        if len(self.history[track_id]) > self.required_stable_frames:
            self.history[track_id].pop(0)
            
    def get_stable_identity(self, track_id):
        if track_id not in self.history:
            return None
            
        recent_ids = self.history[track_id]
        if len(recent_ids) < self.required_stable_frames:
            return None
            
        # Check if all recent frames have the same identity
        if all(x == recent_ids[0] for x in recent_ids):
            return recent_ids[0]
            
        return None
