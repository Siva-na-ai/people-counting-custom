import time
import math

class MovementValidator:
    def __init__(self, max_speed_pixels_per_sec=500.0):
        self.max_speed_pixels_per_sec = max_speed_pixels_per_sec
        self.history = {}

    def is_valid_movement(self, track_id, current_center):
        """
        Check if the movement from the last known position is physically plausible.
        current_center: (x, y)
        """
        now = time.time()
        
        if track_id not in self.history:
            self.history[track_id] = (current_center, now)
            return True
            
        last_center, last_time = self.history[track_id]
        dt = now - last_time
        
        if dt <= 0.01:
            return True # Too fast to judge reliably
            
        dx = current_center[0] - last_center[0]
        dy = current_center[1] - last_center[1]
        distance = math.hypot(dx, dy)
        
        speed = distance / dt
        
        # Update history
        self.history[track_id] = (current_center, now)
        
        # If speed exceeds reasonable human movement limits in pixel space, reject
        return speed <= self.max_speed_pixels_per_sec
        
    def clear_history(self, track_id):
        if track_id in self.history:
            del self.history[track_id]
