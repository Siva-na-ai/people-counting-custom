import logging
import time
import numpy as np

logger = logging.getLogger(__name__)

class MovementValidator:
    def __init__(self, max_speed_pixels_per_sec=1000):
        self.max_speed = max_speed_pixels_per_sec
        self.last_positions = {}

    def is_valid_movement(self, track_id: int, current_box: np.ndarray) -> bool:
        """
        Calculates if the movement since the last observation is physically plausible.
        current_box: [x1, y1, x2, y2]
        """
        now = time.time()
        center_x = (current_box[0] + current_box[2]) / 2.0
        center_y = (current_box[1] + current_box[3]) / 2.0
        current_pos = np.array([center_x, center_y])
        
        if track_id not in self.last_positions:
            self.last_positions[track_id] = (current_pos, now)
            return True
            
        last_pos, last_time = self.last_positions[track_id]
        dt = now - last_time
        
        if dt > 0:
            distance = np.linalg.norm(current_pos - last_pos)
            speed = distance / dt
            self.last_positions[track_id] = (current_pos, now)
            if speed > self.max_speed:
                logger.debug(f"Track {track_id} exceeded max speed: {speed:.2f} px/s")
                return False
                
        return True
