import numpy as np
from typing import Tuple, Optional
import config

class MovementValidator:
    """
    Validates physical movement constraints (speed, acceleration, camera transition limits)
    normalized by object size (body diagonal) to reject physically impossible associations.
    """
    def __init__(self):
        pass

    def validate_track_motion(
        self, 
        track_box: np.ndarray, 
        det_box: np.ndarray, 
        time_since_update: int, 
        img_w: int, 
        img_h: int
    ) -> bool:
        """
        Validates frame-to-frame motion on the same camera.
        Prevents teleportation.
        """
        if track_box is None or det_box is None:
            return True
            
        t_cx = (track_box[0] + track_box[2]) / 2.0
        t_cy = (track_box[1] + track_box[3]) / 2.0
        d_cx = (det_box[0] + det_box[2]) / 2.0
        d_cy = (det_box[1] + det_box[3]) / 2.0
        
        # Calculate pixel distance
        dist_px = np.sqrt(((d_cx - t_cx) * img_w)**2 + ((d_cy - t_cy) * img_h)**2)
        
        # Bounding box diagonal as proxy for distance scale
        w_px = (track_box[2] - track_box[0]) * img_w
        h_px = (track_box[3] - track_box[1]) * img_h
        diag_px = np.sqrt(w_px**2 + h_px**2)
        
        if diag_px <= 0:
            return True
            
        # Determine maximum walking speed limit
        if time_since_update <= 3:
            max_limit = config.MAX_MOTION_DIAGS * diag_px
        else:
            max_limit = (config.MAX_MOTION_DIAGS_LOST_BASE + config.MAX_MOTION_DIAGS_LOST_STEP * time_since_update) * diag_px
            
        return dist_px <= max_limit

    def validate_acceleration(
        self, 
        history: list, 
        current_center: Tuple[float, float], 
        old_velocity: np.ndarray, 
        img_w: int, 
        img_h: int, 
        diag_px: float
    ) -> bool:
        """
        Rejects associations causing physically impossible acceleration spikes.
        """
        if len(history) < 2 or np.all(old_velocity == 0.0) or diag_px <= 0:
            return True
            
        prev_cx, prev_cy = history[-1]
        cx, cy = current_center
        
        current_vel = np.array([(cx - prev_cx) * img_w, (cy - prev_cy) * img_h])
        old_vel_px = np.array([old_velocity[0] * img_w, old_velocity[1] * img_h])
        
        # Acceleration is change in velocity
        accel = np.linalg.norm(current_vel - old_vel_px)
        
        # Max acceleration limit: a human cannot accelerate by more than 2.0 body diagonals/frame^2
        return accel <= 2.0 * diag_px

    def validate_camera_transition(
        self, 
        last_box: Optional[np.ndarray], 
        det_box: np.ndarray, 
        dt_seconds: float, 
        img_w: int, 
        img_h: int
    ) -> bool:
        """
        Validates camera transition speed (long-term gallery reidentification).
        Assumes maximum human running speed is ~8.0 m/s (~10.0 body diagonals/second).
        """
        if last_box is None or dt_seconds <= 0:
            return True
            
        t_cx = (last_box[0] + last_box[2]) / 2.0
        t_cy = (last_box[1] + last_box[3]) / 2.0
        d_cx = (det_box[0] + det_box[2]) / 2.0
        d_cy = (det_box[1] + det_box[3]) / 2.0
        
        dist_px = np.sqrt(((d_cx - t_cx) * img_w)**2 + ((d_cy - t_cy) * img_h)**2)
        
        w_px = (last_box[2] - last_box[0]) * img_w
        h_px = (last_box[3] - last_box[1]) * img_h
        diag_px = np.sqrt(w_px**2 + h_px**2)
        
        if diag_px <= 0:
            return True
            
        # Max walking/running distance over time
        max_allowed_diags = 3.0 + 8.0 * dt_seconds
        return dist_px <= max_allowed_diags * diag_px
