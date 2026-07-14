import numpy as np
from scipy.optimize import linear_sum_assignment
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import config

# Helper function to compute IoU
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

class Track:
    def __init__(self, track_id: int, box: np.ndarray, score: float, class_id: int, state: str = 'Tentative'):
        self.track_id = track_id
        self.box = box.copy()
        self.score = score
        self.class_id = class_id
        self.time_since_update = 0
        self.history = []  # List of (cx, cy) center points
        self.state = state
        self.hits = 1 if score >= 0.70 else 0
        self.velocity = np.zeros(2)
        
        # Associated persistent global identity
        self.person_id: Optional[int] = None

class BoTSORTTracker:
    """
    Refactored BoT-SORT tracker for temporary spatial tracking and Hungarian association.
    """
    def __init__(self, w_iou: float = 0.5, w_motion: float = 0.5, max_age: int = 900):
        self.w_iou = w_iou
        self.w_motion = w_motion
        self.max_age = max_age
        self.tracks: List[Track] = []
        self.next_track_id = 1

    def update(self, detections_raw: List[Dict[str, Any]], img_w: int, img_h: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Runs prediction and Hungarian association (Step 3 matching).
        Returns (matched_pairs, unmatched_detections).
        matched_pairs: list of dict {"track": Track, "detection": dict, "det_idx": int}
        unmatched_detections: list of dict {"bbox": box, "score": score, "class_id": class, "det_idx": int}
        """
        num_dets = len(detections_raw)
        
        # 1. Age existing tracks
        for track in self.tracks:
            track.time_since_update += 1
            
        # 2. Extract coordinates
        boxes = [det["bbox"] for det in detections_raw]
        scores = [det["score"] for det in detections_raw]
        class_ids = [det["class_id"] for det in detections_raw]
        
        active_tracks = [t for t in self.tracks if t.time_since_update <= 30]
        num_tracks = len(active_tracks)
        
        matched_pairs = []
        matched_det_indices = set()
        
        if num_tracks > 0 and num_dets > 0:
            cost_matrix = np.zeros((num_tracks, num_dets))
            
            for t_idx, track in enumerate(active_tracks):
                # Project position using smoothed velocity model
                if len(track.history) >= 1 and not np.all(track.velocity == 0.0):
                    c_last = track.history[-1]
                    pred_cx = c_last[0] + track.velocity[0] * track.time_since_update
                    pred_cy = c_last[1] + track.velocity[1] * track.time_since_update
                else:
                    pred_cx = (track.box[0] + track.box[2]) / 2.0
                    pred_cy = (track.box[1] + track.box[3]) / 2.0
                    
                track_w = track.box[2] - track.box[0]
                track_h = track.box[3] - track.box[1]
                diag = np.sqrt(track_w**2 + track_h**2)
                
                for d_idx in range(num_dets):
                    det_box = boxes[d_idx]
                    
                    # IoU Cost
                    C_iou = float(1.0 - compute_iou(track.box, det_box))
                    
                    # Motion Cost
                    det_cx = (det_box[0] + det_box[2]) / 2.0
                    det_cy = (det_box[1] + det_box[3]) / 2.0
                    dist = np.sqrt((det_cx - pred_cx)**2 + (det_cy - pred_cy)**2)
                    dist_norm = dist / diag if diag > 0 else dist
                    C_motion = float(1.0 - np.exp(-2.0 * dist_norm))
                    
                    # Combined spatial cost
                    cost = self.w_iou * C_iou + self.w_motion * C_motion
                    
                    # Motion Validation Check: reject matches violating maximum speed
                    dist_px = np.sqrt(((det_cx - pred_cx) * img_w)**2 + ((det_cy - pred_cy) * img_h)**2)
                    diag_px = np.sqrt((track_w * img_w)**2 + (track_h * img_h)**2)
                    max_motion = (4.0 + 0.2 * track.time_since_update) * diag_px
                    
                    if dist_px > max_motion or cost > 0.70:
                        cost = 1e5
                        
                    cost_matrix[t_idx, d_idx] = cost
                    
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 1e4:
                    track = active_tracks[r]
                    matched_pairs.append({
                        "track": track,
                        "detection": detections_raw[c],
                        "det_idx": c
                    })
                    matched_det_indices.add(c)
                    
                    # Update track state
                    track.box = boxes[c]
                    track.score = scores[c]
                    track.class_id = class_ids[c]
                    track.time_since_update = 0
                    if track.score >= 0.70:
                        track.hits += 1
                        
                    cx = (track.box[0] + track.box[2]) / 2.0
                    cy = (track.box[1] + track.box[3]) / 2.0
                    track.history.append((cx, cy))
                    if len(track.history) > 10:
                        track.history.pop(0)
                        
                    # Smooth velocity update using EMA
                    if len(track.history) >= 2:
                        prev_cx, prev_cy = track.history[-2]
                        current_vel = np.array([cx - prev_cx, cy - prev_cy])
                        if np.all(track.velocity == 0.0):
                            track.velocity = current_vel
                        else:
                            track.velocity = 0.7 * track.velocity + 0.3 * current_vel
                            
        # Clean up expired tracks (150 frames limit)
        self.tracks = [
            t for t in self.tracks 
            if (t.state == 'Confirmed' and t.time_since_update <= 150)
            or (t.state == 'Tentative' and t.time_since_update == 0)
        ]
        
        # Package unmatched detections
        unmatched_dets = []
        for d_idx in range(num_dets):
            if d_idx not in matched_det_indices:
                unmatched_dets.append({
                    "bbox": boxes[d_idx],
                    "score": scores[d_idx],
                    "class_id": class_ids[d_idx],
                    "det_idx": d_idx
                })
                
        return matched_pairs, unmatched_dets

    def create_track(self, bbox: np.ndarray, score: float, class_id: int) -> Track:
        """
        Creates and registers a new Track object.
        """
        track = Track(self.next_track_id, bbox, score, class_id)
        self.next_track_id += 1
        self.tracks.append(track)
        return track
