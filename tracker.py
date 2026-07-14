import numpy as np

class Track:
    def __init__(self, track_id, bbox, score, class_id):
        self.track_id = track_id
        self.bbox = bbox # [x1, y1, x2, y2]
        self.score = score
        self.class_id = class_id
        self.time_since_update = 0
        self.hits = 1

    def update(self, bbox, score):
        self.bbox = bbox
        self.score = score
        self.time_since_update = 0
        self.hits += 1

class Tracker:
    def __init__(self, max_age=30, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        """
        A placeholder for a real tracking algorithm like BoT-SORT or SORT.
        detections: list of dicts {'bbox': [x1,y1,x2,y2], 'score': float, 'class_id': int}
        """
        # In a real implementation (like from track_stream_reid.py), we'd use Hungarian matching / Kalman filters.
        # This stub just assigns new IDs to everything for architectural completeness, 
        # unless user drops in their byte_tracker.
        
        # Increment age
        for t in self.tracks:
            t.time_since_update += 1

        active_tracks = []
        for det in detections:
            # Naive match (for stub): if overlapping bounding box
            matched = False
            for t in self.tracks:
                if self._iou(det['bbox'], t.bbox) > 0.3:
                    t.update(det['bbox'], det['score'])
                    active_tracks.append(t)
                    matched = True
                    break
            if not matched:
                new_track = Track(self.next_id, det['bbox'], det['score'], det['class_id'])
                active_tracks.append(new_track)
                self.next_id += 1
                
        # Keep tracks that haven't aged out
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        # Merge active back
        for t in active_tracks:
            if t not in self.tracks:
                self.tracks.append(t)

        return self.tracks

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
        area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
        return inter / float(area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0
