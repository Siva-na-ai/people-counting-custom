import cv2
import torch
import numpy as np
import json
import time
import threading
import urllib.request
from scipy.optimize import linear_sum_assignment

# ----------------- Import Torchreid -----------------
try:
    import torchreid
    try:
        from torchreid.utils import FeatureExtractor
    except ImportError:
        from torchreid.reid.utils import FeatureExtractor
except ImportError as e:
    raise ImportError(
        f"Please install torch and torchreid. (Original error: {e}). "
        "Run: pip install torch torchvision torchreid"
    )

# ----------------- Import YOLOv8 -----------------
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Please install ultralytics package to run YOLOv8 detection. Run: pip install ultralytics")


# ----------------- Helper function to compute IoU -----------------
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


# ----------------- Tracking Classes -----------------
class Track:
    def __init__(self, track_id, box, embedding, score, class_id):
        self.track_id = track_id
        self.box = box
        self.score = score
        self.class_id = class_id
        self.embeddings = [embedding] if embedding is not None else []
        self.time_since_update = 0


class BoTSORTTracker:
    def __init__(self, reid_model_name='osnet_x1_0', reid_threshold=0.58, device='cpu', max_age=900):
        self.max_age = max_age
        self.reid_threshold = reid_threshold
        self.extractor = FeatureExtractor(
            model_name=reid_model_name,
            device=device
        )
        self.tracks = []
        self.next_track_id = 1
        
    def get_crop(self, image, box):
        h, w, _ = image.shape
        xmin = int(max(0, box[0]))
        ymin = int(max(0, box[1]))
        xmax = int(min(w, box[2]))
        ymax = int(min(h, box[3]))
            
        if xmax <= xmin or ymax <= ymin:
            return None
            
        crop = image[ymin:ymax, xmin:xmax]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return crop_rgb

    def extract_embedding(self, crop):
        with torch.no_grad():
            features = self.extractor([crop])
            embedding = features[0].cpu().numpy()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding

    def update(self, frame_image, detections):
        num_dets = len(detections)
        descr = [('box', '<f4', (4,)), ('confidence', '<f4'), ('class_id', '<i4'), ('track_id', '<i4')]
        new_dtype = np.dtype(descr)
        
        if num_dets == 0:
            for track in self.tracks:
                track.time_since_update += 1
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return np.zeros(0, dtype=new_dtype)
            
        boxes = []
        scores = []
        class_ids = []
        
        for det in detections:
            det_tuple = tuple(det)
            boxes.append(det_tuple[0])
            scores.append(det_tuple[1])
            class_ids.append(det_tuple[2])
                
        active_tracks = self.tracks
        
        matched_track_indices = []
        matched_det_indices = []
        
        # Stage 1: IoU / Motion Matching
        iou_tracks = [t for t in active_tracks if t.time_since_update == 0]
        num_iou_tracks = len(iou_tracks)
        
        if num_iou_tracks > 0 and num_dets > 0:
            iou_matrix = np.zeros((num_iou_tracks, num_dets))
            for t_idx, track in enumerate(iou_tracks):
                for d_idx in range(num_dets):
                    det_box = boxes[d_idx]
                    if isinstance(det_box, np.ndarray) and det_box.ndim > 1:
                        det_box = det_box.flatten()
                    iou_matrix[t_idx, d_idx] = compute_iou(track.box, det_box)
            
            cost_matrix = 1.0 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= 0.50:
                    matched_track_indices.append(r)
                    matched_det_indices.append(c)
                    
        matched_tracks = [iou_tracks[r] for r in matched_track_indices]
        matched_det_indices = list(matched_det_indices)
        
        # Stage 2: ReID / Appearance Matching for remaining unmatched tracks/detections
        unmatched_tracks = [t for t in active_tracks if t not in matched_tracks]
        unmatched_det_indices = [d for d in range(num_dets) if d not in matched_det_indices]
        
        if len(unmatched_tracks) > 0 and len(unmatched_det_indices) > 0:
            det_embeddings = []
            valid_det_indices = []
            
            for d_idx in unmatched_det_indices:
                det_box = boxes[d_idx]
                if isinstance(det_box, np.ndarray) and det_box.ndim > 1:
                    det_box = det_box.flatten()
                crop = self.get_crop(frame_image, det_box)
                if crop is not None:
                    emb = self.extract_embedding(crop)
                    det_embeddings.append(emb)
                    valid_det_indices.append(d_idx)
                    
            if len(det_embeddings) > 0 and any(len(t.embeddings) > 0 for t in unmatched_tracks):
                reid_cost_matrix = np.ones((len(unmatched_tracks), len(det_embeddings)))
                
                for i, track in enumerate(unmatched_tracks):
                    if len(track.embeddings) == 0:
                        continue
                    for j, det_emb in enumerate(det_embeddings):
                        sims = [np.dot(det_emb, stored_emb) for stored_emb in track.embeddings]
                        max_sim = max(sims) if sims else 0.0
                        reid_cost_matrix[i, j] = 1.0 - max_sim
                
                r_ind, c_ind = linear_sum_assignment(reid_cost_matrix)
                
                for r, c in zip(r_ind, c_ind):
                    max_sim = 1.0 - reid_cost_matrix[r, c]
                    if max_sim >= self.reid_threshold:
                        matched_tracks.append(unmatched_tracks[r])
                        matched_det_indices.append(valid_det_indices[c])
        
        # Update matched tracks
        final_matched_det_indices = []
        for i, track in enumerate(matched_tracks):
            det_idx = matched_det_indices[i]
            det_box = boxes[det_idx]
            if isinstance(det_box, np.ndarray) and det_box.ndim > 1:
                det_box = det_box.flatten()
            track.box = det_box
            track.score = scores[det_idx]
            track.time_since_update = 0
            
            crop = self.get_crop(frame_image, det_box)
            if crop is not None:
                emb = self.extract_embedding(crop)
                track.embeddings.append(emb)
                if len(track.embeddings) > 10:
                    track.embeddings.pop(0)
            final_matched_det_indices.append(det_idx)
            
        # Create new tracks for unmatched detections
        for d_idx in range(num_dets):
            if d_idx not in final_matched_det_indices:
                det_box = boxes[d_idx]
                if isinstance(det_box, np.ndarray) and det_box.ndim > 1:
                    det_box = det_box.flatten()
                crop = self.get_crop(frame_image, det_box)
                emb = self.extract_embedding(crop) if crop is not None else None
                
                new_track = Track(
                    track_id=self.next_track_id,
                    box=det_box,
                    embedding=emb,
                    score=scores[d_idx],
                    class_id=class_ids[d_idx]
                )
                self.tracks.append(new_track)
                self.next_track_id += 1
                
        # Age out old tracks
        for track in self.tracks:
            if track not in matched_tracks and track.time_since_update > 0:
                track.time_since_update += 1
                
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Sort and build outputs
        sorted_pairs = []
        for det_idx in range(num_dets):
            det_box = boxes[det_idx]
            if isinstance(det_box, np.ndarray) and det_box.ndim > 1:
                det_box = det_box.flatten()
            matching_track = None
            for track in self.tracks:
                if track.time_since_update == 0 and np.allclose(track.box, det_box):
                    matching_track = track
                    break
            if matching_track:
                sorted_pairs.append((det_idx, matching_track))
                
        if len(sorted_pairs) == 0:
            return np.zeros(0, dtype=new_dtype)
            
        tracked_arr = np.zeros(len(sorted_pairs), dtype=new_dtype)
        for i, (d_idx, track) in enumerate(sorted_pairs):
            tracked_arr['box'][i] = track.box
            tracked_arr['confidence'][i] = track.score
            tracked_arr['class_id'][i] = track.class_id
            tracked_arr['track_id'][i] = track.track_id
        return tracked_arr


# ----------------- Logical Zones Class -----------------
class Area:
    def __init__(self, points):
        # points: list of normalized [x, y] coordinates
        self.points = np.array(points, dtype=np.float32)

    def contains(self, x, y, width, height):
        # Check if point (x, y) is inside the polygon defined by self.points scaled to (width, height)
        poly_pixels = np.array([[p[0] * width, p[1] * height] for p in self.points], dtype=np.int32)
        dist = cv2.pointPolygonTest(poly_pixels, (float(x), float(y)), False)
        return dist >= 0


# ----------------- Send Data to Dashboard Server -----------------
def send_data_to_dashboard(inside, outside, unique, visitors=None):
    if visitors is None:
        visitors = []
    
    payload = {
        "inside": inside,
        "outside": outside,
        "unique": unique,
        "visitors": visitors
    }
    
    def run():
        try:
            req = urllib.request.Request(
                'http://localhost:8000/api/update',
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=0.5) as r:
                r.read()
        except Exception:
            pass  # Fail silently if local dashboard server is not running

    t = threading.Thread(target=run)
    t.daemon = True
    t.start()


LAST_FRAME_TIME = 0.0
FRAME_INTERVAL = 0.1  # Max 10 fps upload to reduce network load

def send_frame_to_dashboard(frame_image):
    global LAST_FRAME_TIME
    current_time = time.time()
    if current_time - LAST_FRAME_TIME < FRAME_INTERVAL:
        return
    LAST_FRAME_TIME = current_time
    
    try:
        _, jpeg_bytes = cv2.imencode('.jpg', frame_image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        data = jpeg_bytes.tobytes()
        
        def run():
            try:
                req = urllib.request.Request(
                    'http://localhost:8000/api/upload_frame',
                    data=data,
                    headers={'Content-Type': 'image/jpeg'},
                    method='POST'
                )
                with urllib.request.urlopen(req, timeout=0.5) as r:
                    r.read()
            except Exception:
                pass
                
        t = threading.Thread(target=run)
        t.daemon = True
        t.start()
    except Exception:
        pass


# ----------------- Threaded Fresh Frame Reader -----------------
class FreshFrameReader:
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.cap = cv2.VideoCapture(stream_url)
        self.latest_frame = None
        self.ret = False
        self.running = True
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self._update)
        self.thread.daemon = True
        self.thread.start()
        
    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Warning: Failed to retrieve frame in reader thread. Reconnecting...")
                time.sleep(2)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.stream_url)
                continue
            with self.lock:
                self.latest_frame = frame
                self.ret = ret
            # Yield CPU slightly
            time.sleep(0.01)
                
    def read(self):
        with self.lock:
            if not self.ret or self.latest_frame is None:
                return False, None
            # Return a copy to avoid thread modification issues during draw
            return self.ret, self.latest_frame.copy()
            
    def isOpened(self):
        return self.cap.isOpened()
        
    def release(self):
        self.running = False
        self.cap.release()


# ----------------- Main Stream Processor -----------------
def main():
    print("Initializing YOLOv8 object detector...")
    detector = YOLO("yolov8n.pt")
    
    print("Initializing BoTSORT Tracker & ReID gallery...")
    tracker = BoTSORTTracker(reid_model_name='osnet_x1_0', reid_threshold=0.58, device='cpu')

    stream_url = 'https://stream.sivabio.in/stream'
    print(f"Connecting to live video stream: {stream_url}")
    cap = FreshFrameReader(stream_url)
    
    if not cap.isOpened():
        print("Error: Could not open the video stream.")
        return
    
    # Load Zones
    areas = []
    try:
        with open('areas.json', 'r') as f:
            json_areas = json.load(f)
            for item in json_areas:
                areas.append(Area(item["points"]))
            print(f"Loaded {len(areas)} areas from areas.json.")
    except Exception:
        # Fallback default areas
        print("Fallback to default areas configuration.")
        areas = [
            Area([[0.10, 0.25], [0.48, 0.25], [0.48, 0.75], [0.10, 0.75]]),  # Area 1
            Area([[0.55, 0.25], [0.90, 0.25], [0.90, 0.75], [0.55, 0.75]])   # Area 2
        ]

    unique_seen_people = set()
    print("Streaming processor started! Uploading results to http://localhost:8000...")
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue
            
        h, w, _ = frame.shape
        
        # Run YOLOv8 detection
        results = detector(frame, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0].item())
                if cls == 0:  # Person class
                    conf = float(box.conf[0].item())
                    if conf > 0.40:
                        xyxy = box.xyxy[0].cpu().numpy()
                        detections.append((xyxy, conf, cls))
                        
        # Update Tracker
        tracked_detections = tracker.update(frame, detections)
        
        # Reset counters for the current frame
        sum_inside = 0
        current_visitors = []
        
        # Render logical zones on frame
        overlay = frame.copy()
        for ID, area in enumerate(areas):
            pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in area.points], dtype=np.int32)
            color = (129, 185, 16) if ID == 0 else (246, 130, 59) # Emerald vs Blue
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(frame, [pts], True, color, 2)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Process tracked persons
        for det in tracked_detections:
            box = det['box']
            s = float(det['confidence'])
            c = int(det['class_id'])
            t = int(det['track_id'])
            
            unique_seen_people.add(t)
            
            xmin, ymin, xmax, ymax = map(int, box)
            px = (xmin + xmax) / 2.0
            py = float(ymax)  # Feet coordinates
            
            # Check if this person is in any zone
            person_area_name = "Main Coverage"
            box_color = (0, 255, 0)
            in_zone = False
            
            for ID, area in enumerate(areas):
                if area.contains(px, py, w, h):
                    sum_inside += 1
                    in_zone = True
                    person_area_name = f"Area {ID + 1}"
                    box_color = (16, 185, 129) if ID == 0 else (246, 130, 59)
                    break
                    
            current_visitors.append({
                "id": t,
                "confidence": s,
                "area": person_area_name,
                "color": "#10b981" if person_area_name == "Area 1" else ("#3b82f6" if person_area_name == "Area 2" else "#a8a29e"),
                "lastSeen": "Just Now"
            })
            
            # Draw box & ID label
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)
            label = f"#{t} ({s:.2f})"
            cv2.putText(frame, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        outside_count = len(tracked_detections) - sum_inside
        
        # Draw stats overlays
        cv2.putText(frame, f"Inside Room: {len(tracked_detections)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Inside Zones: {sum_inside}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Total Unique: {len(unique_seen_people)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Send details to local server
        send_data_to_dashboard(
            inside=len(tracked_detections),
            outside=0,
            unique=len(unique_seen_people),
            visitors=current_visitors
        )
        
        # Send frame to local server
        send_frame_to_dashboard(frame)
        
        # Delay to cap frame rate
        time.sleep(0.05)


if __name__ == "__main__":
    main()
