#
# Copyright 2026 Sony Semiconductor Solutions Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cv2
import torch
import numpy as np
import subprocess
import argparse
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320
from scipy.optimize import linear_sum_assignment

try:
    import torchreid
    try:
        from torchreid.utils import FeatureExtractor
    except ImportError:
        # Fallback for nested package structure in some PyPI versions of torchreid
        from torchreid.reid.utils import FeatureExtractor
except ImportError as e:
    import traceback
    traceback.print_exc()
    raise ImportError(
        f"Please install torch and torchreid to run this script. (Original error: {e}). "
        "Run: pip install torch torchvision torchreid"
    )

# Helper function to compute IoU
def compute_iou(box1, box2):
    # box format: [xmin, ymin, xmax, ymax]
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

class StreamingHandler(BaseHTTPRequestHandler):
    streamer = None

    def log_message(self, format, *args):
        # Suppress logging of incoming HTTP requests to avoid console spamming
        pass

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = """
            <html>
              <head>
                <title>Raspberry Pi AI Camera Stream</title>
                <style>
                  body {
                    margin: 0;
                    background-color: #121212;
                    color: #ffffff;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    min-height: 100vh;
                  }
                  h1 {
                    margin-bottom: 20px;
                    font-weight: 300;
                  }
                  .stream-container {
                    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                    border-radius: 8px;
                    overflow: hidden;
                    border: 1px solid #333;
                    max-width: 100%;
                  }
                  img {
                    display: block;
                    width: 100%;
                    height: auto;
                    max-height: 80vh;
                  }
                </style>
              </head>
              <body>
                <h1>Live Camera Stream</h1>
                <div class="stream-container">
                  <img src="/stream" />
                </div>
              </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with self.streamer.condition:
                        self.streamer.condition.wait()
                        frame_bytes = self.streamer.frame_bytes
                    
                    if frame_bytes is None:
                        break
                    
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame_bytes)))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n')
            except Exception:
                # Connection reset or browser closed tab
                pass
        else:
            self.send_error(404)

class HTTPStreamer:
    def __init__(self, port=8000):
        self.port = port
        self.frame_bytes = None
        self.condition = threading.Condition()
        self.server = None
        self.server_thread = None
        self.start()

    def start(self):
        class CustomHandler(StreamingHandler):
            streamer = self

        try:
            self.server = HTTPServer(('0.0.0.0', self.port), CustomHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            print(f"[*] Started HTTP MJPEG server at http://localhost:{self.port}")
        except Exception as e:
            print(f"[-] Error starting HTTP server on port {self.port}: {e}")

    def push_frame(self, frame):
        try:
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            with self.condition:
                self.frame_bytes = frame_bytes
                self.condition.notify_all()
        except Exception as e:
            print(f"[-] Error encoding frame to JPEG: {e}")

    def close(self):
        if self.server:
            print("[*] Stopping HTTP server...")
            self.server.shutdown()
            self.server.server_close()
            with self.condition:
                self.frame_bytes = None
                self.condition.notify_all()
            self.server_thread.join(timeout=2)
            self.server = None
            print("[*] HTTP server stopped.")


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
        
        # Initialize ReID FeatureExtractor
        self.extractor = FeatureExtractor(
            model_name=reid_model_name,
            device=device
        )
        self.tracks = []
        self.next_track_id = 1
        
    def get_crop(self, image, box):
        """
        Extract and preprocess the crop from the image.
        """
        h, w, _ = image.shape
        if any(coord > 2.0 for coord in box):
            # Absolute coordinates
            xmin = int(max(0, box[0]))
            ymin = int(max(0, box[1]))
            xmax = int(min(w, box[2]))
            ymax = int(min(h, box[3]))
        else:
            # Normalized coordinates
            xmin = int(max(0, box[0] * w))
            ymin = int(max(0, box[1] * h))
            xmax = int(min(w, box[2] * w))
            ymax = int(min(h, box[3] * h))
            
        if xmax <= xmin or ymax <= ymin:
            return None
            
        crop = image[ymin:ymax, xmin:xmax]
        # Convert BGR (OpenCV default) to RGB (Torchreid expects RGB)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return crop_rgb

    def extract_embedding(self, crop):
        """
        Extract 512-dim embedding from a crop.
        """
        with torch.no_grad():
            features = self.extractor([crop])
            embedding = features[0].cpu().numpy()
            # L2 normalization for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding

    def update(self, frame_image, detections):
        try:
            return self._update_impl(frame_image, detections)
        except Exception as e:
            print("DEBUG: Exception in BoTSORTTracker.update:", type(e), str(e))
            print("DEBUG: detections type:", type(detections))
            try:
                print("DEBUG: detections dir:", dir(detections))
                if len(detections) > 0:
                    item = detections[0]
                    print("DEBUG: item type:", type(item))
                    print("DEBUG: item dir:", dir(item))
                    if hasattr(detections, 'coords'):
                        print("DEBUG: coords:", detections.coords)
                    else:
                        print("DEBUG: item tuple:", tuple(item))
            except Exception as e2:
                print("DEBUG: Failed to inspect detections:", str(e2))
            raise e

    def _update_impl(self, frame_image, detections):
        """
        detections: Detections object or NumPy structured array.
        Returns:
            Detections object or NumPy structured array with updated track IDs.
        """
        num_dets = len(detections)
        
        # Desired structured output dtype (only used if fallback to numpy array is active)
        descr = [('box', '<f4', (4,)), ('confidence', '<f4'), ('class_id', '<i4'), ('track_id', '<i4')]
        new_dtype = np.dtype(descr)
        
        if num_dets == 0:
            for track in self.tracks:
                track.time_since_update += 1
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            if isinstance(detections, np.ndarray):
                return np.zeros(0, dtype=new_dtype)
            return detections
            
        boxes = []
        scores = []
        class_ids = []
        
        # Extract fields
        is_numpy = isinstance(detections, np.ndarray)
        
        if not is_numpy and hasattr(detections, 'coords') and hasattr(detections, 'confidence'):
            boxes = detections.coords
            scores = detections.confidence
            class_ids = detections.class_id
        elif is_numpy and detections.dtype.names is not None:
            names = detections.dtype.names
            box_field = 'box' if 'box' in names else (names[0] if len(names) > 0 else None)
            score_field = 'confidence' if 'confidence' in names else ('score' if 'score' in names else (names[1] if len(names) > 1 else None))
            class_field = 'class_id' if 'class_id' in names else (names[2] if len(names) > 2 else None)
            
            for d_idx in range(num_dets):
                boxes.append(detections[d_idx][box_field] if box_field else detections[d_idx][0])
                scores.append(detections[d_idx][score_field] if score_field else detections[d_idx][1])
                class_ids.append(detections[d_idx][class_field] if class_field else detections[d_idx][2])
        else:
            for det in detections:
                det_tuple = tuple(det)
                boxes.append(det_tuple[0])
                scores.append(det_tuple[1])
                class_ids.append(det_tuple[2])
                
        active_tracks = self.tracks
        num_tracks = len(active_tracks)
        
        matched_track_indices = []
        matched_det_indices = []
        
        # Stage 1: IoU / Motion Matching (only for tracks that were active in the previous frame)
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
                    
        # Map stage 1 matches back to the original active_tracks
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
                        track = unmatched_tracks[r]
                        d_idx = valid_det_indices[c]
                        
                        matched_tracks.append(track)
                        matched_det_indices.append(d_idx)
                        track.embeddings.append(det_embeddings[c])
                        if len(track.embeddings) > 10:
                            track.embeddings.pop(0)
                            
        # Stage 3: Update matched track states
        final_matched_tracks = []
        final_matched_det_indices = []
        matched_pairs = sorted(zip(matched_tracks, matched_det_indices), key=lambda x: x[1])
        
        for track, d_idx in matched_pairs:
            det_box = boxes[d_idx]
            if isinstance(det_box, np.ndarray) and det_box.ndim > 1:
                det_box = det_box.flatten()
            det_score = scores[d_idx]
            if isinstance(det_score, np.ndarray):
                det_score = float(det_score.item())
            det_class = class_ids[d_idx]
            if isinstance(det_class, np.ndarray):
                det_class = int(det_class.item())
            
            track.box = det_box
            track.score = det_score
            track.class_id = det_class
            track.time_since_update = 0
            
            final_matched_tracks.append(track)
            final_matched_det_indices.append(d_idx)
            
        # Stage 4: Create new tracks for unmatched detections
        all_unmatched_det_indices = [d for d in range(num_dets) if d not in final_matched_det_indices]
        for d_idx in all_unmatched_det_indices:
            det_box = boxes[d_idx]
            if isinstance(det_box, np.ndarray) and det_box.ndim > 1:
                det_box = det_box.flatten()
            det_score = scores[d_idx]
            if isinstance(det_score, np.ndarray):
                det_score = float(det_score.item())
            det_class = class_ids[d_idx]
            if isinstance(det_class, np.ndarray):
                det_class = int(det_class.item())
            
            crop = self.get_crop(frame_image, det_box)
            emb = self.extract_embedding(crop) if crop is not None else None
            
            new_track = Track(self.next_track_id, det_box, emb, det_score, det_class)
            self.next_track_id += 1
            
            self.tracks.append(new_track)
            final_matched_tracks.append(new_track)
            final_matched_det_indices.append(d_idx)
            
        # Stage 5: Age unmatched tracks
        for track in self.tracks:
            if track not in final_matched_tracks:
                track.time_since_update += 1
            
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Sort matched tracks parallel to input detection order
        sorted_pairs = sorted(zip(final_matched_det_indices, final_matched_tracks), key=lambda x: x[0])
        tracker_ids_list = [track.track_id for _, track in sorted_pairs]
        tracker_ids = np.array(tracker_ids_list, dtype=np.int32)
        
        # Stage 6: Update tracker IDs and return
        if not is_numpy:
            # Set the tracker IDs on the detections object in-place
            if hasattr(detections, '_tracker_id'):
                detections._tracker_id = tracker_ids
            if hasattr(detections, 'tracker_id'):
                try:
                    detections.tracker_id = tracker_ids
                except AttributeError:
                    pass
            return detections
        else:
            # Build and return NumPy structured array
            if len(sorted_pairs) == 0:
                return np.zeros(0, dtype=new_dtype)
            tracked_arr = np.zeros(len(sorted_pairs), dtype=new_dtype)
            for i, (d_idx, track) in enumerate(sorted_pairs):
                tracked_arr['box'][i] = track.box
                tracked_arr['confidence'][i] = track.score
                tracked_arr['class_id'][i] = track.class_id
                tracked_arr['track_id'][i] = track.track_id
            return tracked_arr


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the live video over HTTP on port 8000 (accessible via cloudflared tunnel)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to stream the live video on (default: 8000)"
    )
    return parser.parse_args()


def main():
    args = get_args()
    
    #-----Camera and AI setup-----
    device = AiCamera()
    model = SSDMobileNetV2FPNLite320x320()
    device.deploy(model)

    # Initialize BoTSORT Tracker (combining tracking and ReID)
    tracker = BoTSORTTracker(reid_model_name='osnet_x1_0', reid_threshold=0.58, device='cpu', max_age=900)
    
    unique_seen_people = set()
    annotator = Annotator(thickness=1, text_thickness=1, text_scale=0.4)

    streamer = None
    try:
        with device as stream:
            for frame in stream:
                #-----Detection Filtering-----
                detections = frame.detections[frame.detections.confidence > 0.55]
                detections = detections[detections.class_id == 0]  # Person
                
                #-----Tracker Update-----
                detections = tracker.update(frame.image, detections)

                #-----ReID / Unique Visitor Count-----
                for idx, (_, s, c, t) in enumerate(detections):
                    unique_seen_people.add(t)

                #-----Display Annotations-----
                annotator.set_label(
                    image=frame.image,
                    x=430,
                    y=30,
                    color=(200, 200, 200),
                    label="Total people detected: " + str(len(unique_seen_people)),
                )

                # Map the track ID in visual annotations
                labels = []
                for idx, (_, s, c, t) in enumerate(detections):
                    labels.append(f"#{t} {model.labels[c]}: {s:0.2f}")
                    
                annotator.annotate_boxes(frame=frame, detections=detections, labels=labels)

                #-----Stream to HTTP-----
                if args.stream:
                    if streamer is None:
                        streamer = HTTPStreamer(port=args.port)
                    streamer.push_frame(frame.image)

                try:
                    frame.display()
                except Exception:
                    # Headless mode
                    pass
    finally:
        if streamer:
            streamer.close()


if __name__ == "__main__":
    main()
