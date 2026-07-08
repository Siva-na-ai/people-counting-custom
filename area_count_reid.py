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

import os
import json
import argparse
import cv2
import torch
import numpy as np
import subprocess
import threading
import queue
import time
from datetime import datetime
import psycopg2
from dotenv import load_dotenv
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

from modlib.apps.annotate import ColorPalette, Annotator, Color
from modlib.apps.area import Area
from modlib.devices import AiCamera
from modlib.models.zoo import NanoDetPlus416x416
from scipy.optimize import linear_sum_assignment

# Load environment variables
load_dotenv()


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
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError) as e:
                print(f"[*] Client disconnected: {e}")
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
            self.server = ThreadingHTTPServer(('0.0.0.0', self.port), CustomHandler)
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


# Global database queue and connection function
db_queue = queue.Queue()

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            database=os.getenv("DB_NAME", "people_counting"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            sslmode=os.getenv("DB_SSLMODE", "prefer")
        )
        return conn
    except Exception as e:
        print(f"[-] Database connection failed: {e}")
        return None


def db_worker():
    while True:
        item = db_queue.get()
        if item is None:
            break
        task_type, data = item
        if task_type == "save_lines":
            camera_id, line_in, line_out = data
            conn = get_db_connection()
            if conn is not None:
                try:
                    with conn.cursor() as cur:
                        # Upsert IN Line
                        cur.execute(
                            "SELECT zone_id FROM public.zones WHERE camera_id = %s AND zone_type = 'line_in'",
                            (camera_id,)
                        )
                        row_in = cur.fetchone()
                        if row_in:
                            cur.execute(
                                "UPDATE public.zones SET points = %s WHERE zone_id = %s",
                                (json.dumps(line_in), row_in[0])
                            )
                        else:
                            cur.execute(
                                "INSERT INTO public.zones (camera_id, zone_name, description, zone_type, points) VALUES (%s, %s, %s, %s, %s)",
                                (camera_id, f"Camera {camera_id} IN Line", "Interactive IN Line", "line_in", json.dumps(line_in))
                            )
                        
                        # Upsert OUT Line
                        cur.execute(
                            "SELECT zone_id FROM public.zones WHERE camera_id = %s AND zone_type = 'line_out'",
                            (camera_id,)
                        )
                        row_out = cur.fetchone()
                        if row_out:
                            cur.execute(
                                "UPDATE public.zones SET points = %s WHERE zone_id = %s",
                                (json.dumps(line_out), row_out[0])
                            )
                        else:
                            cur.execute(
                                "INSERT INTO public.zones (camera_id, zone_name, description, zone_type, points) VALUES (%s, %s, %s, %s, %s)",
                                (camera_id, f"Camera {camera_id} OUT Line", "Interactive OUT Line", "line_out", json.dumps(line_out))
                            )
                        conn.commit()
                        print(f"[+] Saved crossing lines for camera {camera_id} to database.")
                except Exception as e:
                    print(f"[-] Failed to save lines to database: {e}")
                finally:
                    conn.close()
                    
        elif task_type == "update_hourly":
            camera_id, report_date, report_hour, total_in, total_out, peak_occ, avg_occ = data
            conn = get_db_connection()
            if conn is not None:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT id FROM public.people_count_hourly WHERE camera_id = %s AND report_date = %s AND report_hour = %s",
                            (camera_id, report_date, report_hour)
                        )
                        row = cur.fetchone()
                        if row:
                            cur.execute(
                                """UPDATE public.people_count_hourly 
                                   SET total_in = %s, total_out = %s, peak_occupancy = %s, avg_occupancy = %s, created_at = NOW() 
                                   WHERE id = %s""",
                                (total_in, total_out, peak_occ, avg_occ, row[0])
                            )
                        else:
                            cur.execute(
                                """INSERT INTO public.people_count_hourly (camera_id, report_date, report_hour, total_in, total_out, peak_occupancy, avg_occupancy, created_at) 
                                   VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())""",
                                (camera_id, report_date, report_hour, total_in, total_out, peak_occ, avg_occ)
                            )
                        conn.commit()
                except Exception as e:
                    print(f"[-] Database hourly count update failed: {e}")
                finally:
                    conn.close()
        db_queue.task_done()

def load_lines_from_db(camera_id):
    conn = get_db_connection()
    if conn is None:
        return None, None
    line_in, line_out = None, None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT zone_type, points FROM public.zones WHERE camera_id = %s AND zone_type IN ('line_in', 'line_out')",
                (camera_id,)
            )
            rows = cur.fetchall()
            for row in rows:
                zone_type, points_data = row
                if isinstance(points_data, str):
                    pts = json.loads(points_data)
                else:
                    pts = points_data
                if zone_type == "line_in":
                    line_in = pts
                elif zone_type == "line_out":
                    line_out = pts
    except Exception as e:
        print(f"[-] Failed to load lines from database: {e}")
    finally:
        conn.close()
    return line_in, line_out

def draw_lines_interactively(device):
    print("[*] Opening camera stream to capture a frame for line definition...")
    frame_image = None
    with device as stream:
        for frame in stream:
            frame_image = frame.image.copy()
            break
            
    if frame_image is None:
        raise Exception("Failed to capture frame from camera for drawing lines.")
        
    line_in = []
    line_out = []
    current_points = []
    phase = "IN"  # "IN" or "OUT"
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_points
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(current_points) >= 2:
                current_points = []
            h, w = frame_image.shape[:2]
            current_points.append([float(x / w), float(y / h)])
            
    window_name = "Define Lines - Click twice to draw. Enter to confirm. Q to Quit"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print(f"[*] Define {phase} Line: Click two points on the image. Press ENTER or 'n' when finished.")
    
    while True:
        temp_img = frame_image.copy()
        h, w = temp_img.shape[:2]
        
        # Draw current points and line
        if len(current_points) > 0:
            cv2.circle(temp_img, (int(current_points[0][0]*w), int(current_points[0][1]*h)), 5, (0, 255, 0) if phase == "IN" else (0, 0, 255), -1)
        if len(current_points) == 2:
            cv2.circle(temp_img, (int(current_points[1][0]*w), int(current_points[1][1]*h)), 5, (0, 255, 0) if phase == "IN" else (0, 0, 255), -1)
            cv2.line(temp_img, (int(current_points[0][0]*w), int(current_points[0][1]*h)),
                     (int(current_points[1][0]*w), int(current_points[1][1]*h)), (0, 255, 0) if phase == "IN" else (0, 0, 255), 2)
            
        # Draw completed IN line if defining OUT line
        if phase == "OUT" and len(line_in) == 2:
            pt1 = (int(line_in[0][0]*w), int(line_in[0][1]*h))
            pt2 = (int(line_in[1][0]*w), int(line_in[1][1]*h))
            cv2.line(temp_img, pt1, pt2, (0, 255, 255), 2)
            
        # Overlay instruction text
        cv2.putText(temp_img, f"Define {phase} Line: Click twice. Press ENTER to confirm.", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(window_name, temp_img)
        key = cv2.waitKey(30) & 0xFF
        
        if key == 13 or key == ord('n'):  # Enter or 'n' key
            if len(current_points) == 2:
                if phase == "IN":
                    line_in = current_points.copy()
                    current_points = []
                    phase = "OUT"
                    print("[*] Define OUT Line: Click two points on the image. Press ENTER when finished.")
                elif phase == "OUT":
                    line_out = current_points.copy()
                    break
            else:
                print(f"[-] Please click exactly two points to define the {phase} line.")
        elif key == ord('q'):
            break
            
    cv2.destroyWindow(window_name)
    return line_in, line_out

def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


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


def json_regions_extraction(json_filename):
    """
    Extract queue regions from json file.
    """
    with open(json_filename, "r") as json_file:
        area_pts = json.load(json_file)
        if len(area_pts) > 0:
            return area_pts
        else:
            raise Exception("Please ensure there are areas to check")
            
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-file",
        type=str,
        required=False,
        default=None,
        help="Json file containing bboxes of areas",
    )
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
    parser.add_argument(
        "--zones",
        action="store_true",
        help="Use IN and OUT zones configuration from DB or interactive setup"
    )
    parser.add_argument(
        "--redraw",
        action="store_true",
        help="Force interactive zone drawing even if zones are already in the DB"
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=1,
        help="Camera identifier for database records"
    )
    return parser.parse_args()

    
def start_area_count_demo():
    #-----Camera and AI setup-----
    args = get_args()

    model = NanoDetPlus416x416()
    device = AiCamera(camera_id="")
    device.deploy(model)

    areas = []
    line_in = None
    line_out = None
    
    # Start the DB worker thread
    db_thread = threading.Thread(target=db_worker, daemon=True)
    db_thread.start()

    camera_id = args.camera_id
    using_zones = args.zones
    
    if using_zones:
        if not args.redraw:
            # Try loading from DB
            line_in, line_out = load_lines_from_db(camera_id)
            if line_in and line_out:
                print(f"[*] Loaded existing IN and OUT lines for camera {camera_id} from DB.")
        
        if not line_in or not line_out:
            # Must draw them
            try:
                line_in, line_out = draw_lines_interactively(device)
                if line_in and line_out and len(line_in) == 2 and len(line_out) == 2:
                    # Save to DB
                    db_queue.put(("save_lines", (camera_id, line_in, line_out)))
                else:
                    print("[-] Lines creation cancelled or failed. Exiting.")
                    db_queue.put(None)
                    return
            except Exception as e:
                print(f"[-] Error defining lines interactively: {e}")
                print("[-] Please run in a GUI environment or provide pre-defined zones/lines.")
                db_queue.put(None)
                return
    elif args.json_file is not None:
        json_areas = json_regions_extraction(args.json_file)
        for area in json_areas: 
            areas.append(Area(area["points"]))

    # Initialize BoTSORT Tracker (combining tracking and ReID)
    tracker = BoTSORTTracker(reid_model_name='osnet_x1_0', reid_threshold=0.58, device='cpu', max_age=900)
    
    annotator = Annotator(
        color=ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.4
    )
    
    # Variables for database metrics and transition counting
    hourly_in_count = 0
    hourly_out_count = 0
    peak_occupancy = 0
    occupancy_records = []
    
    # track_id -> last center point coordinates: e.g. [cx, cy]
    track_last_center = {}
    # track_id -> timestamp of last transition trigger
    track_last_trigger = {}
    
    current_time = datetime.now()
    current_hour = current_time.hour
    current_date = current_time.date()
    
    last_db_write_time = time.time()
    
    streamer = None
    try:
        with device as stream:
            for frame in stream:
                #-----Camera and AI setup-----
                detections = frame.detections[frame.detections.confidence > 0.5]
                detections = detections[detections.class_id == 0]
                
                #-----Tracker Update-----
                detections = tracker.update(frame.image, detections)
                
                # Track current occupancy (total active people in frame)
                current_occupancy = len(detections)
                occupancy_records.append(current_occupancy)
                if current_occupancy > peak_occupancy:
                    peak_occupancy = current_occupancy
                
                current_time_secs = time.time()
                
                # Check for hour rollover
                now = datetime.now()
                if now.hour != current_hour or now.date() != current_date:
                    # Sync previous hour's data one last time
                    avg_occ = round(sum(occupancy_records) / len(occupancy_records), 2) if occupancy_records else 0.0
                    db_queue.put(("update_hourly", (camera_id, current_date, current_hour, hourly_in_count, hourly_out_count, peak_occupancy, avg_occ)))
                    
                    # Reset counters for the new hour
                    current_hour = now.hour
                    current_date = now.date()
                    hourly_in_count = 0
                    hourly_out_count = 0
                    peak_occupancy = current_occupancy
                    occupancy_records = [current_occupancy]
                    track_last_center.clear()
                    track_last_trigger.clear()
                
                if using_zones and line_in and line_out:
                    for idx, (box, _, _, t) in enumerate(detections):
                        # Center of bbox
                        cx = float((box[0] + box[2]) / 2.0)
                        cy = float((box[1] + box[3]) / 2.0)
                        current_center = [cx, cy]
                        
                        prev_center = track_last_center.get(t)
                        track_last_center[t] = current_center
                        
                        if prev_center is not None:
                            last_trig = track_last_trigger.get(t, 0.0)
                            if current_time_secs - last_trig >= 5.0:
                                # Check if path crossed the IN line (Green)
                                if intersect(line_in[0], line_in[1], prev_center, current_center):
                                    hourly_in_count += 1
                                    track_last_trigger[t] = current_time_secs
                                    print(f"[+] Person #{t} Crossed IN Line. Hourly IN: {hourly_in_count}")
                                    avg_occ = round(sum(occupancy_records) / len(occupancy_records), 2) if occupancy_records else 0.0
                                    db_queue.put(("update_hourly", (camera_id, current_date, current_hour, hourly_in_count, hourly_out_count, peak_occupancy, avg_occ)))
                                
                                # Check if path crossed the OUT line (Red)
                                elif intersect(line_out[0], line_out[1], prev_center, current_center):
                                    hourly_out_count += 1
                                    track_last_trigger[t] = current_time_secs
                                    print(f"[-] Person #{t} Crossed OUT Line. Hourly OUT: {hourly_out_count}")
                                    avg_occ = round(sum(occupancy_records) / len(occupancy_records), 2) if occupancy_records else 0.0
                                    db_queue.put(("update_hourly", (camera_id, current_date, current_hour, hourly_in_count, hourly_out_count, peak_occupancy, avg_occ)))
                                
                    # Clean up tracked histories for aged-out tracks to prevent memory leak
                    active_track_ids = {track.track_id for track in tracker.tracks}
                    track_last_center = {tid: pt for tid, pt in track_last_center.items() if tid in active_track_ids}
                    track_last_trigger = {tid: t_val for tid, t_val in track_last_trigger.items() if tid in active_track_ids}
                    
                # Periodic database sync (every 10 seconds)
                if current_time_secs - last_db_write_time > 10.0:
                    avg_occ = round(sum(occupancy_records) / len(occupancy_records), 2) if occupancy_records else 0.0
                    db_queue.put(("update_hourly", (camera_id, current_date, current_hour, hourly_in_count, hourly_out_count, peak_occupancy, avg_occ)))
                    last_db_write_time = current_time_secs

                #-----Display Annotations-----
                labels = []
                for idx, (_, s, c, t) in enumerate(detections):
                    labels.append(f"#{t} {model.labels[c]}: {s:0.2f}")

                frame.image = annotator.annotate_boxes(
                    frame=frame,
                    detections=detections,
                    labels=labels,
                    color=Color(0, 255, 255),
                    alpha=0.2,
                )
                
                if using_zones and line_in and line_out:
                    h_img, w_img = frame.image.shape[:2]
                    
                    # Draw IN line (Green)
                    pt_in1 = (int(line_in[0][0] * w_img), int(line_in[0][1] * h_img))
                    pt_in2 = (int(line_in[1][0] * w_img), int(line_in[1][1] * h_img))
                    cv2.line(frame.image, pt_in1, pt_in2, (0, 255, 0), 3)
                    cv2.putText(frame.image, "IN LINE", (pt_in1[0] + 10, pt_in1[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Draw OUT line (Red)
                    pt_out1 = (int(line_out[0][0] * w_img), int(line_out[0][1] * h_img))
                    pt_out2 = (int(line_out[1][0] * w_img), int(line_out[1][1] * h_img))
                    cv2.line(frame.image, pt_out1, pt_out2, (0, 0, 255), 3)
                    cv2.putText(frame.image, "OUT LINE", (pt_out1[0] + 10, pt_out1[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    # Labels on top left
                    annotator.set_label(
                        image=frame.image,
                        x=20,
                        y=30,
                        color=(0, 255, 0),
                        label=f"Hourly IN: {hourly_in_count}",
                    )
                    annotator.set_label(
                        image=frame.image,
                        x=20,
                        y=55,
                        color=(0, 0, 255),
                        label=f"Hourly OUT: {hourly_out_count}",
                    )
                    annotator.set_label(
                        image=frame.image,
                        x=20,
                        y=80,
                        color=(255, 255, 255),
                        label=f"Occupancy: {current_occupancy}",
                    )
                else:
                    if len(areas) == 0:
                        #-----Count and show all people-----
                        total_people = len(detections)
                        label = f"Total People Count: {total_people}"
                        annotator.set_label(
                            image=frame.image,
                            x=20,
                            y=40,
                            color=(0, 255, 255),
                            label=label,
                        )
                    else:
                        for ID, area in enumerate(areas):
                            #-----Area-----
                            d = detections[area.contains(detections)]
                            #-----Visualize Detections-----
                            frame.image = annotator.annotate_area(
                                frame=frame, area=area, color=(0, 255, 255), alpha = 0.2,
                            )
                            text_labels = [
                                "In Area: " + str(sum(1 for x in d if x)), #Get Number of people in each Area
                                "Area ID: " + str(ID + 1),
                            ]

                            for index, label in enumerate(text_labels):
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                text_width, text_height = cv2.getTextSize(
                                    text=label,
                                    fontFace=font,
                                    fontScale=0.5,
                                    thickness=1,
                                )[0]
                                annotator.set_label(
                                    image=frame.image,
                                    x=int(((area.points[0][0] +  area.points[1][0]) / 2) * frame.width) - int(text_width/2),
                                    y=int(((area.points[0][1] +  area.points[2][1]) / 2)* frame.height + ((index) * 25)) - int(2 * text_height),
                                    color=(0, 255, 255),
                                    label=label,
                                )
                
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
        # Stop background worker
        db_queue.put(None)
        db_thread.join(timeout=2)
        if streamer:
            streamer.close()


if __name__ == "__main__":
    start_area_count_demo()
