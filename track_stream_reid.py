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
import cv2
import numpy as np
import subprocess
import argparse
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320
from scipy.optimize import linear_sum_assignment

try:
    import hailo_platform as hpf
except ImportError:
    hpf = None

class HailoReID:
    """
    Class managing ReID inference on Hailo-8L using a compiled HEF model.
    """
    def __init__(self, hef_path: str):
        self.hef_path = os.path.expanduser(hef_path)
        if not os.path.exists(self.hef_path):
            raise FileNotFoundError(f"Hailo ReID HEF model not found at {self.hef_path}")
            
        if hpf is None:
            raise ImportError(
                "hailo_platform is not installed. Please install HailoRT Python bindings to run this script."
            )
            
        self.hef = hpf.HEF(self.hef_path)
        self.target = hpf.VDevice()
        
        # Configure network group
        self.configure_params = hpf.ConfigureParams.create_from_hef(
            self.hef, interface=hpf.HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, self.configure_params)[0]
        self.network_group_params = self.network_group.create_params()
        
        # Stream info
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()[0]
        
        # Configure vstream parameters
        self.input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
        )
        self.output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
        )
        
        # Activate network group
        self.activated_network_group = self.network_group.activate(self.network_group_params)
        self.activated_network_group.__enter__()
        
        # Create virtual streams
        self.infer_pipeline = hpf.InferVStreams(
            self.network_group, self.input_vstreams_params, self.output_vstreams_params
        )
        self.infer_pipeline.__enter__()
        
        # Determine shape
        shape = self.input_vstream_info.shape
        if len(shape) == 4:
            self.batch, self.height, self.width, self.channels = shape
        else:
            self.height, self.width, self.channels = shape
            self.batch = 1
            
        self.input_name = self.input_vstream_info.name
        self.output_name = self.output_vstream_info.name

    def infer(self, crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
            
        try:
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            # Resize to expected shape (self.width, self.height)
            crop_resized = cv2.resize(crop_rgb, (self.width, self.height))
            # Add batch dimension and convert to float32
            input_data = {
                self.input_name: np.expand_dims(crop_resized, axis=0).astype(np.float32)
            }
            
            # Run inference
            results = self.infer_pipeline.infer(input_data)
            embedding = results[self.output_name][0].flatten()
            
            # L2 normalization for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
        except Exception as e:
            print(f"[-] HailoReID inference exception: {e}")
            return None

    def close(self):
        if hasattr(self, 'infer_pipeline') and self.infer_pipeline:
            try:
                self.infer_pipeline.__exit__(None, None, None)
            except Exception:
                pass
            self.infer_pipeline = None
        if hasattr(self, 'activated_network_group') and self.activated_network_group:
            try:
                self.activated_network_group.__exit__(None, None, None)
            except Exception:
                pass
            self.activated_network_group = None
        if hasattr(self, 'target') and self.target:
            try:
                self.target.close()
            except Exception:
                pass
            self.target = None

    def __del__(self):
        self.close()

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


class Track:
    def __init__(self, track_id: int, box: np.ndarray, embedding: np.ndarray, score: float, class_id: int):
        self.track_id = track_id
        self.box = box
        self.score = score
        self.class_id = class_id
        self.embeddings = [embedding] if embedding is not None else []
        self.quality_scores = [1.0] if embedding is not None else []
        self.time_since_update = 0
        self.update_count = 0
        self.history = [] # list of (cx, cy) center points

    def add_embedding(self, embedding: np.ndarray, quality_score: float):
        if embedding is None:
            return
        if len(self.embeddings) < 5:
            self.embeddings.append(embedding)
            self.quality_scores.append(quality_score)
        else:
            # Replace the lowest quality embedding if the new one is better
            min_idx = np.argmin(self.quality_scores)
            if quality_score > self.quality_scores[min_idx]:
                self.embeddings[min_idx] = embedding
                self.quality_scores[min_idx] = quality_score

class BoTSORTTracker:
    def __init__(self, reid_model_path: str = '~/models/repvgg_a0_person_reid_512.hef', reid_threshold: float = 0.70, max_age: int = 900, w_iou: float = 0.4, w_app: float = 0.4, w_motion: float = 0.2):
        self.max_age = max_age
        self.reid_threshold = reid_threshold
        self.w_iou = w_iou
        self.w_app = w_app
        self.w_motion = w_motion
        self.gating_threshold = 0.70
        
        # Initialize ReID engine
        self.reid = HailoReID(hef_path=reid_model_path)
        self.tracks = []
        self.next_track_id = 1
        
        # Persistent global gallery: person_id -> {best_embedding, embeddings, quality_scores, last_seen, hit_count}
        self.global_gallery = {}
        
    def close(self):
        if hasattr(self, 'reid') and self.reid:
            self.reid.close()
            
    def __del__(self):
        self.close()
        
    def get_crop(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        Extract the crop from the image as BGR (OpenCV default).
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
            
        return image[ymin:ymax, xmin:xmax]

    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract 512-dim embedding from a crop.
        """
        if crop is None or crop.size == 0:
            return None
        return self.reid.infer(crop)

    def _get_crop_quality(self, crop: np.ndarray, confidence: float) -> tuple[bool, float]:
        if crop is None or crop.size == 0:
            return False, 0.0
        h, w, _ = crop.shape
        # Reject tiny detections
        if w < 32 or h < 64:
            return False, 0.0
        # Reject low-confidence detections
        if confidence < 0.65:
            return False, 0.0
            
        # Reject blurred crops using Laplacian variance
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception:
            return False, 0.0
            
        if lap_var < 100.0:
            return False, 0.0
            
        quality_score = float(confidence * lap_var)
        return True, quality_score

    def _update_gallery(self, person_id: int, embedding: np.ndarray, quality_score: float):
        if person_id not in self.global_gallery:
            self.global_gallery[person_id] = {
                'best_embedding': None,
                'embeddings': [],
                'quality_scores': [],
                'last_seen': datetime.now(),
                'hit_count': 0
            }
            
        entry = self.global_gallery[person_id]
        entry['last_seen'] = datetime.now()
        entry['hit_count'] += 1
        
        if embedding is not None:
            if len(entry['embeddings']) < 5:
                entry['embeddings'].append(embedding)
                entry['quality_scores'].append(quality_score)
            else:
                min_idx = np.argmin(entry['quality_scores'])
                if quality_score > entry['quality_scores'][min_idx]:
                    entry['embeddings'][min_idx] = embedding
                    entry['quality_scores'][min_idx] = quality_score
                    
            # Recalculate best_embedding
            max_idx = np.argmax(entry['quality_scores'])
            entry['best_embedding'] = entry['embeddings'][max_idx]

    def update(self, frame_image: np.ndarray, detections) -> np.ndarray:
        try:
            return self._update_impl(frame_image, detections)
        except Exception as e:
            print("DEBUG: Exception in BoTSORTTracker.update:", type(e), str(e))
            raise e

    def _update_impl(self, frame_image: np.ndarray, detections) -> np.ndarray:
        num_dets = len(detections)
        
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
                
        # 1. Pre-extract embeddings and calculate crop quality metrics
        det_embeddings = []
        det_qualities = []
        
        for d_idx in range(num_dets):
            det_box = boxes[d_idx]
            if isinstance(det_box, np.ndarray) and det_box.ndim > 1:
                det_box = det_box.flatten()
            crop = self.get_crop(frame_image, det_box)
            
            emb = self.extract_embedding(crop)
            is_valid, quality = self._get_crop_quality(crop, float(scores[d_idx]))
            
            det_embeddings.append(emb)
            det_qualities.append((is_valid, quality))
            
        num_tracks = len(self.tracks)
        matched_tracks = []
        matched_det_indices = []
        
        final_mapped_ids = [0] * num_dets
        
        # 2. Hungarian matching on active and recently lost tracks
        if num_tracks > 0:
            cost_matrix = np.zeros((num_tracks, num_dets))
            
            for t_idx, track in enumerate(self.tracks):
                # Predict current position using velocity model
                if len(track.history) >= 2:
                    c_last = track.history[-1]
                    c_prev = track.history[-2]
                    v_x = c_last[0] - c_prev[0]
                    v_y = c_last[1] - c_prev[1]
                    pred_cx = c_last[0] + v_x * (track.time_since_update + 1)
                    pred_cy = c_last[1] + v_y * (track.time_since_update + 1)
                elif len(track.history) == 1:
                    pred_cx, pred_cy = track.history[0]
                else:
                    pred_cx = (track.box[0] + track.box[2]) / 2.0
                    pred_cy = (track.box[1] + track.box[3]) / 2.0
                    
                track_w = track.box[2] - track.box[0]
                track_h = track.box[3] - track.box[1]
                diag = np.sqrt(track_w**2 + track_h**2)
                
                for d_idx in range(num_dets):
                    det_box = boxes[d_idx]
                    if isinstance(det_box, np.ndarray) and det_box.ndim > 1:
                        det_box = det_box.flatten()
                        
                    # IoU Cost
                    C_iou = float(1.0 - compute_iou(track.box, det_box))
                    
                    # Motion Cost
                    det_cx = (det_box[0] + det_box[2]) / 2.0
                    det_cy = (det_box[1] + det_box[3]) / 2.0
                    dist = np.sqrt((det_cx - pred_cx)**2 + (det_cy - pred_cy)**2)
                    dist_norm = dist / diag if diag > 0 else dist
                    C_motion = float(1.0 - np.exp(-2.0 * dist_norm))
                    
                    # Appearance Cost
                    det_emb = det_embeddings[d_idx]
                    if len(track.embeddings) > 0 and det_emb is not None:
                        sims = [np.dot(det_emb, stored_emb) for stored_emb in track.embeddings]
                        max_sim = max(sims) if sims else 0.0
                        C_app = float(1.0 - max_sim)
                        has_app = True
                    else:
                        C_app = 1.0
                        has_app = False
                        
                    if track.time_since_update <= 3:
                        # Frame-to-frame matching combining IoU, Appearance, and Motion
                        if has_app:
                            cost = self.w_iou * C_iou + self.w_app * C_app + self.w_motion * C_motion
                        else:
                            total_w = self.w_iou + self.w_motion
                            if total_w > 0:
                                cost = (self.w_iou / total_w) * C_iou + (self.w_motion / total_w) * C_motion
                            else:
                                cost = C_iou
                    else:
                        # Long-lost track matching (ReID-only)
                        if has_app:
                            cost = C_app
                            if max_sim < self.reid_threshold:
                                cost = 1e5
                        else:
                            cost = 1e5
                            
                    if track.time_since_update <= 3 and cost > self.gating_threshold:
                        cost = 1e5
                        
                    cost_matrix[t_idx, d_idx] = cost
                    
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 1e4:
                    track = self.tracks[r]
                    matched_tracks.append(track)
                    matched_det_indices.append(c)
                    final_mapped_ids[c] = track.track_id
                    
        # Update matched tracks
        for track, d_idx in zip(matched_tracks, matched_det_indices):
            det_box = boxes[d_idx]
            if isinstance(det_box, np.ndarray) and det_box.ndim > 1:
                det_box = det_box.flatten()
            det_score = float(scores[d_idx].item()) if isinstance(scores[d_idx], np.ndarray) else float(scores[d_idx])
            det_class = int(class_ids[d_idx].item()) if isinstance(class_ids[d_idx], np.ndarray) else int(class_ids[d_idx])
            
            track.box = det_box
            track.score = det_score
            track.class_id = det_class
            track.time_since_update = 0
            
            cx = (det_box[0] + det_box[2]) / 2.0
            cy = (det_box[1] + det_box[3]) / 2.0
            track.history.append((cx, cy))
            if len(track.history) > 10:
                track.history.pop(0)
                
            det_emb = det_embeddings[d_idx]
            is_valid, quality = det_qualities[d_idx]
            if is_valid and det_emb is not None:
                track.add_embedding(det_emb, quality)
                self._update_gallery(track.track_id, det_emb, quality)
            else:
                self._update_gallery(track.track_id, None, 0.0)
                
            track.update_count += 1
            
        # 3. Re-identify remaining unmatched detections via Global ReID Gallery
        unmatched_det_indices = [d for d in range(num_dets) if d not in matched_det_indices]
        valid_unmatched_det_indices = [d for d in unmatched_det_indices if det_embeddings[d] is not None]
        
        active_ids = {t.track_id for t in matched_tracks}
        inactive_gallery_ids = [pid for pid in self.global_gallery.keys() if pid not in active_ids]
        
        if len(valid_unmatched_det_indices) > 0 and len(inactive_gallery_ids) > 0:
            gallery_cost = np.ones((len(inactive_gallery_ids), len(valid_unmatched_det_indices)))
            
            for i, pid in enumerate(inactive_gallery_ids):
                entry = self.global_gallery[pid]
                for j, d_idx in enumerate(valid_unmatched_det_indices):
                    det_emb = det_embeddings[d_idx]
                    sims = [np.dot(det_emb, stored_emb) for stored_emb in entry['embeddings']]
                    max_sim = max(sims) if sims else 0.0
                    if max_sim >= self.reid_threshold:
                        gallery_cost[i, j] = float(1.0 - max_sim)
                    else:
                        gallery_cost[i, j] = 1e5
                        
            r_ind, c_ind = linear_sum_assignment(gallery_cost)
            
            for r, c in zip(r_ind, c_ind):
                if gallery_cost[r, c] < 1e4:
                    pid = inactive_gallery_ids[r]
                    d_idx = valid_unmatched_det_indices[c]
                    
                    det_box = boxes[d_idx]
                    if isinstance(det_box, np.ndarray) and det_box.ndim > 1:
                        det_box = det_box.flatten()
                    det_score = float(scores[d_idx].item()) if isinstance(scores[d_idx], np.ndarray) else float(scores[d_idx])
                    det_class = int(class_ids[d_idx].item()) if isinstance(class_ids[d_idx], np.ndarray) else int(class_ids[d_idx])
                    det_emb = det_embeddings[d_idx]
                    is_valid, quality = det_qualities[d_idx]
                    
                    new_track = Track(pid, det_box, None, det_score, det_class)
                    entry = self.global_gallery[pid]
                    new_track.embeddings = list(entry['embeddings'])
                    new_track.quality_scores = list(entry['quality_scores'])
                    
                    cx = (det_box[0] + det_box[2]) / 2.0
                    cy = (det_box[1] + det_box[3]) / 2.0
                    new_track.history.append((cx, cy))
                    
                    if is_valid and det_emb is not None:
                        new_track.add_embedding(det_emb, quality)
                        self._update_gallery(pid, det_emb, quality)
                    else:
                        self._update_gallery(pid, None, 0.0)
                        
                    self.tracks.append(new_track)
                    matched_tracks.append(new_track)
                    final_mapped_ids[d_idx] = pid
                    unmatched_det_indices.remove(d_idx)
                    
        # 4. Create new tracks for the remaining unmatched detections
        for d_idx in unmatched_det_indices:
            det_box = boxes[d_idx]
            if isinstance(det_box, np.ndarray) and det_box.ndim > 1:
                det_box = det_box.flatten()
            det_score = float(scores[d_idx].item()) if isinstance(scores[d_idx], np.ndarray) else float(scores[d_idx])
            det_class = int(class_ids[d_idx].item()) if isinstance(class_ids[d_idx], np.ndarray) else int(class_ids[d_idx])
            det_emb = det_embeddings[d_idx]
            is_valid, quality = det_qualities[d_idx]
            
            new_track = Track(self.next_track_id, det_box, None, det_score, det_class)
            cx = (det_box[0] + det_box[2]) / 2.0
            cy = (det_box[1] + det_box[3]) / 2.0
            new_track.history.append((cx, cy))
            
            if det_emb is not None:
                if is_valid:
                    new_track.add_embedding(det_emb, quality)
                    self._update_gallery(self.next_track_id, det_emb, quality)
                else:
                    new_track.add_embedding(det_emb, 0.1)
                    self._update_gallery(self.next_track_id, det_emb, 0.1)
            else:
                self._update_gallery(self.next_track_id, None, 0.0)
                
            self.tracks.append(new_track)
            matched_tracks.append(new_track)
            final_mapped_ids[d_idx] = self.next_track_id
            self.next_track_id += 1
            
        # Age unmatched tracks
        for track in self.tracks:
            if track not in matched_tracks:
                track.time_since_update += 1
                
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        tracker_ids = np.array(final_mapped_ids, dtype=np.int32)
        
        if not is_numpy:
            if hasattr(detections, '_tracker_id'):
                detections._tracker_id = tracker_ids
            if hasattr(detections, 'tracker_id'):
                try:
                    detections.tracker_id = tracker_ids
                except AttributeError:
                    pass
            return detections
        else:
            tracked_arr = np.zeros(num_dets, dtype=new_dtype)
            for i in range(num_dets):
                tracked_arr['box'][i] = boxes[i]
                tracked_arr['confidence'][i] = scores[i]
                tracked_arr['class_id'][i] = class_ids[i]
                tracked_arr['track_id'][i] = final_mapped_ids[i]
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
    # Add configurable ReID similarity threshold
    parser.add_argument(
        "--reid-threshold",
        type=float,
        default=0.70,
        help="Similarity threshold for ReID matching (default: 0.70)"
    )
    # Add configurable maximum track age to prevent early embedding removal (default: 24 hours at 30 FPS)
    parser.add_argument(
        "--max-age",
        type=int,
        default=2592000,
        help="Maximum frames to keep an inactive track in memory (default: 2592000)"
    )
    parser.add_argument(
        "--reid-model-path",
        type=str,
        default="~/models/repvgg_a0_person_reid_512.hef",
        help="Path to compiled ReID HEF model (default: ~/models/repvgg_a0_person_reid_512.hef)"
    )
    parser.add_argument(
        "--w-iou",
        type=float,
        default=0.4,
        help="Hungarian matching weight for IoU cost (default: 0.4)"
    )
    parser.add_argument(
        "--w-app",
        type=float,
        default=0.4,
        help="Hungarian matching weight for appearance cost (default: 0.4)"
    )
    parser.add_argument(
        "--w-motion",
        type=float,
        default=0.2,
        help="Hungarian matching weight for motion consistency cost (default: 0.2)"
    )
    return parser.parse_args()


def main():
    args = get_args()
    
    #-----Camera and AI setup-----
    device = AiCamera()
    
    # Monkey-patch device.deploy to prevent a bug in modlib where start() redeploys with camera_id=None
    original_deploy = device.deploy
    def patched_deploy(model_obj, camera_id=None, *args, **kwargs):
        if camera_id is None:
            camera_id = ""
        device.camera_id = camera_id
        return original_deploy(model_obj, *args, **kwargs)
    device.deploy = patched_deploy
    
    device.camera_id = ""
    model = SSDMobileNetV2FPNLite320x320()
    device.deploy(model)

    # Initialize BoTSORT Tracker (combining tracking and ReID) with user-configurable parameters to prevent early embedding removal
    tracker = BoTSORTTracker(
        reid_model_path=args.reid_model_path,
        reid_threshold=args.reid_threshold,
        max_age=args.max_age,
        w_iou=args.w_iou,
        w_app=args.w_app,
        w_motion=args.w_motion
    )
    
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
        if 'tracker' in locals():
            tracker.close()


if __name__ == "__main__":
    main()
