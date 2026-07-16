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
from typing import Optional
import cv2
import numpy as np
import subprocess
import argparse
import threading
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models.zoo import NanoDetPlus416x416
from yolo_detector import HailoYOLOv8Detector
from scipy.optimize import linear_sum_assignment

try:
    import hailo_platform as hpf
except ImportError:
    hpf = None

class HailoReID:
    """
    Class managing ReID inference on Hailo-8L using a compiled HEF model.
    """
    def __init__(self, hef_path: str, target: Optional[hpf.VDevice] = None):
        self.hef_path = os.path.expanduser(hef_path)
        if not os.path.exists(self.hef_path):
            raise FileNotFoundError(f"Hailo ReID HEF model not found at {self.hef_path}")
            
        if hpf is None:
            raise ImportError(
                "hailo_platform is not installed. Please install HailoRT Python bindings to run this script."
            )
            
        self.hef = hpf.HEF(self.hef_path)
        self.owns_target = (target is None)
        self.target = target if target is not None else hpf.VDevice()
        
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
        
        self.activated_network_group = None
        self.pipeline_active = False
        
        self.infer_pipeline = hpf.InferVStreams(
            self.network_group, self.input_vstreams_params, self.output_vstreams_params
        )
        
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
            
        self._activate_network()
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

    def _activate_network(self):
        active_group = getattr(self.target, "_active_group", None)
        if active_group is not self.network_group:
            if active_group is not None:
                other_instance = getattr(self.target, "_active_instance", None)
                if other_instance is not None:
                    other_instance._deactivate_network()
            
            self.activated_network_group = self.network_group.activate(self.network_group_params)
            self.activated_network_group.__enter__()
            self.infer_pipeline.__enter__()
            self.pipeline_active = True
            
            self.target._active_group = self.network_group
            self.target._active_instance = self

    def _deactivate_network(self):
        if self.pipeline_active:
            try:
                self.infer_pipeline.__exit__(None, None, None)
            except Exception:
                pass
            self.pipeline_active = False
        if self.activated_network_group:
            try:
                self.activated_network_group.__exit__(None, None, None)
            except Exception:
                pass
            self.activated_network_group = None
        if getattr(self.target, "_active_group", None) is self.network_group:
            self.target._active_group = None
            self.target._active_instance = None

    def close(self):
        self._deactivate_network()
        if hasattr(self, 'infer_pipeline') and self.infer_pipeline:
            self.infer_pipeline = None
        if hasattr(self, 'target') and self.target:
            if hasattr(self, 'owns_target') and self.owns_target:
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
    def __init__(self, track_id: int, box: np.ndarray, embedding: np.ndarray, score: float, class_id: int, state: str = 'Tentative'):
        self.track_id = track_id
        self.box = box
        self.score = score
        self.class_id = class_id
        self.embeddings = [embedding] if embedding is not None else []
        self.quality_scores = [1.0] if embedding is not None else []
        self.embedding_times = [datetime.now()] if embedding is not None else []
        self.time_since_update = 0
        self.update_count = 0
        self.history = [] # list of (cx, cy) center points
        self.state = state
        self.hits = 1 if score >= 0.70 else 0
        self.velocity = np.zeros(2)
        self.person_id = None
        self.identity_state = 'NEW_TRACK'

    def add_embedding(self, embedding: np.ndarray, quality_score: float):
        if embedding is None:
            return
            
        # Prevent gallery pollution: check similarity against existing embeddings using Top-3 average
        if len(self.embeddings) > 0:
            sims = [np.dot(embedding, stored_emb) for stored_emb in self.embeddings]
            sorted_sims = sorted(sims, reverse=True)
            top_sim = np.mean(sorted_sims[:3]) if sorted_sims else 0.0
            if top_sim < 0.50:
                return  # Reject polluted embedding
                
        now = datetime.now()
        # Compute time-decayed quality scores
        decayed_qs = []
        for q, t in zip(self.quality_scores, self.embedding_times):
            dt_hours = (now - t).total_seconds() / 3600.0
            decayed_qs.append(q * np.exp(-0.105 * dt_hours))
            
        if len(self.embeddings) < 5:
            self.embeddings.append(embedding)
            self.quality_scores.append(quality_score)
            self.embedding_times.append(now)
        else:
            # Replace the lowest decayed quality embedding if the new one is better
            min_idx = np.argmin(decayed_qs)
            if quality_score > decayed_qs[min_idx]:
                self.embeddings[min_idx] = embedding
                self.quality_scores[min_idx] = quality_score
                self.embedding_times[min_idx] = now

import config
from qdrant_client import QdrantClient
from worker_pool import WorkerPool
from gallery_manager import GalleryManager
from person_registry import PersonRegistry
from movement_validator import MovementValidator
from temporal_validator import TemporalValidator
from embedding_cache import EmbeddingCache
from event_logger import EventLogger
from face_detector import HailoFaceDetector
from face_recognition import HailoFaceRecognizer
from identity_matcher import IdentityMatcher
from fusion_engine import FusionEngine
from identity_manager import IdentityManager
from duplicate_resolver import DuplicateIdentityResolver

class BoTSORTTracker:
    def __init__(self, reid_model_path: str = '~/models/repvgg_a0_person_reid_512.hef', reid_threshold: float = 0.70, max_age: int = 900, w_iou: float = 0.4, w_app: float = 0.4, w_motion: float = 0.2, camera_id: int = 1):
        self.max_age = max_age
        self.reid_threshold = reid_threshold
        self.w_iou = w_iou
        self.w_app = w_app
        self.w_motion = w_motion
        self.gating_threshold = 0.70
        self.camera_id = camera_id
        
        # Initialize Shared Hailo target if available
        self.shared_target = hpf.VDevice() if hpf is not None else None
        
        # Initialize ReID engine (body)
        self.reid = HailoReID(hef_path=reid_model_path, target=self.shared_target)
        self.tracks = []
        self.next_track_id = 1
        self.next_person_id = 1
        self.global_gallery = {}
        
        # Initialize Qdrant and production components
        self.qdrant = QdrantClient()
        self.worker_pool = WorkerPool()
        self.gallery_mgr = GalleryManager()
        self.registry = PersonRegistry(qdrant_client=self.qdrant)
        self.registry.lock = threading.Lock()
        self.event_logger = EventLogger()
        
        self.movement_val = MovementValidator()
        self.temporal_val = TemporalValidator()
        self.cache = EmbeddingCache()
        
        self.matcher = IdentityMatcher(self.qdrant, self.movement_val, self.registry)
        self.fusion = FusionEngine(self.matcher, self.temporal_val, self.registry)
        
        # Load Hailo Face and YOLOv8 engines
        self.face_det = HailoFaceDetector(hef_path=config.SCRFD_HEF_PATH, target=self.shared_target)
        self.face_rec = HailoFaceRecognizer(hef_path=config.ARCFACE_HEF_PATH, target=self.shared_target)
        self.yolo_detector = HailoYOLOv8Detector(target=self.shared_target)
        
        # Pipeline Coordinator
        self.identity_mgr = IdentityManager(
            self.qdrant, self.worker_pool, self.gallery_mgr, self.registry, self.fusion,
            self.face_det, self.face_rec, self.reid, self.cache, self.event_logger
        )
        
    def close(self):
        if hasattr(self, 'yolo_detector') and self.yolo_detector:
            try:
                self.yolo_detector.close()
            except Exception:
                pass
        if hasattr(self, 'reid') and self.reid:
            self.reid.close()
        if hasattr(self, 'face_det') and self.face_det:
            self.face_det.close()
        if hasattr(self, 'face_rec') and self.face_rec:
            self.face_rec.close()
        if hasattr(self, 'shared_target') and self.shared_target:
            try:
                self.shared_target.close()
            except Exception:
                pass
            self.shared_target = None
        if hasattr(self, 'worker_pool') and self.worker_pool:
            self.worker_pool.stop()
            
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
            
    def _resolve_duplicates_task(self, active_pids):
        resolver = DuplicateIdentityResolver(self.registry, self.gallery_mgr, self.qdrant)
        resolver.check_and_resolve_duplicates(active_pids)
        
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

    def _validate_bbox(self, box: np.ndarray, img_h: int, img_w: int) -> bool:
        if box is None or len(box) < 4:
            return False
            
        if any(coord > 2.0 for coord in box):
            xmin, ymin, xmax, ymax = box
        else:
            xmin = box[0] * img_w
            ymin = box[1] * img_h
            xmax = box[2] * img_w
            ymax = box[3] * img_h
            
        box_w = xmax - xmin
        box_h = ymax - ymin
        
        if box_w <= 0 or box_h <= 0 or xmin >= xmax or ymin >= ymax:
            return False
            
        # 1. Aspect ratio validation (persons are vertical, typically aspect ratio height/width >= 1.0 and <= 5.0)
        aspect_ratio = box_h / box_w
        if aspect_ratio < 1.0: # Reject horizontal or too-square boxes (shadows, reflections, bags, head-only)
            return False
        if aspect_ratio > 5.0: # Reject extremely thin vertical lines (poles, shelf edges)
            return False
            
        # 2. Scale check (reject tiny boxes)
        if box_w < 20 or box_h < 50:
            return False
            
        # 3. Visibility check (ensure box is mostly inside image frame)
        inter_xmin = max(0, xmin)
        inter_ymin = max(0, ymin)
        inter_xmax = min(img_w, xmax)
        inter_ymax = min(img_h, ymax)
        
        inter_area = max(0.0, inter_xmax - inter_xmin) * max(0.0, inter_ymax - inter_ymin)
        box_area = box_w * box_h
        visible_pct = inter_area / box_area if box_area > 0 else 0.0
        
        if visible_pct < 0.50:
            return False
            
        return True

    def _get_crop_quality(self, crop: np.ndarray, confidence: float) -> tuple[bool, float]:
        if crop is None or crop.size == 0:
            return False, 0.0
        h, w, _ = crop.shape
        
        # Scale check for ReID quality
        if w < 30 or h < 60:
            return False, 0.0
            
        # Low confidence check
        if confidence < 0.65:
            return False, 0.0
            
        # Sharpness/Blur check via Laplacian variance
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception:
            return False, 0.0
            
        if lap_var < 45.0:
            return False, 0.0
            
        # Lighting check (mean brightness)
        mean_brightness = np.mean(gray)
        if mean_brightness < 30.0 or mean_brightness > 225.0:
            return False, 0.0
            
        quality_score = float(confidence * lap_var * (w * h) / 10000.0)
        return True, quality_score

    def _run_nms(self, boxes: list, scores: list, iou_threshold: float = 0.45) -> list[int]:
        if len(boxes) == 0:
            return []
            
        boxes_arr = np.array(boxes)
        scores_arr = np.array(scores)
        
        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = boxes_arr[:, 2]
        y2 = boxes_arr[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores_arr.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
            
        return keep



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
            
        boxes_raw = []
        scores_raw = []
        class_ids_raw = []
        
        is_numpy = isinstance(detections, np.ndarray) or isinstance(detections, list)
        
        if isinstance(detections, list) and len(detections) > 0 and isinstance(detections[0], dict):
            for det in detections:
                boxes_raw.append(det["bbox"])
                scores_raw.append(det["score"])
                class_ids_raw.append(det["class_id"])
        elif not is_numpy and hasattr(detections, 'coords') and hasattr(detections, 'confidence'):
            boxes_raw = detections.coords
            scores_raw = detections.confidence
            class_ids_raw = detections.class_id
        elif is_numpy and detections.dtype.names is not None:
            names = detections.dtype.names
            box_field = 'box' if 'box' in names else (names[0] if len(names) > 0 else None)
            score_field = 'confidence' if 'confidence' in names else ('score' if 'score' in names else (names[1] if len(names) > 1 else None))
            class_field = 'class_id' if 'class_id' in names else (names[2] if len(names) > 2 else None)
            
            for d_idx in range(num_dets):
                boxes_raw.append(detections[d_idx][box_field] if box_field else detections[d_idx][0])
                scores_raw.append(detections[d_idx][score_field] if score_field else detections[d_idx][1])
                class_ids_raw.append(detections[d_idx][class_field] if class_field else detections[d_idx][2])
        else:
            for det in detections:
                det_tuple = tuple(det)
                boxes_raw.append(det_tuple[0])
                scores_raw.append(det_tuple[1])
                class_ids_raw.append(det_tuple[2])
                
        # 1. Validation & CPU NMS Filtering to eliminate false person detections & duplicate IDs
        img_h, img_w, _ = frame_image.shape
        valid_indices = []
        for i in range(num_dets):
            det_box = boxes_raw[i]
            if isinstance(det_box, np.ndarray) and det_box.ndim > 1:
                det_box = det_box.flatten()
            if self._validate_bbox(det_box, img_h, img_w):
                valid_indices.append(i)
                
        if len(valid_indices) > 0:
            nms_boxes = []
            for idx in valid_indices:
                b = boxes_raw[idx]
                if isinstance(b, np.ndarray) and b.ndim > 1:
                    b = b.flatten()
                if any(coord > 2.0 for coord in b):
                    nms_boxes.append(b)
                else:
                    nms_boxes.append([b[0]*img_w, b[1]*img_h, b[2]*img_w, b[3]*img_h])
            nms_scores = [float(scores_raw[idx].item()) if isinstance(scores_raw[idx], np.ndarray) else float(scores_raw[idx]) for idx in valid_indices]
            
            keep_sub_indices = self._run_nms(nms_boxes, nms_scores, iou_threshold=0.45)
            final_valid_indices = [valid_indices[idx] for idx in keep_sub_indices]
        else:
            final_valid_indices = []
            
        boxes = [boxes_raw[i] for i in final_valid_indices]
        scores = [scores_raw[i] for i in final_valid_indices]
        class_ids = [class_ids_raw[i] for i in final_valid_indices]
        num_dets = len(final_valid_indices)
        
        # Pre-compute current frame occluded detection indices
        occluded_det_indices = set()
        for i in range(num_dets):
            for j in range(num_dets):
                if i != j:
                    if compute_iou(boxes[i], boxes[j]) > 0.40:
                        occluded_det_indices.add(i)
                        occluded_det_indices.add(j)
        
        if not is_numpy:
            detections = detections[final_valid_indices]
            
        # 2. Pre-extract embeddings and calculate crop quality metrics
        det_embeddings = []
        det_qualities = []
        
        for d_idx in range(num_dets):
            det_box = boxes[d_idx]
            if isinstance(det_box, np.ndarray) and det_box.ndim > 1:
                det_box = det_box.flatten()
            crop = self.get_crop(frame_image, det_box)
            
            is_valid, quality = self._get_crop_quality(crop, float(scores[d_idx]))
            if is_valid:
                emb = self.extract_embedding(crop)
            else:
                emb = None
                
            det_embeddings.append(emb)
            det_qualities.append((is_valid, quality))
            
        # Filter tracks for main Hungarian matching to those updated within the last 30 frames
        main_stage_tracks = [t for t in self.tracks if t.time_since_update <= 30]
        num_tracks = len(main_stage_tracks)
        matched_tracks = []
        matched_det_indices = []
        
        final_mapped_ids = [0] * num_dets
        final_mapped_states = ['Tentative'] * num_dets
        final_mapped_confs = [0.0] * num_dets
        
        # Identify occluded tracks to prevent ID swaps
        occluded_tracks = set()
        for i, t1 in enumerate(self.tracks):
            if t1.time_since_update > 0:
                continue
            for j, t2 in enumerate(self.tracks):
                if i != j and t2.time_since_update == 0:
                    t1_box = t1.box.flatten() if isinstance(t1.box, np.ndarray) else t1.box
                    t2_box = t2.box.flatten() if isinstance(t2.box, np.ndarray) else t2.box
                    if compute_iou(t1_box, t2_box) > 0.20:
                        occluded_tracks.add(t1.track_id)
                        
        # 3. Hungarian matching on active and recently lost tracks
        if num_tracks > 0:
            cost_matrix = np.zeros((num_tracks, num_dets))
            
            for t_idx, track in enumerate(main_stage_tracks):
                # Predict current position using smoothed velocity model
                if len(track.history) >= 1 and not np.all(track.velocity == 0.0):
                    c_last = track.history[-1]
                    pred_cx = c_last[0] + track.velocity[0] * (track.time_since_update + 1)
                    pred_cy = c_last[1] + track.velocity[1] * (track.time_since_update + 1)
                else:
                    pred_cx = (track.box[0] + track.box[2]) / 2.0
                    pred_cy = (track.box[1] + track.box[3]) / 2.0
                    
                track_w = track.box[2] - track.box[0]
                track_h = track.box[3] - track.box[1]
                diag = np.sqrt(track_w**2 + track_h**2)
                
                # Dynamic ReID threshold based on track state
                if track.time_since_update <= 1:
                    reid_thresh = 0.82
                elif track.time_since_update <= 3:
                    reid_thresh = 0.80
                else:
                    reid_thresh = 0.76
                
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
                        sorted_sims = sorted(sims, reverse=True)
                        sim_score = np.mean(sorted_sims[:3]) if sorted_sims else 0.0
                        C_app = float(1.0 - sim_score)
                        has_app = True
                    else:
                        C_app = 1.0
                        has_app = False
                        
                    # Occlusion-aware weight shift
                    is_occluded = track.track_id in occluded_tracks
                    if is_occluded and has_app:
                        w_iou_f = 0.1
                        w_app_f = 0.6
                        w_motion_f = 0.3
                    else:
                        w_iou_f = self.w_iou
                        w_app_f = self.w_app
                        w_motion_f = self.w_motion
                        
                    # Motion validation check (reject matches violating maximum velocity normalized by object size)
                    dist_px = np.sqrt(((det_cx - pred_cx) * img_w)**2 + ((det_cy - pred_cy) * img_h)**2)
                    diag_px = np.sqrt((track_w * img_w)**2 + (track_h * img_h)**2)
                    if track.time_since_update <= 3:
                        max_motion = 3.0 * diag_px
                    else:
                        max_motion = (4.0 + 0.2 * track.time_since_update) * diag_px
                        
                    if dist_px > max_motion:
                        cost = 1e5
                    elif track.time_since_update <= 3:
                        if has_app:
                            if sim_score < reid_thresh:
                                cost = 1e5
                            else:
                                cost = w_iou_f * C_iou + w_app_f * C_app + w_motion_f * C_motion
                        else:
                            total_w = w_iou_f + w_motion_f
                            if total_w > 0:
                                cost = (w_iou_f / total_w) * C_iou + (w_motion_f / total_w) * C_motion
                            else:
                                cost = C_iou
                    else:
                        if has_app:
                            cost = C_app
                            if sim_score < reid_thresh:
                                cost = 1e5
                        else:
                            cost = 1e5
                            
                    if track.time_since_update <= 3 and cost > self.gating_threshold:
                        cost = 1e5
                        
                    cost_matrix[t_idx, d_idx] = cost
                    
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Process matches

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 1e4:
                    track = main_stage_tracks[r]
                    matched_tracks.append(track)
                    matched_det_indices.append(c)
                    
                    track.box = boxes[c]
                    track.score = float(scores[c].item()) if isinstance(scores[c], np.ndarray) else float(scores[c])
                    track.class_id = int(class_ids[c].item()) if isinstance(class_ids[c], np.ndarray) else int(class_ids[c])
                    track.time_since_update = 0
                    
                    if track.score >= 0.70:
                        track.hits += 1
                    if track.state == 'Tentative' and track.hits >= 5:
                        track.state = 'Confirmed'
                        
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
                            
                    # Run modular Face + Body Fusion Identity Manager pipeline
                    is_occluded = (c in occluded_det_indices)
                    
                    def get_pid():
                        if track.person_id is not None:
                            return track.person_id
                        # Allocate and increment atomically
                        new_pid = self.next_person_id
                        self.next_person_id += 1
                        return new_pid

                    pid, state, conf = self.identity_mgr.process_observation(
                        frame_image, track.track_id, track.box, track.score, track.time_since_update, 
                        self.camera_id, get_pid, is_occluded, current_person_id=track.person_id
                    )
                    
                    # Update Track identity state machine
                    if track.identity_state == 'NEW_TRACK':
                        track.identity_state = 'TRACKING'
                        
                    if pid is not None:
                        if track.identity_state in ['LOST', 'TRACKING', 'NEW_TRACK']:
                            if track.person_id is not None:
                                track.identity_state = 'REIDENTIFIED'
                            else:
                                track.identity_state = 'CONFIRMED_PERSON'
                        else:
                            track.identity_state = 'CONFIRMED_PERSON'
                    else:
                        # Check if face is visible or passed quality gates to show progress
                        person_crop = self.get_crop(frame_image, track.box)
                        if person_crop is not None and person_crop.size > 0:
                            face_dets = self.face_det.detect(person_crop, threshold=0.55)
                            if len(face_dets) > 0:
                                track.identity_state = 'FACE_VISIBLE'
                                best_face = max(face_dets, key=lambda f: f["score"])
                                fx1 = max(0, int(best_face["bbox"][0]))
                                fy1 = max(0, int(best_face["bbox"][1]))
                                fx2 = min(person_crop.shape[1], int(best_face["bbox"][2]))
                                fy2 = min(person_crop.shape[0], int(best_face["bbox"][3]))
                                face_crop = person_crop[fy1:fy2, fx1:fx2]
                                if face_crop.size > 0:
                                    from embedding_quality import evaluate_face_quality
                                    face_ok, _, _ = evaluate_face_quality(face_crop, best_face["score"], best_face["landmarks"])
                                    if face_ok:
                                        track.identity_state = 'FACE_QUALITY_PASSED'
                    
                    track.person_id = pid
                    
                    final_mapped_ids[c] = track.person_id if track.person_id is not None else -int(track.track_id)
                    final_mapped_states[c] = 'Confirmed' if state in ['CONFIRMED', 'REIDENTIFIED'] or track.state == 'Confirmed' else 'Tentative'
                    final_mapped_confs[c] = float(conf)
                    
        # 4 & 5. Process unmatched detections (re-identify or create new tracks)
        unmatched_det_indices = [d for d in range(num_dets) if d not in matched_det_indices]
        for d_idx in unmatched_det_indices:
            det_box = boxes[d_idx]
            if isinstance(det_box, np.ndarray) and det_box.ndim > 1:
                det_box = det_box.flatten()
            det_score = float(scores[d_idx].item()) if isinstance(scores[d_idx], np.ndarray) else float(scores[d_idx])
            det_class = int(class_ids[d_idx].item()) if isinstance(class_ids[d_idx], np.ndarray) else int(class_ids[d_idx])
            
            new_track = Track(self.next_track_id, det_box, None, det_score, det_class, state='Tentative')
            new_track.identity_state = 'NEW_TRACK'
            self.next_track_id += 1
            
            cx = (det_box[0] + det_box[2]) / 2.0
            cy = (det_box[1] + det_box[3]) / 2.0
            new_track.history.append((cx, cy))
            
            is_occluded = (d_idx in occluded_det_indices)
            
            def get_new_pid():
                new_pid = self.next_person_id
                self.next_person_id += 1
                return new_pid

            pid, state, conf = self.identity_mgr.process_observation(
                frame_image, new_track.track_id, new_track.box, new_track.score, new_track.time_since_update, 
                self.camera_id, get_new_pid, is_occluded, current_person_id=new_track.person_id
            )
            
            if pid is not None:
                new_track.identity_state = 'CONFIRMED_PERSON'
            else:
                person_crop = self.get_crop(frame_image, new_track.box)
                if person_crop is not None and person_crop.size > 0:
                    face_dets = self.face_det.detect(person_crop, threshold=0.55)
                    if len(face_dets) > 0:
                        new_track.identity_state = 'FACE_VISIBLE'
                        best_face = max(face_dets, key=lambda f: f["score"])
                        fx1 = max(0, int(best_face["bbox"][0]))
                        fy1 = max(0, int(best_face["bbox"][1]))
                        fx2 = min(person_crop.shape[1], int(best_face["bbox"][2]))
                        fy2 = min(person_crop.shape[0], int(best_face["bbox"][3]))
                        face_crop = person_crop[fy1:fy2, fx1:fx2]
                        if face_crop.size > 0:
                            from embedding_quality import evaluate_face_quality
                            face_ok, _, _ = evaluate_face_quality(face_crop, best_face["score"], best_face["landmarks"])
                            if face_ok:
                                new_track.identity_state = 'FACE_QUALITY_PASSED'
                
            new_track.person_id = pid
            if state in ['CONFIRMED', 'REIDENTIFIED']:
                new_track.state = 'Confirmed'
                new_track.hits = 5
                
            self.tracks.append(new_track)
            matched_tracks.append(new_track)
            
            final_mapped_ids[d_idx] = pid if pid is not None else -int(new_track.track_id)
            final_mapped_states[d_idx] = 'Confirmed' if state in ['CONFIRMED', 'REIDENTIFIED'] else 'Tentative'
            final_mapped_confs[d_idx] = float(conf)
            
        # Post-match offline duplicate resolving
        active_pids = [t.person_id for t in self.tracks if t.time_since_update == 0 and t.person_id is not None]
        self.worker_pool.submit_task(self._resolve_duplicates_task, active_pids)
            
        # Age unmatched tracks
        for track in self.tracks:
            if track not in matched_tracks:
                track.time_since_update += 1
                track.identity_state = 'LOST'
                
        # Prune inactive tracks from the active list if they are inactive for > 150 frames (5 seconds)
        # to prevent memory build-up and CPU overhead, while keeping them in the global gallery for max_age.
        pruned_tracks = [
            t for t in self.tracks 
            if (t.state == 'Confirmed' and t.time_since_update > 150)
            or (t.state == 'Tentative' and t.time_since_update > 0)
        ]
        for t in pruned_tracks:
            if t.person_id is not None:
                try:
                    self.registry.handle_track_lost(t.person_id)
                except Exception:
                    pass
                    
        self.tracks = [
            t for t in self.tracks 
            if (t.state == 'Confirmed' and t.time_since_update <= 150) 
            or (t.state == 'Tentative' and t.time_since_update == 0)
        ]
        
        # Clean up old gallery entries to prevent infinite memory growth (convert max_age frames to seconds at 30 FPS)
        now = datetime.now()
        max_age_sec = self.max_age / 30.0
        self.global_gallery = {
            pid: entry for pid, entry in self.global_gallery.items()
            if (now - entry['last_seen']).total_seconds() <= max_age_sec
        }
        
        # 6. Filter Output: Only Confirmed tracks are sent downstream (probation mechanism)
        confirmed_mask = np.array([final_mapped_states[i] == 'Confirmed' for i in range(num_dets)], dtype=bool)
        confirmed_indices = [i for i in range(num_dets) if confirmed_mask[i]]
        
        # Build tracker IDs where tentative are represented as -1 (or filtered out below)
        tracker_ids_full = np.array([final_mapped_ids[i] if final_mapped_states[i] == 'Confirmed' else -1 for i in range(num_dets)], dtype=np.int32)
        
        if not is_numpy:
            if hasattr(detections, '_tracker_id'):
                detections._tracker_id = tracker_ids_full
            if hasattr(detections, 'tracker_id'):
                try:
                    detections.tracker_id = tracker_ids_full
                except AttributeError:
                    pass
            # Return custom detections object filtered to confirmed only
            return detections[confirmed_mask]
        else:
            result = []
            for d_idx in confirmed_indices:
                box_val = boxes[d_idx]
                if isinstance(box_val, np.ndarray):
                    box_val = box_val.flatten().tolist()
                result.append((
                    box_val,
                    float(scores[d_idx]),
                    int(class_ids[d_idx]),
                    int(final_mapped_ids[d_idx])
                ))
            return result


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
        default="/home/assimilate/models/repvgg_a0_person_reid_512.hef",
        help="Path to compiled ReID HEF model (default: /home/assimilate/models/repvgg_a0_person_reid_512.hef)"
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
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level (default: INFO). Use DEBUG to see face pipeline details."
    )
    return parser.parse_args()


def main():
    args = get_args()

    # Apply log level from --log-level argument
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s"
    )
    #-----Camera and AI setup-----
    device = AiCamera(frame_rate=15)
    
    # Monkey-patch device.deploy to prevent a bug in modlib where start() redeploys with camera_id=None
    original_deploy = device.deploy
    def patched_deploy(model_obj, camera_id=None, *args, **kwargs):
        if camera_id is None:
            camera_id = ""
        device.camera_id = camera_id
        return original_deploy(model_obj, *args, **kwargs)
    device.deploy = patched_deploy
    
    device.camera_id = ""
    model = NanoDetPlus416x416()
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
                #-----YOLOv8 Detection (Host-Side Hailo-8L)-----
                yolo_dets = tracker.yolo_detector.detect(frame.image, threshold=0.55)
                
                #-----Tracker Update-----
                detections = tracker.update(frame.image, yolo_dets)

                #-----ReID / Unique Visitor Count-----
                for idx, (_, s, c, t) in enumerate(detections):
                    if t > 0:
                        unique_seen_people.add(t)

                #-----Display Annotations-----
                annotator.set_label(
                    image=frame.image,
                    x=430,
                    y=30,
                    color=(200, 200, 200),
                    label="Total unique people detected: " + str(len(unique_seen_people)),
                )

                # Map the track ID in visual annotations
                labels = []
                for idx, (_, s, c, t) in enumerate(detections):
                    if t > 0:
                        # Confirmed person — check if face was detected
                        track_obj = None
                        for tk in tracker.tracks:
                            if tk.person_id == t:
                                track_obj = tk
                                break
                        has_face = track_obj is not None and track_obj.identity_state in ('FACE_VISIBLE', 'FACE_QUALITY_PASSED', 'CONFIRMED_PERSON', 'REIDENTIFIED')
                        face_label = "face" if has_face else "no face"
                        labels.append(f"#Person {t} {face_label}: {s:0.2f}")
                    else:
                        # Unconfirmed track — check identity_state for face feedback
                        track_obj = None
                        for tk in tracker.tracks:
                            if tk.track_id == abs(t):
                                track_obj = tk
                                break
                        if track_obj is not None and track_obj.identity_state in ('FACE_VISIBLE', 'FACE_QUALITY_PASSED'):
                            labels.append(f"#Track {abs(t)} face: {s:0.2f}")
                        else:
                            labels.append(f"#Track {abs(t)} no face: {s:0.2f}")
                    
                # Draw bounding boxes and labels using OpenCV
                for idx, (box, s, c, t) in enumerate(detections):
                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 255, 255) if t > 0 else (255, 0, 255) # Cyan for person, Magenta for track
                    cv2.rectangle(frame.image, (x1, y1), (x2, y2), color, 2)
                    if idx < len(labels):
                        cv2.putText(frame.image, labels[idx], (x1, max(y1 - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

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
