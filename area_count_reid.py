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
import json
import argparse
import cv2
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
        try:
            item = db_queue.get()
            if item is None:
                db_queue.task_done()
                break
            task_type, data = item
            if task_type == "save_gate_line":
                camera_id, gate_line = data
                conn = get_db_connection()
                if conn is not None:
                    try:
                        with conn.cursor() as cur:
                            # Upsert Gate Line
                            cur.execute(
                                "SELECT zone_id FROM public.zones WHERE camera_id = %s AND zone_type = 'gate_line'",
                                (camera_id,)
                            )
                            row = cur.fetchone()
                            if row:
                                cur.execute(
                                    "UPDATE public.zones SET points = %s WHERE zone_id = %s",
                                    (json.dumps(gate_line), row[0])
                                )
                            else:
                                cur.execute(
                                    "INSERT INTO public.zones (camera_id, zone_name, description, zone_type, points) VALUES (%s, %s, %s, %s, %s)",
                                    (camera_id, f"Camera {camera_id} Gate Line", "Interactive Gate Line", "gate_line", json.dumps(gate_line))
                                )
                            conn.commit()
                            print(f"[+] Saved gate line for camera {camera_id} to database.")
                    except Exception as e:
                        print(f"[-] Failed to save gate line to database: {e}")
                    finally:
                        conn.close()
                        
            elif task_type == "save_double_lines":
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
                            print(f"[+] Saved double crossing lines for camera {camera_id} to database.")
                    except Exception as e:
                        print(f"[-] Failed to save double lines to database: {e}")
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
                            print(f"[+] Successfully saved hourly count to DB: camera_id={camera_id}, IN={total_in}, OUT={total_out}, peak={peak_occ}, avg={avg_occ}")
                    except Exception as e:
                        print(f"[-] Database hourly count update failed: {e}")
                    finally:
                        conn.close()
            db_queue.task_done()
        except Exception as queue_err:
            print(f"[-] Exception in db_worker loop: {queue_err}")

def load_gate_line_from_db(camera_id):
    conn = get_db_connection()
    if conn is None:
        return None
    gate_line = None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT zone_type, points FROM public.zones WHERE camera_id = %s AND zone_type = 'gate_line'",
                (camera_id,)
            )
            row = cur.fetchone()
            if row:
                points_data = row[1]
                if isinstance(points_data, str):
                    gate_line = json.loads(points_data)
                else:
                    gate_line = points_data
            else:
                # Fallback: if 'gate_line' is not found, try loading 'line_in' as fallback
                cur.execute(
                    "SELECT points FROM public.zones WHERE camera_id = %s AND zone_type = 'line_in'",
                    (camera_id,)
                )
                row_fallback = cur.fetchone()
                if row_fallback:
                    points_data = row_fallback[0]
                    if isinstance(points_data, str):
                        gate_line = json.loads(points_data)
                    else:
                        gate_line = points_data
    except Exception as e:
        print(f"[-] Failed to load gate line from database: {e}")
    finally:
        conn.close()
    return gate_line

def draw_line_interactively(device):
    print("[*] Opening camera stream to capture a frame for gate line definition...")
    frame_image = None
    with device as stream:
        for frame in stream:
            frame_image = frame.image.copy()
            break
            
    if frame_image is None:
        raise Exception("Failed to capture frame from camera for drawing lines.")
        
    gate_line = []
    current_points = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_points
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(current_points) >= 2:
                current_points = []
            h, w = frame_image.shape[:2]
            current_points.append([float(x / w), float(y / h)])
            
    window_name = "Define Gate Line - Click twice to draw. Enter to confirm. Q to Quit"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("[*] Define Gate Line: Click two points on the image. Press ENTER when finished.")
    
    while True:
        temp_img = frame_image.copy()
        h, w = temp_img.shape[:2]
        
        # Draw current points and line
        if len(current_points) > 0:
            cv2.circle(temp_img, (int(current_points[0][0]*w), int(current_points[0][1]*h)), 5, (0, 255, 255), -1)
        if len(current_points) == 2:
            cv2.circle(temp_img, (int(current_points[1][0]*w), int(current_points[1][1]*h)), 5, (0, 255, 255), -1)
            cv2.line(temp_img, (int(current_points[0][0]*w), int(current_points[0][1]*h)),
                     (int(current_points[1][0]*w), int(current_points[1][1]*h)), (0, 255, 255), 2)
            
        # Overlay instruction text
        cv2.putText(temp_img, "Define Gate Line: Click twice. Press ENTER to confirm.", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(window_name, temp_img)
        key = cv2.waitKey(30) & 0xFF
        
        if key == 13 or key == ord('n'):  # Enter or 'n' key
            if len(current_points) == 2:
                gate_line = current_points.copy()
                break
            else:
                print("[-] Please click exactly two points to define the gate line.")
        elif key == ord('q'):
            break
            
    cv2.destroyWindow(window_name)
    return gate_line

def load_double_lines_from_db(camera_id):
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
                pts = json.loads(points_data) if isinstance(points_data, str) else points_data
                if zone_type == "line_in":
                    line_in = pts
                elif zone_type == "line_out":
                    line_out = pts
    except Exception as e:
        print(f"[-] Failed to load double lines from database: {e}")
    finally:
        conn.close()
    return line_in, line_out

def draw_double_lines_interactively(device):
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
    phase = "IN"
    
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
        
        # Load Hailo Face engines
        self.face_det = HailoFaceDetector(hef_path=config.SCRFD_HEF_PATH, target=self.shared_target)
        self.face_rec = HailoFaceRecognizer(hef_path=config.ARCFACE_HEF_PATH, target=self.shared_target)
        
        # Pipeline Coordinator
        self.identity_mgr = IdentityManager(
            self.qdrant, self.worker_pool, self.gallery_mgr, self.registry, self.fusion,
            self.face_det, self.face_rec, self.reid, self.cache, self.event_logger
        )
        
    def close(self):
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
        
        is_numpy = isinstance(detections, np.ndarray)
        
        if not is_numpy and hasattr(detections, 'coords') and hasattr(detections, 'confidence'):
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
            
            # Pre-compute current frame occluded detection indices
            occluded_det_indices = set()
            for i in range(num_dets):
                for j in range(num_dets):
                    if i != j:
                        if compute_iou(boxes[i], boxes[j]) > 0.40:
                            occluded_det_indices.add(i)
                            occluded_det_indices.add(j)

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
                        return self.next_person_id

                    pid, state, conf = self.identity_mgr.process_observation(
                        frame_image, track.track_id, track.box, track.score, track.time_since_update, self.camera_id, get_pid, is_occluded
                    )
                    
                    if pid == self.next_person_id:
                        self.next_person_id += 1
                        
                    track.person_id = pid
                    
                    final_mapped_ids[c] = track.person_id
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
            self.next_track_id += 1
            
            cx = (det_box[0] + det_box[2]) / 2.0
            cy = (det_box[1] + det_box[3]) / 2.0
            new_track.history.append((cx, cy))
            
            is_occluded = (d_idx in occluded_det_indices)
            
            pid, state, conf = self.identity_mgr.process_observation(
                frame_image, new_track.track_id, new_track.box, new_track.score, new_track.time_since_update, self.camera_id, lambda: self.next_person_id, is_occluded
            )
            
            if pid == self.next_person_id:
                self.next_person_id += 1
                
            new_track.person_id = pid
            if state in ['CONFIRMED', 'REIDENTIFIED']:
                new_track.state = 'Confirmed'
                new_track.hits = 5
                
            self.tracks.append(new_track)
            matched_tracks.append(new_track)
            
            final_mapped_ids[d_idx] = pid
            final_mapped_states[d_idx] = 'Confirmed' if state in ['CONFIRMED', 'REIDENTIFIED'] else 'Tentative'
            final_mapped_confs[d_idx] = float(conf)
            
        # Post-match offline duplicate resolving
        active_pids = [t.person_id for t in self.tracks if t.time_since_update == 0 and t.person_id is not None]
        self.worker_pool.submit_task(self._resolve_duplicates_task, active_pids)
            
        # Age unmatched tracks
        for track in self.tracks:
            if track not in matched_tracks:
                track.time_since_update += 1
                
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
            if len(confirmed_indices) == 0:
                return np.zeros(0, dtype=new_dtype)
            tracked_arr = np.zeros(len(confirmed_indices), dtype=new_dtype)
            for idx, d_idx in enumerate(confirmed_indices):
                tracked_arr['box'][idx] = boxes[d_idx]
                tracked_arr['confidence'][idx] = final_mapped_confs[d_idx]
                tracked_arr['class_id'][idx] = class_ids[d_idx]
                tracked_arr['track_id'][idx] = final_mapped_ids[d_idx]
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
    parser.add_argument(
        "--single-line",
        action="store_true",
        help="Use a single gate line for counting (IN/OUT based on direction)"
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

    
def start_area_count_demo():
    #-----Camera and AI setup-----
    args = get_args()
    
    print("[*] Starting Application. Loaded Database configuration:")
    print(f"    DB_HOST: {os.getenv('DB_HOST')}")
    print(f"    DB_PORT: {os.getenv('DB_PORT')}")
    print(f"    DB_NAME: {os.getenv('DB_NAME')}")
    print(f"    DB_USER: {os.getenv('DB_USER')}")
    print(f"    DB_SSLMODE: {os.getenv('DB_SSLMODE')}")

    model = NanoDetPlus416x416()
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
    device.deploy(model)

    areas = []
    gate_line = None
    line_in = None
    line_out = None
    
    # Start the DB worker thread
    db_thread = threading.Thread(target=db_worker, daemon=True)
    db_thread.start()

    camera_id = args.camera_id
    using_zones = args.zones
    single_line_mode = args.single_line
    
    if using_zones:
        if single_line_mode:
            if not args.redraw:
                # Try loading from DB
                gate_line = load_gate_line_from_db(camera_id)
                if gate_line:
                    print(f"[*] Loaded existing Gate Line for camera {camera_id} from DB.")
            
            if not gate_line:
                # Must draw it
                try:
                    gate_line = draw_line_interactively(device)
                    if gate_line and len(gate_line) == 2:
                        # Save to DB
                        db_queue.put(("save_gate_line", (camera_id, gate_line)))
                    else:
                        print("[-] Gate Line creation cancelled or failed. Exiting.")
                        db_queue.put(None)
                        return
                except Exception as e:
                    print(f"[-] Error defining Gate Line interactively: {e}")
                    print("[-] Please run in a GUI environment or provide pre-defined zones/lines.")
                    db_queue.put(None)
                    return
        else:
            if not args.redraw:
                # Try loading from DB
                line_in, line_out = load_double_lines_from_db(camera_id)
                if line_in and line_out:
                    print(f"[*] Loaded existing IN and OUT lines for camera {camera_id} from DB.")
            
            if not line_in or not line_out:
                # Must draw them
                try:
                    line_in, line_out = draw_double_lines_interactively(device)
                    if line_in and line_out and len(line_in) == 2 and len(line_out) == 2:
                        # Save to DB
                        db_queue.put(("save_double_lines", (camera_id, line_in, line_out)))
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

    # Initialize BoTSORT Tracker (combining tracking and ReID) with user-configurable parameters to prevent early embedding removal
    tracker = BoTSORTTracker(
        reid_model_path=args.reid_model_path,
        reid_threshold=args.reid_threshold,
        max_age=args.max_age,
        w_iou=args.w_iou,
        w_app=args.w_app,
        w_motion=args.w_motion
    )
    
    annotator = Annotator(
        color=ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.4
    )
    
    # Variables for database metrics and transition counting
    hourly_in_count = 0
    hourly_out_count = 0
    session_in_count = 0
    session_out_count = 0
    peak_occupancy = 0
    occupancy_records = []
    
    # track_id -> last center point coordinates: e.g. [cx, cy]
    track_last_center = {}
    # track_id -> timestamp of last transition trigger
    track_last_trigger = {}
    # track_id -> list of recent center points (absolute coordinates for drawing)
    track_history = {}
    # track_id -> starting cx coordinate (normalized)
    track_start_x = {}
    # track_id -> starting cy coordinate (normalized)
    track_start_y = {}
    # track_id -> "IN" or "OUT" (representing how they were counted)
    counted_tracks = {}
    # track_id -> side where they were first detected ("inside" or "outside")
    track_start_side = {}
    
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
                    # Sync previous hour's data one last time if there was activity
                    if hourly_in_count > 0 or hourly_out_count > 0 or peak_occupancy > 0:
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
                    track_history.clear()
                    track_start_x.clear()
                    track_start_y.clear()
                    counted_tracks.clear()
                    track_start_side.clear()
                
                if using_zones:
                    if single_line_mode and gate_line:
                        for idx, (box, _, _, t) in enumerate(detections):
                            # Center of bbox
                            cx = float((box[0] + box[2]) / 2.0)
                            cy = float((box[1] + box[3]) / 2.0)
                            current_center = [cx, cy]
                            
                            if t not in track_start_side:
                                pts = sorted(gate_line, key=lambda p: (p[0], p[1]))
                                A = pts[0]
                                B = pts[1]
                                dx = B[0] - A[0]
                                dy = B[1] - A[1]
                                start_score = (cx - A[0]) * (-dy) + (cy - A[1]) * dx
                                track_start_side[t] = "inside" if start_score > 0 else "outside"
                            
                            prev_center = track_last_center.get(t)
                            track_last_center[t] = current_center
                            
                            if prev_center is not None:
                                last_trig = track_last_trigger.get(t, 0.0)
                                if current_time_secs - last_trig >= 5.0:
                                    # Check if path crossed the gate line
                                    if intersect(gate_line[0], gate_line[1], prev_center, current_center):
                                        pts = sorted(gate_line, key=lambda p: (p[0], p[1]))
                                        A = pts[0]
                                        B = pts[1]
                                        
                                        dx = B[0] - A[0]
                                        dy = B[1] - A[1]
                                        
                                        prev_score = (prev_center[0] - A[0]) * (-dy) + (prev_center[1] - A[1]) * dx
                                        curr_score = (current_center[0] - A[0]) * (-dy) + (current_center[1] - A[1]) * dx
                                        
                                        start_side = track_start_side.get(t)
                                        
                                        if prev_score < 0 and curr_score > 0:
                                            # Outside to Inside -> IN
                                            if start_side == "outside":
                                                hourly_in_count += 1
                                                track_last_trigger[t] = current_time_secs
                                                track_start_side[t] = "inside"
                                                print(f"[+] Person #{t} Crossed Gate Line (IN). Hourly IN: {hourly_in_count}")
                                                avg_occ = round(sum(occupancy_records) / len(occupancy_records), 2) if occupancy_records else 0.0
                                                db_queue.put(("update_hourly", (camera_id, current_date, current_hour, hourly_in_count, hourly_out_count, peak_occupancy, avg_occ)))
                                        elif prev_score > 0 and curr_score < 0:
                                            # Inside to Outside -> OUT
                                            if start_side == "inside":
                                                hourly_out_count += 1
                                                track_last_trigger[t] = current_time_secs
                                                track_start_side[t] = "outside"
                                                print(f"[-] Person #{t} Crossed Gate Line (OUT). Hourly OUT: {hourly_out_count}")
                                                avg_occ = round(sum(occupancy_records) / len(occupancy_records), 2) if occupancy_records else 0.0
                                                db_queue.put(("update_hourly", (camera_id, current_date, current_hour, hourly_in_count, hourly_out_count, peak_occupancy, avg_occ)))
                    
                    elif not single_line_mode and line_in and line_out:
                        for idx, (box, _, _, t) in enumerate(detections):
                            cx = float((box[0] + box[2]) / 2.0)
                            cy = float((box[1] + box[3]) / 2.0)
                            current_center = [cx, cy]
                            
                            prev_center = track_last_center.get(t)
                            track_last_center[t] = current_center
                            
                            if prev_center is not None:
                                last_trig = track_last_trigger.get(t, 0.0)
                                if current_time_secs - last_trig >= 5.0:
                                    if intersect(line_in[0], line_in[1], prev_center, current_center):
                                        hourly_in_count += 1
                                        track_last_trigger[t] = current_time_secs
                                        print(f"[+] Person #{t} Crossed IN Line. Hourly IN: {hourly_in_count}")
                                        avg_occ = round(sum(occupancy_records) / len(occupancy_records), 2) if occupancy_records else 0.0
                                        db_queue.put(("update_hourly", (camera_id, current_date, current_hour, hourly_in_count, hourly_out_count, peak_occupancy, avg_occ)))
                                    elif intersect(line_out[0], line_out[1], prev_center, current_center):
                                        hourly_out_count += 1
                                        track_last_trigger[t] = current_time_secs
                                        print(f"[-] Person #{t} Crossed OUT Line. Hourly OUT: {hourly_out_count}")
                                        avg_occ = round(sum(occupancy_records) / len(occupancy_records), 2) if occupancy_records else 0.0
                                        db_queue.put(("update_hourly", (camera_id, current_date, current_hour, hourly_in_count, hourly_out_count, peak_occupancy, avg_occ)))
                                        
                    # Clean up tracked histories for aged-out tracks to prevent memory leak
                    active_track_ids = {track.person_id for track in tracker.tracks if track.person_id is not None}
                    track_last_center = {tid: pt for tid, pt in track_last_center.items() if tid in active_track_ids}
                    track_last_trigger = {tid: t_val for tid, t_val in track_last_trigger.items() if tid in active_track_ids}
                    track_start_side = {tid: s_val for tid, s_val in track_start_side.items() if tid in active_track_ids}
                    
                else:
                    # Non-zones mode: track movement direction horizontally (right to left is IN, left to right is OUT) and vertically (top to bottom is IN, bottom to top is OUT)
                    for idx, (box, _, _, t) in enumerate(detections):
                        cx = float((box[0] + box[2]) / 2.0)
                        cy = float((box[1] + box[3]) / 2.0)
                        current_center = [cx, cy]
                        
                        # Store starting coordinates
                        if t not in track_start_x:
                            track_start_x[t] = cx
                        if t not in track_start_y:
                            track_start_y[t] = cy
                            
                        # Store history for trail drawing (absolute coordinates)
                        h_img, w_img = frame.image.shape[:2]
                        pt_abs = (int(cx * w_img), int(cy * h_img))
                        if t not in track_history:
                            track_history[t] = []
                        track_history[t].append(pt_abs)
                        if len(track_history[t]) > 30:
                            track_history[t].pop(0)
                            
                        prev_center = track_last_center.get(t)
                        track_last_center[t] = current_center
                        
                        # Check for crossing detection using state machine (checks both X and Y movement)
                        x_start = track_start_x[t]
                        y_start = track_start_y[t]
                        x_diff = cx - x_start
                        y_diff = cy - y_start
                        current_status = counted_tracks.get(t)
                        
                        if current_status is None:
                            # Not yet counted: can trigger IN or OUT
                            # Coming near (IN): X decreases (moves right-to-left) or Y increases (moves downwards)
                            if x_diff <= -0.15 or y_diff >= 0.15:
                                hourly_in_count += 1
                                session_in_count += 1
                                counted_tracks[t] = "IN"
                                track_start_x[t] = cx  # Reset baseline
                                track_start_y[t] = cy
                                print(f"[+] Person #{t} moved near (IN). Net movement: x_diff={x_diff:.2f}, y_diff={y_diff:.2f}")
                                avg_occ = round(sum(occupancy_records) / len(occupancy_records), 2) if occupancy_records else 0.0
                                db_queue.put(("update_hourly", (camera_id, current_date, current_hour, hourly_in_count, hourly_out_count, peak_occupancy, avg_occ)))
                            # Going far (OUT): X increases (moves left-to-right) or Y decreases (moves upwards)
                            elif x_diff >= 0.15 or y_diff <= -0.15:
                                hourly_out_count += 1
                                session_out_count += 1
                                counted_tracks[t] = "OUT"
                                track_start_x[t] = cx  # Reset baseline
                                track_start_y[t] = cy
                                print(f"[-] Person #{t} moved far (OUT). Net movement: x_diff={x_diff:.2f}, y_diff={y_diff:.2f}")
                                avg_occ = round(sum(occupancy_records) / len(occupancy_records), 2) if occupancy_records else 0.0
                                db_queue.put(("update_hourly", (camera_id, current_date, current_hour, hourly_in_count, hourly_out_count, peak_occupancy, avg_occ)))
                        
                        elif current_status == "IN":
                            # Last counted as IN: update limits
                            track_start_x[t] = min(track_start_x[t], cx) # furthest left
                            track_start_y[t] = max(track_start_y[t], cy) # furthest down
                            # Can trigger OUT if they walk back away from camera
                            turnaround_x = cx - track_start_x[t]
                            turnaround_y = cy - track_start_y[t]
                            if turnaround_x >= 0.15 or turnaround_y <= -0.15:
                                hourly_out_count += 1
                                session_out_count += 1
                                counted_tracks[t] = "OUT"
                                track_start_x[t] = cx  # Reset baseline
                                track_start_y[t] = cy
                                print(f"[-] Person #{t} turned around and moved far (OUT). Net turnaround: tx={turnaround_x:.2f}, ty={turnaround_y:.2f}")
                                avg_occ = round(sum(occupancy_records) / len(occupancy_records), 2) if occupancy_records else 0.0
                                db_queue.put(("update_hourly", (camera_id, current_date, current_hour, hourly_in_count, hourly_out_count, peak_occupancy, avg_occ)))
                                
                        elif current_status == "OUT":
                            # Last counted as OUT: update limits
                            track_start_x[t] = max(track_start_x[t], cx) # furthest right
                            track_start_y[t] = min(track_start_y[t], cy) # furthest up
                            # Can trigger IN if they walk back towards camera
                            turnaround_x = cx - track_start_x[t]
                            turnaround_y = cy - track_start_y[t]
                            if turnaround_x <= -0.15 or turnaround_y >= 0.15:
                                hourly_in_count += 1
                                session_in_count += 1
                                counted_tracks[t] = "IN"
                                track_start_x[t] = cx  # Reset baseline
                                track_start_y[t] = cy
                                print(f"[+] Person #{t} turned around and moved near (IN). Net turnaround: tx={turnaround_x:.2f}, ty={turnaround_y:.2f}")
                                avg_occ = round(sum(occupancy_records) / len(occupancy_records), 2) if occupancy_records else 0.0
                                db_queue.put(("update_hourly", (camera_id, current_date, current_hour, hourly_in_count, hourly_out_count, peak_occupancy, avg_occ)))
                                
                    # Clean up tracked histories for aged-out tracks to prevent memory leak
                    active_track_ids = {track.person_id for track in tracker.tracks if track.person_id is not None}
                    track_last_center = {tid: pt for tid, pt in track_last_center.items() if tid in active_track_ids}
                    track_start_x = {tid: x_val for tid, x_val in track_start_x.items() if tid in active_track_ids}
                    track_start_y = {tid: y_val for tid, y_val in track_start_y.items() if tid in active_track_ids}
                    track_history = {tid: pts for tid, pts in track_history.items() if tid in active_track_ids}
                    counted_tracks = {tid: status for tid, status in counted_tracks.items() if tid in active_track_ids}
                    
                # Periodic database sync (every 10 seconds) - ONLY if database activity exists
                if current_time_secs - last_db_write_time > 10.0:
                    if hourly_in_count > 0 or hourly_out_count > 0 or peak_occupancy > 0:
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
                
                if using_zones:
                    h_img, w_img = frame.image.shape[:2]
                    if single_line_mode and gate_line:
                        # Draw Gate Line
                        pt1 = (int(gate_line[0][0] * w_img), int(gate_line[0][1] * h_img))
                        pt2 = (int(gate_line[1][0] * w_img), int(gate_line[1][1] * h_img))
                        cv2.line(frame.image, pt1, pt2, (0, 255, 255), 3)
                        cv2.putText(frame.image, "GATE LINE", (pt1[0] + 10, pt1[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    elif not single_line_mode and line_in and line_out:
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
                        # Draw tracking dots and trails (lines) for all active tracks
                        for t_id, pts in track_history.items():
                            # Determine color based on counted status
                            status = counted_tracks.get(t_id)
                            if status == "IN":
                                color_trail = (0, 255, 0)  # Green
                                color_dot = (0, 255, 0)
                            elif status == "OUT":
                                color_trail = (0, 0, 255)  # Red
                                color_dot = (0, 0, 255)
                            else:
                                color_trail = (255, 0, 255) # Magenta
                                color_dot = (0, 255, 255)   # Yellow
                                
                            if len(pts) > 1:
                                # Draw trail
                                for i in range(1, len(pts)):
                                    cv2.line(frame.image, pts[i-1], pts[i], color_trail, 2)
                            if len(pts) > 0:
                                # Draw a dot at the current location
                                cv2.circle(frame.image, pts[-1], 5, color_dot, -1)
                        
                        # Labels on top left
                        annotator.set_label(
                            image=frame.image,
                            x=20,
                            y=30,
                            color=(0, 255, 0),
                            label=f"Hourly IN: {hourly_in_count} | Session IN: {session_in_count}",
                        )
                        annotator.set_label(
                            image=frame.image,
                            x=20,
                            y=55,
                            color=(0, 0, 255),
                            label=f"Hourly OUT: {hourly_out_count} | Session OUT: {session_out_count}",
                        )
                        annotator.set_label(
                            image=frame.image,
                            x=20,
                            y=80,
                            color=(255, 255, 255),
                            label=f"Occupancy: {current_occupancy}",
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
        if 'tracker' in locals():
            tracker.close()


if __name__ == "__main__":
    start_area_count_demo()
