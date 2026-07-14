import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import config

try:
    import hailo_platform as hpf
except ImportError:
    hpf = None

class HailoFaceDetector:
    """
    Hailo-accelerated SCRFD face detector with multi-scale post-processing and landmark decoding.
    """
    def __init__(self, hef_path: str = config.SCRFD_HEF_PATH):
        self.hef_path = os.path.expanduser(hef_path)
        if not os.path.exists(self.hef_path):
            raise FileNotFoundError(f"SCRFD HEF model not found at {self.hef_path}")
            
        if hpf is None:
            raise ImportError("hailo_platform is not installed.")
            
        self.hef = hpf.HEF(self.hef_path)
        self.target = hpf.VDevice()
        
        # Configure PCIe interface
        self.configure_params = hpf.ConfigureParams.create_from_hef(
            self.hef, interface=hpf.HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, self.configure_params)[0]
        self.network_group_params = self.network_group.create_params()
        
        # Stream info
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_infos = self.hef.get_output_vstream_infos()
        
        # Format input and output streams
        self.input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
        )
        self.output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
        )
        
        self.activated_network_group = self.network_group.activate(self.network_group_params)
        self.activated_network_group.__enter__()
        
        self.infer_pipeline = hpf.InferVStreams(
            self.network_group, self.input_vstreams_params, self.output_vstreams_params
        )
        self.infer_pipeline.__enter__()
        
        # Input Dimensions
        shape = self.input_vstream_info.shape
        if len(shape) == 4:
            self.batch, self.height, self.width, self.channels = shape
        else:
            self.height, self.width, self.channels = shape
            self.batch = 1
            
        self.input_name = self.input_vstream_info.name
        self._group_output_streams()

    def _group_output_streams(self):
        """
        Groups output streams dynamically by their channel count.
        SCRFD output stride groups have 3 stream types:
        - Score: 1 or 2 channels (class scores)
        - BBox: 4 or 8 channels (box coordinates)
        - Landmarks: 10 or 20 channels (5 x 2 landmark positions)
        """
        self.score_streams = []
        self.bbox_streams = []
        self.kps_streams = []
        
        # Sorting outputs by name or spatial dimensions to match strides (8, 16, 32)
        sorted_streams = sorted(self.output_vstream_infos, key=lambda info: info.shape[0] if len(info.shape) == 3 else info.shape[1], reverse=True)
        
        for info in sorted_streams:
            channels = info.shape[-1]
            if channels in [1, 2]:
                self.score_streams.append(info.name)
            elif channels in [4, 8]:
                self.bbox_streams.append(info.name)
            elif channels in [10, 20]:
                self.kps_streams.append(info.name)
                
        # Strides configuration
        self.strides = [8, 16, 32]
        self.anchor_num = 2  # SCRFD default anchors per scale cell

    def _generate_anchors(self, height: int, width: int) -> Dict[int, np.ndarray]:
        """
        Pre-computes anchors for each stride.
        """
        anchors = {}
        for stride in self.strides:
            f_h = height // stride
            f_w = width // stride
            
            # Grid generation
            grid_y, grid_x = np.meshgrid(np.arange(f_h), np.arange(f_w), indexing='ij')
            grid = np.stack([grid_x, grid_y], axis=-1) * stride
            # Shape: (f_h, f_w, 2) -> repeat for anchors
            grid = np.repeat(grid[:, :, np.newaxis, :], self.anchor_num, axis=2)
            anchors[stride] = grid.reshape(-1, 2)
            
        return anchors

    def detect(self, person_crop_bgr: np.ndarray, threshold: float = 0.50) -> List[Dict[str, Any]]:
        """
        Detects faces inside a person crop.
        Returns: list of detections, each with:
          - "bbox": [xmin, ymin, xmax, ymax]
          - "score": confidence score
          - "landmarks": np.ndarray of shape (5, 2)
        """
        if person_crop_bgr is None or person_crop_bgr.size == 0:
            return []
            
        crop_h, crop_w = person_crop_bgr.shape[:2]
        
        # Prepare input
        crop_rgb = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(crop_rgb, (self.width, self.height))
        input_data = {
            self.input_name: np.expand_dims(crop_resized, axis=0).astype(np.float32)
        }
        
        # Inference
        raw_outputs = self.infer_pipeline.infer(input_data)
        
        # Parse multi-scale results
        anchors = self._generate_anchors(self.height, self.width)
        all_boxes = []
        all_scores = []
        all_kps = []
        
        # Loop through strides (8, 16, 32)
        for idx, stride in enumerate(self.strides):
            if idx >= len(self.score_streams) or idx >= len(self.bbox_streams) or idx >= len(self.kps_streams):
                continue
                
            score_out = raw_outputs[self.score_streams[idx]][0]
            bbox_out = raw_outputs[self.bbox_streams[idx]][0]
            kps_out = raw_outputs[self.kps_streams[idx]][0]
            
            # Reshape tensors to flat lists corresponding to anchors
            scores = score_out.reshape(-1, 1)
            bboxes = bbox_out.reshape(-1, 4)
            kps = kps_out.reshape(-1, 10)
            
            stride_anchors = anchors[stride]
            
            # Filter by confidence threshold
            keep_indices = np.where(scores[:, 0] > threshold)[0]
            for k_idx in keep_indices:
                score = float(scores[k_idx, 0])
                anchor = stride_anchors[k_idx]
                
                # Decoded box: anchor_cx_cy + offset * stride
                offset_box = bboxes[k_idx]
                x1 = anchor[0] - offset_box[0] * stride
                y1 = anchor[1] - offset_box[1] * stride
                x2 = anchor[0] + offset_box[2] * stride
                y2 = anchor[1] + offset_box[3] * stride
                
                # Rescale boxes back to original crop size
                x1_scaled = (x1 / self.width) * crop_w
                y1_scaled = (y1 / self.height) * crop_h
                x2_scaled = (x2 / self.width) * crop_w
                y2_scaled = (y2 / self.height) * crop_h
                
                # Decoded landmarks (5 landmarks x 2 coordinates)
                offset_kps = kps[k_idx].reshape(5, 2)
                landmarks_scaled = np.zeros((5, 2))
                for pt_idx in range(5):
                    pt_x = anchor[0] + offset_kps[pt_idx, 0] * stride
                    pt_y = anchor[1] + offset_kps[pt_idx, 1] * stride
                    landmarks_scaled[pt_idx, 0] = (pt_x / self.width) * crop_w
                    landmarks_scaled[pt_idx, 1] = (pt_y / self.height) * crop_h
                    
                all_boxes.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled])
                all_scores.append(score)
                all_kps.append(landmarks_scaled)
                
        if len(all_boxes) == 0:
            return []
            
        # Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(
            bboxes=all_boxes,
            scores=all_scores,
            score_threshold=threshold,
            nms_threshold=0.45
        )
        
        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                detections.append({
                    "bbox": all_boxes[idx],
                    "score": all_scores[idx],
                    "landmarks": all_kps[idx]
                })
                
        return detections

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
