import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import config

try:
    import hailo_platform as hpf
except ImportError:
    hpf = None

class HailoYOLOv8Detector:
    """
    Hailo-accelerated YOLOv8 person detector using the integrated Hailo NMS layer.
    """
    def __init__(self, hef_path: str = config.YOLOv8_HEF_PATH, target: Optional[hpf.VDevice] = None):
        self.hef_path = os.path.expanduser(hef_path)
        if not os.path.exists(self.hef_path):
            raise FileNotFoundError(f"YOLOv8 HEF model not found at {self.hef_path}")
            
        if hpf is None:
            raise ImportError("hailo_platform is not installed.")
            
        self.hef = hpf.HEF(self.hef_path)
        self.owns_target = (target is None)
        self.target = target if target is not None else hpf.VDevice()
        
        # Configure PCIe interface
        self.configure_params = hpf.ConfigureParams.create_from_hef(
            self.hef, interface=hpf.HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, self.configure_params)[0]
        self.network_group_params = self.network_group.create_params()
        
        # Stream info
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()[0]
        
        # Format input (UINT8, quantized) and output (FLOAT32, unquantized)
        self.input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
            self.network_group, quantized=True, format_type=hpf.FormatType.UINT8
        )
        self.output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
        )
        
        self.activated_network_group = None
        self.pipeline_active = False
        
        self.infer_pipeline = hpf.InferVStreams(
            self.network_group, self.input_vstreams_params, self.output_vstreams_params
        )
        
        # Input Dimensions (typically 640x640)
        shape = self.input_vstream_info.shape
        if len(shape) == 4:
            self.batch, self.height, self.width, self.channels = shape
        else:
            self.height, self.width, self.channels = shape
            self.batch = 1
            
        self.input_name = self.input_vstream_info.name
        self.output_name = self.output_vstream_info.name

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

    def detect(self, frame_bgr: np.ndarray, threshold: float = 0.55) -> List[Dict[str, Any]]:
        """
        Runs YOLOv8 object detection on the frame.
        Filters class_id = 0 (Person).
        Returns: list of detections, each with:
          - "bbox": [xmin, ymin, xmax, ymax] (absolute pixel coords)
          - "score": confidence score
          - "class_id": class ID (0)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []
            
        self._activate_network()
        h_img, w_img = frame_bgr.shape[:2]
        
        # Prepare input (RGB format, resize to 640x640, UINT8)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
        input_data = {
            self.input_name: np.expand_dims(frame_resized, axis=0).astype(np.uint8)
        }
        
        # Inference
        raw_outputs = self.infer_pipeline.infer(input_data)
        nms_results = raw_outputs[self.output_name][0]
        
        # Unpack integrated NMS by class (class 0 = Person)
        # Structure is 501 floats per class
        flat_results = nms_results.flatten()
        class_offset = 0 * 501
        num_detections = int(flat_results[class_offset])
        
        detections = []
        for b in range(num_detections):
            offset = class_offset + 1 + b * 5
            ymin, xmin, ymax, xmax, score = flat_results[offset : offset + 5]
            
            if score >= threshold:
                # Convert normalized coordinates (0 to 1) to absolute pixels
                x1 = max(0.0, float(xmin * w_img))
                y1 = max(0.0, float(ymin * h_img))
                x2 = min(float(w_img), float(xmax * w_img))
                y2 = min(float(h_img), float(ymax * h_img))
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "score": float(score),
                    "class_id": 0
                })
                
        return detections

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
