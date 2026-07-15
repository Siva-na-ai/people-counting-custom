import os
import cv2
import numpy as np
import config
from typing import Optional

try:
    import hailo_platform as hpf
except ImportError:
    hpf = None

class HailoFaceRecognizer:
    """
    Hailo-accelerated ArcFace feature extractor for facial recognition.
    """
    def __init__(self, hef_path: str = config.ARCFACE_HEF_PATH, target: Optional[hpf.VDevice] = None):
        self.hef_path = os.path.expanduser(hef_path)
        if not os.path.exists(self.hef_path):
            raise FileNotFoundError(f"ArcFace HEF model not found at {self.hef_path}")
            
        if hpf is None:
            raise ImportError("hailo_platform is not installed.")
            
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
        
        # Format parameters
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
        
        # Shape definition
        shape = self.input_vstream_info.shape
        if len(shape) == 4:
            self.batch, self.height, self.width, self.channels = shape
        else:
            self.height, self.width, self.channels = shape
            self.batch = 1
            
        self.input_name = self.input_vstream_info.name
        self.output_name = self.output_vstream_info.name

    def extract_embedding(self, aligned_face_crop_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts a normalized 512-dimensional face embedding from an aligned 112x112 BGR face crop.
        """
        if aligned_face_crop_bgr is None or aligned_face_crop_bgr.size == 0:
            return None
            
        self._activate_network()
        try:
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(aligned_face_crop_bgr, cv2.COLOR_BGR2RGB)
            # Ensure resizing to expected model input dimensions (usually 112x112)
            crop_resized = cv2.resize(crop_rgb, (self.width, self.height))
            
            input_data = {
                self.input_name: np.expand_dims(crop_resized, axis=0).astype(np.float32)
            }
            
            # Execute inference
            results = self.infer_pipeline.infer(input_data)
            embedding = results[self.output_name][0].flatten()
            
            # L2 Normalization for Cosine Similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
        except Exception as e:
            print(f"[-] Face recognition embedding extraction failed: {e}")
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
