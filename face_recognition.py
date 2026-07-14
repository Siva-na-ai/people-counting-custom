import os
import cv2
import numpy as np
import config

try:
    import hailo_platform as hpf
except ImportError:
    hpf = None

class HailoFaceRecognizer:
    """
    Hailo-accelerated ArcFace feature extractor for facial recognition.
    """
    def __init__(self, hef_path: str = config.ARCFACE_HEF_PATH):
        self.hef_path = os.path.expanduser(hef_path)
        if not os.path.exists(self.hef_path):
            raise FileNotFoundError(f"ArcFace HEF model not found at {self.hef_path}")
            
        if hpf is None:
            raise ImportError("hailo_platform is not installed.")
            
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
        
        # Format parameters
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
