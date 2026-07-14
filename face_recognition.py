import insightface
import numpy as np

class FaceRecognition:
    def __init__(self, model_name='buffalo_l'):
        # For ArcFace, we extract the recognition model from the Buffalo_L pack
        self.app = insightface.app.FaceAnalysis(name=model_name, root='~/.insightface', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        # The recognizer component
        self.recognizer = self.app.models['recognition']

    def get_embedding(self, img, face_obj=None):
        """
        Extract ArcFace embedding (512D) from an aligned face image.
        InsightFace recognizes directly from the Face object which contains bbox and kps.
        If face_obj is provided, it extracts from the original unaligned image.
        If we only have the aligned crop, we pass it as a raw Face object.
        """
        # If we just have the aligned crop of 112x112
        if face_obj is None:
            # Create a mock face object for the recognizer if passing pre-aligned image
            class MockFace:
                pass
            face_obj = MockFace()
            face_obj.bbox = np.array([0, 0, img.shape[1], img.shape[0]])
            face_obj.kps = None
            
        embedding = self.recognizer.get(img, face_obj)
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
