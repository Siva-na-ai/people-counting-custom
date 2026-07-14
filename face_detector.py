import numpy as np
import insightface

class FaceDetector:
    def __init__(self, model_name='buffalo_l', det_size=(640, 640), threshold=0.5):
        # We initialize the InsightFace app but only use the detection component (SCRFD)
        self.app = insightface.app.FaceAnalysis(name=model_name, root='~/.insightface', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=det_size)
        self.threshold = threshold

    def detect(self, img):
        """
        Detect faces in the image.
        Returns a list of dicts: [{'bbox': [x1, y1, x2, y2], 'kps': [[x,y],...], 'det_score': float}]
        """
        faces = self.app.get(img)
        results = []
        for face in faces:
            if face.det_score >= self.threshold:
                results.append({
                    'bbox': face.bbox.astype(np.int32).tolist(),
                    'kps': face.kps.astype(np.float32), # 5 facial landmarks
                    'det_score': float(face.det_score)
                })
        return results
