import cv2
import numpy as np
from skimage import transform as trans

class FaceAlignment:
    def __init__(self, output_size=(112, 112)):
        self.output_size = output_size
        # Standard landmarks for 112x112 ArcFace
        self.src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

    def align(self, img, landmarks):
        """
        Align face using 5 facial landmarks.
        landmarks: numpy array of shape (5, 2)
        """
        if landmarks is None or len(landmarks) != 5:
            return None
            
        tform = trans.SimilarityTransform()
        tform.estimate(landmarks, self.src)
        M = tform.params[0:2, :]
        
        aligned_face = cv2.warpAffine(
            img, 
            M, 
            self.output_size, 
            borderValue=0.0
        )
        return aligned_face
