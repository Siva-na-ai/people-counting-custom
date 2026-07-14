import cv2
import numpy as np

def align_face(image: np.ndarray, landmarks: np.ndarray, output_size=(112, 112)) -> np.ndarray:
    """
    Align face based on 5 landmarks using similarity transform.
    landmarks: shape (5, 2)
    Returns normalized 112x112 face crop.
    """
    # Standard 5 landmarks for ArcFace
    src = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)
    
    # If standard output size is different from 112x112, scale the src landmarks
    if output_size[0] != 112 or output_size[1] != 112:
        src[:, 0] *= output_size[0] / 112
        src[:, 1] *= output_size[1] / 112
        
    tform, _ = cv2.estimateAffinePartial2D(landmarks, src)
    if tform is None:
        return cv2.resize(image, output_size)
        
    aligned = cv2.warpAffine(image, tform, output_size, borderValue=0.0)
    return aligned
