import cv2
import numpy as np

# Standard coordinates for 5 landmarks in a normalized 112x112 face template
REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],  # Left Eye
    [73.5318, 51.5014],  # Right Eye
    [56.0252, 71.7366],  # Nose
    [41.5493, 92.3655],  # Left Mouth Corner
    [70.7299, 92.2041]   # Right Mouth Corner
], dtype=np.float32)

def align_face(person_crop: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Performs similarity transform (rigid alignment with rotation, translation, and scaling)
    to align the face based on 5 landmarks, warping the output to 112x112.
    """
    if person_crop is None or person_crop.size == 0 or landmarks is None:
        return None
        
    src_pts = landmarks.astype(np.float32)
    dst_pts = REFERENCE_LANDMARKS.copy()
    
    # Calculate similarity transform matrix M (2x3 matrix)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    
    if M is None:
        # Fallback to standard affine transform if similarity estimation fails
        M = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])
        
    # Warp image to normalized dimensions (112x112)
    aligned_crop = cv2.warpAffine(person_crop, M, (112, 112), borderMode=cv2.BORDER_REPLICATE)
    
    return aligned_crop
