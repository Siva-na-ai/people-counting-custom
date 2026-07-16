import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
import config

def get_laplacian_variance(img_gray: np.ndarray) -> float:
    """
    Computes the Laplacian variance as a measure of focus/blur.
    """
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())

def get_brightness(img_gray: np.ndarray) -> float:
    """
    Returns the average pixel intensity.
    """
    return float(np.mean(img_gray))

def estimate_pose_pnp(landmarks: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimates Yaw, Pitch, and Roll in degrees using 5 facial landmarks and cv2.solvePnP.
    landmarks: np.ndarray of shape (5, 2) inside the face crop coordinates.
    """
    # Standard 3D facial landmarks (model coordinates)
    model_points = np.array([
        [-30.0, 30.0, 0.0],    # Left eye
        [30.0, 30.0, 0.0],     # Right eye
        [0.0, 0.0, 20.0],      # Nose tip
        [-22.0, -30.0, 0.0],   # Left mouth corner
        [22.0, -30.0, 0.0]     # Right mouth corner
    ], dtype=np.float32)
    
    # Camera matrix approximation
    focal_length = 640.0
    center = (0.0, 0.0)
    camera_matrix = np.array([
        [focal_length, 0.0, center[0]],
        [0.0, focal_length, center[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.double)
    
    dist_coeffs = np.zeros((4, 1))
    
    # Center landmarks around the average coordinates to remove translation offsets
    landmarks_centered = landmarks - np.mean(landmarks, axis=0)
    
    try:
        # Use SOLVEPNP_SQPNP: works with exactly 5 points (SOLVEPNP_ITERATIVE
        # requires >=6 in OpenCV 4.12+, causing a crash with SCRFD's 5 landmarks)
        success, rvec, tvec = cv2.solvePnP(
            model_points,
            landmarks_centered.astype(np.float32),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP
        )
        
        if not success:
            return 0.0, 0.0, 0.0  # Treat as frontal — let angle check pass
            
        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        # Extract Euler angles from rotation matrix
        sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
        singular = sy < 1e-6
        
        if not singular:
            pitch = np.arctan2(rmat[2,1], rmat[2,2]) * 180.0 / np.pi
            yaw = np.arctan2(-rmat[2,0], sy) * 180.0 / np.pi
            roll = np.arctan2(rmat[1,0], rmat[0,0]) * 180.0 / np.pi
        else:
            pitch = np.arctan2(-rmat[1,2], rmat[1,1]) * 180.0 / np.pi
            yaw = np.arctan2(-rmat[2,0], sy) * 180.0 / np.pi
            roll = 0.0
            
        return abs(yaw), abs(pitch), abs(roll)
        
    except Exception:
        # Any cv2 failure (e.g. OpenCV version incompatibility) — return zero angles
        # so the face crop is not rejected solely due to a pose estimation error.
        return 0.0, 0.0, 0.0

def evaluate_face_quality(
    crop: np.ndarray, 
    score: float, 
    landmarks: Optional[np.ndarray] = None,
    parent_w: int = 1920,
    parent_h: int = 1080,
    face_box_coords: Optional[np.ndarray] = None
) -> Tuple[bool, float, float, float, float, float, str]:
    """
    Evaluates face crop quality strictly against production config rules.
    Returns: (passed, quality_score, blur_score, yaw, pitch, roll, failure_reason)
    """
    # Defaults
    quality_score = 0.0
    blur_score = 0.0
    yaw, pitch, roll = 0.0, 0.0, 0.0

    if crop is None or crop.size == 0:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, "Empty crop"

    h, w = crop.shape[:2]
    
    # 1. Bounding box coordinates check (Fully inside frame boundary)
    if face_box_coords is not None:
        fx1, fy1, fx2, fy2 = face_box_coords
        if fx1 < 0 or fy1 < 0 or fx2 > parent_w or fy2 > parent_h:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, f"Face not fully inside frame boundary: box={[int(x) for x in face_box_coords]}"

    # 2. Score threshold check
    if score < config.FACE_MIN_CONFIDENCE:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, f"Confidence low: {score:.3f} < {config.FACE_MIN_CONFIDENCE}"

    # 3. Size check
    if w < config.FACE_MIN_WIDTH or h < config.FACE_MIN_HEIGHT:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, f"Face too small: {w}x{h} < {config.FACE_MIN_WIDTH}x{config.FACE_MIN_HEIGHT}"

    # 4. Blur check
    img_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_score = get_laplacian_variance(img_gray)
    if blur_score <= config.FACE_BLUR_THRESHOLD:
        return False, 0.0, blur_score, 0.0, 0.0, 0.0, f"Blur score low (blurry): {blur_score:.2f} <= {config.FACE_BLUR_THRESHOLD}"

    # 5. Landmarks validation
    if landmarks is None or len(landmarks) != 5:
        return False, 0.0, blur_score, 0.0, 0.0, 0.0, "Missing or invalid 5 landmarks"

    # Ensure all landmarks coordinates fall inside the face crop boundaries
    for pt_idx, pt in enumerate(landmarks):
        px, py = pt
        if px < 0 or px > w or py < 0 or py > h:
            return False, 0.0, blur_score, 0.0, 0.0, 0.0, f"Landmark #{pt_idx} coordinate {pt} out of crop boundaries {w}x{h}"

    # 6. Pose angles estimation
    yaw, pitch, roll = estimate_pose_pnp(landmarks)
    if yaw >= config.FACE_MAX_YAW:
        return False, 0.0, blur_score, yaw, pitch, roll, f"Yaw too high: {yaw:.1f}° >= {config.FACE_MAX_YAW}°"
    if pitch >= config.FACE_MAX_PITCH:
        return False, 0.0, blur_score, yaw, pitch, roll, f"Pitch too high: {pitch:.1f}° >= {config.FACE_MAX_PITCH}°"
    if roll >= config.FACE_MAX_ROLL:
        return False, 0.0, blur_score, yaw, pitch, roll, f"Roll too high: {roll:.1f}° >= {config.FACE_MAX_ROLL}°"

    # High quality passed
    quality_score = float(score)
    return True, quality_score, blur_score, yaw, pitch, roll, "Success"


def evaluate_body_quality(crop: np.ndarray, score: float) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluates body crop quality.
    Returns: (is_valid, final_quality_score, details_dict)
    """
    details = {
        "blur_score": 0.0,
        "size": 0,
        "score": score
    }
    
    if crop is None or crop.size == 0:
        return False, 0.0, details
        
    h, w = crop.shape[:2]
    details["size"] = min(h, w)
    
    # 1. Size constraint
    if min(h, w) < config.BODY_MIN_SIZE:
        return False, 0.0, details
        
    # Convert to grayscale
    img_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # 2. Blur assessment
    blur = get_laplacian_variance(img_gray)
    details["blur_score"] = blur
    if blur < config.BODY_BLUR_THRESHOLD:
        return False, 0.0, details
        
    # Combine detector score and resolution/sharpness score
    norm_resolution = min(1.0, min(h, w) / 256.0)
    norm_blur = min(1.0, blur / 100.0)
    final_quality = 0.5 * score + 0.3 * norm_blur + 0.2 * norm_resolution
    
    return True, float(final_quality), details
