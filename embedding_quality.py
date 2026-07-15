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
    # We assume a standard pinhole camera with focal length approximately equal to the crop width
    focal_length = 640.0
    center = (0.0, 0.0)
    camera_matrix = np.array([
        [focal_length, 0.0, center[0]],
        [0.0, focal_length, center[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.double)
    
    dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion
    
    # Center landmarks around the average coordinates to remove translation offsets
    landmarks_centered = landmarks - np.mean(landmarks, axis=0)
    
    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        model_points, 
        landmarks_centered.astype(np.float32), 
        camera_matrix, 
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return 99.0, 99.0, 99.0 # Fail-safe high angles
        
    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)
    
    # Extract Euler angles from rotation matrix
    # Yaw (around Y), Pitch (around X), Roll (around Z)
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

def evaluate_face_quality(
    crop: np.ndarray, 
    score: float, 
    landmarks: Optional[np.ndarray] = None
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluates face crop quality according to strict production rules.
    Returns: (is_valid, final_quality_score, details_dict)
    """
    details = {
        "blur_score": 0.0,
        "brightness": 0.0,
        "size": 0,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "score": score
    }
    
    if crop is None or crop.size == 0:
        return False, 0.0, details
        
    h, w = crop.shape[:2]
    details["size"] = min(h, w)
    
    # 1. Size constraint (Both dimensions must be >= config.FACE_MIN_SIZE)
    if h < config.FACE_MIN_SIZE or w < config.FACE_MIN_SIZE:
        return False, 0.0, details
        
    # 2. Strict score/occlusion gate: SCRFD confidence must be high
    if score < 0.70:
        return False, 0.0, details
        
    # Convert to grayscale for blur and brightness
    img_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # 3. Brightness assessment
    brightness = get_brightness(img_gray)
    details["brightness"] = brightness
    if not (config.FACE_BRIGHTNESS_MIN <= brightness <= config.FACE_BRIGHTNESS_MAX):
        return False, 0.0, details
        
    # 4. Blur assessment (Laplacian Variance >= 80)
    blur = get_laplacian_variance(img_gray)
    details["blur_score"] = blur
    if blur < config.FACE_BLUR_THRESHOLD:
        return False, 0.0, details
        
    # 5. Landmark sanity, boundary, and pose estimation
    if landmarks is not None:
        # Landmarks sanity validation:
        # [left_eye, right_eye, nose, left_mouth, right_mouth]
        if len(landmarks) < 5:
            return False, 0.0, details
            
        le, re, nose, lm, rm = landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4]
        
        # Instability checks:
        # Eyes must be above nose
        if le[1] >= nose[1] or re[1] >= nose[1]:
            return False, 0.0, details
        # Mouth must be below nose
        if lm[1] <= nose[1] or rm[1] <= nose[1]:
            return False, 0.0, details
        # Horizontal spacing of eyes
        if abs(re[0] - le[0]) < 0.15 * w:
            return False, 0.0, details
            
        # Border crop / partial face check:
        # Reject if any landmark is within 2 pixels of the crop boundaries
        for pt in landmarks:
            if pt[0] <= 2.0 or pt[0] >= w - 2.0 or pt[1] <= 2.0 or pt[1] >= h - 2.0:
                return False, 0.0, details
                
        # 3D PnP Pose estimation
        yaw, pitch, roll = estimate_pose_pnp(landmarks)
        details["yaw"] = yaw
        details["pitch"] = pitch
        details["roll"] = roll
        
        # Strict angle thresholds: Yaw < 20, Pitch < 20, Roll < 15
        if yaw >= config.FACE_ANGLE_YAW_MAX:
            return False, 0.0, details
        if pitch >= config.FACE_ANGLE_PITCH_MAX:
            return False, 0.0, details
        if roll >= config.FACE_ANGLE_ROLL_MAX:
            return False, 0.0, details
            
    # Compute combined quality score
    norm_blur = min(1.0, blur / 200.0)
    pose_penalty = 1.0 - (details.get("yaw", 0.0) / config.FACE_ANGLE_YAW_MAX)
    final_quality = 0.4 * score + 0.4 * norm_blur + 0.2 * max(0.0, pose_penalty)
    
    return True, float(final_quality), details

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
