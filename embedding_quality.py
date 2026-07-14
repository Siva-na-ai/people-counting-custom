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

def estimate_face_pose_symmetry(landmarks: np.ndarray) -> Tuple[float, float]:
    """
    Estimates yaw and pitch symmetry ratios from 5 facial landmarks:
    [left_eye, right_eye, nose, left_mouth, right_mouth]
    Yaw symmetry: symmetry of left/right eye distance to nose.
    Pitch ratio: eye-to-nose vertical distance compared to nose-to-mouth vertical distance.
    """
    if landmarks is None or len(landmarks) < 5:
        return 0.0, 1.0

    # Extract coordinates
    le, re, nose, lm, rm = landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4]
    
    # Yaw symmetry check (horizontal distance eye-to-nose)
    dx_left = abs(nose[0] - le[0])
    dx_right = abs(re[0] - nose[0])
    yaw_symmetry = (dx_left - dx_right) / (dx_left + dx_right + 1e-5)
    
    # Pitch ratio check (vertical distance eye-to-nose and nose-to-mouth)
    eye_y = (le[1] + re[1]) / 2.0
    mouth_y = (lm[1] + rm[1]) / 2.0
    dy_eye_nose = abs(nose[1] - eye_y)
    dy_nose_mouth = abs(mouth_y - nose[1])
    pitch_ratio = dy_eye_nose / (dy_nose_mouth + 1e-5)
    
    return float(abs(yaw_symmetry)), float(pitch_ratio)

def evaluate_face_quality(
    crop: np.ndarray, 
    score: float, 
    landmarks: Optional[np.ndarray] = None
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluates face crop quality.
    Returns: (is_valid, final_quality_score, details_dict)
    """
    details = {
        "blur_score": 0.0,
        "brightness": 0.0,
        "size": 0,
        "yaw_symmetry": 0.0,
        "pitch_ratio": 1.0,
        "score": score
    }
    
    if crop is None or crop.size == 0:
        return False, 0.0, details
        
    h, w = crop.shape[:2]
    details["size"] = min(h, w)
    
    # 1. Size constraint
    if min(h, w) < config.FACE_MIN_SIZE:
        return False, 0.0, details
        
    # Convert to grayscale for blur and brightness
    img_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # 2. Blur assessment
    blur = get_laplacian_variance(img_gray)
    details["blur_score"] = blur
    if blur < config.FACE_BLUR_THRESHOLD:
        return False, 0.0, details
        
    # 3. Brightness assessment
    brightness = get_brightness(img_gray)
    details["brightness"] = brightness
    if not (config.FACE_BRIGHTNESS_MIN <= brightness <= config.FACE_BRIGHTNESS_MAX):
        return False, 0.0, details
        
    # 4. Landmark-based pose/angle constraints
    if landmarks is not None:
        yaw_sym, pitch_ratio = estimate_face_pose_symmetry(landmarks)
        details["yaw_symmetry"] = yaw_sym
        details["pitch_ratio"] = pitch_ratio
        
        # Extreme yaw check: horizontal displacement > 0.45
        if yaw_sym > 0.45:
            return False, 0.0, details
            
        # Extreme pitch check: eye-to-nose vs nose-to-mouth ratio outside [0.2, 3.5]
        if not (0.2 <= pitch_ratio <= 3.5):
            return False, 0.0, details
            
    # Compute combined quality score
    # Score is weighted average of normalized blur, detector score, and pose alignment
    norm_blur = min(1.0, blur / 200.0)
    pose_penalty = 1.0 - details["yaw_symmetry"]
    final_quality = 0.4 * score + 0.4 * norm_blur + 0.2 * pose_penalty
    
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
