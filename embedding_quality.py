import cv2
import numpy as np

def calculate_blur_laplacian(image: np.ndarray) -> float:
    """Calculates variance of Laplacian to estimate blur."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(image, cv2.CV_64F).var()

def calculate_brightness(image: np.ndarray) -> float:
    """Calculates average brightness."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return np.mean(image[:, :, 2])
    return np.mean(image)

def assess_face_quality(face_crop: np.ndarray) -> float:
    """
    Assesses face crop quality returning a score between 0.0 and 1.0.
    In a real implementation this would check blur, angles (pose), occlusion.
    """
    if face_crop.size == 0:
        return 0.0
    blur = calculate_blur_laplacian(face_crop)
    brightness = calculate_brightness(face_crop)
    
    # Normalize score based on empirical thresholds
    blur_score = min(1.0, blur / 500.0) 
    brightness_score = 1.0 - abs(brightness - 127.5) / 127.5
    
    return (blur_score * 0.6) + (brightness_score * 0.4)

def assess_body_quality(body_crop: np.ndarray) -> float:
    """
    Assesses body crop quality.
    """
    if body_crop.size == 0:
        return 0.0
    blur = calculate_blur_laplacian(body_crop)
    return min(1.0, blur / 300.0)
