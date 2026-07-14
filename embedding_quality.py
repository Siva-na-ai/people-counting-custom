import cv2
import numpy as np

class EmbeddingQuality:
    def __init__(self, min_size=60, blur_threshold=100.0, min_brightness=40, max_brightness=220):
        self.min_size = min_size
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def assess_crop_quality(self, img_crop):
        """
        Assess the quality of an image crop before embedding extraction.
        Returns a float between 0.0 (worst) and 1.0 (best).
        """
        if img_crop is None or img_crop.size == 0:
            return 0.0

        h, w = img_crop.shape[:2]
        
        # 1. Size Penalty
        size_score = min(h, w) / float(self.min_size)
        if size_score < 1.0:
            return 0.0 # Reject outright if too small

        # 2. Blur / Sharpness check (Laplacian variance)
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(variance / self.blur_threshold, 1.0)
        
        # 3. Brightness check
        mean_brightness = np.mean(gray)
        if mean_brightness < self.min_brightness or mean_brightness > self.max_brightness:
            brightness_score = 0.5
        else:
            brightness_score = 1.0

        # Weighted combination
        final_score = (blur_score * 0.7) + (brightness_score * 0.3)
        return float(np.clip(final_score, 0.0, 1.0))
