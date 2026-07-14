import logging

logger = logging.getLogger(__name__)

class FusionEngine:
    def __init__(self, face_weight=0.7, body_weight=0.3):
        self.face_weight = face_weight
        self.body_weight = body_weight

    def fuse_similarities(self, face_sim: float, body_sim: float, 
                          face_quality: float, body_quality: float) -> float:
        """
        Fuses face and body similarity scores weighted by their quality.
        """
        if face_sim == 0.0 and body_sim == 0.0:
            return 0.0
            
        if face_sim == 0.0:
            return body_sim * body_quality
            
        if body_sim == 0.0:
            return face_sim * face_quality
            
        # Dynamically adjust weights based on quality
        total_q = (self.face_weight * face_quality) + (self.body_weight * body_quality)
        if total_q == 0:
            return 0.0
            
        w_f = (self.face_weight * face_quality) / total_q
        w_b = (self.body_weight * body_quality) / total_q
        
        return (face_sim * w_f) + (body_sim * w_b)
