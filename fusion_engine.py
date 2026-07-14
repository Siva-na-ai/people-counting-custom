import numpy as np

class FusionEngine:
    def __init__(self, face_weight=0.7, body_weight=0.3):
        self.face_weight = face_weight
        self.body_weight = body_weight

    def compute_similarity(self, face_score, body_score):
        """
        Compute weighted fusion similarity score.
        If one score is missing (None), it falls back to the other.
        Scores are expected to be cosine similarity [0, 1] mapped or raw distances.
        """
        if face_score is not None and body_score is not None:
            return (self.face_weight * face_score) + (self.body_weight * body_score)
        elif face_score is not None:
            return face_score
        elif body_score is not None:
            return body_score
        else:
            return 0.0

    def compute_cosine_similarity(self, emb1, emb2):
        if emb1 is None or emb2 is None:
            return None
        # Assuming both are normalized
        return np.dot(emb1, emb2)
