import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional
from config import FACE_MATCH_THRESHOLD, BODY_MATCH_THRESHOLD

logger = logging.getLogger(__name__)

class IdentityMatcher:
    def __init__(self, qdrant_client):
        self.db = qdrant_client
        logger.info("Initialized IdentityMatcher")

    def find_match(self, face_emb: Optional[np.ndarray], body_emb: Optional[np.ndarray]) -> Tuple[Optional[str], float]:
        """
        Searches the DB for matching embeddings.
        Returns (person_id, confidence_score) or (None, 0.0)
        """
        best_match_id = None
        highest_score = 0.0

        if face_emb is not None:
            results = self.db.search_face(face_emb, top_k=5)
            if results and results[0]["score"] >= FACE_MATCH_THRESHOLD:
                return results[0]["person_id"], results[0]["score"]

        if body_emb is not None:
            results = self.db.search_body(body_emb, top_k=5)
            if results and results[0]["score"] >= BODY_MATCH_THRESHOLD:
                # If we also searched face, we might fuse scores, but for now body match is sufficient
                if highest_score == 0.0 or results[0]["score"] > highest_score:
                    best_match_id = results[0]["person_id"]
                    highest_score = results[0]["score"]

        return best_match_id, highest_score
