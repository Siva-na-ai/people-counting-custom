class IdentityMatcher:
    def __init__(self, qdrant_client, face_threshold=0.6, body_threshold=0.6):
        self.qdrant_client = qdrant_client
        self.face_threshold = face_threshold
        self.body_threshold = body_threshold

    def find_match(self, face_emb=None, body_emb=None, top_k=5):
        """
        Queries Qdrant for Top-K candidates for both face and body embeddings.
        Returns the best matched person_id or None.
        """
        candidates = {}

        # 1. Query Face
        if face_emb is not None:
            face_hits = self.qdrant_client.search_face(face_emb, top_k=top_k)
            for hit in face_hits:
                if hit.score >= self.face_threshold:
                    pid = hit.payload.get("person_id")
                    if pid not in candidates:
                        candidates[pid] = {'face_score': hit.score, 'body_score': None}
                    else:
                        candidates[pid]['face_score'] = max(candidates[pid].get('face_score', 0) or 0, hit.score)

        # 2. Query Body
        if body_emb is not None:
            body_hits = self.qdrant_client.search_body(body_emb, top_k=top_k)
            for hit in body_hits:
                if hit.score >= self.body_threshold:
                    pid = hit.payload.get("person_id")
                    if pid not in candidates:
                        candidates[pid] = {'face_score': None, 'body_score': hit.score}
                    else:
                        candidates[pid]['body_score'] = max(candidates[pid].get('body_score', 0) or 0, hit.score)

        if not candidates:
            return None, 0.0

        return candidates
