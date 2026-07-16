import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import config

logger = logging.getLogger("identity_matcher")

class IdentityMatcher:
    """
    Performs similarity search in Qdrant, merges face and body match vectors,
    resolves priority (face > body), and runs spatiotemporal movement gates.
    """
    def __init__(self, qdrant_client, movement_validator, registry, gallery_mgr):
        self.qdrant = qdrant_client
        self.movement_validator = movement_validator
        self.registry = registry
        self.gallery_mgr = gallery_mgr

    def match_identity(
        self, 
        face_emb: Optional[np.ndarray], 
        body_emb: Optional[np.ndarray], 
        det_box: np.ndarray,
        time_since_update: int,
        img_w: int, 
        img_h: int
    ) -> Tuple[Optional[int], float, Dict[str, Any]]:
        """
        Retrieves Top-5 matches from Qdrant, groups scores by person_id,
        applies ambiguity check, body ReID verification, and calculates final fusion.
        Returns: (person_id, confidence, details_dict)
        """
        details = {
            "top1_pid": None,
            "top1_score": 0.0,
            "top2_pid": None,
            "top2_score": 0.0,
            "gap": 0.0,
            "body_sim": 0.0,
            "fusion_score": 0.0,
            "decision": "Rejected",
            "reason": "No face embedding"
        }

        if face_emb is None:
            return None, 0.0, details

        # 1. Search Top-K face embeddings from Qdrant
        face_hits = self.qdrant.search_similar("face_embeddings", face_emb.tolist(), limit=config.TOP_K)
        
        # 2. Group by PID and keep the highest score
        pid_scores = {}
        for hit in face_hits:
            pid = hit["payload"]["person_id"]
            score = hit["score"]
            if pid not in pid_scores or score > pid_scores[pid]:
                pid_scores[pid] = score

        if not pid_scores:
            details["reason"] = "No face hits in Qdrant"
            return None, 0.0, details

        # Sort candidate PIDs by face similarity
        sorted_candidates = sorted(pid_scores.items(), key=lambda x: x[1], reverse=True)
        top1_pid, top1_score = sorted_candidates[0]
        top2_pid = None
        top2_score = 0.0
        if len(sorted_candidates) > 1:
            top2_pid, top2_score = sorted_candidates[1]

        gap = top1_score - top2_score
        
        details.update({
            "top1_pid": top1_pid,
            "top1_score": top1_score,
            "top2_pid": top2_pid,
            "top2_score": top2_score,
            "gap": gap
        })

        # 3. Spatiotemporal validate motion for candidate PID
        if not self._validate_motion_for_pid(top1_pid, det_box, time_since_update, img_w, img_h):
            details["reason"] = f"Motion validation failed for PID #{top1_pid}"
            return None, 0.0, details

        # 4. Ambiguity check
        if gap < config.AMBIGUITY_GAP:
            details["reason"] = f"Ambiguous match (gap {gap:.3f} < {config.AMBIGUITY_GAP:.2f})"
            return None, 0.0, details

        # 5. Threshold check
        if top1_score < config.FACE_MATCH_THRESHOLD:
            details["reason"] = f"Face similarity too low ({top1_score:.3f} < {config.FACE_MATCH_THRESHOLD:.2f})"
            return None, 0.0, details

        # 6. Body ReID Verification
        body_sim = 1.0
        has_body_templates = False
        if top1_pid in self.registry.persons:
            has_body_templates = self.registry.persons[top1_pid].get("body_embedding_count", 0) > 0

        if has_body_templates and body_emb is not None:
            body_sim = 0.0
            # Retrieve body templates from Qdrant
            body_hits = self.qdrant.search_similar("body_embeddings", body_emb.tolist(), limit=20)
            for hit in body_hits:
                if hit["payload"]["person_id"] == top1_pid:
                    body_sim = max(body_sim, hit["score"])
            # Also retrieve from local gallery memory
            local_bodies = self.gallery_mgr.get_body_embeddings(top1_pid)
            if len(local_bodies) > 0:
                local_sims = [np.dot(body_emb, b) for b in local_bodies]
                body_sim = max(body_sim, float(np.max(local_sims)))

        details["body_sim"] = body_sim

        # 7. Verification criteria checks
        # Reject if body similarity strongly disagrees
        if has_body_templates and body_emb is not None and body_sim < config.BODY_REJECT_THRESHOLD:
            details["reason"] = f"Body ReID rejected (sim {body_sim:.3f} < reject {config.BODY_REJECT_THRESHOLD:.2f})"
            return None, 0.0, details

        # Check hybrid fusion criteria
        # Strong face match overrides moderate body difference
        if top1_score >= config.FACE_STRONG_MATCH:
            # Face is strong, body verify passed body_sim >= BODY_REJECT_THRESHOLD
            pass
        else:
            # Moderate face, requires strong body agreement
            if has_body_templates and body_emb is not None and body_sim < config.BODY_MATCH_THRESHOLD:
                details["reason"] = f"Body similarity insufficient for moderate face match ({body_sim:.3f} < match {config.BODY_MATCH_THRESHOLD:.2f})"
                return None, 0.0, details

        # 8. Compute FusionScore
        fusion_score = 0.70 * top1_score + 0.30 * body_sim
        details.update({
            "fusion_score": fusion_score,
            "decision": "Accepted",
            "reason": "Verified candidate"
        })

        return top1_pid, fusion_score, details

    def _validate_motion_for_pid(
        self, 
        person_id: int, 
        det_box: np.ndarray, 
        time_since_update: int, 
        img_w: int, 
        img_h: int
    ) -> bool:
        """
        Helper method to retrieve person's last location and validate physical motion constraints.
        """
        if person_id not in self.registry.persons:
            # Fallback if registry profile not fully loaded yet in memory
            return True
            
        # Get metadata
        p = self.registry.persons[person_id]
        
        # Check active track coordinates first (if they are currently mapped to a track)
        last_box = None
        if self.qdrant:
            # Look up last_box in the registry/global_gallery
            # We can read 'last_box' payload from person_registry collection
            # Retrieve from registry memory (fast cache)
            # (We will store 'last_box' inside the registry persons metadata dict)
            last_box = p.get("last_box", None)
            
        if last_box is None:
            return True
            
        # Get last seen timestamp
        last_seen_str = p.get("last_seen", None)
        dt_seconds = 0.0
        if last_seen_str is not None:
            try:
                from datetime import datetime
                last_seen_dt = datetime.fromisoformat(last_seen_str)
                dt_seconds = (datetime.now() - last_seen_dt).total_seconds()
            except Exception:
                pass
                
        is_lost = p.get("identity_state", None) == "LOST"
        
        if is_lost or dt_seconds > 2.0:
            # Use long-term camera transition validation
            return self.movement_validator.validate_camera_transition(
                last_box, det_box, dt_seconds, img_w, img_h
            )
        else:
            # Validate same-camera short-term walking speed limit
            return self.movement_validator.validate_track_motion(
                last_box, det_box, time_since_update, img_w, img_h
            )
