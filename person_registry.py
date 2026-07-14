import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import config

logger = logging.getLogger("person_registry")

class PersonRegistry:
    """
    Manages identity lifecycle states (NEW -> CANDIDATE -> CONFIRMED -> LOST -> REIDENTIFIED -> ARCHIVED)
    and handles metadata updates, camera transitions, and profile merges.
    """
    def __init__(self, qdrant_client=None):
        self.qdrant = qdrant_client
        self.persons: Dict[int, Dict[str, Any]] = {}
        self.lock = threading_lock = None  # To be initialized inside tracker context or thread-safe

    def create_person(self, person_id: int, camera_id: int) -> Dict[str, Any]:
        """
        Creates a new person profile in the registry.
        """
        now = datetime.now().isoformat()
        metadata = {
            "person_id": person_id,
            "identity_state": "NEW",
            "confidence": 0.50,
            "stability_score": 0.50,
            "visit_count": 1,
            "first_seen": now,
            "last_seen": now,
            "camera_history": [camera_id],
            "active_track_ids": [],
            "face_embedding_count": 0,
            "body_embedding_count": 0
        }
        self.persons[person_id] = metadata
        return metadata

    def update_person(
        self, 
        person_id: int, 
        camera_id: int, 
        track_id: int, 
        face_count: int, 
        body_count: int, 
        confidence: float,
        hits: int
    ):
        """
        Updates metadata and manages identity lifecycle transitions.
        """
        if person_id not in self.persons:
            self.create_person(person_id, camera_id)
            
        p = self.persons[person_id]
        p["last_seen"] = datetime.now().isoformat()
        
        # Track history updates
        if camera_id not in p["camera_history"]:
            p["camera_history"].append(camera_id)
        if track_id not in p["active_track_ids"]:
            p["active_track_ids"].append(track_id)
            
        p["face_embedding_count"] = face_count
        p["body_embedding_count"] = body_count
        
        # Smooth confidence updating
        p["confidence"] = 0.8 * p["confidence"] + 0.2 * confidence
        
        # Lifecycle State Machine Transitions
        state = p["identity_state"]
        if state == "NEW":
            if hits >= 2:
                p["identity_state"] = "CANDIDATE"
                logger.info(f"[PersonRegistry] Person #{person_id} promoted to CANDIDATE")
        elif state == "CANDIDATE":
            if hits >= config.CONFIRMATION_THRESHOLD:
                p["identity_state"] = "CONFIRMED"
                logger.info(f"[PersonRegistry] Person #{person_id} promoted to CONFIRMED")
        elif state == "LOST":
            p["identity_state"] = "REIDENTIFIED"
            p["visit_count"] += 1
            logger.info(f"[PersonRegistry] Person #{person_id} REIDENTIFIED (Welcome back)")
        elif state == "REIDENTIFIED":
            if hits >= config.CONFIRMATION_THRESHOLD:
                p["identity_state"] = "CONFIRMED"

    def handle_track_lost(self, person_id: int):
        """
        Transitions active/confirmed identities to LOST when their active tracks expire.
        """
        if person_id in self.persons:
            p = self.persons[person_id]
            if p["identity_state"] in ["NEW", "CANDIDATE", "CONFIRMED", "REIDENTIFIED"]:
                p["identity_state"] = "LOST"
                p["active_track_ids"] = []
                logger.info(f"[PersonRegistry] Person #{person_id} set to LOST")

    def archive_expired_persons(self, timeout_seconds: float = 7200.0):
        """
        Transitions LOST identities to ARCHIVED if they have not been seen for a long period.
        """
        now = datetime.now()
        for pid, p in self.persons.items():
            if p["identity_state"] == "LOST":
                last_seen_dt = datetime.fromisoformat(p["last_seen"])
                if (now - last_seen_dt).total_seconds() > timeout_seconds:
                    p["identity_state"] = "ARCHIVED"
                    logger.info(f"[PersonRegistry] Person #{pid} ARCHIVED (exceeded {timeout_seconds}s inactivity)")

    def merge_persons(self, src_id: int, dst_id: int):
        """
        Consolidates two duplicate registry profiles.
        """
        if src_id not in self.persons or dst_id not in self.persons:
            return
            
        src = self.persons[src_id]
        dst = self.persons[dst_id]
        
        # Merge stats
        dst["visit_count"] += src["visit_count"]
        
        # Keep earliest first_seen and latest last_seen
        src_first = datetime.fromisoformat(src["first_seen"])
        dst_first = datetime.fromisoformat(dst["first_seen"])
        dst["first_seen"] = min(src_first, dst_first).isoformat()
        
        src_last = datetime.fromisoformat(src["last_seen"])
        dst_last = datetime.fromisoformat(dst["last_seen"])
        dst["last_seen"] = max(src_last, dst_last).isoformat()
        
        # Camera history union
        for cam in src["camera_history"]:
            if cam not in dst["camera_history"]:
                dst["camera_history"].append(cam)
                
        # Merge track IDs
        for tid in src["active_track_ids"]:
            if tid not in dst["active_track_ids"]:
                dst["active_track_ids"].append(tid)
                
        # Delete source profile
        del self.persons[src_id]
