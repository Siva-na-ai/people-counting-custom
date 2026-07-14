import logging
from typing import Dict, Any, List, Optional
import time

logger = logging.getLogger(__name__)

class PersonRegistry:
    def __init__(self):
        # Maps person_id to metadata dict
        self.registry: Dict[str, Dict[str, Any]] = {}
        logger.info("Initialized PersonRegistry")
        
    def create_person(self, person_id: str, camera_id: str, track_id: int) -> Dict[str, Any]:
        """Creates a new person in the registry with NEW state."""
        now = time.time()
        metadata = {
            "person_id": person_id,
            "first_seen": now,
            "last_seen": now,
            "visit_count": 1,
            "camera_history": [camera_id],
            "active_track_ids": {track_id},
            "identity_state": "NEW",
            "confidence": 0.0,
            "face_count": 0,
            "body_count": 0
        }
        self.registry[person_id] = metadata
        logger.debug(f"Created new person: {person_id}")
        return metadata
        
    def update_person(self, person_id: str, track_id: int, camera_id: str, has_face: bool = False, has_body: bool = False):
        """Updates person stats."""
        if person_id not in self.registry:
            return
            
        person = self.registry[person_id]
        person["last_seen"] = time.time()
        person["active_track_ids"].add(track_id)
        
        if camera_id not in person["camera_history"]:
            person["camera_history"].append(camera_id)
            
        if has_face:
            person["face_count"] += 1
        if has_body:
            person["body_count"] += 1
            
        # State transitions
        if person["identity_state"] == "NEW" and (person["face_count"] > 3 or person["body_count"] > 3):
            person["identity_state"] = "CANDIDATE"
        elif person["identity_state"] == "CANDIDATE" and (person["face_count"] > 10 or person["body_count"] > 10):
            person["identity_state"] = "CONFIRMED"
            
    def get_person(self, person_id: str) -> Optional[Dict[str, Any]]:
        return self.registry.get(person_id)
        
    def merge_persons(self, primary_id: str, secondary_id: str):
        """Merge secondary person into primary person."""
        if primary_id not in self.registry or secondary_id not in self.registry:
            return
            
        primary = self.registry[primary_id]
        secondary = self.registry[secondary_id]
        
        primary["last_seen"] = max(primary["last_seen"], secondary["last_seen"])
        primary["first_seen"] = min(primary["first_seen"], secondary["first_seen"])
        primary["visit_count"] += secondary["visit_count"]
        
        for cam in secondary["camera_history"]:
            if cam not in primary["camera_history"]:
                primary["camera_history"].append(cam)
                
        primary["active_track_ids"].update(secondary["active_track_ids"])
        primary["face_count"] += secondary["face_count"]
        primary["body_count"] += secondary["body_count"]
        
        # Promote state if necessary
        if secondary["identity_state"] == "CONFIRMED":
            primary["identity_state"] = "CONFIRMED"
            
        # Remove secondary
        del self.registry[secondary_id]
        logger.info(f"Merged {secondary_id} into {primary_id}")
        
    def cleanup_archived(self, expiration_seconds: float):
        """Transition old persons to ARCHIVED or remove them."""
        now = time.time()
        to_archive = []
        for pid, p in self.registry.items():
            if now - p["last_seen"] > expiration_seconds:
                p["identity_state"] = "ARCHIVED"
                p["active_track_ids"].clear()
                to_archive.append(pid)
        return to_archive
