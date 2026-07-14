import uuid
from datetime import datetime

class PersonState:
    NEW = "NEW"
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"
    REIDENTIFIED = "REIDENTIFIED"
    ARCHIVED = "ARCHIVED"

class PersonRegistry:
    def __init__(self, timeout_sec=300):
        self.persons = {}
        self.timeout_sec = timeout_sec

    def create_person(self, track_id, camera_id):
        person_id = str(uuid.uuid4())
        now = datetime.utcnow()
        self.persons[person_id] = {
            "person_id": person_id,
            "state": PersonState.NEW,
            "first_seen": now,
            "last_seen": now,
            "visit_count": 1,
            "camera_history": [camera_id],
            "active_track_ids": {track_id}
        }
        return person_id

    def update_person(self, person_id, track_id, camera_id):
        if person_id not in self.persons:
            return False
            
        person = self.persons[person_id]
        person["last_seen"] = datetime.utcnow()
        person["active_track_ids"].add(track_id)
        
        if camera_id not in person["camera_history"]:
            person["camera_history"].append(camera_id)
            
        # State transitions
        if person["state"] == PersonState.NEW:
            person["state"] = PersonState.CANDIDATE
        elif person["state"] == PersonState.CANDIDATE:
            person["state"] = PersonState.CONFIRMED
        elif person["state"] == PersonState.LOST:
            person["state"] = PersonState.REIDENTIFIED
            person["visit_count"] += 1
            
        return True

    def mark_lost(self, person_id):
        if person_id in self.persons:
            self.persons[person_id]["state"] = PersonState.LOST

    def cleanup_stale_persons(self):
        """Archive persons who haven't been seen for timeout_sec"""
        now = datetime.utcnow()
        archived = []
        for pid, pdata in list(self.persons.items()):
            dt = (now - pdata["last_seen"]).total_seconds()
            if dt > self.timeout_sec:
                pdata["state"] = PersonState.ARCHIVED
                archived.append(pid)
                # In a real app we might pop them from memory entirely 
                # or move to a long-term DB
        return archived
