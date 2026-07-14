import logging
import json
import os
from datetime import datetime

class EventLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.logger = logging.getLogger("IdentityPipeline")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = os.path.join(log_dir, f"identity_events_{datetime.now().strftime('%Y%m%d')}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_event(self, event_type, person_id, metadata=None):
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "person_id": person_id,
            "metadata": metadata or {}
        }
        self.logger.info(json.dumps(event))
        
    def log_creation(self, person_id):
        self.log_event("CREATED", person_id)
        
    def log_reidentification(self, person_id, old_track_id, new_track_id):
        self.log_event("REIDENTIFIED", person_id, {"old_track": old_track_id, "new_track": new_track_id})
        
    def log_merge(self, primary_id, secondary_id):
        self.log_event("MERGED", primary_id, {"merged_with": secondary_id})
        
    def log_lost(self, person_id, last_camera_id):
        self.log_event("LOST", person_id, {"last_camera_id": last_camera_id})
