import logging
import json
import time

logger = logging.getLogger(__name__)

class EventLogger:
    def __init__(self, log_file="events.log"):
        self.log_file = log_file

    def log_event(self, event_type: str, person_id: str, metadata: dict = None):
        """
        Logs an identity event.
        Types: Created, Updated, Reidentified, Merged, Left, Returned
        """
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "person_id": person_id,
            "metadata": metadata or {}
        }
        
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
