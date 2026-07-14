import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Set up logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("event_logger")

class EventLogger:
    """
    Asynchronous-friendly production logger for tracking and reidentification events.
    Appends events to a local JSON Lines (JSONL) file for auditability.
    """
    def __init__(self, log_path: str = "identity_events.jsonl"):
        self.log_path = log_path
        
    def log_event(
        self, 
        event_type: str, 
        person_id: int, 
        camera_id: int, 
        confidence: float, 
        processing_time_ms: float, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Formats and logs the identity event.
        Can be safely delegated to the WorkerPool to avoid disk IO blocks.
        """
        payload = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "person_id": person_id,
            "camera_id": camera_id,
            "confidence": float(confidence),
            "processing_time_ms": float(processing_time_ms),
            "metadata": metadata or {}
        }
        
        # Log to standard output/syslog
        msg = (
            f"[Event: {event_type}] Person #{person_id} | Camera {camera_id} | "
            f"Conf: {confidence:.2f} | Latency: {processing_time_ms:.1f}ms"
        )
        if metadata:
            msg += f" | Meta: {metadata}"
        logger.info(msg)
        
        # Append to file
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            logger.error(f"Failed to write event to file: {e}")
