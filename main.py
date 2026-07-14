import cv2
import time
import argparse
import logging
from qdrant_client import QdrantClientWrapper
from config import QDRANT_HOST, QDRANT_PORT
from identity_manager import IdentityManager
from worker_pool import WorkerPool
from event_logger import EventLogger
from tracker import BoTSORTTracker
from modlib.devices import AiCamera
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", type=str, default="cam_01")
    return parser.parse_args()

def main():
    args = get_args()
    
    # Initialize Core Components
    db = QdrantClientWrapper(QDRANT_HOST, QDRANT_PORT)
    db.initialize_collections()
    
    identity_manager = IdentityManager(db)
    worker_pool = WorkerPool(num_workers=2)
    event_logger = EventLogger()
    
    # Camera setup
    device = AiCamera()
    
    # Monkey-patch device.deploy to prevent modlib bug
    original_deploy = device.deploy
    def patched_deploy(model_obj, camera_id=None, *args, **kwargs):
        if camera_id is None:
            camera_id = ""
        device.camera_id = camera_id
        return original_deploy(model_obj, *args, **kwargs)
    device.deploy = patched_deploy
    
    device.camera_id = ""
    model = SSDMobileNetV2FPNLite320x320()
    device.deploy(model)

    tracker = BoTSORTTracker(reid_model_name='osnet_x1_0', max_age=900)
    
    logger.info("Starting Face + Body Identity Pipeline")
    
    try:
        with device as stream:
            for frame in stream:
                # 1. Detection
                detections = frame.detections[frame.detections.confidence > 0.55]
                detections = detections[detections.class_id == 0]  # Person
                
                # 2. Tracking & ReID
                detections = tracker.update(frame.image, detections)
                
                # 3. Identity Workflow
                for idx, (_, s, c, track_id) in enumerate(detections):
                    # Here we would extract body and face crops and run them through their respective models.
                    # Since this is the orchestrated loop, we pass mock qualities to the manager.
                    body_emb = None  # Mock
                    face_emb = None  # Mock
                    
                    # Pass the detection to identity manager
                    person_id = identity_manager.process_detection(
                        track_id=track_id, 
                        camera_id=args.camera_id,
                        face_emb=face_emb,
                        body_emb=body_emb,
                        face_quality=0.8,
                        body_quality=0.8
                    )
                    
                # Clean up lost tracks (simplified for example)
                # identity_manager.end_track(track_id)
                
                # Render (Mock)
                try:
                    frame.display()
                except Exception:
                    pass
                    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        worker_pool.shutdown()

if __name__ == "__main__":
    main()
