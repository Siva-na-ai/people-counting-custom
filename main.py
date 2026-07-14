import os
import cv2
import time
import argparse
import logging
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import threading
import numpy as np

# Import custom pipeline modules
import config
from qdrant_client import QdrantClient
from worker_pool import WorkerPool
from gallery_manager import GalleryManager
from person_registry import PersonRegistry
from movement_validator import MovementValidator
from temporal_validator import TemporalValidator
from embedding_cache import EmbeddingCache
from event_logger import EventLogger
from face_detector import HailoFaceDetector
from face_recognition import HailoFaceRecognizer
from tracker import BoTSORTTracker, Track
from identity_matcher import IdentityMatcher
from fusion_engine import FusionEngine
from identity_manager import IdentityManager

# Import ReID class from modified body script
from track_stream_reid import HailoReID

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("main_runner")

class PipelineRunner:
    def __init__(self, args):
        self.args = args
        self.camera_id = args.camera_id
        
        # 1. Initialize DB and Background Thread Pool
        self.qdrant = QdrantClient(url=args.qdrant_url)
        self.worker_pool = WorkerPool()
        self.gallery_mgr = GalleryManager()
        self.registry = PersonRegistry(qdrant_client=self.qdrant)
        self.event_logger = EventLogger()
        
        # Lock for registry state synchronization
        self.registry.lock = threading.Lock()
        
        # 2. Initialize Validators and Caching
        self.movement_val = MovementValidator()
        self.temporal_val = TemporalValidator()
        self.cache = EmbeddingCache()
        
        # 3. Initialize Identity Decision Engines
        self.matcher = IdentityMatcher(self.qdrant, self.movement_val, self.registry)
        self.fusion = FusionEngine(self.matcher, self.temporal_val, self.registry)
        
        # 4. Load Hailo HEF Engines
        logger.info("Initializing Hailo-8L accelerators...")
        self.body_reid = HailoReID(hef_path=config.REPVGG_HEF_PATH)
        self.face_det = HailoFaceDetector(hef_path=config.SCRFD_HEF_PATH)
        self.face_rec = HailoFaceRecognizer(hef_path=config.ARCFACE_HEF_PATH)
        
        # 5. Core Tracker
        self.tracker = BoTSORTTracker(max_age=config.MAX_AGE)
        
        # 6. Pipeline Coordinator
        self.identity_mgr = IdentityManager(
            self.qdrant, self.worker_pool, self.gallery_mgr, self.registry, self.fusion,
            self.face_det, self.face_rec, self.body_reid, self.cache, self.event_logger
        )
        
        # Person counter ID source
        self.next_pid = 1

    def get_next_person_id(self) -> int:
        pid = self.next_pid
        self.next_pid += 1
        return pid

    def process_frame(self, frame: np.ndarray, yolo_detections: list) -> np.ndarray:
        """
        Processes a single frame: runs YOLO -> BoTSORT Tracker -> Face/Body ReID Fusion.
        Returns the annotated frame image.
        """
        img_h, img_w = frame.shape[:2]
        
        # Convert YOLO detections to tracker format: List[Dict[str, Any]]
        # Filter classes to keep only person (usually class_id = 0 in COCO)
        person_detections = []
        for det in yolo_detections:
            box, score, class_id = det["box"], det["score"], det["class_id"]
            if class_id == 0:  # COCO Person class
                person_detections.append({
                    "bbox": box,
                    "score": score,
                    "class_id": class_id
                })
                
        # Update BoTSORT Spatial Tracker (Step 3 Hungarian mapping)
        matched_pairs, unmatched_dets = self.tracker.update(person_detections, img_w, img_h)
        
        # Active tracks set for cache cleanup
        active_tids = []
        
        # Process matched spatial tracks
        for pair in matched_pairs:
            track = pair["track"]
            det = pair["detection"]
            active_tids.append(track.track_id)
            
            # Reresolve identity or update existing track status
            if track.person_id is None:
                # First time matching, run full fusion
                pid, state, conf = self.identity_mgr.process_observation(
                    frame, track.track_id, track.box, track.score, track.time_since_update, self.camera_id, self.get_next_person_id
                )
                track.person_id = pid
            else:
                # Update existing track (offload gallery updates asynchronously)
                pid, state, conf = self.identity_mgr.process_observation(
                    frame, track.track_id, track.box, track.score, track.time_since_update, self.camera_id, lambda: track.person_id
                )
                
            # Draw labels
            cv2.rectangle(frame, (int(track.box[0]), int(track.box[1])), (int(track.box[2]), int(track.box[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"Person #{track.person_id} [{state}]", (int(track.box[0]), int(track.box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        # Process unmatched detections (create new tracks or match global gallery)
        for det in unmatched_dets:
            # Create new temporary spatial track
            new_track = self.tracker.create_track(det["bbox"], det["score"], det["class_id"])
            active_tids.append(new_track.track_id)
            
            # Resolve global identity
            pid, state, conf = self.identity_mgr.process_observation(
                frame, new_track.track_id, new_track.box, new_track.score, new_track.time_since_update, self.camera_id, self.get_next_person_id
            )
            new_track.person_id = pid
            
            # Draw labels
            cv2.rectangle(frame, (int(new_track.box[0]), int(new_track.box[1])), (int(new_track.box[2]), int(new_track.box[3])), (0, 165, 255), 2)
            cv2.putText(frame, f"Person #{new_track.person_id} [{state}]", (int(new_track.box[0]), int(new_track.box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Cleanup cache entry for expired tracks
        self.cache.tick(active_tids)
        
        # Registry offline transition for lost tracks
        # Track ID to Person ID cross reference check
        active_pids = {t.person_id for t in self.tracker.tracks if t.person_id is not None}
        for pid in list(self.registry.persons.keys()):
            if pid not in active_pids:
                self.registry.handle_track_lost(pid)
                
        # Periodic duplicate check (delegated to worker pool)
        self.worker_pool.submit_task(self._offline_resolve_duplicates)
        
        return frame

    def _offline_resolve_duplicates(self):
        active_pids = [t.person_id for t in self.tracker.tracks if t.person_id is not None]
        resolver = DuplicateIdentityResolver(self.registry, self.gallery_mgr, self.qdrant)
        merges = resolver.check_and_resolve_duplicates(active_pids)
        for src, dst in merges:
            logger.info(f"[Offline Worker] Consolidated duplicate Identity: Person #{src} merged into Person #{dst}")

    def close(self):
        logger.info("Cleaning up resources...")
        self.worker_pool.stop()
        self.body_reid.close()
        self.face_det.close()
        self.face_rec.close()
