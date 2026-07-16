import logging
import time
import uuid
import numpy as np
from typing import Dict, Any, Tuple, Optional
import config
from embedding_quality import evaluate_face_quality, evaluate_body_quality
from face_alignment import align_face

logger = logging.getLogger("identity_manager")

class IdentityManager:
    """
    Coordinator controlling the overall identity workflow.
    Manages cache lookup, routes crops through face and body pipelines,
    resolves identity with the FusionEngine, and offloads heavy database IO to background workers.
    """
    def __init__(
        self,
        qdrant,
        worker_pool,
        gallery_mgr,
        registry,
        fusion_engine,
        scrfd_detector,
        arcface_recognizer,
        repvgg_reid,
        embedding_cache,
        event_logger
    ):
        self.qdrant = qdrant
        self.worker_pool = worker_pool
        self.gallery_mgr = gallery_mgr
        self.registry = registry
        self.fusion = fusion_engine
        
        # Hailo models
        self.scrfd = scrfd_detector
        self.arcface = arcface_recognizer
        self.repvgg = repvgg_reid
        
        # Cache and logging
        self.cache = embedding_cache
        self.logger = event_logger

    def process_observation(
        self,
        frame_bgr: np.ndarray,
        track_id: int,
        det_box: np.ndarray,
        det_score: float,
        time_since_update: int,
        camera_id: int,
        next_person_id_callback: callable,
        is_occluded: bool = False,
        current_person_id: Optional[int] = None
    ) -> Tuple[Optional[int], str, float]:
        """
        Coordinates face/body pipelines and decides final Person ID, State, and Confidence.
        """
        start_time = time.time()
        
        # Crop person from frame
        h, w = frame_bgr.shape[:2]
        x1 = max(0, int(det_box[0]))
        y1 = max(0, int(det_box[1]))
        x2 = min(w, int(det_box[2]))
        y2 = min(h, int(det_box[3]))
        person_crop = frame_bgr[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return None, "Tentative", 0.0
            
        face_emb = None
        body_emb = None
        face_quality = 0.0
        body_quality = 0.0
        face_ok = False
        
        # If this track is already verified as a positive Person ID:
        # We reuse the local mapping directly (skipping Qdrant search to match)
        # But we still run face detection to potentially update/enrich database templates
        existing_pid = current_person_id if (current_person_id is not None and current_person_id > 0) else None
        
        # 1. Run SCRFD Face Detector inside person crop
        face_dets = self.scrfd.detect(person_crop, threshold=0.55)
        if len(face_dets) > 0:
            # Process the highest score face
            best_face = max(face_dets, key=lambda f: f["score"])
            face_bbox = best_face["bbox"]
            landmarks = best_face["landmarks"]  # in person_crop coordinates
            
            # Extract face crop from person crop
            fx1 = max(0, int(face_bbox[0]))
            fy1 = max(0, int(face_bbox[1]))
            fx2 = min(person_crop.shape[1], int(face_bbox[2]))
            fy2 = min(person_crop.shape[0], int(face_bbox[3]))
            face_crop = person_crop[fy1:fy2, fx1:fx2]
            
            # Transform landmarks from person_crop coords → face_crop coords.
            # Without this, boundary checks in evaluate_face_quality compare
            # person_crop landmark values against face_crop width/height → always fail.
            import numpy as np
            landmarks_in_face = landmarks - np.array([fx1, fy1], dtype=np.float32)
            
            face_ok, face_quality, _ = evaluate_face_quality(
                face_crop, best_face["score"], landmarks_in_face
            )
            
            if face_ok:
                # Align and extract ArcFace features (uses original person_crop landmarks)
                aligned_face = align_face(person_crop, landmarks)
                face_emb = self.arcface.extract_embedding(aligned_face)
                
        # Evaluate body quality
        body_ok, body_quality, _ = evaluate_body_quality(person_crop, det_score)
        
        # 2. RUN BODY PIPELINE
        # Only run Body ReID if config.USE_BODY_REID is enabled, OR if we are restoring an existing Person ID
        if (config.USE_BODY_REID or existing_pid is not None) and body_ok:
            body_emb = self.repvgg.infer(person_crop)
            
        # 3. Resolve identity
        if existing_pid is not None:
            # Reusing the local mapping, skipping Qdrant search for matching
            person_id = existing_pid
            confidence = 1.0
            p_state = "CONFIRMED"
        else:
            # Run FusionEngine to resolve final Person ID (searches Qdrant)
            # face_detected=True even if face_emb is None (quality check may have failed)
            person_id, confidence = self.fusion.resolve_identity(
                track_id, face_emb, body_emb, body_quality, det_box, time_since_update, w, h,
                next_person_id_callback,
                face_detected=(len(face_dets) > 0)
            )
            
        if person_id is None:
            return None, "Tentative", 0.0
            
        # 4. Update Person Registry Metadata and state machine
        hits = 5
        self.registry.update_person(
            person_id, camera_id, track_id, 
            len(self.gallery_mgr.get_face_embeddings(person_id)),
            len(self.gallery_mgr.get_body_embeddings(person_id)),
            confidence, hits
        )
        self.registry.persons[person_id]["last_box"] = det_box.copy()
        p_state = self.registry.persons[person_id]["identity_state"]
        
        # 5. Asynchronously update vector galleries and Qdrant DB if not occluded
        latency = (time.time() - start_time) * 1000.0
        
        if not is_occluded:
            if face_emb is not None and face_quality > 0.50:
                self.worker_pool.submit_task(
                    self._async_update_face_gallery, person_id, face_emb, face_quality, camera_id
                )
                
            if body_emb is not None and body_quality > 0.40:
                self.worker_pool.submit_task(
                    self._async_update_body_gallery, person_id, body_emb, body_quality, camera_id
                )
        else:
            logger.info(f"[IdentityManager] Gallery update frozen for Person #{person_id} due to occlusion.")
            
        self.worker_pool.submit_task(
            self._async_update_metadata_db, person_id
        )
        self.worker_pool.submit_task(
            self.logger.log_event, "Person_Updated", person_id, camera_id, confidence, latency
        )
        
        return person_id, p_state, confidence

    def _async_update_face_gallery(self, person_id: int, embedding: np.ndarray, quality: float, camera_id: int):
        # Retrieve existing face templates for this person
        gallery = self.gallery_mgr.get_or_create_gallery(person_id)
        
        # If we already have 3 or more face templates, check if the new face quality is better than average
        if len(gallery.faces) >= 3:
            avg_quality = np.mean([f["quality"] for f in gallery.faces])
            if quality <= avg_quality:
                logger.info(f"[IdentityManager] Ignoring lower-quality face template ({quality:.2f} <= average {avg_quality:.2f}) for Person #{person_id}.")
                return
                
        # 1. Update local gallery manager
        inserted = self.gallery_mgr.add_face(person_id, embedding, quality)
        if inserted and self.qdrant:
            # 2. Write to Qdrant Face Collection
            point_id = str(uuid.uuid4())
            self.qdrant.upsert_point(
                "face_embeddings", 
                point_id, 
                embedding.tolist(), 
                {
                    "person_id": person_id,
                    "quality_score": quality,
                    "timestamp": time.time(),
                    "camera_id": camera_id
                }
            )

    def _async_update_body_gallery(self, person_id: int, embedding: np.ndarray, quality: float, camera_id: int):
        inserted = self.gallery_mgr.add_body(person_id, embedding, quality)
        if inserted and self.qdrant:
            point_id = str(uuid.uuid4())
            self.qdrant.upsert_point(
                "body_embeddings", 
                point_id, 
                embedding.tolist(), 
                {
                    "person_id": person_id,
                    "quality_score": quality,
                    "timestamp": time.time(),
                    "camera_id": camera_id
                }
            )

    def _async_update_metadata_db(self, person_id: int):
        if self.qdrant and person_id in self.registry.persons:
            p = self.registry.persons[person_id]
            # Write metadata point (using 1D dummy vector [0.0] for compatibility)
            self.qdrant.upsert_point(
                "person_metadata",
                person_id,
                [0.0],
                p
            )
