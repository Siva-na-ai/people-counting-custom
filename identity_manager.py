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
    resolves identity with the FusionEngine, and offloads database IO to background workers.
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
            return None, "TENTATIVE", 0.0

        # Retrieve or initialize the track's temporal state machine
        t_state = self.fusion.temporal_validator.get_state(track_id)
        if t_state is None:
            from temporal_validator import TrackIdentityState
            t_state = TrackIdentityState(track_id)
            self.fusion.temporal_validator.track_states[track_id] = t_state

        # Initialize cooldown if not present
        if not hasattr(t_state, "search_cooldown"):
            t_state.search_cooldown = 0

        # 1. Enforce Track Identity Lock (skip all searches & models if PID is confirmed and locked)
        if config.TRACK_LOCK_ENABLED and t_state.state in ("TRACK_LOCKED", "CONFIRMED") and t_state.confirmed_id is not None:
            person_id = t_state.confirmed_id
            p_state = t_state.state
            confidence = t_state.identity_confidence
            
            # Keep registry updated
            self.registry.update_person(
                person_id, camera_id, track_id, 
                len(self.gallery_mgr.get_face_embeddings(person_id)),
                len(self.gallery_mgr.get_body_embeddings(person_id)),
                confidence, hits=5
            )
            self.registry.persons[person_id]["last_box"] = det_box.copy()
            
            # Simple metadata update
            self.worker_pool.submit_task(self._async_update_metadata_db, person_id)
            return person_id, p_state, confidence

        # 2. Enforce Search Cooldown (if cooldown active, skip Qdrant/embedding logic and just track visually)
        if t_state.search_cooldown > 0:
            t_state.search_cooldown -= 1
            return None, t_state.state, t_state.identity_confidence

        face_emb = None
        body_emb = None
        face_quality = 0.0
        body_quality = 0.0
        face_ok = False
        
        # Face quality metrics
        blur_score = 0.0
        yaw, pitch, roll = 0.0, 0.0, 0.0
        face_size = 0
        face_conf = 0.0
        quality_reason = "No face detected"
        
        # 3. Run SCRFD Face Detector inside person crop
        face_dets = self.scrfd.detect(person_crop, threshold=0.55)
        logger.debug(f"[Track {track_id}] SCRFD: {len(face_dets)} face(s) detected in person crop {person_crop.shape}")

        if len(face_dets) > 0:
            best_face = max(face_dets, key=lambda f: f["score"])
            face_bbox = best_face["bbox"]
            landmarks = best_face["landmarks"]
            face_conf = best_face["score"]

            fx1 = max(0, int(face_bbox[0]))
            fy1 = max(0, int(face_bbox[1]))
            fx2 = min(person_crop.shape[1], int(face_bbox[2]))
            fy2 = min(person_crop.shape[0], int(face_bbox[3]))
            face_crop = person_crop[fy1:fy2, fx1:fx2]
            face_size = min(face_crop.shape[0], face_crop.shape[1])

            landmarks_in_face = landmarks - np.array([fx1, fy1], dtype=np.float32)
            
            # Calculate coordinates in parent frame to verify if face is fully inside boundaries
            parent_fx1 = x1 + fx1
            parent_fy1 = y1 + fy1
            parent_fx2 = x1 + fx2
            parent_fy2 = y1 + fy2
            face_box_parent = np.array([parent_fx1, parent_fy1, parent_fx2, parent_fy2])

            # Call strict quality gate
            face_ok, face_quality, blur_score, yaw, pitch, roll, quality_reason = evaluate_face_quality(
                face_crop, face_conf, landmarks_in_face,
                parent_w=w, parent_h=h, face_box_coords=face_box_parent
            )

            if face_ok:
                aligned_face = align_face(person_crop, landmarks)
                face_emb = self.arcface.extract_embedding(aligned_face)
            else:
                logger.debug(f"[Track {track_id}] Face quality check FAILED: {quality_reason}")

        # 4. Run Body Pipeline (Extract body embedding if quality is ok)
        body_ok, body_quality, _ = evaluate_body_quality(person_crop, det_score)
        if body_ok:
            body_emb = self.repvgg.infer(person_crop)

        # Update consecutive good face confirmations counter
        if face_ok and face_emb is not None:
            t_state.good_face_confirmations += 1
            logger.debug(f"[Track {track_id}] Good face count: {t_state.good_face_confirmations}")
        else:
            if t_state.state not in ("CONFIRMED", "TRACK_LOCKED", "REIDENTIFIED"):
                t_state.good_face_confirmations = 0

        # 5. Resolve Identity (using Qdrant matcher and state validator)
        person_id, confidence, match_details = self.fusion.resolve_identity(
            track_id, face_emb, body_emb, body_quality, det_box, time_since_update, w, h,
            next_person_id_callback,
            face_quality_passed=face_ok
        )

        p_state = t_state.state

        # 6. Apply Search Cooldown on actual search failure / rejection
        if person_id is None and t_state.good_face_confirmations >= config.GOOD_FACE_CONFIRMATIONS:
            t_state.search_cooldown = config.SEARCH_RETRY_INTERVAL
            t_state.good_face_confirmations = 0

        latency = (time.time() - start_time) * 1000.0

        # 7. Logging ReID decisions
        decision = match_details.get("decision", "Rejected")
        reason = match_details.get("reason", quality_reason)
        logger.info(
            f"[ReID Decision] Track={track_id} | State={p_state} | "
            f"Current PID={t_state.confirmed_id} | Candidate PID={match_details.get('top1_pid')} | "
            f"FaceSim={match_details.get('top1_score'):.3f} | BodySim={match_details.get('body_sim'):.3f} | "
            f"Fusion={match_details.get('fusion_score'):.3f} | Top1={match_details.get('top1_score'):.3f} | "
            f"Top2={match_details.get('top2_score'):.3f} | Gap={match_details.get('gap'):.3f} | "
            f"Size={face_size} | Blur={blur_score:.1f} | Yaw={yaw:.1f} | Pitch={pitch:.1f} | Roll={roll:.1f} | "
            f"Decision={decision} | Reason={reason} | Latency={latency:.1f}ms"
        )

        if person_id is None:
            return None, p_state, confidence

        # 8. Update Person Registry Metadata
        self.registry.update_person(
            person_id, camera_id, track_id, 
            len(self.gallery_mgr.get_face_embeddings(person_id)),
            len(self.gallery_mgr.get_body_embeddings(person_id)),
            confidence, hits=5
        )
        self.registry.persons[person_id]["last_box"] = det_box.copy()
        
        # 9. Asynchronously update vector galleries and Qdrant DB if stable/confirmed
        # Enforce Template Protection: never save templates unless the track is confirmed or locked,
        # and similarity matches our FACE_TEMPLATE_UPDATE rules, and the face quality is verified
        is_stable_state = p_state in ("CONFIRMED", "TRACK_LOCKED", "REIDENTIFIED")
        
        if not is_occluded and is_stable_state:
            # Only save a new face template if:
            # - Face quality passed.
            # - It's a new person OR similarity to existing template is high (>= FACE_TEMPLATE_UPDATE)
            is_new_person = (match_details.get("top1_score", 0.0) == 0.0)
            is_good_template = match_details.get("top1_score", 0.0) >= config.FACE_TEMPLATE_UPDATE
            
            if face_emb is not None and face_ok and (is_new_person or is_good_template):
                brightness = float(np.mean(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)))
                self.worker_pool.submit_task(
                    self._async_update_face_gallery, person_id, face_emb, face_quality, camera_id,
                    yaw, pitch, roll, blur_score, brightness
                )
                
            # Only save body templates for verified/stable tracks
            if body_emb is not None and body_ok:
                self.worker_pool.submit_task(
                    self._async_update_body_gallery, person_id, body_emb, body_quality, camera_id
                )
        
        self.worker_pool.submit_task(self._async_update_metadata_db, person_id)
        
        return person_id, p_state, confidence

    def _async_update_face_gallery(
        self, 
        person_id: int, 
        embedding: np.ndarray, 
        quality: float, 
        camera_id: int,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
        blur: float = 0.0,
        brightness: float = 128.0
    ):
        gallery = self.gallery_mgr.get_or_create_gallery(person_id)
        before_count = len(gallery.faces)

        # 1. Update local gallery manager (runs diversity checks)
        inserted = self.gallery_mgr.add_face(person_id, embedding, quality, yaw, pitch, roll, brightness)
        after_count = len(gallery.faces)

        if inserted and after_count > before_count:
            logger.info(f"[IdentityManager] Person #{person_id} face gallery: {before_count} → {after_count} templates")
            if self.qdrant:
                # 2. Write to Qdrant Face Collection with separate metadata and versioning
                point_id = str(uuid.uuid4())
                ok = self.qdrant.upsert_point(
                    "face_embeddings",
                    point_id,
                    embedding.tolist(),
                    {
                        "person_id": person_id,
                        "quality_score": quality,
                        "timestamp": time.time(),
                        "camera_id": camera_id,
                        "yaw": yaw,
                        "pitch": pitch,
                        "roll": roll,
                        "blur": blur,
                        "brightness": brightness,
                        "embedding_version": "arcface_mobilefacenet_h8l_v1"
                    }
                )
                if ok:
                    logger.info(f"[IdentityManager] Person #{person_id} face template saved to Qdrant ✓ (id={point_id[:8]}...)")
                else:
                    logger.warning(f"[IdentityManager] Person #{person_id} face Qdrant upsert FAILED")

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
                    "camera_id": camera_id,
                    "embedding_version": "repvgg_a0_v1"
                }
            )

    def _async_update_metadata_db(self, person_id: int):
        if self.qdrant and person_id in self.registry.persons:
            p = self.registry.persons[person_id]
            self.qdrant.upsert_point(
                "person_registry",
                person_id,
                [0.0],
                p
            )
