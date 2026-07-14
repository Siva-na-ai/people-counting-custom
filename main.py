import cv2
import numpy as np
import time
from face_detector import FaceDetector
from face_alignment import FaceAlignment
from face_recognition import FaceRecognition
from tracker import Tracker
from identity_manager import IdentityManager
from qdrant_db import QdrantIdentityClient
from embedding_quality import EmbeddingQuality
import config

class PipelineRunner:
    def __init__(self):
        print("[*] Initializing AI Models...")
        self.face_detector = FaceDetector(threshold=config.DETECTION_CONFIDENCE)
        self.face_aligner = FaceAlignment()
        self.face_rec = FaceRecognition()
        self.quality = EmbeddingQuality()
        
        print("[*] Initializing Identity System (Local Qdrant Mode)...")
        self.qdrant = QdrantIdentityClient(path="qdrant_local_db")
        self.id_manager = IdentityManager(self.qdrant)
        self.tracker = Tracker(max_age=config.TRACKING_MAX_AGE, min_hits=config.TRACKING_MIN_HITS)
        
        self.camera_id = "cam_01"
        
    def run(self, source=0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
            
        print("[*] Streaming started...")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Failed to read frame from camera. (Is the camera plugged in and accessible?)")
                    break
                    
                # 1. Detection
                faces = self.face_detector.detect(frame)
                
                # Convert faces to format expected by tracker
                detections = []
                for face in faces:
                    detections.append({
                        'bbox': face['bbox'],
                        'score': face['det_score'],
                        'class_id': 0,
                        'kps': face['kps'] # Inject kps for alignment later
                    })
                    
                # 2. Tracking
                tracks = self.tracker.update(detections)
                
                # 3. Recognition & Identity Fusion
                for track in tracks:
                    bbox = track.bbox
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Crop face
                    face_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                    
                    # Quality Check
                    q_score = self.quality.assess_crop_quality(face_crop)
                    
                    face_emb = None
                    # Find kps from detections if matched (naive match for demo)
                    matched_kps = None
                    for d in detections:
                        if d['bbox'] == track.bbox:
                            matched_kps = d.get('kps')
                            break
                            
                    if q_score >= config.MIN_FACE_QUALITY and matched_kps is not None:
                        aligned = self.face_aligner.align(frame, matched_kps)
                        if aligned is not None:
                            face_emb = self.face_rec.get_embedding(aligned)
                            
                    # Body ReID would go here
                    body_emb = None 
                    body_q_score = 0.0

                    # 4. Identity Management
                    person_id, conf = self.id_manager.process_track(
                        track.track_id, 
                        self.camera_id,
                        face_emb, body_emb,
                        q_score, body_q_score
                    )
                    
                    # 5. Drawing
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"ID:{str(person_id)[:6]} ({conf:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow("Face + Body Identity Fusion Pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.id_manager.worker.shutdown()

if __name__ == "__main__":
    runner = PipelineRunner()
    runner.run(0)
