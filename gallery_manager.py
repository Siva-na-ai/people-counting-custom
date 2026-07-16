import numpy as np
import threading
from datetime import datetime
from typing import List, Dict, Any, Tuple
import config

class PersonGallery:
    """
    Template gallery for a single person holding face and body embeddings.
    """
    def __init__(self, person_id: int):
        self.person_id = person_id
        self.faces: List[Dict[str, Any]] = []   # Elements: {"embedding": np.ndarray, "quality": float, "time": datetime}
        self.bodies: List[Dict[str, Any]] = []  # Elements: {"embedding": np.ndarray, "quality": float, "time": datetime}

class GalleryManager:
    """
    Centralized gallery manager supporting similarity-based clustering,
    duplicate elimination, and age-decay quality pruning.
    """
    def __init__(self):
        self.galleries: Dict[int, PersonGallery] = {}
        self.lock = threading.Lock()

    def get_or_create_gallery(self, person_id: int) -> PersonGallery:
        with self.lock:
            if person_id not in self.galleries:
                self.galleries[person_id] = PersonGallery(person_id)
            return self.galleries[person_id]

    def add_face(
        self, 
        person_id: int, 
        embedding: np.ndarray, 
        quality_score: float,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
        brightness: float = 128.0
    ) -> bool:
        """
        Adds a face embedding to the person's local gallery.
        Deduplicates templates: rejects identical poses/lighting (sim > 0.93, pose diff < 10 deg, bright diff < 15).
        """
        if embedding is None:
            return False
            
        gallery = self.get_or_create_gallery(person_id)
        now = datetime.now()
        
        with self.lock:
            # 1. Cluster / deduplicate check
            duplicate_idx = -1
            for idx, item in enumerate(gallery.faces):
                sim = np.dot(embedding, item["embedding"])
                if sim > 0.93:
                    # Enforce strict gallery diversity check (sim > 0.93, pose < 10, lighting < 15)
                    dyaw = abs(yaw - item.get("yaw", 0.0))
                    dpitch = abs(pitch - item.get("pitch", 0.0))
                    droll = abs(roll - item.get("roll", 0.0))
                    dbright = abs(brightness - item.get("brightness", 128.0))
                    if dyaw < 10.0 and dpitch < 10.0 and droll < 10.0 and dbright < 15.0:
                        duplicate_idx = idx
                        break
                    
            if duplicate_idx >= 0:
                # Update existing cluster template if the new one is of higher quality
                if quality_score > gallery.faces[duplicate_idx]["quality"]:
                    gallery.faces[duplicate_idx] = {
                        "embedding": embedding,
                        "quality": quality_score,
                        "time": now,
                        "yaw": yaw,
                        "pitch": pitch,
                        "roll": roll,
                        "brightness": brightness
                    }
                return True
                
            # 2. Add new template view
            gallery.faces.append({
                "embedding": embedding,
                "quality": quality_score,
                "time": now,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "brightness": brightness
            })
            
            # 3. Prune if count exceeds limit (20)
            if len(gallery.faces) > config.FACE_GALLERY_MAX:
                # Value function: weight quality heavily but penalize extreme age
                def get_value(item):
                    age_hours = (now - item["time"]).total_seconds() / 3600.0
                    return item["quality"] - (0.05 * age_hours)
                    
                # Evict the item with the lowest value
                gallery.faces.sort(key=get_value)
                gallery.faces.pop(0)
                
            return True

    def add_body(self, person_id: int, embedding: np.ndarray, quality_score: float) -> bool:
        """
        Adds a body embedding to the person's local gallery.
        Checks for pose diversification using a similarity threshold (0.85).
        Different camera angles (front, back, left, right) naturally form separate clusters.
        """
        if embedding is None:
            return False
            
        gallery = self.get_or_create_gallery(person_id)
        now = datetime.now()
        
        with self.lock:
            # 1. Identify if it falls into an existing view cluster (similarity > 0.85)
            cluster_idx = -1
            for idx, item in enumerate(gallery.bodies):
                sim = np.dot(embedding, item["embedding"])
                if sim > 0.85:
                    cluster_idx = idx
                    break
                    
            if cluster_idx >= 0:
                # Update existing pose template if the new view is of higher quality
                if quality_score > gallery.bodies[cluster_idx]["quality"]:
                    gallery.bodies[cluster_idx] = {
                        "embedding": embedding,
                        "quality": quality_score,
                        "time": now
                    }
                return True
                
            # 2. Add as a new view cluster
            gallery.bodies.append({
                "embedding": embedding,
                "quality": quality_score,
                "time": now
            })
            
            # 3. Prune if count exceeds limit (30)
            if len(gallery.bodies) > config.BODY_GALLERY_MAX:
                def get_value(item):
                    age_hours = (now - item["time"]).total_seconds() / 3600.0
                    return item["quality"] - (0.05 * age_hours)
                    
                gallery.bodies.sort(key=get_value)
                gallery.bodies.pop(0)
                
            return True

    def get_face_embeddings(self, person_id: int) -> List[np.ndarray]:
        gallery = self.get_or_create_gallery(person_id)
        with self.lock:
            return [item["embedding"] for item in gallery.faces]

    def get_body_embeddings(self, person_id: int) -> List[np.ndarray]:
        gallery = self.get_or_create_gallery(person_id)
        with self.lock:
            return [item["embedding"] for item in gallery.bodies]
            
    def remove_person(self, person_id: int):
        with self.lock:
            if person_id in self.galleries:
                del self.galleries[person_id]
                
    def merge_galleries(self, src_id: int, dst_id: int):
        """
        Merges the templates of a duplicate identity src_id into dst_id.
        """
        src_gallery = self.get_or_create_gallery(src_id)
        for item in src_gallery.faces:
            self.add_face(dst_id, item["embedding"], item["quality"])
        for item in src_gallery.bodies:
            self.add_body(dst_id, item["embedding"], item["quality"])
        self.remove_person(src_id)
