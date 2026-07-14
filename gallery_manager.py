import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)

class GalleryManager:
    def __init__(self, max_face_size=20, max_body_size=30):
        self.max_face_size = max_face_size
        self.max_body_size = max_body_size
        self.face_galleries: Dict[str, List[Dict]] = {}
        self.body_galleries: Dict[str, List[Dict]] = {}

    def add_face(self, person_id: str, embedding: np.ndarray, quality: float):
        if person_id not in self.face_galleries:
            self.face_galleries[person_id] = []
            
        gallery = self.face_galleries[person_id]
        gallery.append({"embedding": embedding, "quality": quality})
        
        # Sort by quality and keep top N
        gallery.sort(key=lambda x: x["quality"], reverse=True)
        if len(gallery) > self.max_face_size:
            gallery.pop()

    def add_body(self, person_id: str, embedding: np.ndarray, quality: float):
        if person_id not in self.body_galleries:
            self.body_galleries[person_id] = []
            
        gallery = self.body_galleries[person_id]
        gallery.append({"embedding": embedding, "quality": quality})
        
        gallery.sort(key=lambda x: x["quality"], reverse=True)
        if len(gallery) > self.max_body_size:
            gallery.pop()
            
    def get_face_gallery(self, person_id: str) -> List[Dict]:
        return self.face_galleries.get(person_id, [])
        
    def get_body_gallery(self, person_id: str) -> List[Dict]:
        return self.body_galleries.get(person_id, [])
