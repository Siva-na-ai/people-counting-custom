import numpy as np

class GalleryManager:
    def __init__(self, max_face_embeddings=20, max_body_embeddings=30):
        self.max_face_embeddings = max_face_embeddings
        self.max_body_embeddings = max_body_embeddings
        
        # Structure: {person_id: [{'embedding': emb, 'quality': q}]}
        self.face_gallery = {}
        self.body_gallery = {}
        
    def add_face(self, person_id, embedding, quality):
        if person_id not in self.face_gallery:
            self.face_gallery[person_id] = []
            
        gallery = self.face_gallery[person_id]
        gallery.append({'embedding': embedding, 'quality': quality})
        
        # Sort by quality descending and keep top K
        gallery.sort(key=lambda x: x['quality'], reverse=True)
        self.face_gallery[person_id] = gallery[:self.max_face_embeddings]
        
    def add_body(self, person_id, embedding, quality):
        if person_id not in self.body_gallery:
            self.body_gallery[person_id] = []
            
        gallery = self.body_gallery[person_id]
        gallery.append({'embedding': embedding, 'quality': quality})
        
        # Sort by quality descending and keep top K
        gallery.sort(key=lambda x: x['quality'], reverse=True)
        self.body_gallery[person_id] = gallery[:self.max_body_embeddings]
        
    def get_best_face(self, person_id):
        if person_id in self.face_gallery and len(self.face_gallery[person_id]) > 0:
            return self.face_gallery[person_id][0]['embedding']
        return None
        
    def get_best_body(self, person_id):
        if person_id in self.body_gallery and len(self.body_gallery[person_id]) > 0:
            return self.body_gallery[person_id][0]['embedding']
        return None
