from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import uuid

class QdrantIdentityClient:
    def __init__(self, host="localhost", port=6333):
        self.client = QdrantClient(host=host, port=port)
        self.face_collection = "face_embeddings"
        self.body_collection = "body_embeddings"
        
        self._ensure_collections()
        
    def _ensure_collections(self):
        # Ensure Face Collection (ArcFace typically 512d)
        if not self.client.collection_exists(self.face_collection):
            self.client.create_collection(
                collection_name=self.face_collection,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )
            
        # Ensure Body Collection (RepVGG typically varies, assuming 512d for consistency)
        if not self.client.collection_exists(self.body_collection):
            self.client.create_collection(
                collection_name=self.body_collection,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )

    def insert_face_embedding(self, person_id, embedding, quality_score, camera_id):
        point_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.face_collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "person_id": person_id,
                        "quality_score": quality_score,
                        "camera_id": camera_id
                    }
                )
            ]
        )
        return point_id

    def insert_body_embedding(self, person_id, embedding, quality_score, camera_id):
        point_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.body_collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "person_id": person_id,
                        "quality_score": quality_score,
                        "camera_id": camera_id
                    }
                )
            ]
        )
        return point_id

    def search_face(self, embedding, top_k=5):
        return self.client.search(
            collection_name=self.face_collection,
            query_vector=embedding.tolist(),
            limit=top_k
        )

    def search_body(self, embedding, top_k=5):
        return self.client.search(
            collection_name=self.body_collection,
            query_vector=embedding.tolist(),
            limit=top_k
        )
