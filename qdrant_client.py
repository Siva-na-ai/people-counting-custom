import logging
import requests
import uuid
import threading
from typing import List, Dict, Any, Optional
import config

logger = logging.getLogger("qdrant_client")

class QdrantClient:
    """
    Thread-safe Qdrant HTTP REST client implementation for production identity storage.
    """
    def __init__(self, url: str = config.QDRANT_URL):
        self.url = url.rstrip('/')
        self.lock = threading.Lock()
        self._initialize_collections()

    def _initialize_collections(self):
        """
        Ensures that face_embeddings, body_embeddings, and person_metadata collections exist.
        """
        collections_to_init = {
            "face_embeddings": {"vectors": {"size": 512, "distance": "Cosine"}},
            "body_embeddings": {"vectors": {"size": 512, "distance": "Cosine"}},
            "person_metadata": {"vectors": {"size": 1, "distance": "Cosine"}}  # 1D dummy vector for pure metadata collection compatibility
        }
        
        for name, config_data in collections_to_init.items():
            try:
                # Check if collection exists
                res = requests.get(f"{self.url}/collections/{name}", timeout=3.0)
                if res.status_code == 200:
                    continue
                    
                # Create collection if not found
                logger.info(f"Creating Qdrant collection: {name}")
                create_res = requests.put(
                    f"{self.url}/collections/{name}", 
                    json=config_data, 
                    headers={"Content-Type": "application/json"},
                    timeout=5.0
                )
                if create_res.status_code != 200:
                    logger.error(f"Failed to create collection {name}: {create_res.text}")
            except Exception as e:
                logger.warning(f"Could not connect to Qdrant at {self.url} during initialization: {e}")

    def upsert_point(self, collection: str, point_id: Any, vector: List[float], payload: Dict[str, Any]) -> bool:
        """
        Upserts a single point to the specified Qdrant collection.
        point_id can be integer or UUID string.
        """
        # Ensure point_id is string representation if UUID object, or clean int
        if isinstance(point_id, uuid.UUID):
            point_id = str(point_id)
            
        data = {
            "points": [
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": payload
                }
            ]
        }
        
        with self.lock:
            try:
                res = requests.put(
                    f"{self.url}/collections/{collection}/points?wait=true",
                    json=data,
                    headers={"Content-Type": "application/json"},
                    timeout=5.0
                )
                return res.status_code == 200
            except Exception as e:
                logger.error(f"Qdrant upsert failed in {collection}: {e}")
                return False

    def search_similar(self, collection: str, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Queries Qdrant for top-K similar points.
        Returns a list of search hits.
        """
        data = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True,
            "with_vector": False
        }
        
        with self.lock:
            try:
                res = requests.post(
                    f"{self.url}/collections/{collection}/points/search",
                    json=data,
                    headers={"Content-Type": "application/json"},
                    timeout=5.0
                )
                if res.status_code == 200:
                    return res.json().get("result", [])
                else:
                    logger.error(f"Qdrant search error: {res.text}")
                    return []
            except Exception as e:
                logger.error(f"Qdrant search connection failed in {collection}: {e}")
                return []

    def retrieve_points(self, collection: str, ids: List[Any]) -> List[Dict[str, Any]]:
        """
        Retrieves points by their explicit IDs.
        """
        cleaned_ids = [str(i) if isinstance(i, uuid.UUID) else i for i in ids]
        data = {"ids": cleaned_ids, "with_payload": True, "with_vector": False}
        
        with self.lock:
            try:
                res = requests.post(
                    f"{self.url}/collections/{collection}/points",
                    json=data,
                    headers={"Content-Type": "application/json"},
                    timeout=5.0
                )
                if res.status_code == 200:
                    return res.json().get("result", [])
                return []
            except Exception as e:
                logger.error(f"Qdrant retrieve failed: {e}")
                return []

    def delete_points(self, collection: str, ids: List[Any]) -> bool:
        """
        Deletes points from a collection by ID.
        """
        cleaned_ids = [str(i) if isinstance(i, uuid.UUID) else i for i in ids]
        data = {"points": cleaned_ids}
        
        with self.lock:
            try:
                res = requests.post(
                    f"{self.url}/collections/{collection}/points/delete",
                    json=data,
                    headers={"Content-Type": "application/json"},
                    timeout=5.0
                )
                return res.status_code == 200
            except Exception as e:
                logger.error(f"Qdrant delete failed in {collection}: {e}")
                return False
