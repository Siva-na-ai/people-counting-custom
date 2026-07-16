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

    def get_max_person_id(self) -> int:
        """
        Scans face_embeddings to find the highest person_id stored.
        Called on startup so next_person_id continues from where it left off.
        Returns 0 if no embeddings exist yet.
        """
        max_pid = 0
        offset = None
        while True:
            scroll_body = {"limit": 256, "with_payload": True, "with_vector": False}
            if offset is not None:
                scroll_body["offset"] = offset
            try:
                res = requests.post(
                    f"{self.url}/collections/face_embeddings/points/scroll",
                    json=scroll_body,
                    headers={"Content-Type": "application/json"},
                    timeout=10.0
                )
                if res.status_code != 200:
                    break
                data = res.json().get("result", {})
                for pt in data.get("points", []):
                    pid = pt.get("payload", {}).get("person_id", 0)
                    if isinstance(pid, int) and pid > max_pid:
                        max_pid = pid
                next_offset = data.get("next_page_offset", None)
                if next_offset is None:
                    break
                offset = next_offset
            except Exception as e:
                logger.warning(f"get_max_person_id failed: {e}")
                break
        logger.info(f"[QdrantClient] Max person_id in DB: {max_pid}")
        return max_pid


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

    def cleanup_old_embeddings(self, ttl_hours: float = None) -> int:
        """
        Deletes all face and body embedding points older than ttl_hours.
        Scrolls through each collection and batch-deletes expired points.
        Returns total number of points deleted.
        """
        import time as _time
        if ttl_hours is None:
            ttl_hours = config.EMBEDDING_TTL_HOURS

        cutoff_ts = _time.time() - (ttl_hours * 3600.0)
        total_deleted = 0

        for collection in ("face_embeddings", "body_embeddings"):
            offset = None
            expired_ids = []

            # Scroll through all points in the collection
            while True:
                scroll_body = {
                    "limit": 256,
                    "with_payload": True,
                    "with_vector": False
                }
                if offset is not None:
                    scroll_body["offset"] = offset

                try:
                    res = requests.post(
                        f"{self.url}/collections/{collection}/points/scroll",
                        json=scroll_body,
                        headers={"Content-Type": "application/json"},
                        timeout=10.0
                    )
                except Exception as e:
                    logger.error(f"Qdrant scroll failed on {collection}: {e}")
                    break

                if res.status_code != 200:
                    break

                data = res.json().get("result", {})
                points = data.get("points", [])

                for pt in points:
                    ts = pt.get("payload", {}).get("timestamp", None)
                    if ts is not None and float(ts) < cutoff_ts:
                        expired_ids.append(pt["id"])

                next_offset = data.get("next_page_offset", None)
                if next_offset is None:
                    break
                offset = next_offset

            # Delete expired points
            if expired_ids:
                deleted = self.delete_points(collection, expired_ids)
                if deleted:
                    logger.info(f"[QdrantClient] Purged {len(expired_ids)} expired points from '{collection}' (TTL={ttl_hours}h)")
                    total_deleted += len(expired_ids)

        return total_deleted

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
