import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import config

logger = logging.getLogger("duplicate_resolver")

class DuplicateIdentityResolver:
    """
    Scans for duplicate person profiles and consolidates/merges them safely,
    preserving galleries, visit stats, and history.
    """
    def __init__(self, registry, gallery_mgr, qdrant):
        self.registry = registry
        self.gallery_mgr = gallery_mgr
        self.qdrant = qdrant

    def check_and_resolve_duplicates(self, active_pids: List[int]) -> List[Tuple[int, int]]:
        """
        Scans all registered identities to find duplicates.
        Returns a list of merges executed: [(src_person_id, dst_person_id)]
        """
        all_pids = list(self.registry.persons.keys())
        merges_executed = []
        
        # O(N^2) comparison - fine since N is small in active window (Raspberry Pi scale)
        for i in range(len(all_pids)):
            for j in range(i + 1, len(all_pids)):
                pid1 = all_pids[i]
                pid2 = all_pids[j]
                
                if pid1 in merges_executed or pid2 in merges_executed:
                    continue
                    
                if self._are_duplicates(pid1, pid2):
                    # Decide destination ID: keep the older one (smaller ID) to maintain identity persistence
                    dst_id = min(pid1, pid2)
                    src_id = max(pid1, pid2)
                    
                    self.merge_identities(src_id, dst_id)
                    merges_executed.append((src_id, dst_id))
                    
        return merges_executed

    def _are_duplicates(self, pid1: int, pid2: int) -> bool:
        """
        Evaluates similarities and spatiotemporal overlap to determine if pid1 and pid2 represent the same human.
        """
        p1 = self.registry.persons[pid1]
        p2 = self.registry.persons[pid2]
        
        # Safety constraint: if they are active simultaneously on the same camera, they CANNOT be the same human
        p1_active_tracks = p1.get("active_track_ids", [])
        p2_active_tracks = p2.get("active_track_ids", [])
        if len(p1_active_tracks) > 0 and len(p2_active_tracks) > 0:
            # Active in the same camera view right now
            return False
            
        # Get face and body similarity
        faces1 = self.gallery_mgr.get_face_embeddings(pid1)
        faces2 = self.gallery_mgr.get_face_embeddings(pid2)
        
        face_sim = 0.0
        if len(faces1) > 0 and len(faces2) > 0:
            sims = [np.dot(f1, f2) for f1 in faces1 for f2 in faces2]
            face_sim = float(np.max(sims))
            
        bodies1 = self.gallery_mgr.get_body_embeddings(pid1)
        bodies2 = self.gallery_mgr.get_body_embeddings(pid2)
        
        body_sim = 0.0
        if len(bodies1) > 0 and len(bodies2) > 0:
            sims = [np.dot(b1, b2) for b1 in bodies1 for b2 in bodies2]
            body_sim = float(np.max(sims))
            
        # Merge criteria: face match must be high confidence (0.82+) to avoid false merges.
        # After ArcFace normalization fix: same-person=0.7-0.9, different-person=0.2-0.4
        if face_sim >= 0.82:
            return True
        elif face_sim == 0.0 and body_sim > 0.88:
            return True

        return False

    def merge_identities(self, src_id: int, dst_id: int):
        """
        Merges src_id data into dst_id across PersonRegistry, GalleryManager, and Qdrant.
        """
        logger.info(f"[DuplicateIdentityResolver] Executing merge: Person #{src_id} -> Person #{dst_id}")
        
        # 1. Merge metadata in registry
        self.registry.merge_persons(src_id, dst_id)
        
        # 2. Merge local template galleries
        self.gallery_mgr.merge_galleries(src_id, dst_id)
        
        # 3. Synchronize vector DB updates (asynchronously done via worker_pool)
        # Update point payloads in Qdrant or delete src points
        # To avoid complex point search, we query Qdrant for src points and delete them,
        # then re-upload the merged gallery for dst_id.
