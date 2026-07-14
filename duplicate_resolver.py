import numpy as np
from sklearn.cluster import DBSCAN

class DuplicateResolver:
    def __init__(self, eps_threshold=0.3, min_samples=2):
        self.eps_threshold = eps_threshold
        self.min_samples = min_samples

    def resolve_duplicates(self, embeddings, person_ids):
        """
        Takes a list of embeddings and their corresponding IDs.
        Uses DBSCAN clustering to find embeddings that are very close (duplicates).
        Returns a list of ID clusters that should be merged.
        """
        if len(embeddings) < 2:
            return []
            
        # DBSCAN works with Euclidean distance by default. 
        # Assuming embeddings are L2 normalized, Euclidean distance relates to Cosine similarity.
        X = np.array(embeddings)
        clustering = DBSCAN(eps=self.eps_threshold, min_samples=self.min_samples, metric='euclidean').fit(X)
        
        labels = clustering.labels_
        clusters = {}
        for idx, label in enumerate(labels):
            if label == -1: # Noise
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(person_ids[idx])
            
        merge_groups = []
        for label, group in clusters.items():
            # Only return groups with more than one unique person_id
            unique_ids = list(set(group))
            if len(unique_ids) > 1:
                merge_groups.append(unique_ids)
                
        return merge_groups
