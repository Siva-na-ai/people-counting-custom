#
# Copyright 2026 Sony Semiconductor Solutions Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cv2
import torch
import numpy as np
from modlib.apps import Annotator, BYTETracker
from modlib.devices import AiCamera
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320

try:
    import torchreid
    try:
        from torchreid.utils import FeatureExtractor
    except ImportError:
        # Fallback for nested package structure in some PyPI versions of torchreid
        from torchreid.reid.utils import FeatureExtractor
except ImportError as e:
    import traceback
    traceback.print_exc()
    raise ImportError(
        f"Please install torch and torchreid to run this script. (Original error: {e}). "
        "Run: pip install torch torchvision torchreid"
    )


class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


class PersonReID:
    def __init__(self, model_name='osnet_x1_0', threshold=0.70, device='cpu'):
        self.device = device
        self.threshold = threshold
        # Initialize torchreid FeatureExtractor
        self.extractor = FeatureExtractor(
            model_name=model_name,
            device=self.device
        )
        # Gallery to store known identities:
        # global_id -> list of feature embeddings (numpy arrays of shape (512,))
        self.gallery = {}
        # Mapping from BYTETracker's track_id to our persistent global_id
        self.track_to_global = {}
        self.next_global_id = 1
        # Cache for the last crop of each track ID
        self.last_crop = {}
        
    def get_crop(self, image, box):
        """
        Extract and preprocess the crop from the image.
        box coordinates can be normalized [xmin, ymin, xmax, ymax] or absolute pixels.
        """
        h, w, _ = image.shape
        if any(coord > 2.0 for coord in box):
            # Absolute coordinates
            xmin = int(max(0, box[0]))
            ymin = int(max(0, box[1]))
            xmax = int(min(w, box[2]))
            ymax = int(min(h, box[3]))
        else:
            # Normalized coordinates
            xmin = int(max(0, box[0] * w))
            ymin = int(max(0, box[1] * h))
            xmax = int(min(w, box[2] * w))
            ymax = int(min(h, box[3] * h))
            
        if xmax <= xmin or ymax <= ymin:
            return None
            
        crop = image[ymin:ymax, xmin:xmax]
        # Convert BGR (OpenCV default) to RGB (Torchreid expects RGB)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return crop_rgb

    def extract_embedding(self, crop):
        """
        Extract 512-dim embedding from a crop.
        """
        with torch.no_grad():
            features = self.extractor([crop])
            embedding = features[0].cpu().numpy()
            # L2 normalization for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding

    def update_tracks(self, frame_image, tracked_detections):
        """
        tracked_detections: iterable of (box, score, class_id, track_id)
        Returns:
            list of global_ids matching the order of tracked_detections
        """
        current_global_ids = []
        active_track_ids = set()
        
        # Populate active track IDs and cache crops for already tracked targets
        for detection in tracked_detections:
            box, score, class_id, track_id = detection
            active_track_ids.add(track_id)
            
            if track_id in self.track_to_global:
                crop = self.get_crop(frame_image, box)
                if crop is not None:
                    self.last_crop[track_id] = crop
                    
        # Identify global IDs already active in this frame (to avoid identity theft)
        active_global_ids = {self.track_to_global[t_id] for t_id in active_track_ids if t_id in self.track_to_global}
        
        for detection in tracked_detections:
            box, score, class_id, track_id = detection
            
            # If we already mapped this track_id, reuse the global_id
            if track_id in self.track_to_global:
                current_global_ids.append(self.track_to_global[track_id])
                continue
                
            # If it's a new track_id, we crop and extract features
            crop = self.get_crop(frame_image, box)
            if crop is None:
                # If we cannot crop, assign a new global ID without embedding
                global_id = self.next_global_id
                self.next_global_id += 1
                self.track_to_global[track_id] = global_id
                current_global_ids.append(global_id)
                active_global_ids.add(global_id)
                continue
                
            # Cache the crop for future updates
            self.last_crop[track_id] = crop
            embedding = self.extract_embedding(crop)
            
            # Match against our gallery of known global_ids
            best_match_id = None
            best_similarity = -1.0
            
            for g_id, embeddings in self.gallery.items():
                # Avoid matching with a global ID that is already active in this frame
                if g_id in active_global_ids:
                    continue
                    
                # Compare similarity against stored embeddings for this global ID
                sims = [np.dot(embedding, stored_emb) for stored_emb in embeddings]
                max_sim = max(sims) if sims else 0.0
                
                if max_sim > best_similarity:
                    best_similarity = max_sim
                    best_match_id = g_id
            
            if best_similarity >= self.threshold and best_match_id is not None:
                # Re-identified!
                global_id = best_match_id
                self.gallery[global_id].append(embedding)
                if len(self.gallery[global_id]) > 5:
                    self.gallery[global_id].pop(0)
            else:
                # New person detected
                global_id = self.next_global_id
                self.next_global_id += 1
                self.gallery[global_id] = [embedding]
                
            self.track_to_global[track_id] = global_id
            current_global_ids.append(global_id)
            active_global_ids.add(global_id)
            
        # Clean up track_to_global map and extract latest embeddings for tracks that are no longer active
        inactive_tracks = set(self.track_to_global.keys()) - active_track_ids
        for t_id in inactive_tracks:
            global_id = self.track_to_global[t_id]
            # When the track is lost, we extract its latest embedding and add it to the gallery
            if t_id in self.last_crop:
                crop = self.last_crop[t_id]
                if crop is not None:
                    try:
                        embedding = self.extract_embedding(crop)
                        self.gallery[global_id].append(embedding)
                        if len(self.gallery[global_id]) > 5:
                            self.gallery[global_id].pop(0)
                    except Exception:
                        pass
                del self.last_crop[t_id]
            del self.track_to_global[t_id]
            
        return current_global_ids


def main():
    #-----Camera and AI setup-----
    device = AiCamera()
    model = SSDMobileNetV2FPNLite320x320()
    device.deploy(model)

    tracker = BYTETracker(BYTETrackerArgs())
    # Initialize ReID module using OSNet (default to cpu for RPi 5 suitability)
    reid = PersonReID(model_name='osnet_x1_0', threshold=0.70, device='cpu')
    
    unique_seen_people = set()
    annotator = Annotator(thickness=1, text_thickness=1, text_scale=0.4)

    with device as stream:
        for frame in stream:
            #-----Detection Filtering-----
            detections = frame.detections[frame.detections.confidence > 0.55]
            detections = detections[detections.class_id == 0]  # Person
            
            #-----Tracker-----
            detections = tracker.update(frame, detections)

            #-----ReID Update-----
            global_ids = reid.update_tracks(frame.image, detections)
            for g_id in global_ids:
                unique_seen_people.add(g_id)

            #-----Display Annotations-----
            annotator.set_label(
                image=frame.image,
                x=430,
                y=30,
                color=(200, 200, 200),
                label="Total people detected: " + str(len(unique_seen_people)),
            )

            # Map the raw track ID to the persistent global ReID ID in visual annotations
            labels = []
            for idx, (_, s, c, t) in enumerate(detections):
                g_id = global_ids[idx]
                labels.append(f"#{g_id} {model.labels[c]}: {s:0.2f}")
                
            annotator.annotate_boxes(frame=frame, detections=detections, labels=labels)

            frame.display()


if __name__ == "__main__":
    main()
