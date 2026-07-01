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
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320
from scipy.optimize import linear_sum_assignment

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

# Helper function to compute IoU
def compute_iou(box1, box2):
    # box format: [xmin, ymin, xmax, ymax]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

class Track:
    def __init__(self, track_id, box, embedding, score, class_id):
        self.track_id = track_id
        self.box = box
        self.score = score
        self.class_id = class_id
        self.embeddings = [embedding] if embedding is not None else []
        self.time_since_update = 0

class BoTSORTTracker:
    def __init__(self, reid_model_name='osnet_x1_0', reid_threshold=0.65, device='cpu', max_age=30):
        self.max_age = max_age
        self.reid_threshold = reid_threshold
        
        # Initialize ReID FeatureExtractor
        self.extractor = FeatureExtractor(
            model_name=reid_model_name,
            device=device
        )
        self.tracks = []
        self.next_track_id = 1
        
    def get_crop(self, image, box):
        """
        Extract and preprocess the crop from the image.
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

    def update(self, frame_image, detections):
        """
        detections: Detections object or NumPy structured array.
        Returns:
            Detections object or NumPy structured array.
        """
        import copy
        
        # Extract the underlying numpy structured array if wrapped in a Detections object
        is_wrapped = False
        data_attr = None
        raw_arr = detections
        
        for attr in ['_data', 'data', '_array', 'array']:
            if hasattr(detections, attr) and isinstance(getattr(detections, attr), np.ndarray):
                is_wrapped = True
                data_attr = attr
                raw_arr = getattr(detections, attr)
                break
                
        if len(raw_arr) == 0:
            for track in self.tracks:
                track.time_since_update += 1
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            if is_wrapped:
                out_detections = copy.copy(detections)
                setattr(out_detections, data_attr, raw_arr.copy()[:0])
                return out_detections
            else:
                return raw_arr.copy()[:0]
            
        active_tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        num_tracks = len(active_tracks)
        num_dets = len(raw_arr)
        
        matched_track_indices = []
        matched_det_indices = []
        
        # Stage 1: IoU / Motion Matching
        if num_tracks > 0 and num_dets > 0:
            iou_matrix = np.zeros((num_tracks, num_dets))
            for t_idx, track in enumerate(active_tracks):
                for d_idx in range(num_dets):
                    det_box = raw_arr[d_idx]['box'] if 'box' in raw_arr.dtype.names else raw_arr[d_idx][0]
                    iou_matrix[t_idx, d_idx] = compute_iou(track.box, det_box)
            
            cost_matrix = 1.0 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= 0.50:
                    matched_track_indices.append(r)
                    matched_det_indices.append(c)
                    
        # Stage 2: ReID / Appearance Matching for unmatched tracks/detections
        unmatched_track_indices = [t for t in range(num_tracks) if t not in matched_track_indices]
        unmatched_det_indices = [d for d in range(num_dets) if d not in matched_det_indices]
        
        if len(unmatched_track_indices) > 0 and len(unmatched_det_indices) > 0:
            det_embeddings = []
            valid_det_indices = []
            
            for d_idx in unmatched_det_indices:
                det_box = raw_arr[d_idx]['box'] if 'box' in raw_arr.dtype.names else raw_arr[d_idx][0]
                crop = self.get_crop(frame_image, det_box)
                if crop is not None:
                    emb = self.extract_embedding(crop)
                    det_embeddings.append(emb)
                    valid_det_indices.append(d_idx)
                    
            if len(det_embeddings) > 0 and any(len(active_tracks[t_idx].embeddings) > 0 for t_idx in unmatched_track_indices):
                reid_cost_matrix = np.ones((len(unmatched_track_indices), len(det_embeddings)))
                
                for i, t_idx in enumerate(unmatched_track_indices):
                    track = active_tracks[t_idx]
                    if len(track.embeddings) == 0:
                        continue
                    for j, det_emb in enumerate(det_embeddings):
                        sims = [np.dot(det_emb, stored_emb) for stored_emb in track.embeddings]
                        max_sim = max(sims) if sims else 0.0
                        reid_cost_matrix[i, j] = 1.0 - max_sim
                
                r_ind, c_ind = linear_sum_assignment(reid_cost_matrix)
                
                for r, c in zip(r_ind, c_ind):
                    max_sim = 1.0 - reid_cost_matrix[r, c]
                    if max_sim >= self.reid_threshold:
                        t_idx = unmatched_track_indices[r]
                        d_idx = valid_det_indices[c]
                        
                        matched_track_indices.append(t_idx)
                        matched_det_indices.append(d_idx)
                        active_tracks[t_idx].embeddings.append(det_embeddings[c])
                        if len(active_tracks[t_idx].embeddings) > 5:
                            active_tracks[t_idx].embeddings.pop(0)
                            
        # Stage 3: Update matched track states
        final_matched_tracks = []
        final_matched_det_indices = []
        matched_pairs = sorted(zip(matched_track_indices, matched_det_indices), key=lambda x: x[1])
        
        for t_idx, d_idx in matched_pairs:
            track = active_tracks[t_idx]
            det_box = raw_arr[d_idx]['box'] if 'box' in raw_arr.dtype.names else raw_arr[d_idx][0]
            det_score = raw_arr[d_idx]['confidence'] if 'confidence' in raw_arr.dtype.names else raw_arr[d_idx][1]
            det_class = raw_arr[d_idx]['class_id'] if 'class_id' in raw_arr.dtype.names else raw_arr[d_idx][2]
            
            track.box = det_box
            track.score = det_score
            track.class_id = det_class
            track.time_since_update = 0
            
            final_matched_tracks.append(track)
            final_matched_det_indices.append(d_idx)
            
        # Stage 4: Create new tracks for unmatched detections
        all_unmatched_det_indices = [d for d in range(num_dets) if d not in final_matched_det_indices]
        for d_idx in all_unmatched_det_indices:
            det_box = raw_arr[d_idx]['box'] if 'box' in raw_arr.dtype.names else raw_arr[d_idx][0]
            det_score = raw_arr[d_idx]['confidence'] if 'confidence' in raw_arr.dtype.names else raw_arr[d_idx][1]
            det_class = raw_arr[d_idx]['class_id'] if 'class_id' in raw_arr.dtype.names else raw_arr[d_idx][2]
            
            crop = self.get_crop(frame_image, det_box)
            emb = self.extract_embedding(crop) if crop is not None else None
            
            new_track = Track(self.next_track_id, det_box, emb, det_score, det_class)
            self.next_track_id += 1
            
            self.tracks.append(new_track)
            final_matched_tracks.append(new_track)
            final_matched_det_indices.append(d_idx)
            
        # Stage 5: Age unmatched tracks
        all_unmatched_track_indices = [t_idx for t_idx in range(num_tracks) if t_idx not in matched_track_indices]
        for t_idx in all_unmatched_track_indices:
            active_tracks[t_idx].time_since_update += 1
            
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Stage 6: Build tracked structured array output
        if len(final_matched_tracks) == 0:
            tracked_arr = raw_arr.copy()[:0]
        elif raw_arr.dtype.names and len(raw_arr.dtype.names) >= 4:
            track_id_field = raw_arr.dtype.names[3]
            tracked_arr = np.copy(raw_arr[final_matched_det_indices])
            for i, track in enumerate(final_matched_tracks):
                tracked_arr[track_id_field][i] = track.track_id
        else:
            descr = raw_arr.dtype.descr
            field_names = [d[0] for d in descr]
            if 'track_id' in field_names:
                track_id_field = 'track_id'
                tracked_arr = np.copy(raw_arr[final_matched_det_indices])
                for i, track in enumerate(final_matched_tracks):
                    tracked_arr[track_id_field][i] = track.track_id
            else:
                descr.append(('track_id', '<i4'))
                new_dtype = np.dtype(descr)
                tracked_arr = np.zeros(len(final_matched_det_indices), dtype=new_dtype)
                for name in raw_arr.dtype.names:
                    tracked_arr[name] = raw_arr[name][final_matched_det_indices]
                for i, track in enumerate(final_matched_tracks):
                    tracked_arr['track_id'][i] = track.track_id
                    
        if is_wrapped:
            out_detections = copy.copy(detections)
            setattr(out_detections, data_attr, tracked_arr)
            return out_detections
        else:
            return tracked_arr


def main():
    #-----Camera and AI setup-----
    device = AiCamera()
    model = SSDMobileNetV2FPNLite320x320()
    device.deploy(model)

    # Initialize BoTSORT Tracker (combining tracking and ReID)
    tracker = BoTSORTTracker(reid_model_name='osnet_x1_0', reid_threshold=0.65, device='cpu', max_age=30)
    
    unique_seen_people = set()
    annotator = Annotator(thickness=1, text_thickness=1, text_scale=0.4)

    with device as stream:
        for frame in stream:
            #-----Detection Filtering-----
            detections = frame.detections[frame.detections.confidence > 0.55]
            detections = detections[detections.class_id == 0]  # Person
            
            #-----Tracker Update-----
            detections = tracker.update(frame.image, detections)

            #-----ReID / Unique Visitor Count-----
            for idx, (_, s, c, t) in enumerate(detections):
                unique_seen_people.add(t)

            #-----Display Annotations-----
            annotator.set_label(
                image=frame.image,
                x=430,
                y=30,
                color=(200, 200, 200),
                label="Total people detected: " + str(len(unique_seen_people)),
            )

            # Map the track ID in visual annotations
            labels = []
            for idx, (_, s, c, t) in enumerate(detections):
                labels.append(f"#{t} {model.labels[c]}: {s:0.2f}")
                
            annotator.annotate_boxes(frame=frame, detections=detections, labels=labels)

            frame.display()


if __name__ == "__main__":
    main()
