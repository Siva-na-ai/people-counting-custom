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

import json
import argparse
import cv2
import torch
import numpy as np

from modlib.apps.annotate import ColorPalette, Annotator, Color
from modlib.apps.area import Area
from modlib.devices import AiCamera
from modlib.models.zoo import NanoDetPlus416x416
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
        try:
            return self._update_impl(frame_image, detections)
        except Exception as e:
            print("DEBUG: Exception in BoTSORTTracker.update:", type(e), str(e))
            print("DEBUG: detections type:", type(detections))
            try:
                print("DEBUG: detections dir:", dir(detections))
                if len(detections) > 0:
                    item = detections[0]
                    print("DEBUG: item type:", type(item))
                    print("DEBUG: item dir:", dir(item))
                    if hasattr(detections, 'coords'):
                        print("DEBUG: coords:", detections.coords)
                    else:
                        print("DEBUG: item tuple:", tuple(item))
            except Exception as e2:
                print("DEBUG: Failed to inspect detections:", str(e2))
            raise e

    def _update_impl(self, frame_image, detections):
        """
        detections: Detections object or NumPy structured array.
        Returns:
            Detections object or NumPy structured array with updated track IDs.
        """
        num_dets = len(detections)
        
        # Desired structured output dtype (only used if fallback to numpy array is active)
        descr = [('box', '<f4', (4,)), ('confidence', '<f4'), ('class_id', '<i4'), ('track_id', '<i4')]
        new_dtype = np.dtype(descr)
        
        if num_dets == 0:
            for track in self.tracks:
                track.time_since_update += 1
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            if isinstance(detections, np.ndarray):
                return np.zeros(0, dtype=new_dtype)
            return detections
            
        boxes = []
        scores = []
        class_ids = []
        
        # Extract fields
        is_numpy = isinstance(detections, np.ndarray)
        
        if not is_numpy and hasattr(detections, 'coords') and hasattr(detections, 'confidence'):
            boxes = detections.coords
            scores = detections.confidence
            class_ids = detections.class_id
        elif is_numpy and detections.dtype.names is not None:
            names = detections.dtype.names
            box_field = 'box' if 'box' in names else (names[0] if len(names) > 0 else None)
            score_field = 'confidence' if 'confidence' in names else ('score' if 'score' in names else (names[1] if len(names) > 1 else None))
            class_field = 'class_id' if 'class_id' in names else (names[2] if len(names) > 2 else None)
            
            for d_idx in range(num_dets):
                boxes.append(detections[d_idx][box_field] if box_field else detections[d_idx][0])
                scores.append(detections[d_idx][score_field] if score_field else detections[d_idx][1])
                class_ids.append(detections[d_idx][class_field] if class_field else detections[d_idx][2])
        else:
            for det in detections:
                det_tuple = tuple(det)
                boxes.append(det_tuple[0])
                scores.append(det_tuple[1])
                class_ids.append(det_tuple[2])
                
        active_tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        num_tracks = len(active_tracks)
        
        matched_track_indices = []
        matched_det_indices = []
        
        # Stage 1: IoU / Motion Matching
        if num_tracks > 0 and num_dets > 0:
            iou_matrix = np.zeros((num_tracks, num_dets))
            for t_idx, track in enumerate(active_tracks):
                for d_idx in range(num_dets):
                    det_box = boxes[d_idx]
                    if isinstance(det_box, np.ndarray) and det_box.ndim > 1:
                        det_box = det_box.flatten()
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
                det_box = boxes[d_idx]
                if isinstance(det_box, np.ndarray) and det_box.ndim > 1:
                    det_box = det_box.flatten()
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
            det_box = boxes[d_idx]
            if isinstance(det_box, np.ndarray) and det_box.ndim > 1:
                det_box = det_box.flatten()
            det_score = scores[d_idx]
            if isinstance(det_score, np.ndarray):
                det_score = float(det_score.item())
            det_class = class_ids[d_idx]
            if isinstance(det_class, np.ndarray):
                det_class = int(det_class.item())
            
            track.box = det_box
            track.score = det_score
            track.class_id = det_class
            track.time_since_update = 0
            
            final_matched_tracks.append(track)
            final_matched_det_indices.append(d_idx)
            
        # Stage 4: Create new tracks for unmatched detections
        all_unmatched_det_indices = [d for d in range(num_dets) if d not in final_matched_det_indices]
        for d_idx in all_unmatched_det_indices:
            det_box = boxes[d_idx]
            if isinstance(det_box, np.ndarray) and det_box.ndim > 1:
                det_box = det_box.flatten()
            det_score = scores[d_idx]
            if isinstance(det_score, np.ndarray):
                det_score = float(det_score.item())
            det_class = class_ids[d_idx]
            if isinstance(det_class, np.ndarray):
                det_class = int(det_class.item())
            
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
        
        # Sort matched tracks parallel to input detection order
        sorted_pairs = sorted(zip(final_matched_det_indices, final_matched_tracks), key=lambda x: x[0])
        tracker_ids_list = [track.track_id for _, track in sorted_pairs]
        tracker_ids = np.array(tracker_ids_list, dtype=np.int32)
        
        # Stage 6: Update tracker IDs and return
        if not is_numpy:
            # Set the tracker IDs on the detections object in-place
            if hasattr(detections, '_tracker_id'):
                detections._tracker_id = tracker_ids
            if hasattr(detections, 'tracker_id'):
                try:
                    detections.tracker_id = tracker_ids
                except AttributeError:
                    pass
            return detections
        else:
            # Build and return NumPy structured array
            if len(sorted_pairs) == 0:
                return np.zeros(0, dtype=new_dtype)
            tracked_arr = np.zeros(len(sorted_pairs), dtype=new_dtype)
            for i, (d_idx, track) in enumerate(sorted_pairs):
                tracked_arr['box'][i] = track.box
                tracked_arr['confidence'][i] = track.score
                tracked_arr['class_id'][i] = track.class_id
                tracked_arr['track_id'][i] = track.track_id
            return tracked_arr


def json_regions_extraction(json_filename):
    """
    Extract queue regions from json file.
    """
    with open(json_filename, "r") as json_file:
        area_pts = json.load(json_file)
        if len(area_pts) > 0:
            return area_pts
        else:
            raise Exception("Please ensure there are areas to check")
            
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-file",
        type=str,
        required=False,
        default=None,
        help="Json file containing bboxes of areas",
    )
    return parser.parse_args()
    
def start_area_count_demo():
    #-----Camera and AI setup-----
    args = get_args()

    model = NanoDetPlus416x416()
    device = AiCamera()
    device.deploy(model)

    areas = []
    if args.json_file is not None:
        json_areas = json_regions_extraction(args.json_file)
        for area in json_areas: 
            areas.append(Area(area["points"]))

    # Initialize BoTSORT Tracker (combining tracking and ReID)
    tracker = BoTSORTTracker(reid_model_name='osnet_x1_0', reid_threshold=0.65, device='cpu', max_age=30)
    
    annotator = Annotator(
        color=ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.4
    )
    with device as stream:
        for frame in stream:
            #-----Camera and AI setup-----
            detections = frame.detections[frame.detections.confidence > 0.5]
            detections = detections[detections.class_id == 0]
            
            #-----Tracker Update-----
            detections = tracker.update(frame.image, detections)
            
            #-----Display Annotations-----
            labels = []
            for idx, (_, s, c, t) in enumerate(detections):
                labels.append(f"#{t} {model.labels[c]}: {s:0.2f}")

            frame.image = annotator.annotate_boxes(
                frame=frame,
                detections=detections,
                labels=labels,
                color=Color(0, 255, 255),
                alpha=0.2,
            )
            
            if len(areas) == 0:
                #-----Count and show all people-----
                total_people = len(detections)
                label = f"Total People Count: {total_people}"
                annotator.set_label(
                    image=frame.image,
                    x=20,
                    y=40,
                    color=(0, 255, 255),
                    label=label,
                )
            else:
                for ID, area in enumerate(areas):
                    #-----Area-----
                    d = detections[area.contains(detections)]
                    #-----Visualize Detections-----
                    frame.image = annotator.annotate_area(
                        frame=frame, area=area, color=(0, 255, 255), alpha = 0.2,
                    )
                    text_labels = [
                        "In Area: " + str(sum(1 for x in d if x)), #Get Number of people in each Area
                        "Area ID: " + str(ID + 1),
                    ]

                    for index, label in enumerate(text_labels):
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text_width, text_height = cv2.getTextSize(
                            text=label,
                            fontFace=font,
                            fontScale=0.5,
                            thickness=1,
                        )[0]
                        annotator.set_label(
                            image=frame.image,
                            x=int(((area.points[0][0] +  area.points[1][0]) / 2) * frame.width) - int(text_width/2),
                            y=int(((area.points[0][1] +  area.points[2][1]) / 2)* frame.height + ((index) * 25)) - int(2 * text_height),
                            color=(0, 255, 255),
                            label=label,
                        )
            frame.display()


if __name__ == "__main__":
    start_area_count_demo()
