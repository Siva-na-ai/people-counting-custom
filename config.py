import os

# Central configuration parameters for Face + Body Fusion ReID system
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Model Paths
REPVGG_HEF_PATH = os.path.expanduser(os.getenv("REPVGG_HEF_PATH", "/home/assimilate/models/repvgg_a0_person_reid_512.hef"))
SCRFD_HEF_PATH = os.path.expanduser(os.getenv("SCRFD_HEF_PATH", "/home/assimilate/models/scrfd_2.5g.hef"))
ARCFACE_HEF_PATH = os.path.expanduser(os.getenv("ARCFACE_HEF_PATH", "/home/assimilate/models/arcface_mobilefacenet_h8l.hef"))

# Tracker settings
MAX_AGE = 900
CONFIRMATION_THRESHOLD = 5  # Hits required to confirm a track

# Embedding Quality Thresholds
FACE_MIN_SIZE = 32          # Minimum face bounding box width/height
FACE_BLUR_THRESHOLD = 40.0  # Minimum Laplacian variance for sharpness
FACE_BRIGHTNESS_MIN = 40    # Minimum average pixel brightness
FACE_BRIGHTNESS_MAX = 220   # Maximum average pixel brightness
FACE_ANGLE_YAW_MAX = 45.0   # Maximum yaw angle
FACE_ANGLE_PITCH_MAX = 35.0 # Maximum pitch angle

BODY_BLUR_THRESHOLD = 0.0   # Disable blur check for body ReID (highly robust to blur)
BODY_MIN_SIZE = 64          # Minimum body size

# Gallery Limits
FACE_GALLERY_MAX = 20
BODY_GALLERY_MAX = 30

# Similarity Thresholds
REID_THRESHOLD_FACE = 0.55  # Threshold for ArcFace cosine similarity match
REID_THRESHOLD_BODY = 0.55  # Threshold for RepVGG body similarity match

# Motion constraints (scaled by object body diagonal size)
MAX_MOTION_DIAGS = 3.0      # Teleport check limit for active tracks (multiplied by diag)
MAX_MOTION_DIAGS_LOST_BASE = 4.0 # Base for lost tracks
MAX_MOTION_DIAGS_LOST_STEP = 0.2 # Step per frame of lost track

# Temporal Confirmation settings
STABILITY_CONFIRM_FRAMES = 5 # Frames required to switch identity
STABILITY_SCORE_THRESHOLD = 0.8
