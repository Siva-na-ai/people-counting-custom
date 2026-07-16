import os

# Central configuration parameters for Face + Body Fusion ReID system
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
USE_BODY_REID = True  # Enable/disable Body ReID matching (Face-only ReID when False)

# Model Paths
YOLOv8_HEF_PATH = os.path.expanduser(os.getenv("YOLOv8_HEF_PATH", "/usr/share/hailo-models/yolov8s_h8l.hef"))
REPVGG_HEF_PATH = os.path.expanduser(os.getenv("REPVGG_HEF_PATH", "/home/assimilate/models/repvgg_a0_person_reid_512.hef"))
SCRFD_HEF_PATH = os.path.expanduser(os.getenv("SCRFD_HEF_PATH", "/home/assimilate/models/scrfd_2.5g.hef"))
ARCFACE_HEF_PATH = os.path.expanduser(os.getenv("ARCFACE_HEF_PATH", "/home/assimilate/models/arcface_mobilefacenet_h8l.hef"))

# Tracker settings
MAX_AGE = 900
CONFIRMATION_THRESHOLD = 3  # Hits required to confirm a track (was 5 - faster confirmation)

# Embedding Quality Thresholds (Entrance Camera Settings)
FACE_MIN_SIZE = 32          # Minimum face bounding box width/height (was 45 - too strict for far/moving people)
FACE_BLUR_THRESHOLD = 15.0  # Minimum Laplacian variance (was 50 - walking motion always causes blur)
FACE_BRIGHTNESS_MIN = 30    # Minimum average pixel brightness (was 40)
FACE_BRIGHTNESS_MAX = 230   # Maximum average pixel brightness (was 220)
FACE_ANGLE_YAW_MAX = 50.0   # Maximum yaw angle (was 35 - entrance angle is rarely frontal)
FACE_ANGLE_PITCH_MAX = 45.0 # Maximum pitch angle (was 30)
FACE_ANGLE_ROLL_MAX = 35.0  # Maximum roll angle (was 20)

BODY_BLUR_THRESHOLD = 0.0   # Disable blur check for body ReID (highly robust to blur)
BODY_MIN_SIZE = 64          # Minimum body size

# Gallery Limits
FACE_GALLERY_MAX = 20
BODY_GALLERY_MAX = 30

# Embedding TTL — points older than this are purged from Qdrant automatically
EMBEDDING_TTL_HOURS = 24   # 24-hour retention window


# Similarity Thresholds
REID_THRESHOLD_FACE = 0.65  # Raised from 0.55 — logs show same-person=0.62-0.68, different-person=0.51-0.62
REID_THRESHOLD_BODY = 0.55  # Threshold for RepVGG body similarity match

# Motion constraints (scaled by object body diagonal size)
MAX_MOTION_DIAGS = 3.0      # Teleport check limit for active tracks (multiplied by diag)
MAX_MOTION_DIAGS_LOST_BASE = 4.0 # Base for lost tracks
MAX_MOTION_DIAGS_LOST_STEP = 0.2 # Step per frame of lost track

# Temporal Confirmation settings
STABILITY_CONFIRM_FRAMES = 5 # Frames required to switch identity
STABILITY_SCORE_THRESHOLD = 0.8
