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

# Production-Grade ReID Configuration Parameters

# Face thresholds
FACE_MATCH_THRESHOLD = 0.82
FACE_STRONG_MATCH = 0.90
FACE_TEMPLATE_UPDATE = 0.93

# Face quality
FACE_MIN_CONFIDENCE = 0.70
FACE_MIN_WIDTH = 35
FACE_MIN_HEIGHT = 35
FACE_BLUR_THRESHOLD = 30.0
FACE_MAX_YAW = 45.0
FACE_MAX_PITCH = 45.0
FACE_MAX_ROLL = 30.0

# Matching
TOP_K = 5
AMBIGUITY_GAP = 0.05

# Body ReID
BODY_MIN_SIZE = 64
BODY_BLUR_THRESHOLD = 0.0
BODY_MATCH_THRESHOLD = 0.50
BODY_REJECT_THRESHOLD = 0.40

# Temporal matching rules
GOOD_FACE_CONFIRMATIONS = 2
MIN_CONSECUTIVE_MATCHES = 4
SEARCH_RETRY_INTERVAL = 5

# Gallery limits (maximum templates stored per person)
MAX_FACE_TEMPLATES = 20
MAX_BODY_TEMPLATES = 20
FACE_GALLERY_MAX = MAX_FACE_TEMPLATES
BODY_GALLERY_MAX = MAX_BODY_TEMPLATES

# Identity state lock and persistence
TRACK_LOCK_ENABLED = True
TRACK_MEMORY_TTL = 120

# Embedding TTL — points older than this are purged from Qdrant automatically
EMBEDDING_TTL_HOURS = 24   # 24-hour retention window

# Motion constraints (scaled by object body diagonal size)
MAX_MOTION_DIAGS = 3.0      # Teleport check limit for active tracks (multiplied by diag)
MAX_MOTION_DIAGS_LOST_BASE = 4.0 # Base for lost tracks
MAX_MOTION_DIAGS_LOST_STEP = 0.2 # Step per frame of lost track

# Legacy temporal confirmation compatibility settings
STABILITY_CONFIRM_FRAMES = MIN_CONSECUTIVE_MATCHES
STABILITY_SCORE_THRESHOLD = 0.8
