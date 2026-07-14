import os

# Qdrant Database Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Collections
FACE_COLLECTION = "face_embeddings"
BODY_COLLECTION = "body_embeddings"
METADATA_COLLECTION = "person_metadata"

# Tracking & Identity Thresholds
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", 0.6))
BODY_MATCH_THRESHOLD = float(os.getenv("BODY_MATCH_THRESHOLD", 0.65))

# Gallery configurations
MAX_FACE_GALLERY_SIZE = 20
MAX_BODY_GALLERY_SIZE = 30

# State transitions
CANDIDATE_FRAMES_REQUIRED = 3
IDENTITY_EXPIRATION_SECONDS = 3600  # 1 hour

# Hardware & Models
DEVICE = os.getenv("DEVICE", "cpu")
REID_MODEL_NAME = os.getenv("REID_MODEL_NAME", "osnet_x1_0")

# Embedding Quality Thresholds
MIN_FACE_QUALITY = float(os.getenv("MIN_FACE_QUALITY", 0.5))
MIN_BODY_QUALITY = float(os.getenv("MIN_BODY_QUALITY", 0.4))
