"""Application configuration and constants.

Values may be overridden through environment variables (.env). Nothing here is
a secret: credentials live only in the environment via python-dotenv.
"""

from __future__ import annotations

import os

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "incident_reconstruction")

YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
YOLO_IMG_SIZE = int(os.getenv("YOLO_IMG_SIZE", "640"))
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))
PERSON_CLASS_ID = 0  # COCO class 0 = person

# Default affected-person classification thresholds (configurable by users).
THRESHOLD_HIGH = float(os.getenv("THRESHOLD_HIGH", "80.0"))
THRESHOLD_MEDIUM = float(os.getenv("THRESHOLD_MEDIUM", "50.0"))

# Travel-time model bounds as factors of the expected (length / speed) time.
TRAVEL_MIN_FACTOR = float(os.getenv("TRAVEL_MIN_FACTOR", "0.75"))
TRAVEL_MAX_FACTOR = float(os.getenv("TRAVEL_MAX_FACTOR", "1.25"))

# Recency decay scale in minutes for probability estimation.
RECENCY_SCALE_MINUTES = float(os.getenv("RECENCY_SCALE_MINUTES", "90.0"))

# Maximum minutes a person may take to cross the affected path before being
# considered to have exited it.
MAX_CROSS_BUFFER_MINUTES = float(os.getenv("MAX_CROSS_BUFFER_MINUTES", "10.0"))

# Route feasibility tolerance added to the observation gap (minutes).
ROUTE_TOLERANCE_MINUTES = float(os.getenv("ROUTE_TOLERANCE_MINUTES", "15.0"))

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")
REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")

RISK_LEVELS = ["HIGH", "MEDIUM", "LOW"]


def validate_thresholds(high: float, medium: float) -> None:
    if not (0 <= medium < high <= 100):
        raise ValueError("Thresholds must satisfy 0 <= medium < high <= 100")


def classify(probability: float, high: float = THRESHOLD_HIGH, medium: float = THRESHOLD_MEDIUM) -> str:
    if probability >= high:
        return "HIGH"
    if probability >= medium:
        return "MEDIUM"
    return "LOW"