import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) in ["app", "scripts"] else SCRIPT_DIR

class Config:
    # Model Artifact Path (defaults to MobileNetV2 Champion)
    MODEL_PATH = os.getenv(
        "MODEL_PATH",
        os.path.join(BASE_DIR, "task2_outputs", "2.3_transfer_learning_best.keras")
    )
    if not os.path.exists(MODEL_PATH):
        # Fallback to alternate naming if present
        fallback_path = os.path.join(BASE_DIR, "task2_outputs", "transfer_learning_best.keras")
        if os.path.exists(fallback_path):
            MODEL_PATH = fallback_path

    # Classification Threshold
    DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))

    # Database Path (defaults to SQLite trail for Task 5)
    DATABASE_PATH = os.getenv(
        "DATABASE_PATH",
        os.path.join(BASE_DIR, "database", "visionguard_history.db")
    )

    # Security & Upload Constraints
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(5 * 1024 * 1024))) # 5 MB
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

    # Server Settings
    PORT = int(os.getenv("PORT", "5000"))
    HOST = os.getenv("HOST", "0.0.0.0")
    DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

    # Cloud API Endpoint (Task 7 & 8: Local Flask calling Cloud Model API)
    REMOTE_API_URL = os.getenv("REMOTE_API_URL", "").strip()

    # Model Name / Metadata
    MODEL_NAME = "MobileNetV2_Champion_v2"
