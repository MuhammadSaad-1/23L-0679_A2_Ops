import os
import sys

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) in ["app", "scripts"] else SCRIPT_DIR
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from app.config import Config
from app.inference import VisionGuardPredictor
import app.database as db_helper

# Initialize Flask application
app = Flask(
    __name__,
    template_folder=os.path.join(SCRIPT_DIR, "templates"),
    static_folder=os.path.join(SCRIPT_DIR, "static")
)

app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH

# -------------------------------------------------------------
# 1. Startup Initialization (One-Time Execution)
# -------------------------------------------------------------
print("=" * 65)
print(" STARTING VISIONGUARD FLASK WEB APPLICATION")
print(f" Environment : Production Server Mode (debug={Config.DEBUG})")
print(f" Model Path  : {Config.MODEL_PATH}")
print(f" Threshold   : {Config.DECISION_THRESHOLD}")
print(f" Database    : {Config.DATABASE_PATH}")
print("=" * 65)

# Initialize database schema (Task 5 requirement)
db_helper.init_db(Config.DATABASE_PATH)

# Load neural network model ONCE into memory (Task 4.1 requirement)
predictor = VisionGuardPredictor(
    model_path=Config.MODEL_PATH,
    threshold=Config.DECISION_THRESHOLD,
    model_name=Config.MODEL_NAME
)

# -------------------------------------------------------------
# 2. Application Routes
# -------------------------------------------------------------
@app.route("/", methods=["GET"])
def upload_page():
    """Renders the main upload interface."""
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Handles image upload, runs validation checks, executes in-memory inference,
    logs the query to the SQLite audit database, and renders the verdict.
    """
    # Check if a file was submitted in the form
    if "file" not in request.files:
        return render_template("error.html",
            error_code=400,
            error_title="No File Uploaded",
            error_message="The request did not contain an image file. Please choose a valid file."
        ), 400

    file = request.files["file"]

    # Check for empty file selection
    if not file or file.filename.strip() == "":
        return render_template("error.html",
            error_code=400,
            error_title="Empty File Selected",
            error_message="No file was selected for upload. Please choose an image to analyze."
        ), 400

    # Sanitize filename (Subtask 4.2 security requirement)
    original_name = file.filename
    clean_filename = secure_filename(original_name)
    if not clean_filename:
        clean_filename = "uploaded_media.png"

    # Validation Failure Case 1: Wrong File Extension
    file_ext = clean_filename.rsplit(".", 1)[-1].lower() if "." in clean_filename else ""
    if file_ext not in Config.ALLOWED_EXTENSIONS:
        return render_template("error.html",
            error_code=400,
            error_title="Unsupported File Type",
            error_message=(
                f"File '{original_name}' has an unsupported extension (.{file_ext}). "
                f"VisionGuard only accepts image formats: {', '.join(sorted(Config.ALLOWED_EXTENSIONS)).upper()}."
            )
        ), 400

    # In-memory byte read (Zero disk retention requirement)
    image_bytes = file.read()

    # Validation Failure Case 2: File exceeds size limit (fallback check)
    if len(image_bytes) > Config.MAX_CONTENT_LENGTH:
        return render_template("error.html",
            error_code=413,
            error_title="File Size Limit Exceeded",
            error_message=(
                f"Uploaded file '{clean_filename}' ({len(image_bytes) / (1024*1024):.2f} MB) exceeds "
                f"the maximum allowed upload limit of {Config.MAX_CONTENT_LENGTH / (1024*1024):.1f} MB."
            )
        ), 413

    # Validation Failure Case 3: Undecodable / Corrupted image byte payload
    try:
        prediction_result = predictor.predict(image_bytes, filename=clean_filename)
    except ValueError as val_err:
        return render_template("error.html",
            error_code=400,
            error_title="Corrupted or Invalid Image",
            error_message=(
                f"File '{clean_filename}' appears to have an image extension but its internal byte structure "
                f"cannot be decoded by the image engine. Reason: {val_err}"
            )
        ), 400
    except Exception as general_err:
        return render_template("error.html",
            error_code=500,
            error_title="Inference Pipeline Error",
            error_message=f"An unexpected internal error occurred during tensor evaluation: {general_err}"
        ), 500

    # Log immutable audit entry into SQLite database (Task 5 requirement)
    try:
        db_helper.log_prediction(
            db_path=Config.DATABASE_PATH,
            filename=clean_filename,
            verdict=prediction_result["verdict"],
            verdict_code=prediction_result["verdict_code"],
            confidence=prediction_result["confidence"],
            raw_prob=prediction_result["raw_probability"],
            latency_ms=prediction_result["latency_ms"],
            model_version=prediction_result["model_name"]
        )
    except Exception as db_err:
        print(f"[Warning] Failed to log audit record to SQLite: {db_err}")

    # Render result page with in-memory base64 preview and honest caveat
    return render_template("result.html", result=prediction_result)

@app.route("/history", methods=["GET"])
def history_page():
    """Renders the query audit trail from SQLite (Task 5 requirement)."""
    records = db_helper.get_history(Config.DATABASE_PATH, limit=100)
    stats = db_helper.get_stats(Config.DATABASE_PATH)
    return render_template("history.html", records=records, stats=stats)

@app.route("/health", methods=["GET"])
def health_check():
    """
    Subtask 4.4 Health Check API endpoint.
    Returns JSON status confirming model readiness, active threshold, and database health.
    """
    db_connected = db_helper.check_connection()
    return jsonify({
        "status": "healthy",
        "service": "VisionGuard AI Classifier",
        "model_loaded": predictor.model is not None,
        "model_architecture": Config.MODEL_NAME,
        "decision_threshold": Config.DECISION_THRESHOLD,
        "database_connected": db_connected,
        "max_content_length_mb": round(Config.MAX_CONTENT_LENGTH / (1024 * 1024), 1)
    }), 200

# -------------------------------------------------------------
# 3. Global HTTP Error Handlers
# -------------------------------------------------------------
@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template("error.html",
        error_code=413,
        error_title="Payload Too Large (HTTP 413)",
        error_message="The uploaded file exceeds the maximum permissible payload size of 5 Megabytes (5MB)."
    ), 413

@app.errorhandler(400)
def bad_request(error):
    return render_template("error.html",
        error_code=400,
        error_title="Bad Request (HTTP 400)",
        error_message="The request is malformed or violates security validation policies."
    ), 400

@app.errorhandler(404)
def not_found(error):
    return render_template("error.html",
        error_code=404,
        error_title="Page Not Found (HTTP 404)",
        error_message="The requested route does not exist on this server."
    ), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template("error.html",
        error_code=500,
        error_title="Internal Server Error (HTTP 500)",
        error_message="A server-side exception occurred. Debug mode is disabled for security."
    ), 500

# -------------------------------------------------------------
# 4. Entrypoint
# -------------------------------------------------------------
if __name__ == "__main__":
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
