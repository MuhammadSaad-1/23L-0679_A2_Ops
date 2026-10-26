import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import tensorflow as tf
from PIL import Image

try:
    from dotenv import load_dotenv
    import mlflow
    import mlflow.keras
    from mlflow.tracking import MlflowClient
except ImportError:
    raise ImportError("Dependencies missing. Run: uv pip install python-dotenv mlflow")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "scripts" else SCRIPT_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "task3_outputs")
TASK2_DIR = os.path.join(BASE_DIR, "task2_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Setup MLflow Tracking
# -------------------------------------------------------------
env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
else:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlruns_dir = os.path.join(BASE_DIR, "mlruns")
    mlflow.set_tracking_uri(f"file:///{mlruns_dir.replace(os.sep, '/')}")

client = MlflowClient()
model_registry_name = "VisionGuard_Classifier"

print("=" * 65)
print(" TASK 3.6: MODEL REGISTRY & ALIAS INFERENCE")
print(f" Tracking URI       : {mlflow.get_tracking_uri()}")
print(f" Registered Model   : {model_registry_name}")
print("=" * 65)

# -------------------------------------------------------------
# 2. Register Two Versions in Model Registry
# -------------------------------------------------------------
baseline_path = os.path.join(TASK2_DIR, "2.2_baseline_best.keras")
transfer_path = os.path.join(TASK2_DIR, "2.3_transfer_learning_best.keras")

if not os.path.exists(transfer_path):
    transfer_path = os.path.join(TASK2_DIR, "transfer_learning_best.keras")
if not os.path.exists(baseline_path):
    baseline_path = os.path.join(TASK2_DIR, "baseline_best.keras")

print("\n--- Registering Version 1 (Baseline CNN) ---")
with mlflow.start_run(run_name="registry_v1_baseline"):
    m1 = tf.keras.models.load_model(baseline_path)
    mlflow.set_tag("version_description", "Baseline CNN from scratch")
    v1_info = mlflow.keras.log_model(
        m1,
        artifact_path="model",
        registered_model_name=model_registry_name
    )
print(f"Version 1 Registered successfully.")

print("\n--- Registering Version 2 (Transfer Learning MobileNetV2) ---")
with mlflow.start_run(run_name="registry_v2_transfer_learning"):
    m2 = tf.keras.models.load_model(transfer_path)
    mlflow.set_tag("version_description", "Fine-tuned MobileNetV2 with top 30 unfrozen layers")
    v2_info = mlflow.keras.log_model(
        m2,
        artifact_path="model",
        registered_model_name=model_registry_name
    )
print(f"Version 2 Registered successfully.")

# Assign alias 'champion' to Version 2
try:
    client.set_registered_model_alias(
        name=model_registry_name,
        alias="champion",
        version="2"
    )
    print(f"\n[Alias Assigned] Alias '@champion' assigned to Version 2 of '{model_registry_name}'")
except Exception as e:
    print(f"Note on alias assignment: {e}")

# -------------------------------------------------------------
# 3. Load Model by Alias and Perform Single Image Inference
# -------------------------------------------------------------
print("\n" + "=" * 65)
print(" LOADING CHAMPION MODEL BY ALIAS: models:/VisionGuard_Classifier@champion")
print("=" * 65)

champion_uri = f"models:/{model_registry_name}@champion"
print(f"Resolving model from: {champion_uri} ...")

try:
    champion_model = mlflow.keras.load_model(champion_uri)
    print("Champion model loaded successfully from MLflow Registry!")
except Exception as e:
    print(f"Loading via URI fallback to latest local version: {e}")
    champion_model = tf.keras.models.load_model(transfer_path)

# Find a test sample image
data_dir = os.path.join(BASE_DIR, "data")
sample_img_path = None
for root, dirs, files in os.walk(data_dir):
    for f in files:
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            sample_img_path = os.path.join(root, f)
            break
    if sample_img_path:
        break

if not sample_img_path:
    raise FileNotFoundError("No image found in data directory for testing.")

print(f"\nTesting Inference on Sample Image:")
print(f" - Image Path: {sample_img_path}")

img = Image.open(sample_img_path).convert("RGB").resize((224, 224))
img_arr = np.array(img, dtype=np.float32)
img_batch = np.expand_dims(img_arr, axis=0)

pred_prob = float(champion_model.predict(img_batch, verbose=0)[0][0])
pred_label = "RealArt (Human Created)" if pred_prob >= 0.5 else "AiArtData (AI Generated)"
confidence = pred_prob if pred_prob >= 0.5 else (1.0 - pred_prob)

print("-" * 50)
print(f" PREDICTION RESULT: {pred_label}")
print(f" Raw Output P(Real): {pred_prob:.4f}")
print(f" Confidence Score  : {confidence * 100:.2f}%")
print("-" * 50)