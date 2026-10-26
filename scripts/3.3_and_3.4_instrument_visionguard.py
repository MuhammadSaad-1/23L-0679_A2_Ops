import os
import sys
import subprocess

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

try:
    from dotenv import load_dotenv
    import mlflow
    import mlflow.keras
    import mlflow.tensorflow
except ImportError:
    raise ImportError("Dependencies missing. Please run:\n  uv pip install python-dotenv mlflow")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "scripts" else SCRIPT_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "task3_outputs")
TASK2_DIR = os.path.join(BASE_DIR, "task2_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Load MLflow Tracking Config
# -------------------------------------------------------------
env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
else:
    # Fallback to local mlruns if .env is not present
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlruns_dir = os.path.join(BASE_DIR, "mlruns")
    mlflow.set_tracking_uri(f"file:///{mlruns_dir.replace(os.sep, '/')}")

experiment_name = "Task3.3_VisionGuard_Tracking"
mlflow.set_experiment(experiment_name)

print("=" * 65)
print(f" TASK 3.3 & 3.4: INSTRUMENT VISIONGUARD TRAINING")
print(f" Tracking URI : {mlflow.get_tracking_uri()}")
print(f" Experiment   : {experiment_name}")
print("=" * 65)

# -------------------------------------------------------------
# 2. Extract Metadata Tags (Git Commit, DVC Data Version, Seed)
# -------------------------------------------------------------
# Git Commit SHA
try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR
    ).decode("utf-8").strip()
except Exception:
    git_commit = "uncommitted_local"

# DVC Data Version from data.dvc
dvc_path = os.path.join(BASE_DIR, "data.dvc")
dvc_version = "unknown"
if os.path.exists(dvc_path):
    try:
        with open(dvc_path, "r") as f:
            dvc_info = yaml.safe_load(f)
            dvc_version = dvc_info["outs"][0]["md5"]
    except Exception as e:
        dvc_version = f"error_reading_dvc: {e}"

RANDOM_SEED = 42

print(f"Metadata Tags Extracted:")
print(f" - Git Commit SHA  : {git_commit}")
print(f" - DVC Data Version: {dvc_version}")
print(f" - Seed            : {RANDOM_SEED}")

# -------------------------------------------------------------
# 3. Load Splits & Build Datasets
# -------------------------------------------------------------
splits_file = os.path.join(TASK2_DIR, "2.1_dataset_splits.csv")
if not os.path.exists(splits_file):
    splits_file = os.path.join(TASK2_DIR, "dataset_splits.csv")
if not os.path.exists(splits_file):
    raise FileNotFoundError(f"Splits not found in {TASK2_DIR}. Run 2.1 first.")

df_splits = pd.read_csv(splits_file)
df_splits["abs_filepath"] = df_splits["filepath"].apply(lambda p: p if os.path.isabs(p) else os.path.join(BASE_DIR, p))

train_df = df_splits[df_splits["split"] == "train"]
val_df = df_splits[df_splits["split"] == "val"]
test_df = df_splits[df_splits["split"] == "test"]

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def load_img(filepath, label):
    raw_img = tf.io.read_file(filepath)
    img = tf.io.decode_image(raw_img, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img, tf.cast(label, tf.float32)

def make_ds(df_sub, is_train=False):
    ds = tf.data.Dataset.from_tensor_slices((df_sub["abs_filepath"].values, df_sub["label"].values))
    if is_train:
        ds = ds.shuffle(len(df_sub), seed=RANDOM_SEED)
    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(train_df, is_train=True)
val_ds = make_ds(val_df, is_train=False)
test_ds = make_ds(test_df, is_train=False)

def create_model():
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

# -------------------------------------------------------------
# 4. RUN A: Manual Logging with Custom Callback & Tags
# -------------------------------------------------------------
print("\n" + "=" * 65)
print(" RUN A: Manual MLflow Logging (Explicit Parameters, Metrics & Tags)")
print("=" * 65)

with mlflow.start_run(run_name="visionguard_manual_logging") as run_a:
    # Log tags
    mlflow.set_tag("git_commit", git_commit)
    mlflow.set_tag("dvc_version", dvc_version)
    mlflow.set_tag("random_seed", str(RANDOM_SEED))
    mlflow.set_tag("logging_method", "manual")
    mlflow.set_tag("architecture", "Custom_CNN_Baseline")
    
    # Log parameters
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("learning_rate", 1e-3)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", 4)
    mlflow.log_param("image_size", "224x224")

    model_manual = create_model()

    class ManualMetricsCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            for metric_name, val in logs.items():
                mlflow.log_metric(f"train_{metric_name}" if not metric_name.startswith("val_") else metric_name,
                                  float(val), step=epoch + 1)

    model_manual.fit(
        train_ds,
        validation_data=val_ds,
        epochs=4,
        callbacks=[ManualMetricsCallback()],
        verbose=1
    )

    # Evaluate on test set
    test_eval = model_manual.evaluate(test_ds, verbose=0)
    mlflow.log_metric("test_loss", float(test_eval[0]))
    mlflow.log_metric("test_accuracy", float(test_eval[1]))
    mlflow.log_metric("test_auc", float(test_eval[2]))

    # Task 3.4: Attach artifacts to Run A
    # 1. Package list
    req_file = os.path.join(OUTPUT_DIR, "3.4_requirements.txt")
    try:
        pip_freeze = subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"]).decode("utf-8")
    except Exception:
        pip_freeze = "tensorflow\npandas\nnumpy\nscikit-learn\nmatplotlib\nmlflow\n"
    with open(req_file, "w") as f:
        f.write(pip_freeze)
    mlflow.log_artifact(req_file, artifact_path="environment")

    # 2. Dataset split lists
    mlflow.log_artifact(splits_file, artifact_path="data_splits")

    # 3. Confusion Matrix and ROC curve from Task 2
    cm_img = os.path.join(TASK2_DIR, "2.5_confusion_matrices.png")
    if os.path.exists(cm_img):
        mlflow.log_artifact(cm_img, artifact_path="evaluation_plots")
    roc_img = os.path.join(TASK2_DIR, "2.5_roc_curves.png")
    if os.path.exists(roc_img):
        mlflow.log_artifact(roc_img, artifact_path="evaluation_plots")

    # 4. Trained model artifact
    mlflow.keras.log_model(model_manual, artifact_path="model")
    print("\n[Run A Complete] Manual metrics, tags, and all artifacts logged successfully!")

# -------------------------------------------------------------
# 5. RUN B: MLflow Autologging
# -------------------------------------------------------------
print("\n" + "=" * 65)
print(" RUN B: MLflow Autologging (mlflow.tensorflow.autolog())")
print("=" * 65)

# Enable automatic logging
mlflow.tensorflow.autolog(log_models=True, log_datasets=False)

with mlflow.start_run(run_name="visionguard_autologging") as run_b:
    mlflow.set_tag("git_commit", git_commit)
    mlflow.set_tag("dvc_version", dvc_version)
    mlflow.set_tag("random_seed", str(RANDOM_SEED))
    mlflow.set_tag("logging_method", "autolog")

    model_auto = create_model()
    model_auto.fit(
        train_ds,
        validation_data=val_ds,
        epochs=4,
        verbose=1
    )
    
    test_eval_auto = model_auto.evaluate(test_ds, verbose=0)
    mlflow.log_metric("test_loss", float(test_eval_auto[0]))
    mlflow.log_metric("test_accuracy", float(test_eval_auto[1]))
    print("\n[Run B Complete] Autologging completed successfully!")

# Disable autolog to prevent interference with downstream scripts
mlflow.tensorflow.autolog(disable=True)
