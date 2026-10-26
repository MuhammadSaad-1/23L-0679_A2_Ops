import os
import sys

# Configure UTF-8 for Windows PowerShell to prevent emoji encoding errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError("python-dotenv is not installed. Please run:\n  uv pip install python-dotenv")

try:
    import mlflow
    import mlflow.keras
except ImportError:
    raise ImportError("mlflow is not installed. Please run:\n  uv pip install mlflow")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "scripts" else SCRIPT_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "task3_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Load Credentials from .env
# -------------------------------------------------------------
env_path = os.path.join(BASE_DIR, ".env")
if not os.path.exists(env_path):
    print(f"\n[WARNING] '.env' file not found at: {env_path}")
    print("Please create '.env' from '.env.example' and add your DagsHub credentials.")
    print("Example content:")
    print("  MLFLOW_TRACKING_URI=https://dagshub.com/<Username>/<Repo>.mlflow")
    print("  MLFLOW_TRACKING_USERNAME=<Username>")
    print("  MLFLOW_TRACKING_PASSWORD=<Your_DagsHub_Token>")
    exit(1)

load_dotenv(dotenv_path=env_path)

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")

if not tracking_uri or not username or not password or "your_dagshub_token_here" in password:
    print("\n[ERROR] Incomplete credentials in .env!")
    print(f"MLFLOW_TRACKING_URI      : {tracking_uri}")
    print(f"MLFLOW_TRACKING_USERNAME : {username}")
    print("Please set your valid DagsHub access token in .env under MLFLOW_TRACKING_PASSWORD.")
    exit(1)

# Configure MLflow remote tracking
os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = password
mlflow.set_tracking_uri(tracking_uri)

experiment_name = "Task3.2_DagsHub_Toy_Experiment"
mlflow.set_experiment(experiment_name)

print("=" * 65)
print(" TASK 3.2: DAGSHUB REMOTE MLFLOW TRACKING LAB")
print(f" Remote Tracking URI : {tracking_uri}")
print(f" Authenticated User  : {username}")
print(f" Experiment Name     : {experiment_name}")
print("=" * 65)

# -------------------------------------------------------------
# 2. Run Remote Toy Experiment on DagsHub
# -------------------------------------------------------------
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train[:2000] / 255.0
y_train = y_train[:2000]
x_val = x_val[:500] / 255.0
y_val = y_val[:500]

with mlflow.start_run(run_name="dagshub_remote_toy_run"):
    print("\nLogging run to DagsHub remote MLflow...")
    mlflow.log_param("dataset", "Fashion-MNIST")
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("epochs", 4)
    mlflow.set_tag("environment", "remote_dagshub")
    mlflow.set_tag("task", "3.2_dagshub_setup")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    class DagsHubEpochCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            for k, v in logs.items():
                mlflow.log_metric(k, float(v), step=epoch + 1)

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=4,
        batch_size=32,
        callbacks=[DagsHubEpochCallback()],
        verbose=1
    )
    
    # Save loss plot and log to DagsHub
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, 5), history.history["loss"], marker="o", label="Train Loss")
    plt.plot(range(1, 5), history.history["val_loss"], marker="s", label="Val Loss")
    plt.title("DagsHub Toy Run Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "3.2_dagshub_toy_loss.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    
    mlflow.log_artifact(plot_path, artifact_path="plots")
    mlflow.keras.log_model(model, artifact_path="model")
    print("\nSuccessfully logged parameters, metrics, plot, and model to DagsHub!")

print("\n" + "=" * 65)
print(f" View your live remote experiment on DagsHub at:")
print(f"   {tracking_uri.replace('.mlflow', '/experiments')}")
print("=" * 65)
