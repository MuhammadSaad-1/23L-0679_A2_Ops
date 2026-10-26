import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

try:
    import mlflow
    import mlflow.keras
except ImportError:
    raise ImportError("mlflow is not installed. Please run:\n  uv pip install mlflow")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "scripts" else SCRIPT_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "task3_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Allow local file store for MLflow 3.x
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

mlruns_dir = os.path.join(BASE_DIR, "mlruns")
mlflow.set_tracking_uri(f"file:///{mlruns_dir.replace(os.sep, '/')}")
experiment_name = "Task3.1_Local_Toy_Experiment"
mlflow.set_experiment(experiment_name)

print("=" * 65)
print(f" TASK 3.1: LOCAL MLFLOW TOOL LAB")
print(f" Tracking URI: {mlflow.get_tracking_uri()}")
print(f" Experiment  : {experiment_name}")
print("=" * 65)

# -------------------------------------------------------------
# 2. Load Built-in Fashion-MNIST Dataset
# -------------------------------------------------------------
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train[:2000] / 255.0
y_train = y_train[:2000]
x_val = x_val[:500] / 255.0
y_val = y_val[:500]

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

# -------------------------------------------------------------
# 3. Train Two Runs with Different Learning Rates
# -------------------------------------------------------------
learning_rates = [0.001, 0.01]

for idx, lr in enumerate(learning_rates, start=1):
    run_name = f"toy_run_lr_{lr}"
    print(f"\n--- Starting Run {idx}: {run_name} ---")
    
    with mlflow.start_run(run_name=run_name):
        # 1. Log Hyperparameters
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("epochs", 5)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("hidden_units", 64)
        
        # 2. Set Tags
        mlflow.set_tag("task", "3.1_local_toy")
        mlflow.set_tag("dataset", "Fashion-MNIST")
        mlflow.set_tag("developer", "MSaad")
        
        # 3. Build & Compile Model
        model = build_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        # 4. Custom Callback to Log Per-Epoch Metrics
        class MLflowEpochCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                for k, v in logs.items():
                    mlflow.log_metric(k, float(v), step=epoch + 1)
        
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=5,
            batch_size=32,
            callbacks=[MLflowEpochCallback()],
            verbose=1
        )
        
        # 5. Save and Log Training Loss Plot as Artifact
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, 6), history.history["loss"], marker="o", label="Train Loss")
        plt.plot(range(1, 6), history.history["val_loss"], marker="s", label="Val Loss")
        plt.title(f"Loss Curve (LR = {lr})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f"3.1_toy_loss_plot_lr_{lr}.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        
        mlflow.log_artifact(plot_path, artifact_path="plots")
        print(f"Logged plot artifact: {plot_path}")
        
        # 6. Log Trained Keras Model Artifact
        mlflow.keras.log_model(model, artifact_path="model")
        print(f"Logged model artifact to MLflow.")

print(" Task 3.1 Complete! To view the MLflow UI locally, run:")
print(f"   mlflow ui --port 5000")
print(" Then open http://127.0.0.1:5000 in your browser to inspect and compare runs.")
print("=" * 65)
