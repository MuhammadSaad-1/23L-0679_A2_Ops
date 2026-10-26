import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

try:
    from dotenv import load_dotenv
    import mlflow
    import mlflow.keras
except ImportError:
    raise ImportError("Dependencies missing. Run: uv pip install python-dotenv mlflow")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "scripts" else SCRIPT_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "task3_outputs")
TASK2_DIR = os.path.join(BASE_DIR, "task2_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Setup MLflow Tracking (Remote DagsHub or Local Fallback)
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

experiment_name = "Task3.5_VisionGuard_Ablation_Study"
mlflow.set_experiment(experiment_name)

print("=" * 65)
print(f" TASK 3.5: EXPERIMENT COMPARISON (6 ABLATION RUNS)")
print(f" Tracking URI : {mlflow.get_tracking_uri()}")
print(f" Experiment   : {experiment_name}")
print("=" * 65)

# -------------------------------------------------------------
# 2. Load Splits & Data Pipelines
# -------------------------------------------------------------
splits_file = os.path.join(TASK2_DIR, "2.1_dataset_splits.csv")
if not os.path.exists(splits_file):
    splits_file = os.path.join(TASK2_DIR, "dataset_splits.csv")

df_splits = pd.read_csv(splits_file)
df_splits["abs_filepath"] = df_splits["filepath"].apply(lambda p: p if os.path.isabs(p) else os.path.join(BASE_DIR, p))

train_df = df_splits[df_splits["split"] == "train"]
val_df = df_splits[df_splits["split"] == "val"]
test_df = df_splits[df_splits["split"] == "test"]

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 42

def load_img(filepath, label):
    raw = tf.io.read_file(filepath)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img, tf.cast(label, tf.float32)

aug_pipeline = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", seed=RANDOM_SEED),
    tf.keras.layers.RandomRotation(0.1, seed=RANDOM_SEED),
    tf.keras.layers.RandomZoom(0.1, seed=RANDOM_SEED)
])

def create_datasets(use_augmentation=True):
    train_ds = tf.data.Dataset.from_tensor_slices((train_df["abs_filepath"].values, train_df["label"].values))
    train_ds = train_ds.shuffle(len(train_df), seed=RANDOM_SEED)
    train_ds = train_ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    if use_augmentation:
        train_ds = train_ds.batch(BATCH_SIZE)
        train_ds = train_ds.map(lambda x, y: (aug_pipeline(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_df["abs_filepath"].values, val_df["label"].values))
    val_ds = val_ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_df["abs_filepath"].values, test_df["label"].values))
    test_ds = test_ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds

# -------------------------------------------------------------
# 3. Model Builders
# -------------------------------------------------------------
def build_baseline_cnn(dropout=0.4):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def build_mobilenet_v2(unfreeze_layers=30, dropout=0.3):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    pre = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    base = tf.keras.applications.MobileNetV2(input_tensor=pre, include_top=False, weights="imagenet")
    base.trainable = True
    for layer in base.layers[:-unfreeze_layers]:
        layer.trainable = False
    for layer in base.layers[-unfreeze_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# -------------------------------------------------------------
# 4. Define 6 Meaningful Ablation Experiments
# -------------------------------------------------------------
experiments = [
    {
        "name": "Run_1_Baseline_LR_1e3",
        "model_type": "Baseline_CNN",
        "lr": 1e-3,
        "augmentation": True,
        "unfrozen_layers": "N/A",
        "dropout": 0.4,
        "description": "Standard baseline CNN with Adam lr=1e-3"
    },
    {
        "name": "Run_2_Baseline_LR_1e4",
        "model_type": "Baseline_CNN",
        "lr": 1e-4,
        "augmentation": True,
        "unfrozen_layers": "N/A",
        "dropout": 0.4,
        "description": "Ablation on Learning Rate: Reduced to 1e-4"
    },
    {
        "name": "Run_3_Baseline_NoAugmentation",
        "model_type": "Baseline_CNN",
        "lr": 1e-3,
        "augmentation": False,
        "unfrozen_layers": "N/A",
        "dropout": 0.4,
        "description": "Ablation on Regularization: Removed data augmentation"
    },
    {
        "name": "Run_4_MobileNetV2_Frozen10",
        "model_type": "MobileNetV2",
        "lr": 5e-4,
        "augmentation": True,
        "unfrozen_layers": 10,
        "dropout": 0.3,
        "description": "Ablation on Backbone: MobileNetV2 with top 10 layers unfrozen"
    },
    {
        "name": "Run_5_MobileNetV2_Frozen30_Champion",
        "model_type": "MobileNetV2",
        "lr": 5e-4,
        "augmentation": True,
        "unfrozen_layers": 30,
        "dropout": 0.3,
        "description": "Champion Configuration: MobileNetV2 with top 30 layers unfrozen"
    },
    {
        "name": "Run_6_MobileNetV2_HighDropout",
        "model_type": "MobileNetV2",
        "lr": 5e-4,
        "augmentation": True,
        "unfrozen_layers": 30,
        "dropout": 0.5,
        "description": "Ablation on Dropout: Increased dropout to 0.50"
    }
]

comparison_results = []

for idx, exp in enumerate(experiments, start=1):
    print(f"\n[{idx}/6] Executing Run: {exp['name']}...")
    train_ds, val_ds, test_ds = create_datasets(use_augmentation=exp["augmentation"])

    if exp["model_type"] == "Baseline_CNN":
        model = build_baseline_cnn(dropout=exp["dropout"])
    else:
        model = build_mobilenet_v2(unfreeze_layers=exp["unfrozen_layers"], dropout=exp["dropout"])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=exp["lr"]),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    with mlflow.start_run(run_name=exp["name"]):
        # Log all params
        mlflow.log_param("model_type", exp["model_type"])
        mlflow.log_param("learning_rate", exp["lr"])
        mlflow.log_param("augmentation", exp["augmentation"])
        mlflow.log_param("unfrozen_layers", exp["unfrozen_layers"])
        mlflow.log_param("dropout", exp["dropout"])
        mlflow.log_param("description", exp["description"])

        # Quick 3-epoch ablation training
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=3,
            verbose=0
        )

        val_loss = float(history.history["val_loss"][-1])
        val_acc = float(history.history["val_accuracy"][-1])
        val_auc = float(history.history["val_auc"][-1])

        test_eval = model.evaluate(test_ds, verbose=0)
        test_loss = float(test_eval[0])
        test_acc = float(test_eval[1])
        test_auc = float(test_eval[2])

        # Log metrics
        mlflow.log_metric("val_loss", val_loss)
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("val_auc", val_auc)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_auc", test_auc)

        comparison_results.append({
            "Run Name": exp["name"],
            "Backbone": exp["model_type"],
            "Learning Rate": exp["lr"],
            "Augmentation": exp["augmentation"],
            "Unfrozen Layers": exp["unfrozen_layers"],
            "Dropout": exp["dropout"],
            "Val Acc": round(val_acc, 4),
            "Test Acc": round(test_acc, 4),
            "Test AUC": round(test_auc, 4),
            "Test Loss": round(test_loss, 4)
        })
        print(f" -> Completed: Test Acc = {test_acc:.4f}, Test AUC = {test_auc:.4f}")

# -------------------------------------------------------------
# 5. Export Comparison Table and Visualization
# -------------------------------------------------------------
df_comp = pd.DataFrame(comparison_results)
comp_csv_path = os.path.join(OUTPUT_DIR, "3.5_experiment_comparison.csv")
df_comp.to_csv(comp_csv_path, index=False)
print("\n" + "=" * 65)
print("             EXPERIMENT COMPARISON SUMMARY TABLE (TASK 3.5)      ")
print("=" * 65)
print(df_comp.to_string(index=False))
print(f"\n[Saved] Comparison table saved to: {comp_csv_path}")

# Plot comparison bar chart
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(df_comp))
width = 0.35

ax.bar(x - width/2, df_comp["Test Acc"], width, label="Test Accuracy", color="#1f77b4")
ax.bar(x + width/2, df_comp["Test AUC"], width, label="Test AUC", color="#2ca02c")
ax.set_ylabel("Score")
ax.set_title("Figure 3.5: Comparison of 6 Ablation Runs across Key Metrics", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(df_comp["Run Name"], rotation=25, ha="right", fontsize=9)
ax.set_ylim(0.5, 1.0)
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "3.5_experiment_comparison_chart.png")
plt.savefig(chart_path, dpi=300)
plt.close()
print(f"[Saved] Comparison chart saved to: {chart_path}")
