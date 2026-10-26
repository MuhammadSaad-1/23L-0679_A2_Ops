import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# -------------------------------------------------------------
# Configuration and Reproducibility (Task 2.4)
# -------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "scripts" else SCRIPT_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "task2_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# -------------------------------------------------------------
# 1. Load Predefined Dataset Splits
# -------------------------------------------------------------
splits_file = os.path.join(OUTPUT_DIR, "2.1_dataset_splits.csv")
if not os.path.exists(splits_file):
    splits_file = os.path.join(OUTPUT_DIR, "dataset_splits.csv")
if not os.path.exists(splits_file):
    raise FileNotFoundError(f"Splits file not found in '{OUTPUT_DIR}'. Please run '2.1_input_pipeline.py' first.")

df_splits = pd.read_csv(splits_file)
# Ensure absolute filepaths
df_splits["abs_filepath"] = df_splits["filepath"].apply(lambda p: p if os.path.isabs(p) else os.path.join(BASE_DIR, p))

train_df = df_splits[df_splits["split"] == "train"]
val_df = df_splits[df_splits["split"] == "val"]

print(f"Loaded splits: {len(train_df)} train images, {len(val_df)} val images.")

# Pipeline helpers
def load_and_preprocess_image(filepath, label):
    raw_img = tf.io.read_file(filepath)
    img = tf.io.decode_image(raw_img, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img, tf.cast(label, tf.float32)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", seed=RANDOM_SEED),
    tf.keras.layers.RandomRotation(0.1, seed=RANDOM_SEED),
    tf.keras.layers.RandomZoom(0.1, seed=RANDOM_SEED),
    tf.keras.layers.RandomContrast(0.1, seed=RANDOM_SEED)
], name="train_augmentation")

def build_ds(df_subset, is_training=False):
    ds = tf.data.Dataset.from_tensor_slices((df_subset["abs_filepath"].values, df_subset["label"].values))
    if is_training:
        ds = ds.shuffle(buffer_size=len(df_subset), seed=RANDOM_SEED)
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        ds = ds.batch(BATCH_SIZE)
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.batch(BATCH_SIZE)
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = build_ds(train_df, is_training=True)
val_ds = build_ds(val_df, is_training=False)

# -------------------------------------------------------------
# 2. Design Baseline CNN Architecture (Task 2.2)
# -------------------------------------------------------------
def build_baseline_cnn():
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="input_image")
    
    # Layer 0: Normalization [0, 255] -> [0, 1]
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescaling_norm")(inputs)
    
    # Block 1: Low-level features (edges, color gradients)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", name="conv1_32")(x)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.Activation("relu", name="relu1")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    
    # Block 2: Mid-level features (textures, geometric patterns)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", name="conv2_64")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.Activation("relu", name="relu2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    
    # Block 3: High-level features (domain-specific semantic cues)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", name="conv3_128")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.Activation("relu", name="relu3")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    
    # Spatial Reduction and Classification Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(0.4, seed=RANDOM_SEED, name="dropout_regularization")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="sigmoid_classifier")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="VisionGuard_Baseline_CNN")
    return model

model = build_baseline_cnn()
model.summary()

# Export Layer Summary Table to CSV for report
layer_records = []
for layer in model.layers:
    layer_records.append({
        "Layer Name": layer.name,
        "Type": layer.__class__.__name__,
        "Output Shape": str(layer.output.shape),
        "Param Count": layer.count_params()
    })
df_layers = pd.DataFrame(layer_records)
layer_table_path = os.path.join(OUTPUT_DIR, "2.2_baseline_cnn_layer_table.csv")
df_layers.to_csv(layer_table_path, index=False)
print(f"\n[Saved] Layer summary table saved to: {layer_table_path}")

# -------------------------------------------------------------
# 3. Compile Model with Justified Loss, Optimizer & Metrics
# -------------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

# -------------------------------------------------------------
# 4. Training Discipline & Callbacks (Task 2.4)
# -------------------------------------------------------------
checkpoint_path = os.path.join(OUTPUT_DIR, "2.2_baseline_best.keras")
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

print("\n--- Starting Baseline CNN Training ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Identify best epoch based on minimum val_loss
val_losses = history.history["val_loss"]
best_epoch_idx = int(np.argmin(val_losses))  # 0-indexed
best_epoch_num = best_epoch_idx + 1
best_val_loss = val_losses[best_epoch_idx]
best_val_acc = history.history["val_accuracy"][best_epoch_idx]
print(f"\n>>> Best Epoch: {best_epoch_num} (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.4f})")

# Save training history
with open(os.path.join(OUTPUT_DIR, "2.2_baseline_history.json"), "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=4)

# -------------------------------------------------------------
# 5. Plot Loss & Accuracy Curves with Best Epoch Marked (Task 2.4)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs_range = range(1, len(history.history["loss"]) + 1)

# Loss Curve
axes[0].plot(epochs_range, history.history["loss"], label="Training Loss", marker="o", color="#1f77b4")
axes[0].plot(epochs_range, history.history["val_loss"], label="Validation Loss", marker="s", color="#ff7f0e")
axes[0].axvline(best_epoch_num, color="red", linestyle="--", label=f"Best Epoch ({best_epoch_num})")
axes[0].scatter(best_epoch_num, best_val_loss, color="red", s=80, zorder=5)
axes[0].set_title("Baseline CNN: Loss Curve")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Binary Crossentropy Loss")
axes[0].legend()
axes[0].grid(True, linestyle=":", alpha=0.6)

# Accuracy Curve
axes[1].plot(epochs_range, history.history["accuracy"], label="Training Accuracy", marker="o", color="#1f77b4")
axes[1].plot(epochs_range, history.history["val_accuracy"], label="Validation Accuracy", marker="s", color="#ff7f0e")
axes[1].axvline(best_epoch_num, color="red", linestyle="--", label=f"Best Epoch ({best_epoch_num})")
axes[1].scatter(best_epoch_num, best_val_acc, color="red", s=80, zorder=5)
axes[1].set_title("Baseline CNN: Accuracy Curve")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, linestyle=":", alpha=0.6)

plt.suptitle("Figure 2.2: Baseline CNN Training & Validation Dynamics", fontsize=14, y=1.02)
plt.tight_layout()
curves_path = os.path.join(OUTPUT_DIR, "2.2_baseline_training_curves.png")
plt.savefig(curves_path, dpi=300)
plt.close()
print(f"[Saved] Training curves saved to: {curves_path}")
print(f"[Saved] Best baseline model saved to: {checkpoint_path}")
print("\nTask 2.2 & 2.4 baseline completed successfully.")
