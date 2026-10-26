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
STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 10

# -------------------------------------------------------------
# 1. Load Predefined Dataset Splits
# -------------------------------------------------------------
splits_file = os.path.join(OUTPUT_DIR, "2.1_dataset_splits.csv")
if not os.path.exists(splits_file):
    splits_file = os.path.join(OUTPUT_DIR, "dataset_splits.csv")
if not os.path.exists(splits_file):
    raise FileNotFoundError(f"Splits file not found in '{OUTPUT_DIR}'. Please run '2.1_input_pipeline.py' first.")

df_splits = pd.read_csv(splits_file)
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
# 2. Transfer Learning: MobileNetV2 Architecture
# -------------------------------------------------------------
# MobileNetV2 expects input normalized to [-1, 1].
# tf.keras.applications.mobilenet_v2.preprocess_input performs: (x / 127.5) - 1.0
inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="input_image")
preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

base_model = tf.keras.applications.MobileNetV2(
    input_tensor=preprocessed,
    include_top=False,
    weights="imagenet"
)

# Freeze base model for Stage 1
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
x = tf.keras.layers.Dropout(0.3, seed=RANDOM_SEED, name="head_dropout")(x)
x = tf.keras.layers.Dense(64, activation="relu", name="head_dense")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="sigmoid_classifier")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="VisionGuard_MobileNetV2")

# -------------------------------------------------------------
# 3. Stage 1: Feature Extraction (Frozen Backbone)
# -------------------------------------------------------------
print("\n" + "=" * 65)
print(" STAGE 1: Feature Extraction (Backbone Frozen, Head Trainable)")
print("=" * 65)

stage1_trainable_layers = [l.name for l in model.layers if l.trainable]
stage1_frozen_layers = [l.name for l in model.layers if not l.trainable]
print(f"Total model layers: {len(model.layers)}")
print(f"Trainable layers in Stage 1: {len(stage1_trainable_layers)}")
print(f"Frozen layers in Stage 1   : {len(stage1_frozen_layers)}")

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

checkpoint_stage1 = os.path.join(OUTPUT_DIR, "2.3_mobilenetv2_stage1.keras")
history_stage1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=STAGE1_EPOCHS,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(checkpoint_stage1, monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1)
    ],
    verbose=1
)

# -------------------------------------------------------------
# 4. Stage 2: Fine-Tuning (Unfreeze Last Block)
# -------------------------------------------------------------
print("\n" + "=" * 65)
print(" STAGE 2: Fine-Tuning (Unfreezing Last 30 Layers of MobileNetV2)")
print("=" * 65)

base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at:]:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    else:
        layer.trainable = True

stage2_trainable_layers = [l.name for l in base_model.layers if l.trainable]
print(f"Unfrozen backbone layers in Stage 2: {len(stage2_trainable_layers)}")

# Record layer statuses to text file for assignment report evidence
layers_doc_path = os.path.join(OUTPUT_DIR, "2.3_transfer_learning_layers.txt")
with open(layers_doc_path, "w") as f:
    f.write("=== Stage 1 Trainable Layers ===\n")
    for name in stage1_trainable_layers:
        f.write(f" - {name}\n")
    f.write(f"\nTotal Trainable in Stage 1: {len(stage1_trainable_layers)}\n\n")
    f.write("=== Stage 2 Unfrozen Backbone Layers (Top 30 Block) ===\n")
    for name in stage2_trainable_layers:
        f.write(f" - {name}\n")
    f.write(f"\nTotal Trainable Backbone in Stage 2: {len(stage2_trainable_layers)}\n")
print(f"[Saved] Layer status document saved to: {layers_doc_path}")

# Re-compile with much lower learning rate to preserve pretrained weights
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

checkpoint_stage2 = os.path.join(OUTPUT_DIR, "2.3_transfer_learning_best.keras")
callbacks_stage2 = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_stage2, monitor="val_loss", save_best_only=True, verbose=1),
    # Also save without prefix for fallback compatibility
    tf.keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_DIR, "transfer_learning_best.keras"), monitor="val_loss", save_best_only=True, verbose=0),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=1)
]

initial_epochs = len(history_stage1.history["loss"])
total_epochs = initial_epochs + STAGE2_EPOCHS

history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=initial_epochs,
    callbacks=callbacks_stage2,
    verbose=1
)

# -------------------------------------------------------------
# 5. Combined Training & Fine-Tuning Curves (Task 2.3 Evidence)
# -------------------------------------------------------------
loss_full = history_stage1.history["loss"] + history_stage2.history["loss"]
val_loss_full = history_stage1.history["val_loss"] + history_stage2.history["val_loss"]
acc_full = history_stage1.history["accuracy"] + history_stage2.history["accuracy"]
val_acc_full = history_stage1.history["val_accuracy"] + history_stage2.history["val_accuracy"]

best_full_epoch = int(np.argmin(val_loss_full)) + 1
best_full_val_loss = val_loss_full[best_full_epoch - 1]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
x_epochs = range(1, len(loss_full) + 1)

# Loss
axes[0].plot(x_epochs, loss_full, label="Training Loss", marker="o", color="#1f77b4")
axes[0].plot(x_epochs, val_loss_full, label="Validation Loss", marker="s", color="#ff7f0e")
axes[0].axvline(initial_epochs, color="green", linestyle=":", linewidth=2, label="Fine-Tuning Start")
axes[0].axvline(best_full_epoch, color="red", linestyle="--", label=f"Best Overall Epoch ({best_full_epoch})")
axes[0].set_title("Transfer Learning (MobileNetV2): Loss Dynamics")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Binary Crossentropy Loss")
axes[0].legend()
axes[0].grid(True, linestyle=":", alpha=0.6)

# Accuracy
axes[1].plot(x_epochs, acc_full, label="Training Accuracy", marker="o", color="#1f77b4")
axes[1].plot(x_epochs, val_acc_full, label="Validation Accuracy", marker="s", color="#ff7f0e")
axes[1].axvline(initial_epochs, color="green", linestyle=":", linewidth=2, label="Fine-Tuning Start")
axes[1].axvline(best_full_epoch, color="red", linestyle="--", label=f"Best Overall Epoch ({best_full_epoch})")
axes[1].set_title("Transfer Learning (MobileNetV2): Accuracy Dynamics")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, linestyle=":", alpha=0.6)

plt.suptitle("Figure 2.3: MobileNetV2 Two-Stage Training Curves (Feature Extraction & Fine-Tuning)", fontsize=14, y=1.02)
plt.tight_layout()
curves_path = os.path.join(OUTPUT_DIR, "2.3_transfer_learning_curves.png")
plt.savefig(curves_path, dpi=300)
plt.close()
print(f"[Saved] Transfer learning curves saved to: {curves_path}")
print(f"\nTask 2.3 completed successfully. Best model saved to: {checkpoint_stage2}")
