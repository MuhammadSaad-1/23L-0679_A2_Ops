import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------
# Path Resolution & Reproducibility
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

# -------------------------------------------------------------
# 1. Dataset Splitting (70% Train, 15% Validation, 15% Test)
# -------------------------------------------------------------
csv_audit = os.path.join(BASE_DIR, "task1_outputs", "1.2_dataset_audit.csv")
if not os.path.exists(csv_audit):
    csv_audit = os.path.join(BASE_DIR, "1.2_dataset_audit.csv")
if not os.path.exists(csv_audit):
    raise FileNotFoundError(f"'{csv_audit}' not found. Please run Task 1.2 audit first.")

df = pd.read_csv(csv_audit)
print(f"Total valid images loaded from audit: {len(df)}")
print(f"Classes found: {df['class'].value_counts().to_dict()}")

# Map classes to binary labels: AiArtData -> 0, RealArt -> 1
class_mapping = {"AiArtData": 0, "RealArt": 1}
df["label"] = df["class"].map(class_mapping)

# Ensure absolute filepaths for robust dataset pipeline loading
df["abs_filepath"] = df["filepath"].apply(lambda p: p if os.path.isabs(p) else os.path.join(BASE_DIR, p))

# Stratified split: Train (70%), Val (15%), Test (15%)
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=RANDOM_SEED, stratify=df["label"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=RANDOM_SEED, stratify=temp_df["label"]
)

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

all_splits = pd.concat([train_df, val_df, test_df], ignore_index=True)
splits_csv_path = os.path.join(OUTPUT_DIR, "2.1_dataset_splits.csv")
all_splits.to_csv(splits_csv_path, index=False)
# Also save without prefix for backwards compatibility
all_splits.to_csv(os.path.join(OUTPUT_DIR, "dataset_splits.csv"), index=False)

print(f"\n[Saved] Dataset splits saved to: {splits_csv_path}")
print(f"Train split count : {len(train_df)} ({len(train_df)/len(df):.1%})")
print(f"Val split count   : {len(val_df)} ({len(val_df)/len(df):.1%})")
print(f"Test split count  : {len(test_df)} ({len(test_df)/len(df):.1%})")

# -------------------------------------------------------------
# 2. tf.data Data Pipeline Definition
# -------------------------------------------------------------
def load_and_preprocess_image(filepath, label):
    raw_img = tf.io.read_file(filepath)
    img = tf.io.decode_image(raw_img, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img, tf.cast(label, tf.float32)

# Training Data Augmentation Layer (Applied to training data only)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", seed=RANDOM_SEED),
    tf.keras.layers.RandomRotation(0.1, seed=RANDOM_SEED),
    tf.keras.layers.RandomZoom(0.1, seed=RANDOM_SEED),
    tf.keras.layers.RandomContrast(0.1, seed=RANDOM_SEED)
], name="data_augmentation")

def build_dataset(filepaths, labels, is_training=False, batch_size=BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(filepaths), seed=RANDOM_SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# -------------------------------------------------------------
# 3. Create Datasets and Visualize One Augmented Batch
# -------------------------------------------------------------
train_ds = build_dataset(train_df["abs_filepath"].values, train_df["label"].values, is_training=True)
val_ds = build_dataset(val_df["abs_filepath"].values, val_df["label"].values, is_training=False)
test_ds = build_dataset(test_df["abs_filepath"].values, test_df["label"].values, is_training=False)

print("\n--- Pipeline Construction Summary ---")
print("1. Filepath loading -> tf.data.Dataset.from_tensor_slices")
print("2. Parallel I/O decoding -> tf.io.decode_image (converts to RGB 3-channel)")
print(f"3. Resizing -> {IMAGE_SIZE} via bilinear interpolation")
print("4. Augmentations -> RandomFlip, RandomRotation(10%), RandomZoom(10%), RandomContrast(10%)")
print(f"5. Batching -> Batch size {BATCH_SIZE}")
print("6. Prefetching -> tf.data.AUTOTUNE (overlaps data ingestion with model compute)")

# Take one batch from train_ds and visualize
for images, labels in train_ds.take(1):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    inv_map = {0: "AI-Generated", 1: "Real Art"}
    for idx in range(16):
        ax = axes[idx]
        img_np = np.clip(images[idx].numpy() / 255.0, 0.0, 1.0)
        lbl = int(labels[idx].numpy())
        ax.imshow(img_np)
        ax.set_title(f"{inv_map[lbl]} (Augmented)", fontsize=10, fontweight="bold")
        ax.axis("off")
    plt.suptitle("Figure 2.1: Sample Augmented Training Batch (VisionGuard Input Pipeline)", fontsize=14, y=0.95)
    plt.tight_layout()
    aug_batch_path = os.path.join(OUTPUT_DIR, "2.1_augmented_batch.png")
    plt.savefig(aug_batch_path, dpi=300)
    plt.close()
    print(f"\n[Saved] Augmented batch figure saved to: {aug_batch_path}")
    break

print("\nTask 2.1 completed successfully.")
