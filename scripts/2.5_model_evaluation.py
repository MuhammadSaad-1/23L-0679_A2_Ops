import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "scripts" else SCRIPT_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "task2_outputs")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

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

val_df = df_splits[df_splits["split"] == "val"].reset_index(drop=True)
test_df = df_splits[df_splits["split"] == "test"].reset_index(drop=True)

print(f"Validation images: {len(val_df)} | Test images: {len(test_df)}")

def load_and_preprocess_image(filepath, label):
    raw_img = tf.io.read_file(filepath)
    img = tf.io.decode_image(raw_img, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img, tf.cast(label, tf.float32)

def build_eval_ds(df_subset):
    ds = tf.data.Dataset.from_tensor_slices((df_subset["abs_filepath"].values, df_subset["label"].values))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    return ds.prefetch(tf.data.AUTOTUNE)

val_ds = build_eval_ds(val_df)
test_ds = build_eval_ds(test_df)

y_val_true = val_df["label"].values
y_test_true = test_df["label"].values

# -------------------------------------------------------------
# 2. Load Saved Best Checkpoints
# -------------------------------------------------------------
baseline_model_path = os.path.join(OUTPUT_DIR, "2.2_baseline_best.keras")
if not os.path.exists(baseline_model_path):
    baseline_model_path = os.path.join(OUTPUT_DIR, "baseline_best.keras")

transfer_model_path = os.path.join(OUTPUT_DIR, "2.3_transfer_learning_best.keras")
if not os.path.exists(transfer_model_path):
    transfer_model_path = os.path.join(OUTPUT_DIR, "transfer_learning_best.keras")

if not os.path.exists(baseline_model_path):
    raise FileNotFoundError(f"Baseline model not found. Please run '2.2_baseline_cnn.py' first.")
if not os.path.exists(transfer_model_path):
    raise FileNotFoundError(f"Transfer model not found. Please run '2.3_transfer_learning.py' first.")

print("\nLoading models from disk...")
baseline_model = tf.keras.models.load_model(baseline_model_path)
transfer_model = tf.keras.models.load_model(transfer_model_path)
print("Models loaded successfully.")

# -------------------------------------------------------------
# 3. Decision Threshold Tuning using VALIDATION DATA ONLY
# -------------------------------------------------------------
print("\n" + "=" * 65)
print(" TUNING DECISION THRESHOLDS ON VALIDATION DATA")
print("=" * 65)

val_preds_base = baseline_model.predict(val_ds, verbose=0).ravel()
val_preds_tl = transfer_model.predict(val_ds, verbose=0).ravel()

def find_best_threshold(y_true, y_probs, model_name):
    thresholds = np.linspace(0.1, 0.9, 81)
    f1_scores = []
    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        f1_scores.append(f1_score(y_true, preds, zero_division=0))
    best_idx = np.argmax(f1_scores)
    best_th = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    print(f"[{model_name}] Optimal Threshold on Val Set: {best_th:.2f} (Val F1: {best_f1:.4f})")
    return best_th, thresholds, f1_scores

th_base, ths_base, f1s_base = find_best_threshold(y_val_true, val_preds_base, "Baseline CNN")
th_tl, ths_tl, f1s_tl = find_best_threshold(y_val_true, val_preds_tl, "Transfer Learning (MobileNetV2)")

# Plot threshold tuning curves
plt.figure(figsize=(9, 4.5))
plt.plot(ths_base, f1s_base, label=f"Baseline CNN (Best Th={th_base:.2f})", color="#1f77b4", linewidth=2)
plt.plot(ths_tl, f1s_tl, label=f"Transfer Learning (Best Th={th_tl:.2f})", color="#2ca02c", linewidth=2)
plt.axvline(0.5, color="gray", linestyle=":", label="Default Threshold (0.50)")
plt.title("Figure 2.5a: Decision Threshold Optimization on Validation Data (F1-Score)", fontsize=12)
plt.xlabel("Decision Threshold")
plt.ylabel("Validation F1-Score")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
th_plot_path = os.path.join(OUTPUT_DIR, "2.5_threshold_tuning_curves.png")
plt.savefig(th_plot_path, dpi=300)
plt.close()
print(f"[Saved] Threshold tuning curves saved to: {th_plot_path}")

# -------------------------------------------------------------
# 4. Final Single Evaluation on TEST SET ONCE
# -------------------------------------------------------------
print("\n" + "=" * 65)
print(" FINAL EVALUATION ON UNSEEN TEST SET (EXACTLY ONCE)")
print("=" * 65)

test_preds_base = baseline_model.predict(test_ds, verbose=0).ravel()
test_preds_tl = transfer_model.predict(test_ds, verbose=0).ravel()

def compute_metrics(y_true, y_probs, threshold, model_name):
    y_pred = (y_probs >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_probs)
    return {
        "Model": model_name,
        "Threshold": round(threshold, 2),
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(auc, 4)
    }

results = [
    compute_metrics(y_test_true, test_preds_base, 0.50, "Baseline CNN (Default Th=0.50)"),
    compute_metrics(y_test_true, test_preds_base, th_base, f"Baseline CNN (Optimal Th={th_base:.2f})"),
    compute_metrics(y_test_true, test_preds_tl, 0.50, "Transfer Learning (Default Th=0.50)"),
    compute_metrics(y_test_true, test_preds_tl, th_tl, f"Transfer Learning (Optimal Th={th_tl:.2f})")
]

df_results = pd.DataFrame(results)
metrics_csv_path = os.path.join(OUTPUT_DIR, "2.5_model_evaluation_metrics.csv")
df_results.to_csv(metrics_csv_path, index=False)
print(df_results.to_string(index=False))
print(f"\n[Saved] Metrics comparison table saved to: {metrics_csv_path}")

# -------------------------------------------------------------
# 5. Confusion Matrices (Both Models)
# -------------------------------------------------------------
cm_base = confusion_matrix(y_test_true, (test_preds_base >= th_base).astype(int))
cm_tl = confusion_matrix(y_test_true, (test_preds_tl >= th_tl).astype(int))
labels = ["AI-Generated (0)", "Real Art (1)"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=labels, yticklabels=labels, cbar=False)
axes[0].set_title(f"Baseline CNN Confusion Matrix (Th={th_base:.2f})")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

sns.heatmap(cm_tl, annot=True, fmt="d", cmap="Greens", ax=axes[1],
            xticklabels=labels, yticklabels=labels, cbar=False)
axes[1].set_title(f"Transfer Learning Confusion Matrix (Th={th_tl:.2f})")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.suptitle("Figure 2.5b: Test Set Confusion Matrices (Optimal Thresholds)", fontsize=13, y=1.02)
plt.tight_layout()
cm_plot_path = os.path.join(OUTPUT_DIR, "2.5_confusion_matrices.png")
plt.savefig(cm_plot_path, dpi=300)
plt.close()
print(f"[Saved] Confusion matrices plot saved to: {cm_plot_path}")

# -------------------------------------------------------------
# 6. ROC Curves (Both Models)
# -------------------------------------------------------------
fpr_base, tpr_base, _ = roc_curve(y_test_true, test_preds_base)
auc_base = roc_auc_score(y_test_true, test_preds_base)

fpr_tl, tpr_tl, _ = roc_curve(y_test_true, test_preds_tl)
auc_tl = roc_auc_score(y_test_true, test_preds_tl)

plt.figure(figsize=(8, 6))
plt.plot(fpr_base, tpr_base, label=f"Baseline CNN (AUC = {auc_base:.4f})", color="#1f77b4", linewidth=2)
plt.plot(fpr_tl, tpr_tl, label=f"Transfer Learning MobileNetV2 (AUC = {auc_tl:.4f})", color="#2ca02c", linewidth=2)
plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random Chance (AUC = 0.50)")
plt.title("Figure 2.5c: Test Set ROC Curves Comparison", fontsize=13)
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity / Recall)")
plt.legend(loc="lower right")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
roc_plot_path = os.path.join(OUTPUT_DIR, "2.5_roc_curves.png")
plt.savefig(roc_plot_path, dpi=300)
plt.close()
print(f"[Saved] ROC curves plot saved to: {roc_plot_path}")

# Save test predictions for downstream error analysis
test_df["pred_prob_baseline"] = test_preds_base
test_df["pred_prob_tl"] = test_preds_tl
test_df["pred_class_tl"] = (test_preds_tl >= th_tl).astype(int)
test_preds_csv = os.path.join(OUTPUT_DIR, "2.5_test_predictions.csv")
test_df.to_csv(test_preds_csv, index=False)
# Also save without prefix for fallback
test_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
print(f"[Saved] Test predictions saved to: {test_preds_csv}")

print("\nTask 2.5 evaluation completed successfully.")
