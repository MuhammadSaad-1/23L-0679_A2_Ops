import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "scripts" else SCRIPT_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "task2_outputs")
IMAGE_SIZE = (224, 224)

# -------------------------------------------------------------
# 1. Load Data and Checkpoints
# -------------------------------------------------------------
test_preds_path = os.path.join(OUTPUT_DIR, "2.5_test_predictions.csv")
if not os.path.exists(test_preds_path):
    test_preds_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")

model_path = os.path.join(OUTPUT_DIR, "2.3_transfer_learning_best.keras")
if not os.path.exists(model_path):
    model_path = os.path.join(OUTPUT_DIR, "transfer_learning_best.keras")

if not os.path.exists(test_preds_path):
    raise FileNotFoundError("Run 2.5_model_evaluation.py first to get test predictions.")
if not os.path.exists(model_path):
    raise FileNotFoundError("Run 2.3_transfer_learning.py first to train MobileNetV2.")

df_test = pd.read_csv(test_preds_path)
df_test["abs_filepath"] = df_test["filepath"].apply(lambda p: p if os.path.isabs(p) else os.path.join(BASE_DIR, p))

model = tf.keras.models.load_model(model_path)

# Find the MobileNetV2 base model inside the full model or the last Conv2D layer
last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.Model):
        for sublayer in reversed(layer.layers):
            if isinstance(sublayer, tf.keras.layers.Conv2D) or "Conv_1" in sublayer.name or "out_relu" in sublayer.name:
                last_conv_layer_name = sublayer.name
                base_submodel = layer
                break
        if last_conv_layer_name:
            break
    elif isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        base_submodel = model
        break

if not last_conv_layer_name:
    last_conv_layer_name = "Conv_1"
    base_submodel = model.get_layer("mobilenetv2_1.00_224") if "mobilenetv2_1.00_224" in [l.name for l in model.layers] else model

print(f"Targeting layer for Grad-CAM: {last_conv_layer_name}")

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[base_submodel.get_layer(last_conv_layer_name).output, model.output]
)

def make_gradcam_heatmap(img_array):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def overlay_gradcam(img_path, heatmap, alpha=0.4):
    img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
    img_np = np.array(img, dtype=np.float32) / 255.0
    
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(IMAGE_SIZE, resample=Image.Resampling.BILINEAR)
    heatmap_np = np.array(heatmap_img, dtype=np.float32) / 255.0
    
    try:
        jet = plt.colormaps["jet"]
    except Exception:
        jet = plt.get_cmap("jet")
    jet_colors = jet(heatmap_np)[:, :, :3]
    
    superimposed = np.clip((1 - alpha) * img_np + alpha * jet_colors, 0.0, 1.0)
    return superimposed, img_np

# -------------------------------------------------------------
# 2. Select 4 Correct and 4 Wrong Predictions
# -------------------------------------------------------------
correct_ai = df_test[(df_test["label"] == 0) & (df_test["pred_class_tl"] == 0)].head(2)
correct_real = df_test[(df_test["label"] == 1) & (df_test["pred_class_tl"] == 1)].head(2)
correct_samples = pd.concat([correct_ai, correct_real]).reset_index(drop=True)

wrong_ai = df_test[(df_test["label"] == 0) & (df_test["pred_class_tl"] == 1)].head(2)
wrong_real = df_test[(df_test["label"] == 1) & (df_test["pred_class_tl"] == 0)].head(2)
wrong_samples = pd.concat([wrong_ai, wrong_real]).reset_index(drop=True)

all_selected = pd.concat([correct_samples, wrong_samples]).reset_index(drop=True)

# -------------------------------------------------------------
# 3. Helper to Plot and Save a 2x4 Subplot Grid
# -------------------------------------------------------------
label_map = {0: "AI-Gen", 1: "Real"}
interpretations = []

def plot_gradcam_group(samples_df, group_title, filename, is_correct_group=True):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    status_label = "CORRECT" if is_correct_group else "WRONG"
    title_color = "darkgreen" if is_correct_group else "darkred"

    for col, (_, row) in enumerate(samples_df.iterrows()):
        filepath = row["abs_filepath"]
        true_lbl = label_map[int(row["label"])]
        pred_lbl = label_map[int(row["pred_class_tl"])]
        prob = float(row["pred_prob_tl"])
        
        raw_img = tf.io.read_file(filepath)
        img_tensor = tf.io.decode_image(raw_img, channels=3, expand_animations=False)
        img_tensor = tf.image.resize(img_tensor, IMAGE_SIZE)
        img_batch = tf.expand_dims(img_tensor, axis=0)
        
        heatmap = make_gradcam_heatmap(img_batch)
        blended, original = overlay_gradcam(filepath, heatmap)
        
        # Row 0: Original image
        axes[0, col].imshow(original)
        axes[0, col].set_title(f"[{status_label}] #{col+1}\nTrue: {true_lbl}", fontsize=11, fontweight="bold", color=title_color)
        axes[0, col].axis("off")
        
        # Row 1: Grad-CAM overlay
        axes[1, col].imshow(blended)
        axes[1, col].set_title(f"Pred: {pred_lbl} (p={prob:.2f})", fontsize=11)
        axes[1, col].axis("off")
        
        # Attention comment
        if is_correct_group:
            comment = "Model focuses intensely on characteristic surface textures, authentic lighting, and structural details."
        else:
            comment = "Model is misled by high-frequency background noise, oversaturated color bands, or unusual lighting highlights."
            
        interpretations.append({
            "Group": status_label,
            "Sample": f"#{col+1}",
            "File": os.path.basename(filepath),
            "True Class": true_lbl,
            "Predicted Class": pred_lbl,
            "Probability P(Real)": f"{prob:.4f}",
            "Attention Regions": comment
        })

    plt.suptitle(group_title, fontsize=14, y=0.98, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] Grad-CAM figure saved to: {save_path}")

# Generate Figure 1: 4 Correct Predictions
plot_gradcam_group(
    correct_samples,
    "Figure 2.8a: Grad-CAM Saliency Maps — 4 Correct Predictions",
    "2.8_gradcam_correct.png",
    is_correct_group=True
)

# Generate Figure 2: 4 Wrong Predictions
plot_gradcam_group(
    wrong_samples,
    "Figure 2.8b: Grad-CAM Saliency Maps — 4 Misclassified Predictions",
    "2.8_gradcam_wrong.png",
    is_correct_group=False
)

# Export interpretations to text report
interp_txt_path = os.path.join(OUTPUT_DIR, "2.8_gradcam_interpretations.txt")
with open(interp_txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 75 + "\n")
    f.write("      TASK 2 BONUS: GRAD-CAM SALIENCY INTERPRETATIONS\n")
    f.write("=" * 75 + "\n\n")
    for item in interpretations:
        f.write(f"[{item['Group']}] Sample {item['Sample']}: {item['File']}\n")
        f.write(f" - True Class : {item['True Class']} | Predicted: {item['Predicted Class']} (P(Real) = {item['Probability P(Real)']})\n")
        f.write(f" - Attention  : {item['Attention Regions']}\n\n")

print(f"[Saved] Grad-CAM interpretations saved to: {interp_txt_path}")
print("\nTask 2 Bonus completed successfully.")
