import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image

# Load the audit records generated in Task 1.2
csv_file = "1.2_dataset_audit.csv" if os.path.exists("1.2_dataset_audit.csv") else "dataset_audit.csv"
df = pd.read_csv(csv_file)
os.makedirs("eda_outputs", exist_ok=True)

# -------------------------------------------------------------
# 1. Figure 1: Sample Grid of Images per Class
# -------------------------------------------------------------
classes = df["class"].unique()
fig, axes = plt.subplots(len(classes), 5, figsize=(15, 6))

for row_idx, cls in enumerate(classes):
    cls_df = df[df["class"] == cls]
    sample_paths = cls_df.sample(min(5, len(cls_df)), random_state=42)["filepath"].tolist()
    for col_idx, img_path in enumerate(sample_paths):
        ax = axes[row_idx, col_idx]
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis("off")
        if col_idx == 0:
            ax.set_title(f"Class: {cls}", fontsize=12, loc="left", fontweight="bold")

plt.suptitle("Figure 1: Sample Images per Class (VisionGuard)", fontsize=14, y=0.98)
plt.tight_layout()
grid_path = "eda_outputs/sample_grid.png"
plt.savefig(grid_path, dpi=300)
plt.close()
print(f"Saved: {grid_path}")

# -------------------------------------------------------------
# 2. Figure 2: Histograms of Dimensions and File Sizes
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Width distribution
sns.histplot(data=df, x="width", hue="class", bins=30, kde=True, ax=axes[0], palette="Set2")
axes[0].set_title("Width Distribution by Class")
axes[0].set_xlabel("Width (pixels)")

# Height distribution
sns.histplot(data=df, x="height", hue="class", bins=30, kde=True, ax=axes[1], palette="Set2")
axes[1].set_title("Height Distribution by Class")
axes[1].set_xlabel("Height (pixels)")

# File Size distribution
sns.histplot(data=df, x="file_size_kb", hue="class", bins=30, kde=True, ax=axes[2], palette="Set2")
axes[2].set_title("File Size (KB) Distribution by Class")
axes[2].set_xlabel("Size (KB)")

plt.suptitle("Figure 2: Dimension and File Size Distributions", fontsize=14, y=1.02)
plt.tight_layout()
dist_path = "eda_outputs/distributions.png"
plt.savefig(dist_path, dpi=300)
plt.close()
print(f"Saved: {dist_path}")

print("\nVisual EDA complete! Both plots are ready in the 'eda_outputs' folder.")
