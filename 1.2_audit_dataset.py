import os
import hashlib
from collections import defaultdict
from PIL import Image
import pandas as pd

DATA_DIR = "data"

records = []
corrupt_files = []
hashes = defaultdict(list)

print(f"Auditing dataset in: '{DATA_DIR}' folder...")

if not os.path.exists(DATA_DIR):
    print(f"ERROR: '{DATA_DIR}' directory does not exist! Please place your dataset inside a folder named 'data'.")
    exit(1)

for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        file_path = os.path.join(root, file)
        label = os.path.basename(root)

        # File size in bytes and KB
        try:
            file_size_bytes = os.path.getsize(file_path)
            file_size_kb = file_size_bytes / 1024.0
        except Exception as e:
            corrupt_files.append((file_path, str(e)))
            continue

        # Content SHA-256 hash
        try:
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            file_hash = hasher.hexdigest()
            hashes[file_hash].append(file_path)
        except Exception as e:
            corrupt_files.append((file_path, f"Hashing error: {e}"))
            continue

        # Image properties verification
        try:
            with Image.open(file_path) as img:
                img.verify()  # Check file integrity
            
            # Reopen to read dimensions and mode (verify closes the image pointer)
            with Image.open(file_path) as img:
                width, height = img.size
                mode = img.mode
                img_format = img.format

            records.append({
                "filepath": file_path,
                "class": label,
                "file_size_kb": round(file_size_kb, 2),
                "width": width,
                "height": height,
                "color_mode": mode,
                "format": img_format,
                "sha256": file_hash
            })
        except Exception as e:
            corrupt_files.append((file_path, f"Image decode error: {e}"))

df = pd.DataFrame(records)
df.to_csv("1.2_dataset_audit.csv", index=False)

# Detect duplicate groups
duplicates = {h: paths for h, paths in hashes.items() if len(paths) > 1}
total_duplicate_files = sum(len(paths) - 1 for paths in duplicates.values())

print("\n" + "=" * 65)
print("             DATASET AUDIT SUMMARY (TASK 1.2)            ")
print("=" * 65)
print(f"Total files scanned        : {len(records) + len(corrupt_files)}")
print(f"Valid images readable      : {len(records)}")
print(f"Corrupt / unreadable files : {len(corrupt_files)}")
print(f"Unique duplicate groups    : {len(duplicates)}")
print(f"Redundant duplicate files  : {total_duplicate_files}")

if corrupt_files:
    print("\n--- Corrupt Files Detected ---")
    for cp, err in corrupt_files[:5]:
        print(f" - {cp}: {err}")
    if len(corrupt_files) > 5:
        print(f" ... and {len(corrupt_files) - 5} more.")

print("\n--- Image Count Per Class ---")
print(df["class"].value_counts())

print("\n--- Summary Statistics (Width, Height, Size in KB) ---")
summary = df.groupby("class").agg({
    "width": ["min", "median", "max"],
    "height": ["min", "median", "max"],
    "file_size_kb": ["min", "median", "max"]
})
print(summary)
print("=" * 65)
print("Audit saved to '1.2_dataset_audit.csv'.")
