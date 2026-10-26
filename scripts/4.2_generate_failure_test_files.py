import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) in ["scripts", "app"] else SCRIPT_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "task4_outputs", "test_samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print(" GENERATING SUBTASK 4.2 FAILURE TEST ARTIFACTS")
print("=" * 65)

# 1. Failure Case 1: Wrong file format (.txt instead of .png/.jpg)
txt_path = os.path.join(OUTPUT_DIR, "invalid_format.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("This is a plain text file, not a valid computer vision image.")
print(f"1. [Generated] Wrong file type: {txt_path}")

# 2. Failure Case 2: Corrupted / Undecodable image (looks like PNG by name, but invalid binary)
corrupt_path = os.path.join(OUTPUT_DIR, "corrupted_image.png")
with open(corrupt_path, "wb") as f:
    # Deliberately malformed binary header
    f.write(b"INVALID_HEADER_NOT_A_REAL_PNG_DATA_STREAM_CORRUPTED_BYTES_123456789")
print(f"2. [Generated] Corrupted undecodable image: {corrupt_path}")

# 3. Failure Case 3: Oversized file (> 5MB limit, e.g. 5.5 MB)
oversized_path = os.path.join(OUTPUT_DIR, "oversized_file_5.5MB.png")
with open(oversized_path, "wb") as f:
    # Write 5.5 MB of dummy data
    f.write(b"\x00" * int(5.5 * 1024 * 1024))
print(f"3. [Generated] Oversized file (5.5 MB): {oversized_path}")

print("\nAll 3 failure case test files are ready in: task4_outputs/test_samples/")
print("Upload each file through the web browser at http://localhost:5000 to capture the 3 failure screenshots.")
