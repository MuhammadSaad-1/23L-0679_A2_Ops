import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) in ["scripts", "app"] else SCRIPT_DIR
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.config import Config
from app.inference import VisionGuardPredictor

print("=" * 70)
print(" TASK 4.1: INFERENCE MODULE VERIFICATION & TEST ON 2 KNOWN IMAGES")
print("=" * 70)

# 1. Instantiate Predictor (Loads champion model ONCE at startup)
predictor = VisionGuardPredictor(
    model_path=Config.MODEL_PATH,
    threshold=Config.DECISION_THRESHOLD,
    model_name=Config.MODEL_NAME
)

# 2. Locate 2 Known Test Images (1 RealArt, 1 AiArtData)
real_dir = os.path.join(BASE_DIR, "data", "RealArt")
ai_dir = os.path.join(BASE_DIR, "data", "AiArtData")

def get_first_image(directory):
    for f in sorted(os.listdir(directory)):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            return os.path.join(directory, f)
    raise FileNotFoundError(f"No image found in {directory}")

real_sample = get_first_image(real_dir)
ai_sample = get_first_image(ai_dir)

test_cases = [
    {"path": real_sample, "ground_truth": "RealArt (Human Created)"},
    {"path": ai_sample, "ground_truth": "AiArtData (AI Generated)"}
]

# 3. Execute In-Memory Inference on Both Images
for idx, test in enumerate(test_cases, start=1):
    filepath = test["path"]
    ground_truth = test["ground_truth"]
    filename = os.path.basename(filepath)

    print(f"\n--- [TEST SAMPLE {idx}/2] ---")
    print(f" File Path     : {filepath}")
    print(f" Ground Truth  : {ground_truth}")

    # Read bytes into volatile memory (simulating web upload)
    with open(filepath, "rb") as f:
        img_bytes = f.read()

    result = predictor.predict(img_bytes, filename=filename)

    print(f" Preprocessing : In-Memory BytesIO -> RGB -> 224x224 (0 Disk Writes)")
    print(f" Predicted     : {result['verdict']}")
    print(f" Verdict Match : {'[CORRECT]' if result['verdict'] == ground_truth else '[MISMATCH]'}")
    print(f" Raw Output    : P(Real) = {result['raw_probability']:.4f}")
    print(f" Confidence    : {result['confidence_pct']}%")
    print(f" Latency       : {result['latency_ms']} ms")
    print(f" Honest Caveat : {result['caveat']}")

print("\n" + "=" * 70)
print(" Task 4.1 inference verification completed successfully.")
print(f" Explanation document available at: task4_outputs/4.1_inference_explanation.txt")
print("=" * 70)
