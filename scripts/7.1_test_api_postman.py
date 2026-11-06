import os
import sys
import argparse
import json
import time
import urllib.request
import urllib.parse
import uuid

# Base directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def send_multipart_file(url, file_path, field_name="file"):
    """Sends a file as multipart/form-data via urllib without third-party deps."""
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    content_type = f"multipart/form-data; boundary={boundary}"
    
    filename = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
    mime_type = f"image/{'jpeg' if ext in ['jpg', 'jpeg'] else ext}"

    # Build RFC 2046 multipart payload
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": content_type,
            "User-Agent": "VisionGuard-PostmanSimulator/1.0"
        }
    )
    
    start_time = time.perf_counter()
    with urllib.request.urlopen(req, timeout=45) as resp:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        response_data = resp.read().decode("utf-8")
        status_code = resp.status
        return status_code, json.loads(response_data), elapsed_ms


def check_health(base_url):
    """Queries /health endpoint."""
    health_url = f"{base_url.rstrip('/')}/health"
    print(f"\n[1/3] Probing Health Endpoint: {health_url}")
    req = urllib.request.Request(health_url, headers={"User-Agent": "VisionGuard-Verifier/1.0"})
    start_time = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        data = json.loads(resp.read().decode("utf-8"))
        print(f"       HTTP Status : {resp.status} OK ({elapsed_ms:.1f}ms)")
        print(f"       Payload     : {json.dumps(data, indent=2)}")
        return data


def test_prediction(base_url, sample_image_path):
    """Sends real test image to /api/predict."""
    predict_url = f"{base_url.rstrip('/')}/api/predict"
    print(f"\n[2/3] Testing REST Prediction Endpoint: {predict_url}")
    print(f"       Sample Image: {sample_image_path}")
    
    status_code, data, elapsed_ms = send_multipart_file(predict_url, sample_image_path)
    print(f"       HTTP Status : {status_code} OK (Round-trip: {elapsed_ms:.1f}ms)")
    print(f"       API Output  : {json.dumps(data, indent=2)}")
    return data


def test_rejection(base_url):
    """Tests rejection on invalid format."""
    predict_url = f"{base_url.rstrip('/')}/api/predict"
    print(f"\n[3/3] Testing Rejection Security (Text file upload)")
    dummy_txt = os.path.join(SCRIPT_DIR, "dummy_test.txt")
    with open(dummy_txt, "w") as f:
        f.write("This is not an image file.")

    try:
        status_code, data, elapsed_ms = send_multipart_file(predict_url, dummy_txt)
        print(f"       Unexpected Success: {data}")
    except urllib.error.HTTPError as http_err:
        err_body = http_err.read().decode("utf-8")
        print(f"       Expected Rejection: HTTP {http_err.code} ({http_err.reason})")
        print(f"       Error JSON        : {err_body}")
    finally:
        if os.path.exists(dummy_txt):
            os.remove(dummy_txt)


def main():
    parser = argparse.ArgumentParser(description="VisionGuard Task 7 API Client Verification")
    parser.add_argument("--url", default="http://localhost:5000", help="Base URL of VisionGuard instance")
    parser.add_argument("--image", default=None, help="Path to sample image for testing")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print("=" * 70)
    print(" VISIONGUARD TASK 7 REST API VERIFICATION SUITE")
    print(f" Target Server : {base_url}")
    print("=" * 70)

    # 1. Health check
    try:
        check_health(base_url)
    except Exception as e:
        print(f" [ERROR] Could not connect to {base_url}/health: {e}")
        sys.exit(1)

    # 2. Find sample image
    sample_img = args.image
    if not sample_img or not os.path.exists(sample_img):
        # Look for existing sample image in eda_outputs or data
        candidates = [
            os.path.join(PROJECT_ROOT, "eda_outputs", "sample_real_0.png"),
            os.path.join(PROJECT_ROOT, "eda_outputs", "sample_ai_0.png"),
            os.path.join(PROJECT_ROOT, "task2_outputs", "2.1_augmented_batch.png"),
            os.path.join(PROJECT_ROOT, "task4_outputs", "oversized_file_5.5MB.png")
        ]
        for c in candidates:
            if os.path.exists(c) and not c.endswith("5.5MB.png"):
                sample_img = c
                break

    if not sample_img or not os.path.exists(sample_img):
        # Create a tiny 64x64 valid test PNG
        from PIL import Image
        sample_img = os.path.join(SCRIPT_DIR, "temp_test_image.png")
        Image.new("RGB", (64, 64), color=(73, 109, 137)).save(sample_img)

    # 3. Test Prediction
    test_prediction(base_url, sample_img)

    # 4. Test Rejection
    test_rejection(base_url)

    print("\n" + "=" * 70)
    print(" [SUCCESS] REST API Endpoint successfully verified!")
    print("=" * 70)


if __name__ == "__main__":
    main()
