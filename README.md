# VisionGuard: AI vs Real Image Classification & MLOps Deployment

[![VisionGuard CI](https://github.com/MSaad/23L-0679_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/MSaad/23L-0679_A2_Ops/actions/workflows/ci.yml)
[![DagsHub Tracking](https://img.shields.io/badge/DagsHub-MLflow%20Tracking-blue)](https://dagshub.com/MSaad/23L-0679_A2_Ops)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

---

## 📌 Project Overview
VisionGuard is an end-to-end production machine learning system designed to distinguish between human-created art (`RealArt`) and generative AI-synthesized imagery (`AiArtData`). The system incorporates full-lifecycle MLOps practices:
* **Task 1**: Data Auditing, EDA, and DVC Remote Versioning (DagsHub S3 remote).
* **Task 2**: Transfer Learning with MobileNetV2, Grad-CAM interpretability, and Keras Tuner.
* **Task 3**: Remote Experiment Tracking on DagsHub MLflow & Model Registry (`@champion` alias).
* **Task 4**: Production Flask Web Service with zero-disk in-memory evaluation and honest caveats.
* **Task 5**: Immutable SQLite & PostgreSQL Audit Trails tracking queries, confidence, and latency.
* **Task 6**: Multi-container Docker & Docker Compose stack with isolated networks and GitHub Actions CI.
* **Task 7**: Cloud Deployment and Public REST API integration.

---

## 🚀 Quickstart: Running with Docker Compose (Task 6)

The entire multi-container production stack (Flask App + PostgreSQL 15) can be launched with a single command:

```bash
# 1. Clone repository
git clone https://github.com/MSaad/23L-0679_A2_Ops.git
cd 23L-0679_A2_Ops

# 2. Configure environment credentials
cp .env.example .env

# 3. Launch isolated multi-service stack
docker compose up -d --build
```

* **Web UI**: Open `http://localhost:5000`
* **Health Check**: Open `http://localhost:5000/health`
* **Audit History**: Open `http://localhost:5000/history`

---

## 🔒 Security Architecture Highlights
1. **Zero Disk Retention**: Uploaded user images are processed entirely in-memory (`io.BytesIO`) as float32 tensors and are never written to physical disk.
2. **Network Isolation**: The PostgreSQL database publishes **no host ports** (`5432` is closed to the host) and communicates exclusively over Compose's internal bridge network.
3. **Non-Root Execution**: Container processes execute under the restricted `appuser` (UID 1000).
4. **CI/CD Automation**: GitHub Actions runs flake8 linting, pytest suites across Python 3.10 and 3.11, and verifies Docker image builds on every push.
