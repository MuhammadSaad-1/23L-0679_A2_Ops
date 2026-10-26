# ==============================================================
# VisionGuard MLOps Web Application - Production Dockerfile
# ==============================================================

# 1. Slim Base Image (Subtask 6.2 Requirement: Minimal attack surface & small footprint)
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    DECISION_THRESHOLD=0.5

# Install system dependencies (curl required for container HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# 2. Dependency Layer Caching (Subtask 6.2 Requirement)
# Copy ONLY requirements.txt first so Docker caches installed wheels.
# Re-running builds when source code changes will NOT re-download dependencies!
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copy Application Source Code and Trained Champion Model Artifact
COPY app/ ./app
COPY task2_outputs/2.3_transfer_learning_best.keras ./task2_outputs/2.3_transfer_learning_best.keras

# 4. Security: Run as Non-Root User (Subtask 6.2 Requirement)
# Running as root inside a container poses container breakout risks.
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/database && \
    chown -R appuser:appuser /app

# Switch to non-privileged user
USER appuser

# Expose internal service port
EXPOSE 5000

# 5. Built-in Container Health Check (Subtask 6.2 Requirement)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# 6. Production Gunicorn Entrypoint with 2 worker threads
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app.app:app"]
