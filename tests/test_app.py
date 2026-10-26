import io
import pytest
from PIL import Image

from app.app import app
from app.config import Config
from app.inference import VisionGuardPredictor

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_config():
    """Verify core application configuration settings."""
    assert Config.PORT == 5000
    assert Config.MAX_CONTENT_LENGTH == 5 * 1024 * 1024
    assert "png" in Config.ALLOWED_EXTENSIONS
    assert "jpg" in Config.ALLOWED_EXTENSIONS

def test_health_endpoint(client):
    """Verify health check endpoint returns 200 OK and expected JSON schema."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_architecture" in data

def test_upload_page_renders(client):
    """Verify upload page returns 200 OK with detector form."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Verify Authenticity of Visual Media" in response.data

def test_invalid_file_type_rejected(client):
    """Verify that uploading a text file triggers HTTP 400 rejection."""
    data = {
        "file": (io.BytesIO(b"fake text content"), "document.txt")
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"Unsupported File Type" in response.data

def test_corrupted_image_rejected(client):
    """Verify that malformed bytes named as .png trigger HTTP 400 undecodable error."""
    data = {
        "file": (io.BytesIO(b"NOT_A_REAL_PNG_HEADER_123456789"), "corrupt.png")
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"Corrupted or Invalid Image" in response.data
