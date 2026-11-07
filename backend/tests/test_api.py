import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.model_router import ModelRouter

@pytest.fixture(scope="session")
def client():
    # In test_api.py, opt in to local general mode to test full dual-mode integration
    router = ModelRouter.get_instance()
    router.allow_local_general = True
    router.initialize()
    with TestClient(app) as test_client:
        yield test_client
    router.allow_local_general = False

def test_root_endpoint_v2(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["version"] == "2.0.0"
    assert "general" in data["modes_available"]
    assert "technical" in data["modes_available"]

def test_health_endpoint_dual_modes(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["technical_model_loaded"] is True
    assert "modes" in data
    assert "general" in data["modes"]
    assert "technical" in data["modes"]

def test_model_info_endpoint_dual_modes(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "modes" in data
    assert "general" in data["modes"]
    assert "DistilGPT-2" in data["modes"]["general"]["name"]
    assert "technical" in data["modes"]
    assert "BiLSTM" in data["modes"]["technical"]["name"]

def test_predict_technical_mode_default(client):
    payload = {
        "text": "deep learning architectures for",
        "top_k": 5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "technical"
    assert len(data["candidates"]) == 5
    for c in data["candidates"]:
        assert c["probability"] > 0
        assert isinstance(c["word"], str)

def test_predict_general_mode(client):
    payload = {
        "text": "The weather is very",
        "mode": "general",
        "top_k": 5,
        "temperature": 1.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "general"
    assert len(data["candidates"]) >= 3
    for c in data["candidates"]:
        assert len(c["word"].strip()) > 0
        assert c["word"] not in ["<|endoftext|>", ""]

def test_generate_technical_mode(client):
    payload = {
        "prompt": "how to build a",
        "mode": "technical",
        "max_tokens": 4,
        "strategy": "greedy"
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "technical"
    assert len(data["completed_text"]) > len(payload["prompt"])

def test_generate_general_mode(client):
    payload = {
        "prompt": "I am writing this email to",
        "mode": "general",
        "max_tokens": 10,
        "strategy": "greedy"
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "general"
    assert len(data["generated_text"]) > 0
    assert "DistilGPT-2" in data["model_name"]

def test_predict_empty_text_validation(client):
    response = client.post("/predict", json={"text": "   ", "mode": "general"})
    assert response.status_code == 400

def test_generate_empty_prompt_validation(client):
    response = client.post("/generate", json={"prompt": "  ", "mode": "technical"})
    assert response.status_code == 400

def test_analytics_tracking_dual_modes(client):
    response = client.get("/analytics")
    assert response.status_code == 200
    data = response.json()
    assert "predictions_by_mode" in data
    assert "generations_by_mode" in data
    assert data["predictions_by_mode"]["general"] >= 1
    assert data["predictions_by_mode"]["technical"] >= 1
    assert data["total_predictions"] >= 2
