import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.model_loader import ModelManager

@pytest.fixture(scope="session")
def client():
    # Force load model manager
    ModelManager.get_instance().load()
    with TestClient(app) as test_client:
        yield test_client

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "NeuroText AI" in data["name"]

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "BiLSTM" in data["architecture"]
    assert data["vocab_size"] > 1000
    assert "metrics" in data
    assert "top1_accuracy" in data["metrics"]
    assert "top5_accuracy" in data["metrics"]

def test_predict_endpoint_success(client):
    payload = {
        "text": "deep learning and neural",
        "top_k": 5,
        "temperature": 1.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["input_text"] == payload["text"]
    assert len(data["candidates"]) == 5
    # Verify probabilities sum to reasonable positive values
    for cand in data["candidates"]:
        assert cand["probability"] > 0
        assert isinstance(cand["word"], str)
        assert cand["word"] not in ["<oov>", "<pad>"]
    # Check ranking
    probs = [c["probability"] for c in data["candidates"]]
    assert probs == sorted(probs, reverse=True)

def test_predict_endpoint_empty_input(client):
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400

def test_generate_endpoint_greedy(client):
    payload = {
        "prompt": "how to build a",
        "max_tokens": 4,
        "strategy": "greedy"
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prompt"] == payload["prompt"]
    assert len(data["tokens_generated"]) <= 4
    assert len(data["completed_text"]) > len(payload["prompt"])
    assert data["strategy_used"] == "greedy"

def test_generate_endpoint_sampling(client):
    payload = {
        "prompt": "artificial intelligence for",
        "max_tokens": 3,
        "strategy": "top_k",
        "top_k": 5,
        "temperature": 0.8
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["tokens_generated"]) <= 3

def test_analytics_endpoint(client):
    response = client.get("/analytics")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "total_generations" in data
    assert "avg_latency_ms" in data
    assert data["total_predictions"] >= 1

