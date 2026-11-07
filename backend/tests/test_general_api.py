import pytest
from fastapi.testclient import TestClient
from backend.app.general_main import app as general_app
from backend.app.general_main import general_svc

@pytest.fixture(scope="session")
def general_client():
    general_svc.load()
    with TestClient(general_app) as client:
        yield client

def test_general_isolated_root(general_client):
    res = general_client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["service"] == "NeuroText AI General Inference Service"
    assert "DistilGPT-2" in data["model"]
    assert data["version"] == "2.0.0"

def test_general_isolated_health(general_client):
    res = general_client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "DistilGPT-2" in data["metadata"]["name"]

def test_general_isolated_model_info(general_client):
    res = general_client.get("/model-info")
    assert res.status_code == 200
    data = res.json()
    assert data["parameters"] == 81912576
    assert data["is_loaded"] is True

def test_general_isolated_predict(general_client):
    payload = {
        "text": "The future of artificial intelligence",
        "mode": "general",
        "top_k": 5
    }
    res = general_client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "general"
    assert len(data["candidates"]) == 5
    for c in data["candidates"]:
        assert 0.0 < c["probability"] <= 1.0
        assert len(c["word"]) > 0

def test_general_isolated_generate(general_client):
    payload = {
        "prompt": "I am writing this email to",
        "mode": "general",
        "max_tokens": 6,
        "strategy": "greedy"
    }
    res = general_client.post("/generate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "general"
    assert len(data["tokens_generated"]) == 6
    assert "I am writing this email to" in data["completed_text"]
    for t in data["tokens_generated"]:
        assert 0.0 < t["probability"] <= 1.0

def test_general_isolated_empty_text_validation(general_client):
    res = general_client.post("/predict", json={"text": "", "mode": "general"})
    assert res.status_code == 400
