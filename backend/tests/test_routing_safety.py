import json
import urllib.request
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.model_router import ModelRouter

@pytest.fixture(autouse=True)
def reset_router():
    router = ModelRouter.get_instance()
    router.general_api_url = ""
    router.allow_local_general = False
    router._local_general_service = None
    router.stats = {
        "total_predictions": 0,
        "total_generations": 0,
        "predictions_by_mode": {"general": 0, "technical": 0},
        "generations_by_mode": {"general": 0, "technical": 0},
        "latency_history": [],
        "start_time": 0
    }
    router.initialize()
    yield
    router.general_api_url = ""
    router.allow_local_general = False

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_unconfigured_general_predict_returns_503(client):
    """1. Empty GENERAL_API_URL returns HTTP 503 for General prediction without loading DistilGPT-2."""
    router = ModelRouter.get_instance()
    assert router.general_api_url == ""
    assert router.allow_local_general is False
    assert router._local_general_service is None

    response = client.post("/predict", json={"text": "artificial intelligence", "mode": "general"})
    assert response.status_code == 503
    data = response.json()
    assert "General English writing mode is currently unavailable" in data["detail"]
    # Model should not have been loaded
    assert router._local_general_service is None

def test_unconfigured_general_generate_returns_503(client):
    """2. Empty GENERAL_API_URL returns HTTP 503 for General generation without loading DistilGPT-2."""
    router = ModelRouter.get_instance()
    response = client.post("/generate", json={"prompt": "artificial intelligence", "mode": "general"})
    assert response.status_code == 503
    data = response.json()
    assert "General English text generation is currently unavailable" in data["detail"]
    assert router._local_general_service is None

def test_technical_works_when_general_unconfigured(client):
    """3. Technical prediction and generation continue working with no General service URL."""
    # Predict
    resp_p = client.post("/predict", json={"text": "deep learning for", "mode": "technical", "top_k": 3})
    assert resp_p.status_code == 200
    data_p = resp_p.json()
    assert data_p["mode"] == "technical"
    assert len(data_p["candidates"]) == 3

    # Generate
    resp_g = client.post("/generate", json={"prompt": "deep learning for", "mode": "technical", "max_tokens": 4})
    assert resp_g.status_code == 200
    data_g = resp_g.json()
    assert data_g["mode"] == "technical"
    assert len(data_g["tokens_generated"]) == 4

def test_remote_general_predict_forwarding(client):
    """4. Configured remote General prediction correctly forwards the request."""
    router = ModelRouter.get_instance()
    router.general_api_url = "https://mock-general-api.internal"

    mock_response_data = {
        "input_text": "The future of AI",
        "mode": "general",
        "tokens": ["The", "Ġfuture", "Ġof", "ĠAI"],
        "candidates": [{"word": "is", "probability": 0.35, "rank": 1}],
        "latency_ms": 42.5,
        "model_name": "DistilGPT-2 (General English)",
        "model_version": "2.0.0"
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_response_data).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        res = client.post("/predict", json={"text": "The future of AI", "mode": "general"})
        assert res.status_code == 200
        data = res.json()
        assert data["mode"] == "general"
        assert data["candidates"][0]["word"] == "is"
        assert data["model_name"] == "DistilGPT-2 (General English)"
        mock_urlopen.assert_called_once()
        # Verify target URL
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://mock-general-api.internal/predict"

def test_remote_general_generate_forwarding(client):
    """5. Configured remote General generation correctly forwards the request."""
    router = ModelRouter.get_instance()
    router.general_api_url = "https://mock-general-api.internal"

    mock_response_data = {
        "prompt": "Artificial intelligence will",
        "mode": "general",
        "generated_text": "transform writing completely.",
        "completed_text": "Artificial intelligence will transform writing completely.",
        "tokens_generated": [{"token": "transform", "probability": 0.4}],
        "strategy_used": "greedy",
        "latency_ms": 120.0,
        "model_name": "DistilGPT-2 (General English)"
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_response_data).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        res = client.post("/generate", json={"prompt": "Artificial intelligence will", "mode": "general"})
        assert res.status_code == 200
        data = res.json()
        assert data["mode"] == "general"
        assert data["generated_text"] == "transform writing completely."
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://mock-general-api.internal/generate"

def test_remote_general_health_status(client):
    """6. Remote General health status is reflected correctly."""
    router = ModelRouter.get_instance()
    router.general_api_url = "https://mock-general-api.internal"

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"status": "healthy", "model_loaded": True}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["general_model_loaded"] is True
        assert data["modes"]["general"]["loaded"] is True

def test_model_info_with_remote_general(client):
    """7. /model-info works with remote General configuration."""
    router = ModelRouter.get_instance()
    router.general_api_url = "https://mock-general-api.internal"

    res = client.get("/model-info")
    assert res.status_code == 200
    data = res.json()
    assert "general" in data["modes"]
    assert data["modes"]["general"]["name"] == "DistilGPT-2 (General English)"
    assert data["modes"]["general"]["is_remote"] is True
    assert data["modes"]["general"]["remote_url"] == "https://mock-general-api.internal"
    assert data["modes"]["general"]["parameters"] == 81912576

def test_model_info_when_general_unconfigured(client):
    """8. /model-info works when General is unconfigured without loading the model."""
    router = ModelRouter.get_instance()
    router.general_api_url = ""
    router.allow_local_general = False

    res = client.get("/model-info")
    assert res.status_code == 200
    data = res.json()
    assert "general" in data["modes"]
    assert data["modes"]["general"]["is_remote"] is False
    assert data["modes"]["general"]["status"] == "unconfigured"
    assert router._local_general_service is None

def test_remote_general_analytics_counters_update(client):
    """9. Remote General analytics counters update after successful requests."""
    router = ModelRouter.get_instance()
    router.general_api_url = "https://mock-general-api.internal"

    mock_p_resp = MagicMock()
    mock_p_resp.read.return_value = json.dumps({
        "input_text": "test", "mode": "general", "tokens": ["test"],
        "candidates": [], "latency_ms": 10.0, "model_name": "DistilGPT-2", "model_version": "2.0.0"
    }).encode("utf-8")
    mock_p_resp.__enter__.return_value = mock_p_resp

    with patch("urllib.request.urlopen", return_value=mock_p_resp):
        client.post("/predict", json={"text": "test", "mode": "general"})

    assert router.stats["total_predictions"] == 1
    assert router.stats["predictions_by_mode"]["general"] == 1
    assert len(router.stats["latency_history"]) == 1

def test_upstream_errors_do_not_fall_back_to_technical(client):
    """10. Upstream errors do not silently fall back to the Technical model."""
    router = ModelRouter.get_instance()
    router.general_api_url = "https://mock-general-api.internal"

    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        res = client.post("/predict", json={"text": "test", "mode": "general"})
        assert res.status_code == 503
        data = res.json()
        assert "Dedicated General Inference Service unavailable" in data["detail"]
        # Must NOT count as a successful prediction in analytics
        assert router.stats["total_predictions"] == 0
        assert router.stats["predictions_by_mode"]["general"] == 0

