import pytest
import json
import urllib.error
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.model_router import ModelRouter
from backend.app.services.cloudflare_workers_ai_service import CloudflareWorkersAIProvider
from backend.app.schemas import PredictionCandidate

@pytest.fixture
def cf_provider():
    return CloudflareWorkersAIProvider(
        account_id="test_account_123",
        api_token="test_secret_token_456",
        model_id="@cf/meta/llama-3.2-1b-instruct"
    )

def test_missing_credentials_raises_503():
    unconfigured = CloudflareWorkersAIProvider(account_id="", api_token="")
    assert unconfigured.is_configured is False
    with pytest.raises(Exception) as exc_info:
        unconfigured._call_chat_completions(messages=[{"role": "user", "content": "hi"}])
    assert "503" in str(exc_info.value) or "credentials" in str(exc_info.value).lower()

def test_invalid_account_id_or_unauthorized(cf_provider):
    with patch("urllib.request.urlopen") as mock_url:
        mock_http_err = urllib.error.HTTPError(
            url="http://test",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=MagicMock(read=lambda: b'{"success": false, "errors": [{"message": "Invalid token"}]}')
        )
        mock_url.side_effect = mock_http_err

        with pytest.raises(Exception) as exc:
            cf_provider._call_chat_completions(messages=[{"role": "user", "content": "hi"}])
        assert "503" in str(exc.value)

def test_upstream_timeout_handling(cf_provider):
    with patch("urllib.request.urlopen") as mock_url:
        mock_url.side_effect = urllib.error.URLError("timed out")
        with pytest.raises(Exception) as exc:
            cf_provider._call_chat_completions(messages=[{"role": "user", "content": "hi"}])
        assert "503" in str(exc.value)
        assert "timeout" in str(exc.value).lower() or "network" in str(exc.value).lower()

def test_quota_exhaustion_returns_429(cf_provider):
    with patch("urllib.request.urlopen") as mock_url:
        mock_http_err = urllib.error.HTTPError(
            url="http://test",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=MagicMock(read=lambda: b'{"success": false, "errors": [{"message": "daily limit exceeded"}]}')
        )
        mock_url.side_effect = mock_http_err

        with pytest.raises(Exception) as exc:
            cf_provider._call_chat_completions(messages=[{"role": "user", "content": "hi"}])
        assert "429" in str(exc.value)
        assert "quota" in str(exc.value).lower() or "limit" in str(exc.value).lower()

def test_successful_general_generation_cloudflare(cf_provider):
    mock_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "is evolving rapidly across multiple industries."
                }
            }
        ]
    }
    with patch.object(cf_provider, "_call_chat_completions", return_value=(mock_resp, 142.5)):
        gen_text, completed, tokens, lat = cf_provider.generate_continuation(
            prompt="The future of artificial intelligence",
            max_tokens=10
        )
        assert gen_text == "is evolving rapidly across multiple industries."
        assert completed == "The future of artificial intelligence is evolving rapidly across multiple industries."
        assert len(tokens) > 0
        # Probabilities are reported honestly as 0.0 for unmeasured chat completions
        assert all(t.probability == 0.0 for t in tokens)
        assert lat > 0

def test_successful_general_suggestions_structured(cf_provider):
    mock_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '["is", "will", "has", "can", "should"]'
                }
            }
        ]
    }
    with patch.object(cf_provider, "_call_chat_completions", return_value=(mock_resp, 95.0)):
        candidates, tokens, lat = cf_provider.predict_next_words("The future of artificial intelligence", top_k=5)
        assert len(candidates) == 5
        words = [c.word for c in candidates]
        assert words == ["is", "will", "has", "can", "should"]
        # Probability is reported as 0.0 honestly, rank is strictly preserved
        for idx, c in enumerate(candidates):
            assert c.probability == 0.0
            assert c.rank == idx + 1

def test_malformed_json_fallback_extraction(cf_provider):
    # Upstream returns bulleted / numbered text instead of clean JSON array
    mock_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "1. creates\n2. enables\n3. transforms"
                }
            }
        ]
    }
    with patch.object(cf_provider, "_call_chat_completions", return_value=(mock_resp, 110.0)):
        candidates, tokens, lat = cf_provider.predict_next_words("Technology", top_k=3)
        assert len(candidates) == 3
        words = [c.word for c in candidates]
        assert words == ["creates", "enables", "transforms"]

def test_model_router_cloudflare_integration(monkeypatch):
    monkeypatch.setenv("GENERAL_PROVIDER", "cloudflare_workers_ai")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "mock_acc_123")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "mock_tok_456")

    router = ModelRouter()
    assert router.is_cloudflare_provider is True
    assert router.general_metadata["name"] == "Llama 3.2 1B Instruct (General English)"

    # Mock predict
    with patch.object(
        router.cloudflare_service,
        "predict_next_words",
        return_value=([PredictionCandidate(word="is", probability=0.0, rank=1)], ["test"], 45.0)
    ):
        res = router.predict("AI", mode="general")
        assert res.mode == "general"
        assert res.candidates[0].word == "is"
        assert "Llama 3.2 1B" in res.model_name
        assert router.stats["total_predictions"] == 1
        assert router.stats["predictions_by_mode"]["general"] == 1

def test_technical_mode_remains_intact_when_cloudflare_active(monkeypatch):
    monkeypatch.setenv("GENERAL_PROVIDER", "cloudflare_workers_ai")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "mock_acc_123")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "mock_tok_456")

    router = ModelRouter()
    # Technical mode must run genuine PyTorch BiLSTM and return genuine non-zero probabilities
    res = router.predict("Deep learning", mode="technical")
    assert res.mode == "technical"
    assert "PyTorch BiLSTM" in res.model_name
    assert len(res.candidates) > 0
    assert any(c.probability > 0.0 for c in res.candidates)

def test_no_secret_exposure_in_responses(monkeypatch):
    secret_token = "secret_cf_token_never_expose_xyz"
    monkeypatch.setenv("GENERAL_PROVIDER", "cloudflare_workers_ai")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "secret_acc_id_789")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", secret_token)

    router = ModelRouter()
    health = router.get_health_status()
    meta = router.general_metadata

    health_str = json.dumps(health.model_dump())
    meta_str = json.dumps(meta)

    assert secret_token not in health_str
    assert secret_token not in meta_str
    assert "secret_acc_id_789" not in health_str
    assert "secret_acc_id_789" not in meta_str
