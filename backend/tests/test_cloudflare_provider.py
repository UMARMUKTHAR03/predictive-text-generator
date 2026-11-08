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
        model_id="@cf/google/gemma-4-26b-a4b-it"
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
    assert "Gemma 4" in router.general_metadata["name"]
    assert router.general_metadata["lifecycle_status"] == "Active Supported (No planned deprecation)"

    # Provider configured but not yet verified -> state is 'configured', not 'online'
    health_before = router.get_health_status()
    assert health_before.general_model_loaded is False
    assert "Configured" in health_before.modes["general"].description

    # Mock successful predict
    with patch.object(
        router.cloudflare_service,
        "predict_next_words",
        return_value=([PredictionCandidate(word="is", probability=0.0, rank=1)], ["test"], 45.0)
    ):
        res = router.predict("AI", mode="general")
        assert res.mode == "general"
        assert res.candidates[0].word == "is"
        assert "Gemma 4" in res.model_name
        assert router.stats["total_predictions"] == 1
        assert router.stats["predictions_by_mode"]["general"] == 1

    # Now that an inference has completed successfully, provider state transitions to 'online'
    router.cloudflare_service.has_verified_online = True
    health_after = router.get_health_status()
    assert health_after.general_model_loaded is True
    assert "Online" in health_after.modes["general"].description

def test_provider_unavailable_state_on_upstream_failure(monkeypatch):
    monkeypatch.setenv("GENERAL_PROVIDER", "cloudflare_workers_ai")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "mock_acc_123")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "mock_tok_456")

    router = ModelRouter()
    # Simulate an upstream error
    router.cloudflare_service.last_error = "HTTP 429: Rate limit exceeded"
    assert router.cloudflare_service.provider_state == "unavailable"
    health = router.get_health_status()
    assert health.general_model_loaded is False
    assert "Unavailable" in health.modes["general"].description

def test_health_polling_does_not_call_inference(monkeypatch):
    monkeypatch.setenv("GENERAL_PROVIDER", "cloudflare_workers_ai")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "mock_acc_123")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "mock_tok_456")

    router = ModelRouter()
    with patch.object(router.cloudflare_service, "_call_chat_completions") as mock_inference:
        # Polling get_health_status multiple times
        for _ in range(5):
            h = router.get_health_status()
        # Zero billable inference calls made
        assert mock_inference.call_count == 0

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

def test_evaluation_unconfigured_produces_prompt_manifest(monkeypatch, tmp_path):
    # Ensure unconfigured evaluation never fabricates evaluation metrics
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    from ml.training.evaluate_cloudflare_general import run_genuine_cloudflare_evaluation
    provider = CloudflareWorkersAIProvider(account_id="", api_token="")
    test_manifest_path = str(tmp_path / "test_manifest.json")
    res = run_genuine_cloudflare_evaluation(provider, output_path=test_manifest_path)
    assert res["status"] == "unevaluated_offline_manifest"
    assert res["model_id"] == "@cf/google/gemma-4-26b-a4b-it"
    assert "prompts" in res
    assert len(res["prompts"]) == 20

def test_evaluation_configured_mocked_execution(monkeypatch, tmp_path):
    from ml.training.evaluate_cloudflare_general import run_genuine_cloudflare_evaluation
    provider = CloudflareWorkersAIProvider(account_id="mock_acc", api_token="mock_tok")
    test_eval_path = str(tmp_path / "test_eval.json")

    with patch.object(provider, "generate_continuation", return_value=("continuation text", "prompt continuation text", [], 110.0)), \
         patch.object(provider, "predict_next_words", return_value=([PredictionCandidate(word="test", probability=0.0, rank=1)], ["test"], 45.0)):
        res = run_genuine_cloudflare_evaluation(provider, output_path=test_eval_path)
        assert res["status"] == "genuine_evaluated"
        assert res["successful_requests"] == 20
        assert res["failed_requests"] == 0
        assert "Gemma 4" in res["methodology"]
        assert len(res["results"]) == 20
        assert res["results"][0]["continuation"] == "continuation text"
        assert res["results"][0]["ranked_suggestions"] == ["test"]


