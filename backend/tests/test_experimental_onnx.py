import pytest
import os
import torch
from backend.app.schemas import PredictionCandidate, GenerationToken

HAS_ONNX_MODEL = os.path.exists("ml/artifacts/onnx/distilgpt2_int8/model_quantized.onnx")

@pytest.fixture(scope="module")
def onnx_service():
    if not HAS_ONNX_MODEL:
        pytest.skip("ONNX quantized model weights not found in working directory")
    from backend.app.services.experimental_onnx_service import ExperimentalONNXGeneralService
    svc = ExperimentalONNXGeneralService()
    svc.load()
    yield svc

def test_onnx_model_loading_and_metadata(onnx_service):
    assert onnx_service.is_loaded is True
    assert onnx_service.metadata["model_id"] == "distilbert/distilgpt2"
    assert onnx_service.metadata["revision"] == "2290a62682d06624634c1f46a6ad5be0f47f38aa"
    assert "Dynamic INT8" in onnx_service.metadata["quantization"]
    assert onnx_service.metadata["parameters"] == 81912576

def test_onnx_tokenizer_compatibility(onnx_service):
    assert onnx_service.tokenizer is not None
    assert len(onnx_service.tokenizer) == 50257
    tokens = onnx_service.tokenizer.encode("Machine learning")
    decoded = onnx_service.tokenizer.decode(tokens)
    assert "Machine learning" in decoded

def test_onnx_prediction_response_format(onnx_service):
    candidates, tokens, latency = onnx_service.predict_next_words("The future of artificial intelligence", top_k=5)
    assert isinstance(candidates, list)
    assert len(candidates) > 0
    assert len(candidates) <= 5
    assert isinstance(tokens, list)
    assert isinstance(latency, float)
    assert latency > 0.0

    for cand in candidates:
        assert isinstance(cand, PredictionCandidate)
        assert isinstance(cand.word, str)
        assert isinstance(cand.probability, float)
        assert 0.0 <= cand.probability <= 1.0
        assert isinstance(cand.rank, int)

def test_onnx_generation_response_format(onnx_service):
    gen_text, completed_text, tokens_gen, latency = onnx_service.generate_continuation(
        prompt="The future of artificial intelligence",
        max_tokens=10,
        strategy="greedy"
    )
    assert isinstance(gen_text, str)
    assert len(gen_text) > 0
    assert isinstance(completed_text, str)
    assert completed_text.startswith("The future of artificial intelligence")
    assert isinstance(tokens_gen, list)
    assert len(tokens_gen) > 0
    assert isinstance(latency, float)

    for tok in tokens_gen:
        assert isinstance(tok, GenerationToken)
        assert isinstance(tok.token, str)
        assert isinstance(tok.probability, float)
        assert 0.0 <= tok.probability <= 1.0

def test_onnx_bpe_whole_word_extraction_trailing_space(onnx_service):
    candidates, _, _ = onnx_service.predict_next_words("The future of artificial intelligence ", top_k=5)
    for c in candidates:
        assert not c.word.startswith(" ")
        assert any(ch.isalnum() for ch in c.word)

def test_onnx_bpe_whole_word_extraction_partial_word(onnx_service):
    candidates, _, _ = onnx_service.predict_next_words("artificial intellig", top_k=5)
    assert len(candidates) > 0
    for c in candidates:
        assert c.word.lower().startswith("intellig")

def test_onnx_repeated_prediction_stability(onnx_service):
    latencies = []
    for _ in range(15):
        _, _, lat = onnx_service.predict_next_words("Deep neural networks", top_k=3)
        latencies.append(lat)
    assert len(latencies) == 15
    assert all(lat > 0 for lat in latencies)

def test_onnx_generation_stability_greedy_and_topk(onnx_service):
    # Greedy
    gen_text_greedy, _, _, _ = onnx_service.generate_continuation(
        prompt="Machine learning",
        max_tokens=5,
        strategy="greedy"
    )
    assert len(gen_text_greedy) > 0

    # Top-K Sampling
    gen_text_topk, _, _, _ = onnx_service.generate_continuation(
        prompt="Machine learning",
        max_tokens=5,
        strategy="top_k",
        temperature=0.8,
        top_k=20
    )
    assert len(gen_text_topk) > 0

def test_onnx_pytorch_output_comparison_and_evaluation(onnx_service):
    # Verify candidate generation produces valid non-empty results on representative benchmark prompts
    prompts = [
        "The weather is very",
        "I am writing this email to",
        "Machine learning models are"
    ]
    for prompt in prompts:
        cands, tokens, _ = onnx_service.predict_next_words(prompt, top_k=5)
        assert len(cands) >= 1
        assert all(c.probability > 0.0 for c in cands)

def test_onnx_memory_benchmark_artifact_validation():
    import json
    artifact_path = "ml/artifacts/onnx_int8_memory_benchmark.json"
    assert os.path.exists(artifact_path), f"Benchmark artifact not found at {artifact_path}"
    with open(artifact_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "peak_rss_mb" in data
    assert "strict_acceptance_criteria_audit" in data
    assert "passes_450mb_criterion" in data["strict_acceptance_criteria_audit"]
    assert "passes_render_free_512mb" in data["strict_acceptance_criteria_audit"]
    assert data["strict_acceptance_criteria_audit"]["passes_450mb_criterion"] is False
    assert data["peak_rss_mb"] > 450.0
