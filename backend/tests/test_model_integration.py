import pytest
import os
import json
import torch
from backend.app.services.model_loader import ModelManager
from backend.app.services.prediction_service import TextGenerationService
from ml.preprocessing.tokenizer import TextTokenizer

def test_production_model_artifacts_exist():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "current")
    assert os.path.exists(os.path.join(models_dir, "model.pt")), "Production weights model.pt must exist"
    assert os.path.exists(os.path.join(models_dir, "tokenizer.json")), "Production tokenizer.json must exist"
    assert os.path.exists(os.path.join(models_dir, "model_metadata.json")), "model_metadata.json must exist"
    assert os.path.exists(os.path.join(models_dir, "metrics.json")), "metrics.json must exist"

def test_tokenizer_production_vocabulary():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "current")
    tokenizer = TextTokenizer.load_json(os.path.join(models_dir, "tokenizer.json"))
    assert tokenizer.total_words == 7299, "Vocabulary size must match the authentic 7,299 token vocabulary"
    assert tokenizer.word_index.get("<oov>") == 1
    assert "neural" in tokenizer.word_index
    assert "learning" in tokenizer.word_index

def test_model_loading_and_shape_compatibility():
    manager = ModelManager.get_instance()
    manager.load()
    assert manager.is_loaded is True
    assert manager.model is not None
    assert manager.tokenizer.total_words == 7299

    # Test dummy forward pass through model
    dummy_input = torch.zeros((1, 39), dtype=torch.long)
    with torch.no_grad():
        logits = manager.model(dummy_input)
    assert logits.shape == (1, 7299), f"Expected logits shape (1, 7299), got {logits.shape}"

def test_prediction_service_genuine_inference():
    svc = TextGenerationService()
    res = svc.predict_next("machine learning for data", top_k=5, temperature=1.0)
    assert res.input_text == "machine learning for data"
    assert len(res.candidates) == 5
    # Verify probability distribution
    probs = [c.probability for c in res.candidates]
    assert all(p > 0 for p in probs)
    assert probs == sorted(probs, reverse=True)
    # Check that candidate words are valid vocabulary words
    for cand in res.candidates:
        assert cand.word not in ["<oov>", "<pad>", ""]

def test_generation_service_continuation():
    svc = TextGenerationService()
    res = svc.generate_completion("deep learning", max_tokens=3, strategy="greedy")
    assert res.prompt == "deep learning"
    assert len(res.tokens_generated) <= 3
    assert len(res.completed_text.split()) > len("deep learning".split())
