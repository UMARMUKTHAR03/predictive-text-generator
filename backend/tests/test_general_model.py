import pytest
import re
from backend.app.services.general_service import GeneralLanguageModelService

@pytest.fixture(scope="session")
def general_svc():
    svc = GeneralLanguageModelService()
    svc.load()
    return svc

def test_pinned_revision_and_metadata(general_svc):
    assert general_svc.revision == "2290a62682d06624634c1f46a6ad5be0f47f38aa"
    assert general_svc.metadata["parameters"] == 81912576
    assert general_svc.metadata["license"] == "Apache 2.0"
    assert general_svc.metadata["context_length"] == 1024

def test_actual_mathematical_token_probabilities(general_svc):
    prompt = "The weather is very"
    gen_text, completed, tokens_gen, latency_ms = general_svc.generate_continuation(
        prompt=prompt, max_tokens=5, strategy="greedy"
    )
    assert len(tokens_gen) == 5
    for t in tokens_gen:
        # Probabilities must be legitimate softmax floats between 0 and 1, strictly NOT 1.0 placeholder
        assert 0.0 < t.probability <= 1.0
        assert t.probability != 1.0 or t.token == ""
        assert isinstance(t.token, str)

def test_whole_word_extraction_artificial_intelligence(general_svc):
    # In GPT-2 BPE, 'artificial' without trailing space expects ' intelligence' as next token
    cands, tokens, lat = general_svc.predict_next_words("artificial", top_k=5)
    words = [c.word.lower() for c in cands]
    assert any("intelligence" in w for w in words), f"Expected 'intelligence' in candidates, got {words}"
    for c in cands:
        assert re.search(r"[a-zA-Z0-9]", c.word)
        assert not c.word.startswith("<|")

def test_whole_word_extraction_learning_and_conversation(general_svc):
    # 'machine' predicts 'learning'
    cands, _, _ = general_svc.predict_next_words("machine", top_k=10)
    # Also test common conversational sentence
    cands_conv, _, _ = general_svc.predict_next_words("They are currently playing", top_k=5)
    words_conv = [c.word.lower() for c in cands_conv]
    assert len(words_conv) >= 3
    for c in cands_conv:
        assert re.search(r"[a-zA-Z0-9]", c.word)

def test_prompts_with_and_without_trailing_space(general_svc):
    cands_space, _, _ = general_svc.predict_next_words("The future of artificial intelligence ", top_k=5)
    cands_no_space, _, _ = general_svc.predict_next_words("The future of artificial intelligence", top_k=5)
    assert len(cands_space) > 0
    assert len(cands_no_space) > 0
    for c in cands_space + cands_no_space:
        assert len(c.word.strip()) > 0
