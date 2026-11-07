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
    # Test greedy decoding
    prompt = "The weather is very"
    gen_text, completed, tokens_gen, latency_ms = general_svc.generate_continuation(
        prompt=prompt, max_tokens=5, strategy="greedy"
    )
    assert len(tokens_gen) == 5
    for t in tokens_gen:
        # Probabilities must be legitimate softmax floats between 0 and 1
        assert 0.0 < t.probability <= 1.0
        assert isinstance(t.token, str)

    # Test top_k sampling decoding
    gen_text_sample, _, tokens_gen_sample, _ = general_svc.generate_continuation(
        prompt=prompt, max_tokens=5, strategy="top_k", top_k=20, temperature=0.8
    )
    assert len(tokens_gen_sample) == 5
    for t in tokens_gen_sample:
        assert 0.0 < t.probability <= 1.0

def test_bpe_whole_word_extraction_artificial_intelligence(general_svc):
    # Test prompt without trailing space: next words must include valid continuations like 'is', 'has', 'and'
    cands, tokens, lat = general_svc.predict_next_words("artificial intelligence", top_k=5)
    words = [c.word.lower() for c in cands]
    assert len(words) >= 3
    assert any(w in words for w in ["is", "has", "and", "will", "can"]), f"Unexpected words: {words}"
    for c in cands:
        assert re.search(r"[a-zA-Z0-9]", c.word)
        assert not c.word.startswith("<|")

    # Test prompt with trailing space
    cands_sp, _, _ = general_svc.predict_next_words("artificial intelligence ", top_k=5)
    words_sp = [c.word.lower() for c in cands_sp]
    assert len(words_sp) >= 3
    assert any(w in words_sp for w in ["is", "has", "and", "will", "can"]), f"Unexpected words: {words_sp}"

def test_bpe_whole_word_extraction_machine_learning(general_svc):
    # Test 'Deep learning and machine': next words must include 'learning'
    cands_m, _, _ = general_svc.predict_next_words("Deep learning and machine", top_k=5)
    words_m = [c.word.lower() for c in cands_m]
    assert any("learning" in w for w in words_m), f"Expected 'learning' in {words_m}"

    # Test 'machine learning' next words
    cands_ml, _, _ = general_svc.predict_next_words("machine learning", top_k=5)
    words_ml = [c.word.lower() for c in cands_ml]
    assert len(words_ml) >= 3

    # Test 'machine learning ' with trailing space
    cands_mls, _, _ = general_svc.predict_next_words("machine learning ", top_k=5)
    words_mls = [c.word.lower() for c in cands_mls]
    assert len(words_mls) >= 3

def test_bpe_whole_word_extraction_playing_learning_conversation(general_svc):
    # Test complete words
    for target in ["playing", "learning", "conversation"]:
        cands, _, _ = general_svc.predict_next_words(target, top_k=5)
        words = [c.word.lower() for c in cands]
        assert len(words) >= 3
        for c in cands:
            assert re.search(r"[a-zA-Z0-9]", c.word)
            assert not c.word.startswith("<|")

def test_bpe_partial_word_continuations(general_svc):
    # 'play' continuation in sentence
    cands_play, _, _ = general_svc.predict_next_words("We were play", top_k=5)
    words_play = [c.word.lower() for c in cands_play]
    # Should complete 'play' into meaningful words like 'playfully', 'playtesting', or start next word
    assert any("play" in w for w in words_play), f"Expected play-prefixed words, got: {words_play}"

    # 'conver' partial continuation
    cands_conver, _, _ = general_svc.predict_next_words("In everyday conver", top_k=5)
    words_conver = [c.word.lower() for c in cands_conver]
    assert any("conver" in w for w in words_conver), f"Expected conver-prefixed words, got: {words_conver}"

def test_bpe_punctuation_handling(general_svc):
    # Prompt with punctuation at end
    cands_punct, _, _ = general_svc.predict_next_words("Hello,", top_k=5)
    words_punct = [c.word.lower() for c in cands_punct]
    assert len(words_punct) >= 3
    for c in cands_punct:
        assert re.search(r"[a-zA-Z0-9]", c.word)

    cands_excl, _, _ = general_svc.predict_next_words("Hello! ", top_k=5)
    words_excl = [c.word.lower() for c in cands_excl]
    assert len(words_excl) >= 3
