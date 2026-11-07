import time
import re
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from backend.app.schemas import PredictionCandidate, GenerationToken

class GeneralLanguageModelService:
    """
    General English Writing Language Model Service powered by DistilGPT-2 (82M params).
    Provides causal autoregressive text generation and next-word candidate extraction.
    """
    def __init__(self, model_id: str = "distilbert/distilgpt2"):
        self.model_id = model_id
        self.device = torch.device("cpu")
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.metadata = {
            "name": "DistilGPT-2 (General English)",
            "model_id": model_id,
            "architecture": "Causal Autoregressive Transformer (6 layers, 768 hidden, 12 heads)",
            "parameters": 81912576,
            "vocab_size": 50257,
            "license": "Apache 2.0",
            "context_length": 1024,
            "domain": "General English (WebText, everyday sentences, emails, articles, creative writing)"
        }

    def load(self):
        if self.is_loaded:
            return
        t0 = time.perf_counter()
        print(f"Loading General Language Model: {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # Ensure pad token is configured cleanly
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.model.eval()
        self.is_loaded = True
        print(f"General Language Model loaded in {time.perf_counter() - t0:.2f}s")

    def unload(self):
        """Unload general model to release memory if needed."""
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def predict_next_words(self, text: str, top_k: int = 5, temperature: float = 1.0) -> Tuple[List[PredictionCandidate], List[str], float]:
        if not self.is_loaded:
            self.load()

        t0 = time.perf_counter()
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        tokens = [self.tokenizer.decode([t]) for t in input_ids[0]]

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :] # [1, vocab_size]
            scaled_logits = logits / max(temperature, 1e-4)
            probs = F.softmax(scaled_logits, dim=-1)[0] # [vocab_size]

        # Extract top 30 tokens and filter for whole-word candidates
        top_probs, top_indices = probs.topk(min(30, probs.size(-1)), dim=-1)

        candidates: List[PredictionCandidate] = []
        seen_words = set()

        for prob, idx in zip(top_probs, top_indices):
            tok_str = self.tokenizer.decode([idx.item()])
            clean_word = tok_str.strip()

            # Must contain alphanumeric characters (exclude isolated punctuation or empty whitespace)
            if not clean_word or not re.search(r"[a-zA-Z0-9]", clean_word):
                continue

            lower_word = clean_word.lower()
            if lower_word not in seen_words:
                seen_words.add(lower_word)
                candidates.append(PredictionCandidate(
                    word=clean_word,
                    probability=round(prob.item(), 4),
                    rank=len(candidates) + 1
                ))

            if len(candidates) >= top_k:
                break

        latency_ms = (time.perf_counter() - t0) * 1000
        return candidates, tokens, round(latency_ms, 2)

    def generate_continuation(
        self,
        prompt: str,
        max_tokens: int = 15,
        temperature: float = 0.8,
        strategy: str = "greedy",
        top_k: int = 20,
        repetition_penalty: float = 1.1
    ) -> Tuple[str, str, List[GenerationToken], float]:
        if not self.is_loaded:
            self.load()

        t0 = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        do_sample = strategy != "greedy"
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": repetition_penalty,
            "attention_mask": attention_mask
        }

        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 0.1)
            if strategy == "top_k":
                gen_kwargs["top_k"] = top_k

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **gen_kwargs)

        generated_ids = output_ids[0][input_ids.shape[-1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        completed_text = (prompt.strip() + " " + generated_text).strip()

        # Token breakdown
        tokens_generated = []
        for tid in generated_ids:
            tok_str = self.tokenizer.decode([tid.item()])
            tokens_generated.append(GenerationToken(token=tok_str, probability=1.0))

        latency_ms = (time.perf_counter() - t0) * 1000
        return generated_text, completed_text, tokens_generated, round(latency_ms, 2)

