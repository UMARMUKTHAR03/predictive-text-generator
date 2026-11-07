import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple
from backend.app.schemas import PredictionCandidate, PredictResponse, GenerationToken, GenerateResponse
from backend.app.services.model_loader import ModelManager
from ml.preprocessing.tokenizer import tokenize_text

class TextGenerationService:
    def __init__(self):
        self.manager = ModelManager.get_instance()
        self.stats = {
            "total_predictions": 0,
            "total_generations": 0,
            "latency_history": [],
            "start_time": time.time()
        }

    def _prepare_input_tensor(self, text: str) -> Tuple[torch.Tensor, List[str]]:
        if not self.manager.is_loaded:
            self.manager.load()

        tokenizer = self.manager.tokenizer
        tokens = tokenize_text(text)
        token_ids = tokenizer.texts_to_sequences([text])[0]

        max_len = self.manager.max_sequence_len - 1
        if len(token_ids) > max_len:
            token_ids = token_ids[-max_len:]
        elif len(token_ids) < max_len:
            token_ids = [0] * (max_len - len(token_ids)) + token_ids

        tensor_x = torch.tensor([token_ids], dtype=torch.long, device=self.manager.device)
        return tensor_x, tokens

    def predict_next(self, text: str, top_k: int = 5, temperature: float = 1.0) -> PredictResponse:
        t0 = time.perf_counter()
        tensor_x, tokens = self._prepare_input_tensor(text)

        with torch.no_grad():
            logits = self.manager.model(tensor_x) # [1, vocab_size]
            scaled_logits = logits / max(temperature, 1e-4)
            # Mask out padding token (idx 0) and OOV token (idx 1) from next-token predictions
            scaled_logits[:, 0] = -1e9
            scaled_logits[:, 1] = -1e9
            probs = F.softmax(scaled_logits, dim=-1) # [1, vocab_size]

        top_probs, top_indices = probs.topk(top_k, dim=-1)
        top_probs = top_probs.squeeze(0).tolist()
        top_indices = top_indices.squeeze(0).tolist()

        candidates = []
        for rank, (idx, prob) in enumerate(zip(top_indices, top_probs), start=1):
            word = self.manager.tokenizer.index_word.get(idx, "<unk>")
            candidates.append(PredictionCandidate(
                word=word,
                probability=round(prob, 4),
                rank=rank
            ))

        latency_ms = (time.perf_counter() - t0) * 1000
        self.stats["total_predictions"] += 1
        self.stats["latency_history"].append(latency_ms)

        return PredictResponse(
            input_text=text,
            tokens=tokens,
            candidates=candidates,
            latency_ms=round(latency_ms, 2),
            model_version=self.manager.model_metadata.get("version", "1.0.0")
        )

    def generate_completion(
        self,
        prompt: str,
        max_tokens: int = 5,
        temperature: float = 0.8,
        strategy: str = "greedy",
        top_k: int = 5
    ) -> GenerateResponse:
        t0 = time.perf_counter()
        current_text = prompt
        generated_tokens: List[GenerationToken] = []

        for _ in range(max_tokens):
            tensor_x, _ = self._prepare_input_tensor(current_text)

            with torch.no_grad():
                logits = self.manager.model(tensor_x)
                scaled_logits = logits / max(temperature, 1e-4)
                scaled_logits[:, 0] = -1e9
                scaled_logits[:, 1] = -1e9
                probs = F.softmax(scaled_logits, dim=-1)

            if strategy == "greedy":
                next_idx = torch.argmax(probs, dim=-1).item()
                prob = probs[0, next_idx].item()
            elif strategy == "top_k":
                top_p, top_i = probs.topk(min(top_k, probs.size(-1)), dim=-1)
                norm_p = top_p / top_p.sum(dim=-1, keepdim=True)
                sample_idx = torch.multinomial(norm_p, num_samples=1).item()
                next_idx = top_i[0, sample_idx].item()
                prob = top_p[0, sample_idx].item()
            else:  # sampling
                next_idx = torch.multinomial(probs, num_samples=1).item()
                prob = probs[0, next_idx].item()

            word = self.manager.tokenizer.index_word.get(next_idx, "")
            if not word or word in ["<oov>", "<pad>"]:
                break

            generated_tokens.append(GenerationToken(token=word, probability=round(prob, 4)))
            current_text = (current_text + " " + word).strip()

        latency_ms = (time.perf_counter() - t0) * 1000
        self.stats["total_generations"] += 1
        self.stats["latency_history"].append(latency_ms)

        gen_text = " ".join([gt.token for gt in generated_tokens])
        return GenerateResponse(
            prompt=prompt,
            generated_text=gen_text,
            completed_text=(prompt + " " + gen_text).strip(),
            tokens_generated=generated_tokens,
            strategy_used=strategy,
            latency_ms=round(latency_ms, 2)
        )

    def get_analytics(self) -> dict:
        lats = self.stats["latency_history"]
        avg_lat = float(np.mean(lats)) if lats else 0.0
        return {
            "total_predictions": self.stats["total_predictions"],
            "total_generations": self.stats["total_generations"],
            "avg_latency_ms": round(avg_lat, 2),
            "uptime_seconds": round(time.time() - self.stats["start_time"], 1)
        }

