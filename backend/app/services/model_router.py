import time
from typing import Dict, Any, Tuple, List, Optional
from backend.app.schemas import (
    PredictResponse, GenerateResponse, PredictionCandidate,
    GenerationToken, HealthResponse, ModeStatus, WritingMode
)
from backend.app.services.model_loader import ModelManager
from backend.app.services.prediction_service import TextGenerationService
from backend.app.services.general_service import GeneralLanguageModelService

class ModelRouter:
    """
    Central router that dynamically dispatches prediction and generation requests
    between the Technical Language Model (PyTorch BiLSTM) and General Language Model (DistilGPT-2).
    """
    _instance: Optional["ModelRouter"] = None

    def __init__(self):
        self.technical_manager = ModelManager.get_instance()
        self.technical_service = TextGenerationService()
        self.general_service = GeneralLanguageModelService()
        self.stats = {
            "total_predictions": 0,
            "total_generations": 0,
            "predictions_by_mode": {"general": 0, "technical": 0},
            "generations_by_mode": {"general": 0, "technical": 0},
            "latency_history": [],
            "start_time": time.time()
        }

    @classmethod
    def get_instance(cls) -> "ModelRouter":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self):
        """Pre-load technical model by default. General model is lazily loaded upon request."""
        print("Initializing ModelRouter...")
        self.technical_manager.load()
        print("Technical model ready.")

    def get_health_status(self) -> HealthResponse:
        modes = {
            "technical": ModeStatus(
                name="Technical Writing (PyTorch BiLSTM)",
                loaded=self.technical_manager.is_loaded,
                description="Bidirectional LSTM trained on technical publications & headline sequences",
                architecture="Embedding (100d) -> BiLSTM (150 units) -> Linear Head",
                parameters=3229299
            ),
            "general": ModeStatus(
                name="General English Writing (DistilGPT-2)",
                loaded=self.general_service.is_loaded,
                description="82M parameter causal autoregressive transformer for open-domain English",
                architecture="DistilGPT-2 (6 layers, 768 hidden, 12 heads)",
                parameters=81912576
            )
        }

        return HealthResponse(
            status="healthy" if self.technical_manager.is_loaded else "degraded",
            version="2.0.0",
            device=str(self.technical_manager.device),
            technical_model_loaded=self.technical_manager.is_loaded,
            general_model_loaded=self.general_service.is_loaded,
            modes=modes
        )

    def predict(self, text: str, mode: WritingMode = "technical", top_k: int = 5, temperature: float = 1.0) -> PredictResponse:
        t0 = time.perf_counter()
        normalized_mode = (mode or "technical").lower()

        if normalized_mode == "general":
            if not self.general_service.is_loaded:
                self.general_service.load()
            candidates, tokens, model_lat = self.general_service.predict_next_words(text, top_k, temperature)
            model_name = "DistilGPT-2 (General English)"
        else:
            # Fallback / default: Technical BiLSTM
            pred_res = self.technical_service.predict_next(text, top_k, temperature)
            candidates = pred_res.candidates
            tokens = pred_res.tokens
            model_lat = pred_res.latency_ms
            model_name = "PyTorch BiLSTM (Technical Writing)"

        latency_ms = (time.perf_counter() - t0) * 1000
        self.stats["total_predictions"] += 1
        self.stats["predictions_by_mode"][normalized_mode] = self.stats["predictions_by_mode"].get(normalized_mode, 0) + 1
        self.stats["latency_history"].append(latency_ms)

        return PredictResponse(
            input_text=text,
            mode=normalized_mode,
            tokens=tokens,
            candidates=candidates,
            latency_ms=round(latency_ms, 2),
            model_name=model_name,
            model_version="2.0.0"
        )

    def generate(
        self,
        prompt: str,
        mode: WritingMode = "technical",
        max_tokens: int = 10,
        temperature: float = 0.8,
        strategy: str = "greedy",
        top_k: int = 5,
        repetition_penalty: float = 1.1
    ) -> GenerateResponse:
        t0 = time.perf_counter()
        normalized_mode = (mode or "technical").lower()

        if normalized_mode == "general":
            if not self.general_service.is_loaded:
                self.general_service.load()
            gen_text, completed_text, tokens_gen, model_lat = self.general_service.generate_continuation(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                strategy=strategy,
                top_k=top_k,
                repetition_penalty=repetition_penalty
            )
            model_name = "DistilGPT-2 (General English)"
        else:
            gen_res = self.technical_service.generate_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                strategy=strategy,
                top_k=top_k
            )
            gen_text = gen_res.generated_text
            completed_text = gen_res.completed_text
            tokens_gen = gen_res.tokens_generated
            model_lat = gen_res.latency_ms
            model_name = "PyTorch BiLSTM (Technical Writing)"

        latency_ms = (time.perf_counter() - t0) * 1000
        self.stats["total_generations"] += 1
        self.stats["generations_by_mode"][normalized_mode] = self.stats["generations_by_mode"].get(normalized_mode, 0) + 1
        self.stats["latency_history"].append(latency_ms)

        return GenerateResponse(
            prompt=prompt,
            mode=normalized_mode,
            generated_text=gen_text,
            completed_text=completed_text,
            tokens_generated=tokens_gen,
            strategy_used=strategy,
            latency_ms=round(latency_ms, 2),
            model_name=model_name
        )

    def get_analytics(self) -> Dict[str, Any]:
        lats = self.stats["latency_history"]
        avg_lat = float(sum(lats) / len(lats)) if lats else 0.0
        return {
            "total_predictions": self.stats["total_predictions"],
            "total_generations": self.stats["total_generations"],
            "predictions_by_mode": self.stats["predictions_by_mode"],
            "generations_by_mode": self.stats["generations_by_mode"],
            "avg_latency_ms": round(avg_lat, 2),
            "uptime_seconds": round(time.time() - self.stats["start_time"], 1)
        }

