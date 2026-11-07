import os
import time
import urllib.request
import json
from typing import Dict, Any, Tuple, List, Optional
from fastapi import HTTPException
from backend.app.schemas import (
    PredictResponse, GenerateResponse, PredictionCandidate,
    GenerationToken, HealthResponse, ModeStatus, WritingMode
)
from backend.app.services.model_loader import ModelManager
from backend.app.services.prediction_service import TextGenerationService

DISTILGPT2_METADATA_CONST = {
    "name": "DistilGPT-2 (General English)",
    "model_id": "distilbert/distilgpt2",
    "revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
    "publisher": "DistilBERT authors / Hugging Face",
    "architecture": "Causal Autoregressive Transformer (6 layers, 768 hidden, 12 heads)",
    "parameters": 81912576,
    "vocab_size": 50257,
    "context_length": 1024,
    "license": "Apache 2.0"
}

class ModelRouter:
    """
    Central router that dynamically dispatches prediction and generation requests.
    Technical Writing is served by the local PyTorch BiLSTM (3.2M params).
    General English Writing can be fulfilled via:
      1. Cloudflare Workers AI managed inference (when GENERAL_PROVIDER=cloudflare_workers_ai or CLOUDFLARE_API_TOKEN is set)
      2. An isolated external microservice via GENERAL_API_URL
      3. Explicit local opt-in via ALLOW_LOCAL_GENERAL=true (requires >=1GB RAM)
    If no provider is configured, General mode fails closed with HTTP 503.
    """
    _instance: Optional["ModelRouter"] = None

    def __init__(self):
        self.technical_manager = ModelManager.get_instance()
        self.technical_service = TextGenerationService()
        self.general_provider = os.environ.get("GENERAL_PROVIDER", "").strip().lower()
        self.general_api_url = os.environ.get("GENERAL_API_URL", "").rstrip("/")
        self.allow_local_general = os.environ.get("ALLOW_LOCAL_GENERAL", "false").lower() in ("true", "1", "yes")
        self._local_general_service = None
        self._cloudflare_service = None
        self.stats = {
            "total_predictions": 0,
            "total_generations": 0,
            "predictions_by_mode": {"general": 0, "technical": 0},
            "generations_by_mode": {"general": 0, "technical": 0},
            "latency_history": [],
            "start_time": time.time()
        }

    @property
    def is_cloudflare_provider(self) -> bool:
        return self.general_provider == "cloudflare_workers_ai" or bool(os.environ.get("CLOUDFLARE_API_TOKEN"))

    @property
    def cloudflare_service(self):
        if self._cloudflare_service is None:
            from backend.app.services.cloudflare_workers_ai_service import CloudflareWorkersAIProvider
            self._cloudflare_service = CloudflareWorkersAIProvider()
        return self._cloudflare_service

    @classmethod
    def get_instance(cls) -> "ModelRouter":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def general_metadata(self) -> Dict[str, Any]:
        """Provides model metadata without instantiating or loading DistilGPT-2 weights."""
        if self.is_cloudflare_provider:
            return self.cloudflare_service.metadata

        meta = dict(DISTILGPT2_METADATA_CONST)
        meta["is_remote"] = bool(self.general_api_url)
        meta["remote_url"] = self.general_api_url if self.general_api_url else None
        meta["status"] = "remote_configured" if self.general_api_url else ("local_opt_in" if self.allow_local_general else "unconfigured")
        return meta

    @property
    def general_service(self):
        """Loads local General model ONLY if ALLOW_LOCAL_GENERAL=true has been explicitly set."""
        if not self.allow_local_general and not self.general_api_url and not self.is_cloudflare_provider:
            raise HTTPException(
                status_code=503,
                detail=(
                    "General English model is not configured in this service. "
                    "In production, use Cloudflare Workers AI (GENERAL_PROVIDER=cloudflare_workers_ai) "
                    "or set GENERAL_API_URL to point to a dedicated General inference service, "
                    "or set ALLOW_LOCAL_GENERAL=true to opt into local loading."
                )
            )
        if self._local_general_service is None and self.allow_local_general:
            from backend.app.services.general_service import GeneralLanguageModelService
            self._local_general_service = GeneralLanguageModelService()
        return self._local_general_service

    def initialize(self):
        """Pre-load technical model by default. Never pre-load DistilGPT-2 in the main service."""
        print("Initializing ModelRouter...")
        self.technical_manager.load()
        if self.is_cloudflare_provider:
            status_desc = "configured" if self.cloudflare_service.is_configured else "missing credentials"
            print(f"General writing mode routed to Cloudflare Workers AI ({self.cloudflare_service.model_id}, {status_desc}).")
        elif self.general_api_url:
            print(f"General writing mode routed to isolated microservice: {self.general_api_url}")
        elif self.allow_local_general:
            print("General writing mode configured for local lazy loading (ALLOW_LOCAL_GENERAL=true, requires >1GB RAM).")
        else:
            print("General writing mode is fail-closed (no external provider set). Only Technical model loaded.")
        print("Technical model ready.")

    def check_remote_general_health(self) -> bool:
        if self.is_cloudflare_provider:
            return self.cloudflare_service.check_health()
        if not self.general_api_url:
            return False
        try:
            req = urllib.request.Request(f"{self.general_api_url}/health", headers={"User-Agent": "NeuroText-Router/2.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                return bool(data.get("status") == "healthy" or data.get("model_loaded"))
        except Exception:
            return False

    def get_health_status(self) -> HealthResponse:
        general_is_loaded = False
        general_name = "General English Writing (DistilGPT-2)"
        general_arch = "DistilGPT-2 (6 layers, 768 hidden, 12 heads)"
        general_params = 81912576
        general_desc = "82M parameter transformer (unconfigured / offline)"

        general_provider_status = "unconfigured"

        if self.is_cloudflare_provider:
            cf = self.cloudflare_service
            general_name = f"General English ({cf.metadata['name']})"
            general_arch = cf.metadata["architecture"]
            general_params = cf.metadata["parameters"]
            state = cf.provider_state
            general_provider_status = state
            if state == "online":
                general_is_loaded = True
                general_desc = f"Managed Cloudflare Workers AI ({cf.model_id}) - Online"
            elif state == "configured":
                general_is_loaded = False
                general_desc = f"Cloudflare Workers AI ({cf.model_id}) - Configured (pending first inference verification)"
            elif state == "unavailable":
                general_is_loaded = False
                general_desc = f"Cloudflare Workers AI ({cf.model_id}) - Unavailable: {cf.last_error}"
            else:
                general_is_loaded = False
                general_desc = "Cloudflare Workers AI unconfigured (missing server credentials)"
        elif self.general_api_url:
            general_is_loaded = self.check_remote_general_health()
            general_provider_status = "online" if general_is_loaded else "unavailable"
            general_desc = f"82M parameter causal autoregressive transformer hosted at {self.general_api_url}"
        elif self.allow_local_general and self._local_general_service is not None:
            general_is_loaded = self._local_general_service.is_loaded
            general_provider_status = "online" if general_is_loaded else "configured"
            general_desc = "82M parameter transformer (local opt-in)"

        modes = {
            "technical": ModeStatus(
                name="Technical Writing (PyTorch BiLSTM)",
                loaded=self.technical_manager.is_loaded,
                description="Bidirectional LSTM trained on technical publications & headline sequences",
                architecture="Embedding (100d) -> BiLSTM (150 units) -> Linear Head",
                parameters=3229299,
                provider_status="online" if self.technical_manager.is_loaded else "unavailable"
            ),
            "general": ModeStatus(
                name=general_name,
                loaded=general_is_loaded,
                description=general_desc,
                architecture=general_arch,
                parameters=general_params,
                provider_status=general_provider_status
            )
        }

        return HealthResponse(
            status="healthy" if self.technical_manager.is_loaded else "degraded",
            version="2.0.0",
            device=str(self.technical_manager.device),
            technical_model_loaded=self.technical_manager.is_loaded,
            general_model_loaded=general_is_loaded,
            modes=modes
        )

    def predict(self, text: str, mode: WritingMode = "technical", top_k: int = 5, temperature: float = 1.0) -> PredictResponse:
        t0 = time.perf_counter()
        normalized_mode = (mode or "technical").lower()

        if normalized_mode == "general":
            if self.is_cloudflare_provider:
                # Cloudflare Workers AI managed inference
                candidates, tokens, model_lat = self.cloudflare_service.predict_next_words(text, top_k, temperature)
                latency_ms = (time.perf_counter() - t0) * 1000
                result = PredictResponse(
                    input_text=text,
                    mode=normalized_mode,
                    tokens=tokens,
                    candidates=candidates,
                    latency_ms=round(latency_ms, 2),
                    model_name=f"{self.cloudflare_service.metadata['name']} (Cloudflare)",
                    model_version="2.0.0"
                )
            elif self.general_api_url:
                # Forward to dedicated isolated General service
                try:
                    payload = json.dumps({"text": text, "mode": "general", "top_k": top_k, "temperature": temperature}).encode("utf-8")
                    req = urllib.request.Request(
                        f"{self.general_api_url}/predict",
                        data=payload,
                        headers={"Content-Type": "application/json", "User-Agent": "NeuroText-Router/2.0"}
                    )
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        res_data = json.loads(resp.read().decode())
                        result = PredictResponse(**res_data)
                except Exception as e:
                    raise HTTPException(status_code=503, detail=f"Dedicated General Inference Service unavailable at {self.general_api_url}: {str(e)}")
            elif self.allow_local_general:
                # Local execution only when explicitly allowed
                if not self.general_service.is_loaded:
                    self.general_service.load()
                candidates, tokens, model_lat = self.general_service.predict_next_words(text, top_k, temperature)
                latency_ms = (time.perf_counter() - t0) * 1000
                result = PredictResponse(
                    input_text=text,
                    mode=normalized_mode,
                    tokens=tokens,
                    candidates=candidates,
                    latency_ms=round(latency_ms, 2),
                    model_name="DistilGPT-2 (General English)",
                    model_version="2.0.0"
                )
            else:
                # Fail closed: never load DistilGPT-2 without config
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "General English writing mode is currently unavailable. "
                        "Neither Cloudflare Workers AI nor a dedicated General microservice is configured. "
                        "Technical Writing mode (PyTorch BiLSTM) is active and available."
                    )
                )
        else:
            # Technical BiLSTM
            pred_res = self.technical_service.predict_next(text, top_k, temperature)
            latency_ms = (time.perf_counter() - t0) * 1000
            result = PredictResponse(
                input_text=text,
                mode=normalized_mode,
                tokens=pred_res.tokens,
                candidates=pred_res.candidates,
                latency_ms=round(latency_ms, 2),
                model_name="PyTorch BiLSTM (Technical Writing)",
                model_version="2.0.0"
            )

        # Update analytics only upon success
        total_lat = (time.perf_counter() - t0) * 1000
        self.stats["total_predictions"] += 1
        self.stats["predictions_by_mode"][normalized_mode] = self.stats["predictions_by_mode"].get(normalized_mode, 0) + 1
        self.stats["latency_history"].append(total_lat)

        return result

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
            if self.is_cloudflare_provider:
                gen_text, completed_text, tokens_gen, model_lat = self.cloudflare_service.generate_continuation(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    strategy=strategy,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                result = GenerateResponse(
                    prompt=prompt,
                    mode=normalized_mode,
                    generated_text=gen_text,
                    completed_text=completed_text,
                    tokens_generated=tokens_gen,
                    strategy_used=strategy,
                    latency_ms=round(latency_ms, 2),
                    model_name=f"{self.cloudflare_service.metadata['name']} (Cloudflare)"
                )
            elif self.general_api_url:
                try:
                    payload = json.dumps({
                        "prompt": prompt,
                        "mode": "general",
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "strategy": strategy,
                        "top_k": top_k,
                        "repetition_penalty": repetition_penalty
                    }).encode("utf-8")
                    req = urllib.request.Request(
                        f"{self.general_api_url}/generate",
                        data=payload,
                        headers={"Content-Type": "application/json", "User-Agent": "NeuroText-Router/2.0"}
                    )
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        res_data = json.loads(resp.read().decode())
                        result = GenerateResponse(**res_data)
                except Exception as e:
                    raise HTTPException(status_code=503, detail=f"Dedicated General Inference Service unavailable at {self.general_api_url}: {str(e)}")
            elif self.allow_local_general:
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
                latency_ms = (time.perf_counter() - t0) * 1000
                result = GenerateResponse(
                    prompt=prompt,
                    mode=normalized_mode,
                    generated_text=gen_text,
                    completed_text=completed_text,
                    tokens_generated=tokens_gen,
                    strategy_used=strategy,
                    latency_ms=round(latency_ms, 2),
                    model_name="DistilGPT-2 (General English)"
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "General English text generation is currently unavailable. "
                        "Neither Cloudflare Workers AI nor a dedicated General microservice is configured. "
                        "Technical Writing mode (PyTorch BiLSTM) is active and available."
                    )
                )
        else:
            gen_res = self.technical_service.generate_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                strategy=strategy,
                top_k=top_k
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            result = GenerateResponse(
                prompt=prompt,
                mode=normalized_mode,
                generated_text=gen_res.generated_text,
                completed_text=gen_res.completed_text,
                tokens_generated=gen_res.tokens_generated,
                strategy_used=strategy,
                latency_ms=round(latency_ms, 2),
                model_name="PyTorch BiLSTM (Technical Writing)"
            )

        total_lat = (time.perf_counter() - t0) * 1000
        self.stats["total_generations"] += 1
        self.stats["generations_by_mode"][normalized_mode] = self.stats["generations_by_mode"].get(normalized_mode, 0) + 1
        self.stats["latency_history"].append(total_lat)

        return result

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

