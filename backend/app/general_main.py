import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.schemas import (
    PredictRequest, PredictResponse,
    GenerateRequest, GenerateResponse,
    PredictionCandidate, GenerationToken,
    HealthResponse, ModeStatus
)
from backend.app.services.general_service import GeneralLanguageModelService

# Singleton service for isolated General inference
general_svc = GeneralLanguageModelService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing NeuroText AI General Inference Service (DistilGPT-2)...")
    try:
        general_svc.load()
        print("General Inference Service online.")
    except Exception as e:
        print(f"Warning: General model failed to pre-load during startup: {e}")
    yield
    print("Shutting down General Inference Service...")
    general_svc.unload()

app = FastAPI(
    title="NeuroText AI General Inference API",
    description="Dedicated microservice for General English writing prediction & text continuation powered by DistilGPT-2 (82M params).",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "https://umarmukthar03.github.io",
        "https://neurotext-ai-api.onrender.com",
        "https://neurotext-ai-general.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/", tags=["General"])
async def root():
    return {
        "service": "NeuroText AI General Inference Service",
        "model": "DistilGPT-2 (General English)",
        "revision": general_svc.revision,
        "version": "2.0.0",
        "status": "online" if general_svc.is_loaded else "ready_to_load",
        "docs_url": "/docs"
    }

@app.get("/health", tags=["Monitoring"])
async def health():
    return {
        "status": "healthy" if general_svc.is_loaded else "degraded",
        "service": "general-inference",
        "version": "2.0.0",
        "model_loaded": general_svc.is_loaded,
        "device": str(general_svc.device),
        "metadata": general_svc.metadata
    }

@app.get("/model-info", tags=["Monitoring"])
async def model_info():
    return {
        "name": "DistilGPT-2 (General English)",
        "model_id": general_svc.model_id,
        "revision": general_svc.revision,
        "parameters": general_svc.metadata["parameters"],
        "architecture": general_svc.metadata["architecture"],
        "context_length": general_svc.metadata["context_length"],
        "is_loaded": general_svc.is_loaded
    }

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        candidates, tokens, lat = general_svc.predict_next_words(
            text=request.text,
            top_k=request.top_k,
            temperature=request.temperature
        )
        return PredictResponse(
            input_text=request.text,
            mode="general",
            tokens=tokens,
            candidates=candidates,
            latency_ms=lat,
            model_name="DistilGPT-2 (General English)",
            model_version="2.0.0"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"General prediction error: {str(e)}")

@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
async def generate(request: GenerateRequest):
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    try:
        gen_text, completed_text, tokens_gen, lat = general_svc.generate_continuation(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            strategy=request.strategy,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty
        )
        return GenerateResponse(
            prompt=request.prompt,
            mode="general",
            generated_text=gen_text,
            completed_text=completed_text,
            tokens_generated=tokens_gen,
            strategy_used=request.strategy,
            latency_ms=lat,
            model_name="DistilGPT-2 (General English)"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"General generation error: {str(e)}")

