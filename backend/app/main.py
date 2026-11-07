import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from backend.app.schemas import (
    PredictRequest, PredictResponse,
    GenerateRequest, GenerateResponse,
    ModelInfoResponse, HealthResponse,
    AnalyticsResponse
)
from backend.app.services.model_loader import ModelManager
from backend.app.services.prediction_service import TextGenerationService

model_manager = ModelManager.get_instance()
generation_service: TextGenerationService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global generation_service
    print("Initializing NeuroText AI backend...")
    model_manager.load()
    generation_service = TextGenerationService()
    print("NeuroText AI backend initialized successfully.")
    yield
    print("Shutting down NeuroText AI backend...")

app = FastAPI(
    title="NeuroText AI API",
    description="Intelligent Predictive Writing & Language Intelligence Platform API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["General"])
async def root():
    return {
        "name": "NeuroText AI API",
        "description": "Intelligent Predictive Writing & Language Intelligence Platform",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        model_loaded=model_manager.is_loaded,
        version="1.0.0",
        device=str(model_manager.device)
    )

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_next_token(request: PredictRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    try:
        return generation_service.predict_next(
            text=request.text,
            top_k=request.top_k,
            temperature=request.temperature
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
async def generate_text(request: GenerateRequest):
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    try:
        return generation_service.generate_completion(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            strategy=request.strategy,
            top_k=request.top_k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    config = model_manager.model.get_config()
    return ModelInfoResponse(
        name="NeuroText AI BiLSTM",
        version="1.0.0",
        architecture=config["architecture"],
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        max_sequence_len=model_manager.max_sequence_len,
        total_parameters=config["total_parameters"],
        device=str(model_manager.device),
        metrics=model_manager.metrics,
        dataset=model_manager.model_metadata.get("dataset", {
            "name": "Medium Articles Dataset",
            "domain": "Data science and tech titles",
            "split": "80/10/10 leak-free split"
        })
    )

@app.get("/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
async def get_analytics():
    if not generation_service:
        raise HTTPException(status_code=503, detail="Service not ready.")
    data = generation_service.get_analytics()
    return AnalyticsResponse(
        total_predictions=data["total_predictions"],
        total_generations=data["total_generations"],
        avg_latency_ms=data["avg_latency_ms"],
        uptime_seconds=data["uptime_seconds"]
    )

