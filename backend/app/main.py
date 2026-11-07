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
from backend.app.services.model_router import ModelRouter

router = ModelRouter.get_instance()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing NeuroText AI V2 dual-mode backend...")
    router.initialize()
    print("NeuroText AI V2 backend online.")
    yield
    print("Shutting down NeuroText AI backend...")

app = FastAPI(
    title="NeuroText AI V2 API",
    description="Intelligent Predictive Writing & Language Intelligence Platform API with Dual Model Modes (General English & Technical Writing)",
    version="2.0.0",
    lifespan=lifespan
)

# Standard strict CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "https://umarmukthar03.github.io",
        "https://neurotext-ai-api.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/", tags=["General"])
async def root():
    return {
        "name": "NeuroText AI V2 API",
        "description": "Dual-Mode Intelligent Predictive Writing Platform (General English & Technical Writing)",
        "version": "2.0.0",
        "status": "online",
        "modes_available": ["general", "technical"],
        "docs_url": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    return router.get_health_status()

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_next_token(request: PredictRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    try:
        return router.predict(
            text=request.text,
            mode=request.mode or "technical",
            top_k=request.top_k,
            temperature=request.temperature
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
async def generate_text(request: GenerateRequest):
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    try:
        return router.generate(
            prompt=request.prompt,
            mode=request.mode or "technical",
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            strategy=request.strategy,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    tech_config = router.technical_manager.model.get_config() if router.technical_manager.is_loaded else {}
    return ModelInfoResponse(
        name="NeuroText AI Dual-Mode Language Engine",
        version="2.0.0",
        device=str(router.technical_manager.device),
        modes={
            "general": router.general_service.metadata,
            "technical": {
                "name": "Technical Writing (PyTorch BiLSTM)",
                "architecture": tech_config.get("architecture", "Bidirectional LSTM"),
                "vocab_size": router.technical_manager.tokenizer.total_words if router.technical_manager.tokenizer else 7299,
                "parameters": tech_config.get("total_parameters", 3229299),
                "metrics": router.technical_manager.metrics,
                "dataset": router.technical_manager.model_metadata.get("dataset", {
                    "name": "Medium Articles Dataset",
                    "domain": "Technical & data science publication titles",
                    "split": "80/10/10 title-level leak-free split"
                })
            }
        }
    )

@app.get("/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
async def get_analytics():
    data = router.get_analytics()
    return AnalyticsResponse(
        total_predictions=data["total_predictions"],
        total_generations=data["total_generations"],
        predictions_by_mode=data["predictions_by_mode"],
        generations_by_mode=data["generations_by_mode"],
        avg_latency_ms=data["avg_latency_ms"],
        uptime_seconds=data["uptime_seconds"]
    )
