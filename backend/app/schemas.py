from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class PredictionCandidate(BaseModel):
    word: str
    probability: float
    rank: int

class PredictRequest(BaseModel):
    text: str = Field(..., description="Prompt or prefix text")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top candidate words to return")
    temperature: float = Field(default=1.0, gt=0.0, le=2.0, description="Softmax temperature scaling")

class PredictResponse(BaseModel):
    input_text: str
    tokens: List[str]
    candidates: List[PredictionCandidate]
    latency_ms: float
    model_version: str

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Starting seed text")
    max_tokens: int = Field(default=5, ge=1, le=30, description="Number of tokens to generate")
    temperature: float = Field(default=0.8, gt=0.0, le=2.0, description="Sampling temperature")
    strategy: str = Field(default="greedy", description="Generation strategy: greedy, sampling, or top_k")
    top_k: int = Field(default=5, ge=1, le=50, description="K value for top_k sampling")

class GenerationToken(BaseModel):
    token: str
    probability: float

class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    completed_text: str
    tokens_generated: List[GenerationToken]
    strategy_used: str
    latency_ms: float

class ModelInfoResponse(BaseModel):
    name: str
    version: str
    architecture: str
    vocab_size: int
    embedding_dim: int
    hidden_dim: int
    max_sequence_len: int
    total_parameters: int
    device: str
    metrics: Dict[str, Any]
    dataset: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    device: str

class AnalyticsResponse(BaseModel):
    total_predictions: int
    total_generations: int
    avg_latency_ms: float
    uptime_seconds: float

