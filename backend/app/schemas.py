from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

WritingMode = Literal["general", "technical"]

class PredictionCandidate(BaseModel):
    word: str
    probability: float
    rank: int

class PredictRequest(BaseModel):
    text: str = Field(..., description="Prompt or prefix text")
    mode: Optional[WritingMode] = Field(default="technical", description="Writing mode: 'general' or 'technical'")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of candidate words to return")
    temperature: float = Field(default=1.0, gt=0.0, le=2.0, description="Softmax temperature scaling")

class PredictResponse(BaseModel):
    input_text: str
    mode: Optional[str] = "technical"
    tokens: List[str]
    candidates: List[PredictionCandidate]
    latency_ms: float
    model_name: Optional[str] = "PyTorch BiLSTM"
    model_version: str = "2.0.0"

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Starting seed text")
    mode: Optional[WritingMode] = Field(default="technical", description="Writing mode: 'general' or 'technical'")
    max_tokens: int = Field(default=10, ge=1, le=50, description="Number of tokens to generate")
    temperature: float = Field(default=0.8, gt=0.0, le=2.0, description="Sampling temperature")
    strategy: str = Field(default="greedy", description="Generation strategy: greedy, sampling, or top_k")
    top_k: int = Field(default=5, ge=1, le=50, description="K value for top_k sampling")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0, description="Penalty for repeating tokens")

class GenerationToken(BaseModel):
    token: str
    probability: float

class GenerateResponse(BaseModel):
    prompt: str
    mode: Optional[str] = "technical"
    generated_text: str
    completed_text: str
    tokens_generated: List[GenerationToken]
    strategy_used: str
    latency_ms: float
    model_name: Optional[str] = "PyTorch BiLSTM"

class ModeStatus(BaseModel):
    name: str
    loaded: bool
    description: str
    architecture: str
    parameters: int

class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    technical_model_loaded: bool
    general_model_loaded: bool
    modes: Dict[str, ModeStatus]

class ModelInfoResponse(BaseModel):
    name: str
    version: str
    modes: Dict[str, Any]
    device: str

class AnalyticsResponse(BaseModel):
    total_predictions: int
    total_generations: int
    predictions_by_mode: Dict[str, int]
    generations_by_mode: Dict[str, int]
    avg_latency_ms: float
    uptime_seconds: float
