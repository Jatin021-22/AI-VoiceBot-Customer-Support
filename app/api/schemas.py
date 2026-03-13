"""Pydantic schemas for API request/response validation."""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class TranscribeResponse(BaseModel):
    text: str
    language: str
    confidence: float
    duration_ms: float
    model: str
    segments: Optional[List[Dict]] = []


class IntentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, example="Where is my order?")


class IntentResponse(BaseModel):
    intent: str
    confidence: float
    all_scores: Dict[str, float]
    duration_ms: float
    backend: str


class ResponseRequest(BaseModel):
    intent: str = Field(..., example="order_status")
    context: Optional[Dict[str, Any]] = None


class ResponseResult(BaseModel):
    text: str
    intent: str
    strategy: str


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    language: Optional[str] = "en"
    speaking_rate: Optional[float] = Field(1.0, ge=0.5, le=2.0)


class PipelineResult(BaseModel):
    success: bool
    transcript: Optional[str]
    intent: Optional[str]
    confidence: Optional[float]
    response_text: Optional[str]
    total_duration_ms: Optional[float]
    error: Optional[str]
    stages: Optional[Dict[str, Any]]


class EvaluationResult(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    labels: List[str]
    classification_report: Optional[Dict]


class WERResult(BaseModel):
    overall_wer: float
    total_samples: int
    samples: Optional[List[Dict]]
