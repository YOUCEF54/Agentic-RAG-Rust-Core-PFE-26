"""Pydantic API contracts for the modular app package."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel

from app.core import config


class IndexRequest(BaseModel):
    rebuild: bool = True
    max_pages: Optional[int] = None
    run_hardware_test: bool = False
    save_hardware_config: bool = True
    hardware_quick_test: bool = True
    hardware_max_runtime_seconds: float = 25.0


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    chat_model: Optional[str] = None
    use_llm: bool = True
    mode: str = "agentic"  # "agentic" or "naive"
    return_trace: bool = True
    min_score: float = 0.7
    max_attempts: int = 3
    crag_correct_threshold: float = config.CRAG_CORRECT_THRESHOLD
    crag_ambiguous_threshold: float = config.CRAG_AMBIGUOUS_THRESHOLD
    crag_external_top_k: int = config.CRAG_EXTERNAL_TOP_K
    crag_enable_external_route: bool = config.CRAG_ENABLE_EXTERNAL_ROUTE
    dartboard_sigma: Optional[float] = None


class QueryResponse(BaseModel):
    answer: Optional[str]
    model_used: Optional[str]
    retrieved: List[dict]
    mode: Optional[str] = None
    refined_query: Optional[str] = None
    score: Optional[float] = None
    attempts: Optional[int] = None
    trace: Optional[List[dict]] = None
    models: Optional[Dict[str, Optional[str]]] = None
    crag_status: Optional[str] = None
    crag_confidence: Optional[float] = None
    crag_reason: Optional[str] = None
    external_retrieved: Optional[List[dict]] = None


class DocumentMeta(BaseModel):
    filename: str
    size_bytes: int
    uploaded_at: Optional[str] = None
    updated_at: Optional[str] = None
    sha256: Optional[str] = None
    indexed_sha256: Optional[str] = None
    pages: Optional[int] = None


class HardwareCalibrationRequest(BaseModel):
    save_config: bool = True
    quick_mode: bool = True
    max_runtime_seconds: float = 25.0

