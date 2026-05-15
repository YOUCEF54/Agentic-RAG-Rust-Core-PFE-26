"""Internal domain-level typing helpers for agentic orchestration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class RetrievedChunk(TypedDict):
    text: str
    source: str
    page: int
    distance: float


class AgentState(TypedDict, total=False):
    query: str
    refined_query: str
    chunks: List[tuple]
    chunks_candidates: List[tuple]
    score: float
    answer: str
    attempts: int
    trace: List[Dict[str, Any]]
    models: Dict[str, Optional[str]]
    crag_status: str
    crag_confidence: float
    crag_reason: str
    external_chunks: List[tuple]
    meta: Dict[str, Any]

