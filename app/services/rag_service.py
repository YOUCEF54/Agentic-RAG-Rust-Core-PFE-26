"""High-level non-stream Agentic RAG service."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import HTTPException

from app.agents.orchestrator import AgentOrchestrator
from app.core import runtime_state
from app.services.chat_service import backend_chat
from app.schemas.api import QueryRequest, QueryResponse
from app.services.query_mode_service import naive_result, retrieval_only_result
from app.services.retrieval_service import retrieve_chunks_with_meta

def run_query(payload: QueryRequest) -> Dict[str, Any]:
    if not runtime_state.INDEX_READY:
        raise HTTPException(status_code=400, detail="Index not built. Call /index first.")

    if not payload.use_llm:
        return retrieval_only_result(payload.question, payload.top_k, payload.dartboard_sigma)

    if payload.mode.lower() == "naive":
        result, _, _, _ = naive_result(
            question=payload.question,
            top_k=payload.top_k,
            chat_model=payload.chat_model,
            dartboard_sigma=payload.dartboard_sigma,
        )
        return result

    orchestrator = AgentOrchestrator(
        payload=payload,
        retrieve_with_meta=lambda query, top_k, sigma: retrieve_chunks_with_meta(
            query,
            top_k,
            dartboard_sigma=sigma,
        ),
        backend_chat=lambda messages, model_override: backend_chat(messages, model_override=model_override),
    )
    state = orchestrator.init_state()
    state = orchestrator.run(state)

    return QueryResponse(
        answer=state.get("answer"),
        model_used=state.get("model_used"),
        retrieved=state.get("retrieved_meta") or [],
        mode="agentic",
        refined_query=state.get("refined_query"),
        score=state.get("score"),
        attempts=state.get("attempts"),
        trace=state.get("trace") if payload.return_trace else None,
        models=state.get("models"),
        crag_status=state.get("crag_status"),
        crag_confidence=state.get("crag_confidence"),
        crag_reason=state.get("crag_reason"),
        external_retrieved=state.get("external_retrieved_meta") or [],
    ).model_dump()
