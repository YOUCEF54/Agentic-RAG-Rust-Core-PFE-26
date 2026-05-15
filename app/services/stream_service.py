"""Streaming query orchestration service."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.agents.orchestrator import AgentOrchestrator
from app.core import config, runtime_state
from app.services.chat_service import backend_chat
from app.schemas.api import QueryRequest, QueryResponse
from app.services.retrieval_service import retrieve_chunks_with_meta


def _sse_event(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def run_query_stream(payload: QueryRequest):
    if not runtime_state.INDEX_READY:
        raise HTTPException(status_code=400, detail="Index not built. Call /index first.")

    def event_stream() -> Iterable[str]:
        yield _sse_event("status", {"state": "started"})
        try:
            if not payload.use_llm:
                _, meta = retrieve_chunks_with_meta(
                    payload.question,
                    payload.top_k,
                    dartboard_sigma=payload.dartboard_sigma,
                )
                yield _sse_event("retrieved", {"items": meta})
                yield _sse_event("final", {"mode": "retrieval_only", "retrieved": meta})
                return

            if payload.mode.lower() == "naive":
                _, meta = retrieve_chunks_with_meta(
                    payload.question,
                    payload.top_k,
                    dartboard_sigma=payload.dartboard_sigma,
                )
                yield _sse_event("retrieved", {"items": meta})
                context_text = "\n".join([f"- {row['text']}" for row in meta])
                system_prompt = (
                    "You are a helpful chatbot.\n"
                    "Use only the following pieces of context to answer the question. "
                    "Don't make up any new information:\n"
                    f"{context_text}"
                )
                naive_model = payload.chat_model or (
                    config.OPENROUTER_CHAT_MODEL if config.API_TYPE == "open_router" else config.OLLAMA_CHAT_MODEL
                )
                answer, model_used = backend_chat(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": payload.question},
                    ],
                    model_override=naive_model,
                )
                yield _sse_event("answer", {"answer": answer, "model_used": model_used})
                yield _sse_event(
                    "final",
                    {
                        "mode": "naive",
                        "answer": answer,
                        "model_used": model_used,
                        "retrieved": meta,
                        "models": {"generator": model_used},
                    },
                )
                return

            pending: List[Dict[str, Any]] = []
            orchestrator = AgentOrchestrator(
                payload=payload,
                retrieve_with_meta=lambda query, top_k, sigma: retrieve_chunks_with_meta(
                    query,
                    top_k,
                    dartboard_sigma=sigma,
                ),
                backend_chat=lambda messages, model_override: backend_chat(messages, model_override=model_override),
            )
            state = orchestrator.init_state(emit=lambda item: pending.append(item))

            state = orchestrator.run_retrieval(state)
            for item in pending:
                yield _sse_event("trace", item)
            pending.clear()
            yield _sse_event("retrieved", {"items": state.get("retrieved_meta") or []})

            state = orchestrator.run_selection(state)
            for item in pending:
                yield _sse_event("trace", item)
            pending.clear()

            state = orchestrator.run_evaluation(state)
            for item in pending:
                yield _sse_event("trace", item)
            pending.clear()
            yield _sse_event(
                "evaluation",
                {
                    "score": state.get("score"),
                    "summary": state.get("judge_summary"),
                    "crag_status": state.get("crag_status"),
                    "crag_confidence": state.get("crag_confidence"),
                    "attempts": state.get("attempts"),
                },
            )

            state = orchestrator.run_external_route(state)
            for item in pending:
                yield _sse_event("trace", item)
            pending.clear()
            if state.get("external_retrieved_meta"):
                yield _sse_event("external_retrieved", {"items": state.get("external_retrieved_meta") or []})

            state = orchestrator.run_generation(state)
            for item in pending:
                yield _sse_event("trace", item)
            pending.clear()
            yield _sse_event("answer", {"answer": state.get("answer"), "model_used": state.get("model_used")})

            final_payload = QueryResponse(
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
            )
            yield _sse_event("final", final_payload.model_dump())
        except HTTPException as exc:
            yield _sse_event(
                "error",
                {
                    "status_code": exc.status_code,
                    "detail": exc.detail,
                    "api_type": config.API_TYPE,
                },
            )
            return
        except Exception as exc:
            yield _sse_event(
                "error",
                {
                    "status_code": 500,
                    "detail": str(exc),
                    "api_type": config.API_TYPE,
                },
            )
            return

    return StreamingResponse(event_stream(), media_type="text/event-stream")
