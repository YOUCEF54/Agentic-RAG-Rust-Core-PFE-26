"""Shared helpers for retrieval-only and naive query modes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.core import config
from app.schemas.api import QueryResponse
from app.services.chat_service import backend_chat
from app.services.retrieval_service import retrieve_chunks_with_meta


def retrieval_only_result(question: str, top_k: int, dartboard_sigma: Optional[float]) -> Dict[str, Any]:
    _, meta = retrieve_chunks_with_meta(question, top_k, dartboard_sigma=dartboard_sigma)
    return QueryResponse(answer=None, model_used=None, retrieved=meta, mode="retrieval_only").model_dump()


def naive_result(
    *,
    question: str,
    top_k: int,
    chat_model: Optional[str],
    dartboard_sigma: Optional[float],
) -> Tuple[Dict[str, Any], str, Optional[str], List[Dict[str, Any]]]:
    _, meta = retrieve_chunks_with_meta(question, top_k, dartboard_sigma=dartboard_sigma)
    context_text = "\n".join([f"- {row['text']}" for row in meta])
    system_prompt = (
        "You are a helpful chatbot.\n"
        "Use only the following pieces of context to answer the question. "
        "Don't make up any new information:\n"
        f"{context_text}"
    )
    naive_model = chat_model or (
        config.OPENROUTER_CHAT_MODEL if config.API_TYPE == "open_router" else config.OLLAMA_CHAT_MODEL
    )
    answer, model_used = backend_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        model_override=naive_model,
    )
    payload = QueryResponse(
        answer=answer,
        model_used=model_used,
        retrieved=meta,
        mode="naive",
        models={"generator": model_used},
    ).model_dump()
    return payload, answer, model_used, meta

