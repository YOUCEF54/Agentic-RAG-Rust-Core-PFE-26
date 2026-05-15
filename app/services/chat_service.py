"""Shared chat backend dispatch for OpenRouter/Ollama."""

from __future__ import annotations

from typing import List, Optional, Tuple

from fastapi import HTTPException

from app.core import config
from app.infrastructure.llm_client import ollama_chat, openrouter_chat


def backend_chat(messages: List[dict], model_override: Optional[str] = None) -> Tuple[str, Optional[str]]:
    if config.API_TYPE == "open_router":
        try:
            return openrouter_chat(
                messages=messages,
                model_override=model_override,
                api_key=config.OPENROUTER_API_KEY,
                base_url=config.OPENROUTER_BASE_URL,
                timeout=config.OPENROUTER_TIMEOUT,
                temperature=config.CHAT_TEMPERATURE,
                default_model=config.OPENROUTER_CHAT_MODEL,
                http_referer=config.OPENROUTER_HTTP_REFERER,
                title=config.OPENROUTER_TITLE,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"OpenRouter unavailable: {exc}") from exc

    try:
        return ollama_chat(
            model=model_override or config.OLLAMA_CHAT_MODEL,
            messages=messages,
            base_url=config.OLLAMA_BASE_URL,
            timeout=config.OLLAMA_TIMEOUT,
            temperature=config.CHAT_TEMPERATURE,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama unavailable: {exc}") from exc

