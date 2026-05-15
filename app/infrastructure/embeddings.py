"""Embedding adapter and normalization helpers."""

from __future__ import annotations

from typing import List

from app.core import config
from app.infrastructure import rust_bridge

_EMBED_MODEL_READY = False


def ensure_model_loaded() -> None:
    global _EMBED_MODEL_READY
    if _EMBED_MODEL_READY:
        return
    rust_bridge.load_embed_model(use_zembed=config.EMBED_MODE)
    _EMBED_MODEL_READY = True


def embed_texts(texts: List[str], batch_size: int) -> List[List[float]]:
    ensure_model_loaded()
    if config.EMBED_MODE:
        return rust_bridge.embed_texts_zembed(texts, batch_size)
    return rust_bridge.embed_texts_local(texts, batch_size)


def embed_query(query: str, batch_size: int) -> List[float]:
    ensure_model_loaded()
    if config.EMBED_MODE:
        return rust_bridge.embed_texts_zembed([query], batch_size)[0]
    prefixed = f"{config.BGE_QUERY_PREFIX}{query}"
    return rust_bridge.embed_texts_local([prefixed], batch_size)[0]
