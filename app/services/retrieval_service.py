"""Retrieval service (internal vector search + optional Dartboard rerank)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from app.core import config
from app.core import runtime_state
from app.infrastructure import embeddings, rust_bridge


def retrieve_chunks_with_meta(
    query: str,
    top_k: int,
    *,
    dartboard_sigma: Optional[float],
) -> Tuple[List[tuple], List[Dict[str, Any]]]:
    query_vector = embeddings.embed_query(query, batch_size=runtime_state.ACTIVE_EMBED_BATCH_SIZE)
    effective_sigma = dartboard_sigma if dartboard_sigma is not None else config.DARTBOARD_SIGMA
    use_dartboard = config.DARTBOARD_ENABLED and effective_sigma > 0.0

    if use_dartboard:
        fetch_n = min(top_k * 3, 50)
        try:
            raw_hits = rust_bridge.lancedb_search(config.DB_DIR, config.TABLE_NAME, query_vector, fetch_n)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LanceDB search failed: {exc}") from exc

        if len(raw_hits) == 0:
            return [], []

        cand_texts = [text for text, _, _, _ in raw_hits]
        cand_vectors = embeddings.embed_texts(cand_texts, batch_size=runtime_state.ACTIVE_EMBED_BATCH_SIZE)
        selected_indices = rust_bridge.dartboard_rerank(query_vector, cand_vectors, top_k, effective_sigma)
        hits = [raw_hits[i] for i in selected_indices]
    else:
        try:
            hits = rust_bridge.lancedb_search(config.DB_DIR, config.TABLE_NAME, query_vector, top_k)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LanceDB search failed: {exc}") from exc

    chunks: List[tuple] = []
    meta: List[Dict[str, Any]] = []
    for text, source, page, dist in hits:
        chunks.append((text, source, page, dist))
        meta.append({"text": text, "distance": dist, "source": source, "page": page})
    return chunks, meta
