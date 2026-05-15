"""Thin wrapper around the `rag_rust` extension.

Keeping Rust calls here helps isolate low-level integration from route logic.
"""

from __future__ import annotations

from typing import List

import rag_rust


def load_embed_model(use_zembed: bool) -> None:
    if use_zembed:
        rag_rust.load_embed_model_zembed()
    else:
        rag_rust.load_embed_model_local()


def embed_texts_local(texts: List[str], batch_size: int) -> List[List[float]]:
    return rag_rust.embed_texts_rust_local(texts, batch_size)


def embed_texts_zembed(texts: List[str], batch_size: int) -> List[List[float]]:
    return rag_rust.embed_texts_rust_zembed(texts, batch_size)


def load_pdf_pages_pdfium_many(paths: List[str]) -> List[list]:
    return rag_rust.load_pdf_pages_pdfium_many(paths)


def semantic_window_chunker_advanced(
    text: str,
    max_chars: int,
    window_size: int,
) -> List[str]:
    return rag_rust.semantic_window_chunker_advanced(text=text, max_chars=max_chars, window_size=window_size)


def lancedb_create_or_open(
    db_dir: str,
    table_name: str,
    texts: List[str],
    sources: List[str],
    pages: List[int],
    embeddings: List[List[float]],
    rebuild: bool,
) -> None:
    rag_rust.lancedb_create_or_open(db_dir, table_name, texts, sources, pages, embeddings, rebuild)


def lancedb_search(db_dir: str, table_name: str, query_vector: List[float], top_k: int) -> list:
    return rag_rust.lancedb_search(db_dir, table_name, query_vector, top_k)


def dartboard_rerank(
    query_vector: List[float],
    candidate_vectors: List[List[float]],
    top_k: int,
    sigma: float,
) -> List[int]:
    return rag_rust.dartboard_rerank(query_vector, candidate_vectors, top_k, sigma)
