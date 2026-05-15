"""Vector store adapter for LanceDB operations via rag_rust."""

from __future__ import annotations

from typing import List

import rag_rust


def create_or_open(
    db_dir: str,
    table_name: str,
    texts: List[str],
    sources: List[str],
    pages: List[int],
    embeddings: List[List[float]],
    rebuild: bool,
) -> None:
    rag_rust.lancedb_create_or_open(db_dir, table_name, texts, sources, pages, embeddings, rebuild)


def search(db_dir: str, table_name: str, query_vector: List[float], top_k: int) -> list:
    return rag_rust.lancedb_search(db_dir, table_name, query_vector, top_k)
