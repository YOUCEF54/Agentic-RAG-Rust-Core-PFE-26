"""External web retrieval adapter service."""

from __future__ import annotations

from typing import List, Tuple


def search_web(query: str, top_k: int) -> List[Tuple[str, str, int, float]]:
    raise NotImplementedError("External search service migration is pending.")

