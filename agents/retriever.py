import os
from typing import Callable, List, Optional

import requests

from .base import Agent


class Retriever(Agent):
    def __init__(
        self,
        retrieve_fn,
        top_k: int,
        top_n: int | None = None,
        web_search_fn: Optional[Callable[[str, int], List[tuple]]] = None,
        external_top_k: int = 5,
    ):
        super().__init__("Retriever")
        self.retrieve_fn    = retrieve_fn
        self.top_k          = top_k
        self.top_n          = top_n or top_k
        self.external_top_k = max(1, int(external_top_k))
        self.web_search_fn  = web_search_fn or self._tavily_web_search

    def _tavily_web_search(self, query: str, top_k: int) -> List[tuple]:
        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not set. External route cannot run.")

        endpoint = os.getenv("TAVILY_ENDPOINT", "https://api.tavily.com/search")
        timeout  = float(os.getenv("TAVILY_TIMEOUT", "12"))
        payload  = {
            "api_key":             api_key,
            "query":               query,
            "max_results":         int(top_k),
            "search_depth":        "basic",
            "include_answer":      False,
            "include_raw_content": False,
        }

        response = requests.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        chunks: List[tuple] = []
        for item in (data.get("results") or []):
            title   = str(item.get("title") or "")
            snippet = str(item.get("content") or "").strip()
            url     = str(item.get("url") or "web")
            score   = float(item.get("score") or 0.0)
            text    = f"{title}\n{snippet}".strip()
            if text:
                chunks.append((text, url, 0, 1.0 - score))
        return chunks

    @staticmethod
    def _filter_external_chunks(chunks: List[tuple], min_content_len: int = 80) -> List[tuple]:
        filtered = []
        for chunk in chunks:
            try:
                text, source, page, dist = chunk
            except (TypeError, ValueError):
                continue
            if len(text.strip()) >= min_content_len and dist <= 0.75:
                filtered.append(chunk)
        return filtered or chunks

    def run(self, state: dict) -> dict:
        """
        Internal retrieval.
        We pass top_k (the final desired count) to retrieve_fn.
        main.py's closure handles Dartboard oversampling internally —
        passing top_n here would double-oversample and fetch far more
        candidates than needed.
        """
        query = state.get("refined_query") or state["query"]

        # Fixed: pass top_k, not top_n — Dartboard oversampling is inside retrieve_fn
        candidates = self.retrieve_fn(query, top_k=self.top_k)

        state["chunks_candidates"] = candidates
        # DPS will override state["chunks"] if enabled; this is the plain fallback
        state["chunks"] = candidates[: self.top_k]

        Agent._trace(
            state,
            self.name,
            f"Retrieved {len(candidates)} internal candidates, {len(state['chunks'])} pre-selected.",
            {"query_used": query, "top_k": self.top_k},
        )
        return state

    def run_external(self, state: dict) -> dict:
        """External web retrieval for Incorrect / Ambiguous CRAG routes."""
        crag_status = (state.get("crag_status") or "").strip()
        if crag_status not in {"Incorrect", "Ambiguous"}:
            state["external_chunks"]         = []
            state["external_retrieved_meta"] = []
            Agent._trace(state, self.name, "Skipped external route (CRAG status is Correct).")
            return state

        query = state.get("refined_query") or state["query"]
        try:
            raw_chunks = self.web_search_fn(query, self.external_top_k)
        except Exception as exc:
            state["external_chunks"]         = []
            state["external_retrieved_meta"] = []
            state["external_error"]          = str(exc)
            Agent._trace(state, self.name, f"External route failed: {exc}")
            return state

        web_chunks = self._filter_external_chunks(raw_chunks)

        state["external_chunks"] = web_chunks
        state["external_retrieved_meta"] = [
            {"text": text, "source": source, "page": page, "distance": dist, "route": "external"}
            for text, source, page, dist in web_chunks
        ]

        Agent._trace(
            state,
            self.name,
            f"External route returned {len(web_chunks)} usable web passages "
            f"({len(raw_chunks)} raw results).",
            {"query_used": query, "external_top_k": self.external_top_k},
        )
        return state