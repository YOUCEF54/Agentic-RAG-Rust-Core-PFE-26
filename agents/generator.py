import re

from .base import Agent


class Generator(Agent):
    def __init__(self, chat_fn):
        super().__init__("Generator")
        self.chat_fn = chat_fn

    @staticmethod
    def _strip_chunk_citations(text: str) -> str:
        if not text:
            return text
        text = re.sub(r"\(\s*chunk\s*\d+\s*\)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[\s*chunk\s*\d+\s*\]",  "", text, flags=re.IGNORECASE)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"[ \t]{2,}", " ",  text)
        return text.strip()

    @staticmethod
    def _format_chunk(chunk: tuple) -> str:
        try:
            text, source, page, _dist = chunk
        except (TypeError, ValueError):
            return str(chunk)
        return f"[source: {source} | page: {page}]\n{text}"

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Split text into sentences without breaking on common abbreviations.
        The naive `(?<=[.!?])\\s+` pattern splits on "et al. 2024", "Fig. 3",
        "e.g. something", etc. This version protects known abbreviations first.
        """
        # Common academic/technical abbreviations that should not trigger splits
        abbrevs = r"(?:et al|e\.g|i\.e|vs|fig|eq|sec|ref|no|vol|pp|approx|dept|prof|dr|mr|ms)"
        # Temporarily replace abbreviation periods with a placeholder
        protected = re.sub(rf"({abbrevs})\.", r"\1<DOT>", text, flags=re.IGNORECASE)
        # Split on sentence boundaries: period/!/? followed by whitespace + capital
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)
        # Restore placeholders
        return [s.replace("<DOT>", ".") for s in sentences]

    @staticmethod
    def _refine_passages(query: str, chunks: list[tuple]) -> str:
        """
        Deterministic knowledge refinement (CRAG paper §3.2):
        Keep only sentences that share at least one significant token with the query.
        No LLM call — fast, lossless, no hallucination risk.
        Falls back to the full chunk if filtering is too aggressive.
        """
        if not chunks:
            return ""

        query_tokens = set(re.findall(r"\b\w{3,}\b", query.lower()))
        refined_parts: list[str] = []

        for chunk in chunks:
            try:
                text, source, page, _dist = chunk
            except (TypeError, ValueError):
                text, source, page = str(chunk), "unknown", 0

            kept: list[str] = []
            for sentence in Generator._split_sentences(text.strip()):
                s_tokens = set(re.findall(r"\b\w{3,}\b", sentence.lower()))
                if s_tokens & query_tokens:
                    kept.append(sentence)

            body = " ".join(kept) if kept else text   # fallback: keep whole chunk
            refined_parts.append(f"[source: {source} | page: {page}]\n{body}")

        return "\n\n".join(refined_parts)

    def _build_external_context(self, external_chunks: list[tuple]) -> str:
        if not external_chunks:
            return ""
        return "\n\n".join(self._format_chunk(c) for c in external_chunks)

    def run(self, state: dict) -> dict:
        query           = state.get("refined_query") or state["query"]
        status          = (state.get("crag_status") or "Correct").strip()
        internal_chunks = state.get("chunks", [])
        external_chunks = state.get("external_chunks", [])

        if status == "Correct":
            refined_context = self._refine_passages(query, internal_chunks)
            route_desc      = "internal_refined"
            system_prompt   = (
                "You are a helpful research assistant.\n"
                "Answer using ONLY the refined internal document context below.\n"
                "If information is missing, say so clearly.\n\n"
                f"Refined Internal Context:\n{refined_context}"
            )

        elif status == "Incorrect":
            external_context = self._build_external_context(external_chunks)
            route_desc       = "external_only"
            if not external_context.strip():
                external_context = "No external web context is available."
            system_prompt = (
                "You are a helpful research assistant.\n"
                "Internal document passages were judged insufficient for this query.\n"
                "Answer using ONLY the external web context below.\n"
                "If the context is insufficient, say so explicitly.\n\n"
                f"External Web Context:\n{external_context}"
            )

        else:  # Ambiguous
            refined_context  = self._refine_passages(query, internal_chunks)
            external_context = self._build_external_context(external_chunks)
            route_desc       = "hybrid_internal_external"
            if not refined_context.strip():
                refined_context  = "No reliable internal context available."
            if not external_context.strip():
                external_context = "No external web context available."
            system_prompt = (
                "You are a helpful research assistant.\n"
                "Combine both the refined internal context and the external web context.\n"
                "Prefer statements supported by both sources when possible.\n"
                "If there is conflict or uncertainty, acknowledge it explicitly.\n\n"
                f"Refined Internal Context:\n{refined_context}\n\n"
                f"External Web Context:\n{external_context}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query},
        ]
        answer, model_used = self.chat_fn(messages)

        state["answer"]           = self._strip_chunk_citations(answer)
        state["model_used"]       = model_used
        state["generation_route"] = route_desc
        state.setdefault("models", {})["generator"] = model_used

        Agent._trace(
            state,
            self.name,
            "Generated answer after CRAG correction routing.",
            {
                "crag_status":      status,
                "generation_route": route_desc,
                "internal_chunks":  len(internal_chunks),
                "external_chunks":  len(external_chunks),
                "model_used":       model_used,
            },
        )
        return state