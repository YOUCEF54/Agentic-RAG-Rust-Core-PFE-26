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
    def _refine_passages(query: str, chunks: list[tuple]) -> str:
        """
        Knowledge refinement as described in the CRAG paper:
        Strip sentences that share no significant tokens with the query,
        keeping sentences that are directly relevant.

        This is deterministic (no extra LLM call), fast, and doesn't risk
        losing or paraphrasing facts the way an LLM rewrite would.
        """
        if not chunks:
            return ""

        query_tokens = set(re.findall(r"\b\w{3,}\b", query.lower()))
        refined_parts: list[str] = []

        for chunk in chunks:
            try:
                text, source, page, _dist = chunk
            except (TypeError, ValueError):
                text   = str(chunk)
                source = "unknown"
                page   = 0

            kept_sentences: list[str] = []
            for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
                sentence_tokens = set(re.findall(r"\b\w{3,}\b", sentence.lower()))
                if sentence_tokens & query_tokens:
                    kept_sentences.append(sentence)

            # If the filter was too aggressive, keep the whole chunk rather than
            # returning empty (e.g. chunk has no keyword overlap but is still useful)
            body = " ".join(kept_sentences) if kept_sentences else text
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

        else:  # Ambiguous — combine both sources
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