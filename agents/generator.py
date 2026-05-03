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
        text = re.sub(r"\[\s*chunk\s*\d+\s*\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    @staticmethod
    def _format_chunk(chunk: tuple) -> str:
        try:
            text, source, page, _dist = chunk
        except (TypeError, ValueError):
            return str(chunk)
        return f"[source: {source} | page: {page}]\n{text}"

    def _compress_internal_context(self, query: str, chunks: list[tuple]) -> tuple[str, str | None]:
        if not chunks:
            return "", None

        raw_context = "\n\n".join(self._format_chunk(c) for c in chunks)
        prompt = (
            "You are a context refiner for CRAG.\n"
            "Keep only facts that are directly useful for answering the question.\n"
            "Remove unrelated or repetitive sentences.\n"
            "Do not add any new facts.\n"
            "Output ONLY the refined context text.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{raw_context}"
        )
        refined, model_used = self.chat_fn([{"role": "user", "content": prompt}])
        return refined.strip(), model_used

    def _build_external_context(self, external_chunks: list[tuple]) -> str:
        return "\n\n".join(self._format_chunk(c) for c in external_chunks)

    def run(self, state: dict) -> dict:
        query = state.get("refined_query") or state["query"]
        status = (state.get("crag_status") or "Correct").strip()

        internal_chunks = state.get("chunks", [])
        external_chunks = state.get("external_chunks", [])

        refinement_model = None
        if status == "Correct":
            refined_context, refinement_model = self._compress_internal_context(query, internal_chunks)
            final_context = refined_context or "\n\n".join(self._format_chunk(c) for c in internal_chunks)
            route_desc = "internal_refined"
            system_prompt = (
                "You are a helpful research assistant.\n"
                "Answer using ONLY the refined internal document context.\n"
                "If information is missing, say so clearly.\n\n"
                f"Refined Internal Context:\n{final_context}"
            )

        elif status == "Incorrect":
            external_context = self._build_external_context(external_chunks)
            route_desc = "external_only"
            if not external_context.strip():
                external_context = "No external web context available."
            system_prompt = (
                "You are a helpful research assistant.\n"
                "Internal passages were judged low-quality for this query.\n"
                "Answer using ONLY external web search context below.\n"
                "If the context is insufficient, say so explicitly.\n\n"
                f"External Web Context:\n{external_context}"
            )

        else:  # Ambiguous
            refined_context, refinement_model = self._compress_internal_context(query, internal_chunks)
            external_context = self._build_external_context(external_chunks)
            route_desc = "hybrid_internal_external"
            if not refined_context.strip():
                refined_context = "No reliable internal context available."
            if not external_context.strip():
                external_context = "No external web context available."
            system_prompt = (
                "You are a helpful research assistant.\n"
                "Combine both refined internal context and external web context.\n"
                "Prefer statements supported by both sources when possible.\n"
                "If there is conflict or uncertainty, acknowledge it.\n\n"
                f"Refined Internal Context:\n{refined_context}\n\n"
                f"External Web Context:\n{external_context}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        answer, model_used = self.chat_fn(messages)

        state["answer"] = self._strip_chunk_citations(answer)
        state["model_used"] = model_used
        state["generation_route"] = route_desc
        state.setdefault("models", {})["generator"] = model_used
        if refinement_model is not None:
            state.setdefault("models", {})["context_refiner"] = refinement_model

        Agent._trace(
            state,
            self.name,
            "Generated answer after CRAG correction routing.",
            {
                "crag_status": status,
                "generation_route": route_desc,
                "internal_chunks": len(internal_chunks),
                "external_chunks": len(external_chunks),
                "model_used": model_used,
            },
        )
        return state
