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

    def run(self, state: dict) -> dict:
        context = "\n\n".join(
            f"[source: {source} | page: {page}]\n{text}"
            for (text, source, page, _dist) in state["chunks"]
        )

        instruction_prompt = (
            "You are a helpful research assistant. Answer the user's question using ONLY the Retrieved Context below.\n\n"
            "RULES:\n"
            "1. Read ALL the chunks carefully. Combine information from multiple chunks.\n"
            "2. Quote or paraphrase directly from the context to support your answer.\n"
            "3. If the context contains relevant information, USE IT - even if it only partially answers the question.\n"
            "4. Only say 'Information not found' as a last resort if the context is 100% irrelevant.\n\n"
            "5. Do NOT add information that is not in the context.\n"
            f"Retrieved Context:\n{context}\n"
        )
        query = state.get("refined_query") or state["query"]
        messages = [
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": query},
        ]
        answer, model_used = self.chat_fn(messages)
        state["answer"] = self._strip_chunk_citations(answer)
        state["model_used"] = model_used
        models = state.setdefault("models", {})
        models["generator"] = model_used
        Agent._trace(
            state,
            self.name,
            "Generated answer from retrieved context.",
            {"model_used": model_used, "context_chunks": len(state["chunks"])},
        )
        return state
