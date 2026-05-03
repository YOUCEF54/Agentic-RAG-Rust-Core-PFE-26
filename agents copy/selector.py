import re

from .base import Agent


class DynamicPassageSelector(Agent):
    """
    Zero-shot DPS: select a variable-length subset from Top-N retrieved passages.
    """

    def __init__(self, chat_fn, max_passages: int = 8, min_passages: int = 1):
        super().__init__("DynamicPassageSelector")
        self.chat_fn = chat_fn
        self.max_passages = max_passages
        self.min_passages = min_passages

    def _build_prompt(self, query: str, candidates: list) -> str:
        passages_text = ""
        for i, (text, source, page, _dist) in enumerate(candidates, start=1):
            truncated = text[:600] + "..." if len(text) > 600 else text
            passages_text += f"[{i}] (Source: {source}, Page: {page})\n{truncated}\n\n"

        return (
            "You are a passage selector for a question answering system.\n\n"
            "Given the query and a numbered list of retrieved passages, select the "
            "MINIMAL set of passages that together are sufficient to answer the query.\n\n"
            "RULES:\n"
            "1. Select only passages that contain information directly needed to answer the query.\n"
            "2. If one passage suffices, select only that one.\n"
            "3. For complex questions requiring multiple pieces of evidence, select all necessary passages.\n"
            "4. Do NOT select redundant or irrelevant passages.\n"
            f"5. Select between {self.min_passages} and {self.max_passages} passages maximum.\n\n"
            f"Query: {query}\n\n"
            f"Passages:\n{passages_text}"
            "Output ONLY a comma-separated list of passage numbers, e.g.: 1,3,5\n"
            "Selected passages: "
        )

    def _parse_indices(self, response: str, n_candidates: int) -> list[int]:
        raw_numbers = re.findall(r"\b(\d+)\b", response)
        indices = []
        seen = set()
        for num_str in raw_numbers:
            idx = int(num_str)
            if 1 <= idx <= n_candidates and idx not in seen:
                indices.append(idx)
                seen.add(idx)
            if len(indices) >= self.max_passages:
                break

        if not indices:
            indices = list(range(1, min(3, n_candidates) + 1))

        return indices

    def run(self, state: dict) -> dict:
        candidates = state.get("chunks_candidates", [])
        if not candidates:
            state["chunks"] = []
            Agent._trace(state, self.name, "No candidates to select from.")
            return state

        query = state.get("refined_query") or state["query"]
        prompt = self._build_prompt(query, candidates)
        messages = [{"role": "user", "content": prompt}]

        try:
            response, model_used = self.chat_fn(messages)
            selected_indices = self._parse_indices(response, len(candidates))
        except Exception as e:
            selected_indices = list(range(1, min(3, len(candidates)) + 1))
            model_used = None
            Agent._trace(state, self.name, f"Selection failed ({e}), falling back to top-3.")

        selected_chunks = [candidates[i - 1] for i in selected_indices]

        state["chunks"] = selected_chunks
        state["dps_selected_indices"] = selected_indices
        state["dps_n_candidates"] = len(candidates)

        models = state.setdefault("models", {})
        models["selector"] = model_used

        Agent._trace(
            state,
            self.name,
            f"Selected {len(selected_chunks)} of {len(candidates)} passages.",
            {
                "query": query,
                "selected_indices": selected_indices,
                "selected_sources": [(s, p) for _, s, p, _ in selected_chunks],
                "model_used": model_used,
            },
        )
        return state
