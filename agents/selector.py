import re
from .base import Agent

class DynamicPassageSelector(Agent):
    """
    Fine-Tuned DPS: select a variable-length subset from Top-N retrieved passages.
    Uses a specific instruction template aligned with the fine-tuned Qwen2.5-7B model.
    Falls back to top min_passages on any error.
    """

    def __init__(self, chat_fn, max_passages: int = 8, min_passages: int = 1):
        super().__init__("DynamicPassageSelector")
        self.chat_fn      = chat_fn
        self.max_passages = max_passages
        self.min_passages = min_passages

    def _build_prompt(self, query: str, candidates: list) -> str:
        passages_list = []
        for i, (text, source, page, _dist) in enumerate(candidates, start=1):
            # Keep truncation to avoid blowing past the 4096 context window
            truncated = text[:600] + "..." if len(text) > 600 else text
            # Format exactly as [1] Text... (injecting metadata cleanly)
            passages_list.append(f"[{i}] (Source: {source}, Page: {page}) {truncated}")

        passages_text = "\n".join(passages_list)

        # Exact template matching the fine-tuning notebook (dps_prompt)
        prompt = (
            "### Instruction:\n"
            "Given the following query and a list of numbered passages, select only the indices of the passages that are necessary to answer the query. Return them as a comma-separated list.\n\n"
            "### Input:\n"
            f"Query: {query}\n"
            "Passages:\n"
            f"{passages_text}\n\n"
            "### Response:\n"
        )
        
        return prompt

    def _parse_indices(self, response: str, n_candidates: int) -> list[int]:
        # The fine-tuned model outputs "1, 3, 5". This regex efficiently captures the digits.
        raw_numbers = re.findall(r"\b(\d+)\b", response)
        indices: list[int] = []
        seen: set[int]     = set()
        
        for num_str in raw_numbers:
            idx = int(num_str)
            if 1 <= idx <= n_candidates and idx not in seen:
                indices.append(idx)
                seen.add(idx)
            if len(indices) >= self.max_passages:
                break
                
        if not indices:
            indices = list(range(1, min(self.min_passages + 2, n_candidates + 1)))
        return indices

    def run(self, state: dict) -> dict:
        candidates = state.get("chunks_candidates", [])
        if not candidates:
            state["chunks"] = []
            Agent._trace(state, self.name, "No candidates to select from.")
            return state

        query  = state.get("refined_query") or state["query"]
        prompt = self._build_prompt(query, candidates)

        model_used       = None
        selected_indices = None
        fallback         = False

        try:
            # Depending on how chat_fn wraps Ollama, you may need to ensure it doesn't 
            # inject its own system prompts that conflict with the ### Instruction format.
            # Passing it as a raw string or single user message usually works best.
            response, model_used = self.chat_fn([{"role": "user", "content": prompt}])
            selected_indices     = self._parse_indices(response, len(candidates))
        except Exception as e:
            fallback         = True
            selected_indices = list(range(1, min(3, len(candidates)) + 1))
            Agent._trace(state, self.name, f"Selection LLM call failed ({e}), falling back to top-3.")

        selected_chunks = [candidates[i - 1] for i in selected_indices]

        state["chunks"]               = selected_chunks
        state["dps_selected_indices"] = selected_indices
        state["dps_n_candidates"]     = len(candidates)
        if model_used is not None:
            state.setdefault("models", {})["selector"] = model_used

        Agent._trace(
            state,
            self.name,
            f"{'[fallback] ' if fallback else ''}Selected {len(selected_chunks)} of {len(candidates)} passages.",
            {
                "query":            query,
                "selected_indices": selected_indices,
                "selected_sources": [(s, p) for _, s, p, _ in selected_chunks],
                "model_used":       model_used,
                "fallback":         fallback,
            },
        )
        return state