from .base import Agent


class Retriever(Agent):
    def __init__(self, retrieve_fn, top_k: int, top_n: int | None = None):
        super().__init__("Retriever")
        self.retrieve_fn = retrieve_fn
        self.top_k = top_k
        self.top_n = top_n or top_k
        self._attempt_top_n_multipliers = [1.0, 1.5, 2.0]

    def run(self, state: dict) -> dict:
        query = state.get("refined_query") or state["query"]

        attempt = state.get("attempts", 0)
        multiplier = self._attempt_top_n_multipliers[
            min(attempt, len(self._attempt_top_n_multipliers) - 1)
        ]
        effective_top_n = int(self.top_n * multiplier)

        candidates = self.retrieve_fn(query, top_k=effective_top_n)

        state["chunks_candidates"] = candidates
        state["chunks"] = candidates[: self.top_k]

        Agent._trace(
            state,
            self.name,
            f"Retrieved {len(candidates)} candidates (attempt={attempt}, top_n={effective_top_n}).",
            {"query_used": query, "effective_top_n": effective_top_n},
        )
        return state
