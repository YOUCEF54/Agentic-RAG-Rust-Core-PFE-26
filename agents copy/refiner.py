from .base import Agent


class QueryRefiner(Agent):
    def __init__(self, chat_fn):
        super().__init__("QueryRefiner")
        self.chat_fn = chat_fn

    def run(self, state: dict) -> dict:
        attempt = state.get("attempts", 0)
        if attempt == 0:
            prompt = (
                "Rewrite the user query to be clearer for document retrieval. "
                "CRITICAL: You must retain all technical terms, language names, and specific entities.\n\n"
                f"User query: {state['query']}"
            )
        else:
            prompt = (
                "The previous retrieval attempt was insufficient. Rewrite this query using "
                "different keywords or a different angle to find relevant document passages. "
                "Return ONLY the rewritten query.\n\n"
                f"Original query: {state['query']}\n"
                f"Previous rewrite: {state.get('refined_query', '')}"
            )
        messages = [{"role": "user", "content": prompt}]
        try:
            refined_query, model_used = self.chat_fn(messages)

            lines = [l.strip() for l in refined_query.strip().splitlines() if l.strip()]
            cleaned = lines[-1] if lines else state["query"]
            for prefix in ("rewritten query:", "query:", "here is", "revised query:", "refined query:"):
                if cleaned.lower().startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    break
            if len(cleaned) > len(state["query"]) * 3:
                cleaned = state["query"]
        except Exception as e:
            print(f"[{self.name}] WARN: Refinement failed ({e}). Using original query.")
            cleaned = state["query"]
            model_used = "error-fallback"

        state["refined_query"] = cleaned
        models = state.setdefault("models", {})
        models["refiner"] = model_used
        Agent._trace(
            state,
            self.name,
            "Refined user query for retrieval.",
            {
                "original_query": state["query"],
                "refined_query": state["refined_query"],
                "model_used": model_used,
            },
        )
        return state
