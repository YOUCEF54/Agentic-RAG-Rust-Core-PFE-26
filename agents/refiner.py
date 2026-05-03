from .base import Agent


class QueryRefiner(Agent):
    def __init__(self, chat_fn):
        super().__init__("QueryRefiner")
        self.chat_fn = chat_fn

    def run(self, state: dict) -> dict:
        attempt = state.get("attempts", 0)

        if attempt == 0:
            # First pass: straightforward clarification for retrieval
            prompt = (
                "Rewrite the user query to be clearer and more specific for document retrieval.\n"
                "CRITICAL: Retain all technical terms, language names, model names, and specific entities exactly.\n"
                "Return ONLY the rewritten query. No explanation.\n\n"
                f"User query: {state['query']}"
            )
        else:
            # Retry pass: the evaluator has already judged the previous attempt.
            # Give the refiner everything it needs to understand what went wrong.
            judge_summary = state.get("judge_summary", "No details available.")
            prev_answer   = state.get("answer", "")
            prev_score    = state.get("score", 0.0)
            prev_refined  = state.get("refined_query", state["query"])

            prompt = (
                "A previous retrieval + generation attempt failed to produce a satisfactory answer.\n\n"
                f"Original query       : {state['query']}\n"
                f"Previous rewrite     : {prev_refined}\n"
                f"Previous answer      : {prev_answer}\n"
                f"Evaluator score      : {prev_score:.2f} / 1.0\n"
                f"Evaluator feedback   : {judge_summary}\n\n"
                "Your task: Rewrite the query using DIFFERENT keywords or a different angle "
                "so the retriever can surface passages that address the evaluator's criticism.\n"
                "- If the evaluator said the answer hallucinated, make the query more specific.\n"
                "- If the evaluator said the context was irrelevant, try synonyms or a broader framing.\n"
                "- If the answer was incomplete, focus the query on the missing aspect.\n\n"
                "Return ONLY the rewritten query. No explanation."
            )

        messages = [{"role": "user", "content": prompt}]
        try:
            refined_query, model_used = self.chat_fn(messages)

            # Clean up common LLM preamble patterns
            lines = [l.strip() for l in refined_query.strip().splitlines() if l.strip()]
            cleaned = lines[-1] if lines else state["query"]
            for prefix in (
                "rewritten query:",
                "query:",
                "here is",
                "revised query:",
                "refined query:",
            ):
                if cleaned.lower().startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    break

            # Sanity guard: if the model went rogue and returned a paragraph, fall back
            if len(cleaned) > len(state["query"]) * 3:
                cleaned = state["query"]

        except Exception as e:
            print(f"[{self.name}] WARN: Refinement failed ({e}). Using original query.")
            cleaned = state["query"]
            model_used = "error-fallback"

        state["refined_query"] = cleaned
        state.setdefault("models", {})["refiner"] = model_used

        Agent._trace(
            state,
            self.name,
            f"Refined query (attempt={attempt}).",
            {
                "original_query": state["query"],
                "refined_query":  state["refined_query"],
                "attempt":        attempt,
                "model_used":     model_used,
            },
        )
        return state