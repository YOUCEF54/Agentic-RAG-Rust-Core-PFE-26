from .base import Agent


class QueryRefiner(Agent):
    """
    Optional pre-processing step that rewrites the user query for better retrieval.

    In a CRAG pipeline this runs ONCE before UserProxy.run() — it is NOT called
    inside the pipeline loop. CRAG corrects retrieval quality by routing to
    external sources, not by retrying with a rewritten query.

    Usage:
        state = refiner.run(state)       # optional, before UserProxy
        state = user_proxy.run(state)    # CRAG pipeline
    """

    def __init__(self, chat_fn):
        super().__init__("QueryRefiner")
        self.chat_fn = chat_fn

    def run(self, state: dict) -> dict:
        attempt = state.get("attempts", 0)

        if attempt == 0:
            prompt = (
                "Rewrite the user query to be clearer and more specific for document retrieval.\n"
                "CRITICAL: Retain all technical terms, language names, model names, and specific entities exactly.\n"
                "Return ONLY the rewritten query. No explanation.\n\n"
                f"User query: {state['query']}"
            )
        else:
            # This branch is only reached if the refiner is used outside CRAG
            # in a custom retry loop. The score here is the CRAG relevance_score
            # (retrieval quality, 0-1), not a generation correctness score.
            judge_summary = state.get("judge_summary", "No details available.")
            prev_refined  = state.get("refined_query", state["query"])
            retrieval_score = state.get("crag_relevance_score", state.get("score", 0.0))

            prompt = (
                "A previous retrieval attempt produced insufficient context.\n\n"
                f"Original query           : {state['query']}\n"
                f"Previous rewrite         : {prev_refined}\n"
                f"Retrieval quality score  : {retrieval_score:.2f} / 1.0\n"
                f"Evaluator feedback       : {judge_summary}\n\n"
                "Rewrite the query using DIFFERENT keywords or a different angle "
                "to help the retriever surface more relevant passages.\n"
                "Return ONLY the rewritten query. No explanation."
            )

        messages = [{"role": "user", "content": prompt}]
        try:
            refined_query, model_used = self.chat_fn(messages)

            lines  = [l.strip() for l in refined_query.strip().splitlines() if l.strip()]
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

            if len(cleaned) > len(state["query"]) * 3:
                cleaned = state["query"]

        except Exception as e:
            print(f"[{self.name}] WARN: Refinement failed ({e}). Using original query.")
            cleaned    = state["query"]
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