from .base import Agent


class UserProxy(Agent):
    """
    Orchestrates the CRAG pipeline in the correct order:

        Retrieve (full candidates)
            ↓
        CRAGEvaluator (scores full candidate pool → Correct / Ambiguous / Incorrect)
            ↓
        DynamicPassageSelector (filters down to the relevant subset)
            ↓
        [External web retrieval if status is Incorrect or Ambiguous]
            ↓
        Generator (routes generation based on crag_status)

    Note: QueryRefiner is NOT part of the CRAG flow. CRAG corrects retrieval
    quality via web search, not by retrying with a rewritten query.
    If you need query rewriting, run it before constructing this pipeline.
    """

    def __init__(
        self,
        retriever,
        evaluator,
        selector,
        generator,
        enable_external_route: bool = True,
    ):
        super().__init__("UserProxy")
        self.retriever             = retriever
        self.evaluator             = evaluator
        self.selector              = selector
        self.generator             = generator
        self.enable_external_route = enable_external_route

    def run(self, state: dict) -> dict:
        Agent._trace(state, self.name, "Starting CRAG pipeline run.")

        # 1. Retrieve full candidate pool
        state = self.retriever.run(state)

        # 2. Evaluate the FULL candidate pool BEFORE any filtering
        #    so the evaluator sees everything the retriever found.
        state = self.evaluator.run(state)

        # 3. DPS filters down to the relevant subset the evaluator confirmed
        if self.selector is not None:
            state = self.selector.run(state)

        # 4. External web retrieval for Incorrect / Ambiguous routes
        crag_status = state.get("crag_status", "Correct")
        if crag_status in {"Incorrect", "Ambiguous"} and self.enable_external_route:
            state = self.retriever.run_external(state)
        else:
            state["external_chunks"]         = []
            state["external_retrieved_meta"] = []

        # 5. Generate answer via the appropriate route
        state = self.generator.run(state)
        state["should_retry"] = False

        Agent._trace(
            state,
            self.name,
            "CRAG pipeline complete.",
            {
                "crag_status":      state.get("crag_status"),
                "generation_route": state.get("generation_route"),
                "score":            state.get("score"),
                "attempts":         state.get("attempts"),
            },
        )
        return state