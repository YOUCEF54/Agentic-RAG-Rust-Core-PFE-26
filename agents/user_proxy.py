from .base import Agent


class UserProxy(Agent):
    def __init__(self, retriever, selector, evaluator, generator):
        super().__init__("UserProxy")
        self.retriever = retriever
        self.selector = selector
        self.evaluator = evaluator
        self.generator = generator

    def run(self, state: dict) -> dict:
        Agent._trace(state, self.name, "Starting CRAG pipeline run.")

        state = self.retriever.run(state)
        if self.selector is not None:
            state = self.selector.run(state)

        state = self.evaluator.run(state)

        if state.get("crag_status") in {"Incorrect", "Ambiguous"} and state.get("crag_enable_external_route", True):
            state = self.retriever.run_external(state)
        else:
            state["external_chunks"] = []
            state["external_retrieved_meta"] = []

        state = self.generator.run(state)
        state["should_retry"] = False

        Agent._trace(
            state,
            self.name,
            "CRAG pipeline complete.",
            {
                "crag_status": state.get("crag_status"),
                "generation_route": state.get("generation_route"),
                "score": state.get("score"),
            },
        )
        return state
