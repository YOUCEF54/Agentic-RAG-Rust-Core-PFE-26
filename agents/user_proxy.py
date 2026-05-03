from .base import Agent


class UserProxy(Agent):
    def __init__(self, refiner, retriever, selector, generator, evaluator):
        super().__init__("UserProxy")
        self.refiner = refiner
        self.retriever = retriever
        self.selector = selector
        self.generator = generator
        self.evaluator = evaluator

    def _run_once(self, state: dict) -> dict:
        state = self.refiner.run(state)
        state = self.retriever.run(state)
        if self.selector is not None:
            state = self.selector.run(state)
        state = self.generator.run(state)
        state = self.evaluator.run(state)
        return state

    def run(self, state: dict) -> dict:
        Agent._trace(state, self.name, "Starting agentic run.")
        state = self._run_once(state)

        while state["should_retry"]:
            Agent._trace(
                state,
                self.name,
                "Retrying due to low evaluator score.",
                {"attempts": state.get("attempts"), "score": state.get("score")},
            )
            state = self._run_once(state)

        return state
