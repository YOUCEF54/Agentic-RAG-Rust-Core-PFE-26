"""Central orchestration entrypoint for the agentic workflow."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from app.agents import CRAGEvaluator, DynamicPassageSelector, Generator, Retriever
from app.core import config
from app.schemas.api import QueryRequest

RetrieveWithMetaFn = Callable[[str, int, Optional[float]], Tuple[List[tuple], List[Dict[str, Any]]]]
ChatFn = Callable[[List[dict], Optional[str]], Tuple[str, Optional[str]]]


class AgentOrchestrator:
    def __init__(
        self,
        *,
        payload: QueryRequest,
        retrieve_with_meta: RetrieveWithMetaFn,
        backend_chat: ChatFn,
    ) -> None:
        self.payload = payload
        self.retrieve_with_meta = retrieve_with_meta
        self.backend_chat = backend_chat
        self._pipeline: Optional[Dict[str, Any]] = None

    def init_state(self, *, emit: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "query": self.payload.question,
            "attempts": 0,
            "should_retry": False,
            "trace": [] if self.payload.return_trace else None,
            "crag_enable_external_route": bool(self.payload.crag_enable_external_route),
        }
        if emit is not None:
            state["emit"] = emit
        return state

    def _build_agent_chat_fns(self) -> Tuple[
        Callable[[List[dict]], Tuple[str, Optional[str]]],
        Callable[[List[dict]], Tuple[str, Optional[str]]],
        Callable[[List[dict]], Tuple[str, Optional[str]]],
        Callable[[List[dict]], Tuple[str, Optional[str]]],
    ]:
        if config.API_TYPE == "open_router":
            shared_model = self.payload.chat_model or config.OPENROUTER_CHAT_MODEL
            shared_chat_fn = lambda msgs: self.backend_chat(msgs, shared_model)
            return shared_chat_fn, shared_chat_fn, shared_chat_fn, shared_chat_fn

        return (
            lambda msgs: self.backend_chat(msgs, config.REFINER_MODEL),
            lambda msgs: self.backend_chat(msgs, config.GENERATOR_MODEL),
            lambda msgs: self.backend_chat(msgs, config.EVALUATOR_MODEL),
            lambda msgs: self.backend_chat(msgs, config.SELECTOR_MODEL),
        )

    def _build_pipeline(self, state: Dict[str, Any]) -> Dict[str, Any]:
        def retrieve_for_agent(query: str, top_k: int) -> List[tuple]:
            hits, meta = self.retrieve_with_meta(query, top_k, self.payload.dartboard_sigma)
            state["retrieved_meta"] = meta
            return hits

        _, chat_generator, chat_evaluator, chat_selector = self._build_agent_chat_fns()
        retriever = Retriever(
            retrieve_fn=lambda q, top_k: retrieve_for_agent(q, top_k),
            top_k=self.payload.top_k,
            top_n=config.TOP_N_RETRIEVAL,
        )
        retriever.external_top_k = int(self.payload.crag_external_top_k)

        selector = (
            DynamicPassageSelector(
                chat_fn=chat_selector,
                max_passages=config.TOP_K_MAX,
                min_passages=config.TOP_K_MIN,
            )
            if config.DPS_ENABLED
            else None
        )
        generator = Generator(chat_fn=chat_generator)
        evaluator = CRAGEvaluator(
            chat_fn=chat_evaluator,
            correct_threshold=self.payload.crag_correct_threshold,
            ambiguous_threshold=self.payload.crag_ambiguous_threshold,
        )
        return {
            "retriever": retriever,
            "selector": selector,
            "generator": generator,
            "evaluator": evaluator,
        }

    def _ensure_pipeline(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if self._pipeline is None:
            self._pipeline = self._build_pipeline(state)
        return self._pipeline

    def run_retrieval(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pipeline = self._ensure_pipeline(state)
        return pipeline["retriever"].run(state)

    def run_selection(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pipeline = self._ensure_pipeline(state)
        selector = pipeline["selector"]
        if selector is None:
            return state
        return selector.run(state)

    def run_evaluation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pipeline = self._ensure_pipeline(state)
        return pipeline["evaluator"].run(state)

    def run_external_route(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pipeline = self._ensure_pipeline(state)
        retriever = pipeline["retriever"]
        if state.get("crag_status") in {"Incorrect", "Ambiguous"} and state.get("crag_enable_external_route", True):
            return retriever.run_external(state)
        state["external_chunks"] = []
        state["external_retrieved_meta"] = []
        return state

    def run_generation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pipeline = self._ensure_pipeline(state)
        return pipeline["generator"].run(state)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state = self.run_retrieval(state)
        state = self.run_selection(state)
        state = self.run_evaluation(state)
        state = self.run_external_route(state)
        state = self.run_generation(state)
        return state
