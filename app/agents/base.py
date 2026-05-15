from typing import Any, Dict, Optional


class Agent:
    def __init__(self, name: str):
        self.name = name

    def run(self, state: dict) -> dict:
        raise NotImplementedError(f"{self.name}.run() not implemented")

    @staticmethod
    def _trace(
        state: Dict[str, Any],
        agent: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        item = {"agent": agent, "message": message}
        if data:
            item["data"] = data
        # Append to trace list if tracing is enabled (trace=[] enables, trace=None disables)
        trace = state.get("trace")
        if isinstance(trace, list):
            trace.append(item)
        # Fire streaming emit hook if present (used by /query/stream in main.py)
        emit = state.get("emit")
        if callable(emit):
            emit(item)