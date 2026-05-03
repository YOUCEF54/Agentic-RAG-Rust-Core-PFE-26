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
        trace = state.get("trace")
        if trace is None:
            trace = None
        item = {"agent": agent, "message": message}
        if data:
            item["data"] = data
        if trace is not None:
            trace.append(item)
        emit = state.get("emit")
        if emit:
            emit(item)
