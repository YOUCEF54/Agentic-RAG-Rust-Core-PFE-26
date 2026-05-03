import json
import re
from typing import Any

from .base import Agent


class CRAGEvaluator(Agent):
    """
    Lightweight CRAG evaluator.
    Scores selected internal passages and classifies retrieval quality into:
    - Correct
    - Ambiguous
    - Incorrect
    """

    def __init__(
        self,
        chat_fn,
        correct_threshold: float = 0.75,
        ambiguous_threshold: float = 0.45,
    ):
        super().__init__("CRAGEvaluator")
        self.chat_fn = chat_fn
        self.correct_threshold = float(correct_threshold)
        self.ambiguous_threshold = float(ambiguous_threshold)

    @staticmethod
    def _chunk_text(item: Any) -> str:
        if isinstance(item, tuple) and item:
            return str(item[0])
        return str(item)

    @staticmethod
    def _normalize_label(label: str) -> str:
        normalized = (label or "").strip().lower()
        if normalized in {"correct", "high", "high_confidence"}:
            return "Correct"
        if normalized in {"incorrect", "low", "low_confidence"}:
            return "Incorrect"
        return "Ambiguous"

    @staticmethod
    def _clamp(v: Any, default: float = 0.0) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, f))

    def _parse_eval(self, raw: str) -> tuple[str, float, float, str]:
        default = ("Ambiguous", 0.5, 0.5, "Could not parse evaluator output reliably.")

        try:
            payload = json.loads(raw.strip())
            if isinstance(payload, dict):
                label = self._normalize_label(str(payload.get("classification") or payload.get("label") or ""))
                confidence = self._clamp(payload.get("confidence"), 0.5)
                relevance = self._clamp(payload.get("relevance_score"), 0.5)
                reason = str(payload.get("reason") or payload.get("summary") or "No reason provided.")
                return label, confidence, relevance, reason
        except Exception:
            pass

        cls_match = re.search(r"(?i)(classification|label)\s*[:=]\s*(correct|incorrect|ambiguous)", raw)
        conf_match = re.search(r"(?i)confidence\s*[:=]\s*([0-9]*\.?[0-9]+)", raw)
        rel_match = re.search(r"(?i)(relevance_score|score)\s*[:=]\s*([0-9]*\.?[0-9]+)", raw)
        reason_match = re.search(r"(?i)reason\s*[:=]\s*(.+)", raw)

        if not cls_match and not conf_match and not rel_match:
            return default

        label = self._normalize_label(cls_match.group(2) if cls_match else "Ambiguous")
        confidence = self._clamp(conf_match.group(1) if conf_match else 0.5, 0.5)
        relevance = self._clamp(rel_match.group(2) if rel_match else 0.5, 0.5)
        reason = reason_match.group(1).strip() if reason_match else "Parsed from non-JSON evaluator output."
        return label, confidence, relevance, reason

    def _deterministic_classification(self, relevance_score: float) -> str:
        if relevance_score >= self.correct_threshold:
            return "Correct"
        if relevance_score < self.ambiguous_threshold:
            return "Incorrect"
        return "Ambiguous"

    def run(self, state: dict) -> dict:
        state["attempts"] = int(state.get("attempts", 0)) + 1

        query = state.get("refined_query") or state["query"]
        chunks = state.get("chunks", [])

        if not chunks:
            state["score"] = 0.0
            state["crag_relevance_score"] = 0.0
            state["crag_confidence"] = 1.0
            state["crag_status"] = "Incorrect"
            state["crag_reason"] = "No internal passages were selected by retriever/selector."
            state["judge_summary"] = state["crag_reason"]
            state["should_retry"] = False
            Agent._trace(
                state,
                self.name,
                "CRAG evaluation: no internal context available.",
                {
                    "status": state["crag_status"],
                    "relevance_score": state["crag_relevance_score"],
                    "confidence": state["crag_confidence"],
                },
            )
            return state

        packed_context = "\n\n".join(
            f"[Passage {i + 1}]\n{self._chunk_text(c)[:900]}"
            for i, c in enumerate(chunks)
        )

        prompt = (
            "You are a CRAG evaluator for retrieval quality.\n"
            "Assess whether the selected passages are sufficient and relevant for answering the question.\n\n"
            "Return STRICT JSON with keys:\n"
            "- classification: Correct | Ambiguous | Incorrect\n"
            "- confidence: float 0..1\n"
            "- relevance_score: float 0..1\n"
            "- reason: one short sentence\n\n"
            f"Question:\n{query}\n\n"
            f"Selected Passages:\n{packed_context}"
        )

        model_used = None
        try:
            raw, model_used = self.chat_fn([{"role": "user", "content": prompt}])
            llm_label, confidence, relevance_score, reason = self._parse_eval(raw)
        except Exception as exc:
            llm_label = "Ambiguous"
            confidence = 0.5
            relevance_score = 0.5
            reason = f"Evaluator call failed: {exc}"

        routed_label = self._deterministic_classification(relevance_score)

        state["score"] = relevance_score
        state["crag_relevance_score"] = relevance_score
        state["crag_confidence"] = confidence
        state["crag_status"] = routed_label
        state["crag_reason"] = reason
        state["judge_summary"] = reason
        state["should_retry"] = False
        state.setdefault("models", {})["evaluator"] = model_used

        Agent._trace(
            state,
            self.name,
            "CRAG evaluation complete.",
            {
                "llm_classification": llm_label,
                "routed_classification": routed_label,
                "relevance_score": relevance_score,
                "confidence": confidence,
                "reason": reason,
                "correct_threshold": self.correct_threshold,
                "ambiguous_threshold": self.ambiguous_threshold,
                "model_used": model_used,
            },
        )
        return state


# Backward compatibility for existing imports.
Evaluator = CRAGEvaluator
