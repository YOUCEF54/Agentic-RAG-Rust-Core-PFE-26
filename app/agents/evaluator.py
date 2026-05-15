import json
import re
from typing import Any

from .base import Agent


class CRAGEvaluator(Agent):
    """
    CRAG retrieval-quality evaluator (Yan et al., 2024).

    Classifies the overall retrieval batch as:
        Correct   — at least one passage is highly relevant
        Ambiguous — passages are partially relevant or uncertain
        Incorrect — no passage is sufficiently relevant

    Must run BEFORE DynamicPassageSelector so it sees the full candidate pool.
    """

    def __init__(
        self,
        chat_fn,
        correct_threshold: float = 0.75,
        ambiguous_threshold: float = 0.45,
    ):
        super().__init__("CRAGEvaluator")
        self.chat_fn             = chat_fn
        self.correct_threshold   = float(correct_threshold)
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

    @staticmethod
    def _extract_json(raw: str) -> dict | None:
        """
        Robustly extract a JSON object from an LLM response.
        Handles: bare JSON, markdown fences (```json ... ```),
        preamble text before the object, trailing text after.
        """
        if not raw:
            return None

        # 1. Strip markdown code fences (```json ... ``` or ``` ... ```)
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if fenced:
            candidate = fenced.group(1)
        else:
            # 2. Find the first { ... } block in the raw string
            brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
            candidate = brace_match.group(0) if brace_match else raw.strip()

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # 3. Last resort: try to repair common issues (trailing commas, single quotes)
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)   # trailing commas
        repaired = repaired.replace("'", '"')                   # single → double quotes
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None

    def _parse_eval(self, raw: str) -> tuple[str, float, float, str]:
        default = ("Ambiguous", 0.5, 0.5, "Could not parse evaluator output reliably.")

        # ── Try structured JSON first (preferred path) ────────────────────────
        payload = self._extract_json(raw)
        if isinstance(payload, dict):
            label      = self._normalize_label(
                str(payload.get("classification") or payload.get("label") or "")
            )
            confidence = self._clamp(payload.get("confidence"), 0.5)
            relevance  = self._clamp(payload.get("relevance_score"), 0.5)
            reason     = str(
                payload.get("reason") or payload.get("summary") or "No reason provided."
            )
            # Only accept if we got at least a meaningful label or score
            if label != "Ambiguous" or relevance != 0.5 or confidence != 0.5:
                return label, confidence, relevance, reason

        # ── Regex fallback — handles plain-text responses ─────────────────────
        cls_match    = re.search(
            r"(?i)(classification|label)\s*[:=]\s*(correct|incorrect|ambiguous)", raw
        )
        conf_match   = re.search(
            r"(?i)confidence\s*[:=]\s*([0-9]*\.?[0-9]+)", raw
        )
        # Fixed: use a single capture group regardless of which keyword matched
        rel_match    = re.search(
            r"(?i)(?:relevance_score|relevance|score)\s*[:=]\s*([0-9]*\.?[0-9]+)", raw
        )
        reason_match = re.search(r"(?i)reason\s*[:=]\s*(.+)", raw)

        if not cls_match and not conf_match and not rel_match:
            return default

        label      = self._normalize_label(cls_match.group(2) if cls_match else "Ambiguous")
        confidence = self._clamp(conf_match.group(1) if conf_match else 0.5, 0.5)
        # Fixed: always group(1) — single capture group
        relevance  = self._clamp(rel_match.group(1) if rel_match else 0.5, 0.5)
        reason     = (
            reason_match.group(1).strip()
            if reason_match
            else "Parsed from non-JSON evaluator output."
        )
        return label, confidence, relevance, reason

    def _route(self, llm_label: str, relevance_score: float, confidence: float) -> str:
        threshold_label = (
            "Correct"   if relevance_score >= self.correct_threshold   else
            "Incorrect" if relevance_score <  self.ambiguous_threshold  else
            "Ambiguous"
        )
        # High relevance but low confidence → stay cautious
        if relevance_score >= self.correct_threshold and confidence < 0.5:
            return "Ambiguous"
        # Both signals agree
        if threshold_label == llm_label:
            return threshold_label
        # LLM breaks ties in the Ambiguous band
        if threshold_label == "Ambiguous" and llm_label in {"Correct", "Incorrect"}:
            return llm_label
        return threshold_label

    def run(self, state: dict) -> dict:
        state["attempts"] = int(state.get("attempts", 0)) + 1

        query      = state.get("refined_query") or state["query"]
        candidates = state.get("chunks_candidates") or state.get("chunks", [])

        if not candidates:
            state.update({
                "score":                0.0,
                "crag_relevance_score": 0.0,
                "crag_confidence":      1.0,
                "crag_status":         "Incorrect",
                "crag_reason":         "No internal passages were retrieved.",
                "judge_summary":       "No internal passages were retrieved.",
                "should_retry":        False,
            })
            Agent._trace(state, self.name, "CRAG evaluation: no candidates available.",
                         {"status": "Incorrect"})
            return state

        packed_context = "\n\n".join(
            f"[Passage {i + 1}]\n{self._chunk_text(c)[:900]}"
            for i, c in enumerate(candidates)
        )

        prompt = (
            "You are a CRAG retrieval-quality evaluator.\n"
            "Assess whether the retrieved passages are sufficient and relevant to answer the question.\n\n"
            "Return ONLY a JSON object with exactly these four keys — no explanation, no markdown:\n"
            '  "classification": "Correct" | "Ambiguous" | "Incorrect"\n'
            '  "confidence":     float between 0 and 1\n'
            '  "relevance_score": float between 0 and 1\n'
            '  "reason":         one short sentence\n\n'
            "Example of valid output:\n"
            '{"classification": "Correct", "confidence": 0.9, "relevance_score": 0.85, '
            '"reason": "Passage 2 directly answers the question."}\n\n'
            f"Question:\n{query}\n\n"
            f"Retrieved Passages:\n{packed_context}\n\n"
            "JSON output:"
        )

        model_used = None
        try:
            raw, model_used = self.chat_fn([{"role": "user", "content": prompt}])
            llm_label, confidence, relevance_score, reason = self._parse_eval(raw)
        except Exception as exc:
            llm_label       = "Ambiguous"
            confidence      = 0.5
            relevance_score = 0.5
            reason          = f"Evaluator call failed: {exc}"

        routed_label = self._route(llm_label, relevance_score, confidence)

        state.update({
            "score":                relevance_score,
            "crag_relevance_score": relevance_score,
            "crag_confidence":      confidence,
            "crag_status":          routed_label,
            "crag_reason":          reason,
            "judge_summary":        reason,
            "should_retry":         False,
        })
        state.setdefault("models", {})["evaluator"] = model_used

        Agent._trace(
            state,
            self.name,
            "CRAG evaluation complete.",
            {
                "llm_classification":    llm_label,
                "routed_classification": routed_label,
                "relevance_score":       relevance_score,
                "confidence":            confidence,
                "reason":                reason,
                "correct_threshold":     self.correct_threshold,
                "ambiguous_threshold":   self.ambiguous_threshold,
                "model_used":            model_used,
            },
        )
        return state


# Backward compatibility
Evaluator = CRAGEvaluator