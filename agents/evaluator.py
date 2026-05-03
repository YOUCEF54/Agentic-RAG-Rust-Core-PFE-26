import re

from utils.stop_words import stop_words

from .base import Agent


class Evaluator(Agent):
    def __init__(self, chat_fn, min_score: float = 0.7, max_attempts: int = 3):
        super().__init__("Evaluator")
        self.chat_fn = chat_fn
        self.min_score = min_score
        self.max_attempts = max_attempts

    def _faithfulness_precheck(self, answer: str, chunks: list) -> float | None:
        answer_tokens = set(answer.lower().split())
        stopwords = set(stop_words)
        answer_tokens -= stopwords
        if not answer_tokens:
            return 0.0

        all_chunk_tokens: set[str] = set()
        for item in chunks:
            text = item[0] if isinstance(item, tuple) else item
            all_chunk_tokens.update(text.lower().split())
        all_chunk_tokens -= stopwords

        if not all_chunk_tokens:
            return None

        overlap = len(answer_tokens & all_chunk_tokens) / len(answer_tokens)
        if overlap < 0.15:
            return 0.0
        return None

    def _score(self, query: str, answer: str, chunks: list) -> tuple[float, str, str | None]:
        if not answer.strip():
            return 0.0, "Empty answer.", None

        precheck = self._faithfulness_precheck(answer, chunks)
        if precheck is not None:
            return precheck, "Failed token-overlap faithfulness check.", None

        context = "\n\n".join(
            f"[Chunk {i+1}]:\n{text}"
            for i, (text, *_) in enumerate(chunks)
        )
        eval_message = (
            "You are a strict auditor. Compare the Generated Answer to the Retrieved Context only.\n\n"
            "Tasks:\n"
            "1. FAITHFULNESS: Is every claim in the answer supported by the context? "
            "If any claim is not in the context, it is a hallucination.\n"
            "2. RELEVANCE: Does the answer address the question?\n\n"
            f"Question: {query}\n\n"
            f"Retrieved Context:\n{context}\n\n"
            f"Generated Answer:\n{answer}\n\n"
            "Respond with exactly two lines:\n"
            "SUMMARY: <one sentence>\n"
            "SCORE: <float 0.0-1.0>"
        )
        messages = [{"role": "user", "content": eval_message}]

        try:
            llm_response, model_used = self.chat_fn(messages)
            summary = "No summary provided"
            for line in llm_response.strip().splitlines():
                if re.search(r"(?i)^\s*\*?\*?SUMMARY", line):
                    parts = re.split(r":", line, maxsplit=1)
                    if len(parts) > 1:
                        summary = parts[1].strip("* ")
                    else:
                        summary = line.strip("* ")
                    break
            match = re.search(r"(?i)SCORE[\s\*:]*(0\.\d+|1\.0|0|1)", llm_response)
            if match:
                score = float(match.group(1))
            else:
                all_numbers = re.findall(r"(0\.\d+|1\.0|[01])", llm_response)
                score = float(all_numbers[-1]) if all_numbers else 0.0
            summary = "No summary provided"
            if "SUMMARY:" in llm_response.upper():
                summary = llm_response.upper().split("SUMMARY:")[1].split("\n")[0].strip()

            return max(0.0, min(1.0, score)), summary, model_used
        except Exception as e:
            return 0.0, f"Evaluation failed: {e}", None

    def run(self, state: dict) -> dict:
        state["attempts"] += 1
        score, summary, model_used = self._score(state["query"], state["answer"], state["chunks"])
        state["score"] = score
        state["judge_summary"] = summary
        models = state.setdefault("models", {})
        models["evaluator"] = model_used

        state["should_retry"] = (state["score"] < self.min_score) and (state["attempts"] < self.max_attempts)
        Agent._trace(
            state,
            self.name,
            "Evaluated answer quality.",
            {
                "score": state["score"],
                "summary": summary,
                "model_used": model_used,
                "should_retry": state["should_retry"],
                "attempts": state["attempts"],
            },
        )
        return state
