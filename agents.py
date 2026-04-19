import re
from typing import Any, Dict, List, Optional
from utils.stop_words import stop_words

class Agent:
    def __init__(self, name: str):
        self.name = name
    
    def run(self, state: dict) -> dict:
        raise NotImplementedError(f"{self.name}.run() not implemented") 

    @staticmethod
    def _trace(state: Dict[str, Any], agent: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
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

class UserProxy(Agent):
    def __init__(self, refiner, retriever, selector, generator, evaluator):
        super().__init__("UserProxy")
        self.refiner   = refiner
        self.retriever = retriever
        self.selector  = selector   # NEW — can be None to disable DPS
        self.generator = generator
        self.evaluator = evaluator

    def _run_once(self, state: dict) -> dict:
        state = self.refiner.run(state)
        state = self.retriever.run(state)
        if self.selector is not None:
            state = self.selector.run(state)   # DPS step
        state = self.generator.run(state)
        state = self.evaluator.run(state)
        return state

    def run(self, state: dict) -> dict:
        Agent._trace(state, self.name, "Starting agentic run.")
        state = self._run_once(state)

        while state["should_retry"]:
            Agent._trace(
                state, self.name,
                "Retrying due to low evaluator score.",
                {"attempts": state.get("attempts"), "score": state.get("score")},
            )
            state = self._run_once(state)

        return state

class Retriever(Agent):
    def __init__(self, retrieve_fn, top_k: int, top_n: int | None = None):
        super().__init__("Retriever")
        self.retrieve_fn = retrieve_fn
        self.top_k = top_k
        # top_n: how many candidates to fetch for DPS reranking
        # if None, falls back to top_k (DPS disabled mode)
        self.top_n = top_n or top_k

    def run(self, state: dict) -> dict:
        query = state.get("refined_query") or state["query"]

        # Fetch Top-N candidates for DPS
        candidates = self.retrieve_fn(query, top_k=self.top_n)
        
        # Store full candidate list for DPS selector
        state["chunks_candidates"] = candidates
        # Also set chunks to top_k subset as fallback if DPS is disabled
        state["chunks"] = candidates[:self.top_k]

        Agent._trace(
            state,
            self.name,
            f"Retrieved {len(candidates)} candidates (top_n={self.top_n}, top_k={self.top_k}).",
            {"query_used": query, "top_n": self.top_n, "top_k": self.top_k},
        )
        return state



class Generator(Agent):
    def __init__(self, chat_fn):
        super().__init__("Generator")
        self.chat_fn = chat_fn
        
    def run(self, state: dict) -> dict:
        context = "\n\n".join(
            f"[Chunk {i+1} | {source} p.{page}]:\n{text}"
            for i, (text, source, page, _dist) in enumerate(state["chunks"])
        )

        instruction_prompt = (
            "You are a helpful research assistant. Answer the user's question using ONLY the Retrieved Context below.\n\n"
            "RULES:\n"
            "1. Read ALL the chunks carefully. Combine information from multiple chunks if needed.\n"
            "2. Quote or paraphrase directly from the context to support your answer.\n"
            "3. If the context contains relevant information, USE IT — even if it only partially answers the question.\n"
            "4. Only say 'The document does not contain this information' if NONE of the chunks are relevant at all.\n"
            "5. Do NOT add information that is not in the context.\n\n"
            f"Retrieved Context:\n{context}\n"
        )
        query = state.get("refined_query") or state["query"]
        messages = [
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": query},
        ]
        answer, model_used = self.chat_fn(messages)
        state["answer"] = answer
        state["model_used"] = model_used
        models = state.setdefault("models", {})
        models["generator"] = model_used
        Agent._trace(
            state,
            self.name,
            "Generated answer from retrieved context.",
            {"model_used": model_used, "context_chunks": len(state["chunks"])},
        )
        return state
    

class Evaluator(Agent):
    def __init__(self, chat_fn, min_score: float = 0.7, max_attempts: int = 3):
        super().__init__("Evaluator")
        self.chat_fn = chat_fn
        # On augmente le seuil d'exigence à 0.7 (70% de qualité minimum)
        self.min_score = min_score
        self.max_attempts = max_attempts

    def _faithfulness_precheck(self, answer: str, chunks: list) -> float | None:
        """
        Fast token-overlap check. If the answer shares very few tokens with 
        any retrieved chunk, it's almost certainly hallucinated — return 0.0
        immediately without calling the LLM.
        Returns None if inconclusive (let LLM judge proceed).
        """
        answer_tokens = set(answer.lower().split())
        # Remove stopwords that appear everywhere
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
        # Less than 15% token overlap → likely hallucination
        # (lowered from 30% to avoid penalizing concise or refusal-style answers)
        if overlap < 0.15:
            return 0.0
        return None  # inconclusive — proceed to LLM judge

    def _score(self, query: str, answer: str, chunks: list) -> tuple[float, str, str | None]:
        if not answer.strip():
            return 0.0, "Empty answer.", None

        # Fast path: catch obvious hallucinations before LLM call
        precheck = self._faithfulness_precheck(answer, chunks)
        if precheck is not None:
            return precheck, "Failed token-overlap faithfulness check.", None

        context = "\n\n".join(
            f"[Chunk {i+1}]:\n{text}"
            for i, (text, *_) in enumerate(chunks)
        )
        # Folded prompt — works for small models
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
                if line.strip().upper().startswith("SUMMARY:"):
                    summary = line.split(":", 1)[1].strip()
                    break
            match = re.search(r"SCORE:\s*(0\.\d+|1\.0|0|1)", llm_response)
            if match:
                score = float(match.group(1))
            else:
                # Parse failed — log it and be conservative
                print(f"[Evaluator] WARN: could not parse score from: {llm_response[:200]!r}")
                score = 0.0   # ← P15 fix: conservative fallback, not 0.5
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
        
        # La condition de retry : score insuffisant ET on a encore des essais disponibles
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
    
class QueryRefiner(Agent):
    def __init__(self, chat_fn):
        super().__init__("QueryRefiner")
        self.chat_fn = chat_fn

    def run(self, state: dict) -> dict:
        prompt = (
            "Rewrite the user query to be clearer and more specific for document retrieval. "
            "Return ONLY the rewritten query, nothing else. No explanation, no preamble.\n\n"
            f"User query: {state['query']}"
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            refined_query, model_used = self.chat_fn(messages)
            
            # Strip preambles small models add: "Here is the rewritten query: ..."
            lines = [l.strip() for l in refined_query.strip().splitlines() if l.strip()]
            # Take last non-empty line — most models put the actual query last
            cleaned = lines[-1] if lines else state["query"]
            # Strip common preamble patterns
            for prefix in ("rewritten query:", "query:", "here is", "revised query:", "refined query:"):
                if cleaned.lower().startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    break
            # Final guard: if it's longer than 3x the original, it's not a query — fall back
            if len(cleaned) > len(state["query"]) * 3:
                cleaned = state["query"]
        except Exception as e:
            print(f"[{self.name}] WARN: Refinement failed ({e}). Using original query.")
            cleaned = state["query"]
            model_used = "error-fallback"

        state["refined_query"] = cleaned
        models = state.setdefault("models", {})
        models["refiner"] = model_used
        Agent._trace(
            state,
            self.name,
            "Refined user query for retrieval.",
            {
                "original_query": state["query"],
                "refined_query": state["refined_query"],
                "model_used": model_used,
            },
        )
        return state


class DynamicPassageSelector(Agent):
    """
    Zero-shot DPS: prompts an LLM to select a variable-length subset of
    passage indices from Top-N candidates, modeling inter-passage dependencies.
    
    Input state keys:
        chunks_candidates: list of (text, source, page, distance) — Top-N from retriever
        query / refined_query
    
    Output state keys:
        chunks: selected subset — replaces the raw Top-N before Generator sees it
        dps_selected_indices: list[int] for tracing
    """
    def __init__(self, chat_fn, max_passages: int = 8, min_passages: int = 1):
        super().__init__("DynamicPassageSelector")
        self.chat_fn = chat_fn
        self.max_passages = max_passages
        self.min_passages = min_passages

    def _build_prompt(self, query: str, candidates: list) -> str:
        # Format passages with 1-based index tokens exactly as in the paper:
        # Input = Query: q || Passages: [1]p1 [2]p2 ... [n]pn
        passages_text = ""
        for i, (text, source, page, _dist) in enumerate(candidates, start=1):
            # Truncate individual passages to avoid token overflow
            truncated = text[:600] + "..." if len(text) > 600 else text
            passages_text += f"[{i}] (Source: {source}, Page: {page})\n{truncated}\n\n"

        return (
            f"You are a passage selector for a question answering system.\n\n"
            f"Given the query and a numbered list of retrieved passages, select the "
            f"MINIMAL set of passages that together are sufficient to answer the query.\n\n"
            f"RULES:\n"
            f"1. Select only passages that contain information directly needed to answer the query.\n"
            f"2. If one passage suffices, select only that one.\n"
            f"3. For complex questions requiring multiple pieces of evidence, select all necessary passages.\n"
            f"4. Do NOT select redundant or irrelevant passages.\n"
            f"5. Select between {self.min_passages} and {self.max_passages} passages maximum.\n\n"
            f"Query: {query}\n\n"
            f"Passages:\n{passages_text}"
            f"Output ONLY a comma-separated list of passage numbers, e.g.: 1,3,5\n"
            f"Selected passages: "
        )

    def _parse_indices(self, response: str, n_candidates: int) -> list[int]:
        """
        Parse the model response into a list of valid 1-based indices.
        Handles: "1,3,5" / "1, 3, 5" / "[1,3,5]" / "Passages 1 and 3" etc.
        """
        # Extract all numbers from the response
        raw_numbers = re.findall(r'\b(\d+)\b', response)
        indices = []
        seen = set()
        for num_str in raw_numbers:
            idx = int(num_str)
            # Valid range: 1-based, within candidate list, not duplicate
            if 1 <= idx <= n_candidates and idx not in seen:
                indices.append(idx)
                seen.add(idx)
            if len(indices) >= self.max_passages:
                break

        # Fallback: if parsing fails completely, return top-3 by distance
        if not indices:
            indices = list(range(1, min(3, n_candidates) + 1))

        return indices

    def run(self, state: dict) -> dict:
        candidates = state.get("chunks_candidates", [])
        if not candidates:
            # No candidates — nothing to select from, pass through empty
            state["chunks"] = []
            Agent._trace(state, self.name, "No candidates to select from.")
            return state

        query = state.get("refined_query") or state["query"]
        prompt = self._build_prompt(query, candidates)
        messages = [{"role": "user", "content": prompt}]

        try:
            response, model_used = self.chat_fn(messages)
            selected_indices = self._parse_indices(response, len(candidates))
        except Exception as e:
            # On failure, fall back to top-3 by distance (already sorted)
            selected_indices = list(range(1, min(3, len(candidates)) + 1))
            model_used = None
            Agent._trace(state, self.name, f"Selection failed ({e}), falling back to top-3.")

        # Convert 1-based indices to selected chunks
        selected_chunks = [candidates[i - 1] for i in selected_indices]

        state["chunks"] = selected_chunks
        state["dps_selected_indices"] = selected_indices
        state["dps_n_candidates"] = len(candidates)

        models = state.setdefault("models", {})
        models["selector"] = model_used

        Agent._trace(
            state,
            self.name,
            f"Selected {len(selected_chunks)} of {len(candidates)} passages.",
            {
                "query": query,
                "selected_indices": selected_indices,
                "selected_sources": [(s, p) for _, s, p, _ in selected_chunks],
                "model_used": model_used,
            },
        )
        return state