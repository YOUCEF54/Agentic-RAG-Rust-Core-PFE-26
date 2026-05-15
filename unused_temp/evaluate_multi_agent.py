# evaluate_multi_agent.py
# Targets main.py backend (FastAPI server).
#
# Key changes vs previous version:
#   - Imports from main, not api_ollama (file no longer exists)
#   - chat_refiner/generator/evaluator/selector built via main.build_agent_chat_fns()
#     (they are not module-level exports in main.py)
#   - Judge calls use main.backend_chat() — respects API_TYPE (Ollama or OpenRouter)
#   - DB guard only warns instead of running script_opt_ollama.py as subprocess
#   - retrieve() mirrors main.py's retrieve_chunks_with_meta() pattern exactly

from __future__ import annotations

import sys
import time
from pathlib import Path

from datasets import load_dataset

# ── Agent classes ─────────────────────────────────────────────────────────────
from agents import (
    DynamicPassageSelector,
    Evaluator,
    Generator,
    QueryRefiner,
    Retriever,
    UserProxy,
)

# ── Backend — import from main.py (the FastAPI server) ───────────────────────
from main import (
    # LLM routing
    backend_chat,
    build_agent_chat_fns,
    # Embedding
    embed_query,
    ensure_embed_model_loaded,
    # Config
    DB_DIR,
    TABLE_NAME,
    TOP_N_RETRIEVAL,
    TOP_K_MAX,
    TOP_K_MIN,
    DPS_ENABLED,
)
import rag_rust


# ─────────────────────────────────────────────────────────────────────────────
# Eval config
# ─────────────────────────────────────────────────────────────────────────────
EVAL_TOP_K        = 5     # passages passed to the Generator
EVAL_MIN_SCORE    = 0.65  # Evaluator self-critique threshold
EVAL_MAX_ATTEMPTS = 2


# ─────────────────────────────────────────────────────────────────────────────
# DB guard — just warn, don't try to build it here.
# Build the index via the /index endpoint or script_opt_ollama.py beforehand.
# ─────────────────────────────────────────────────────────────────────────────
if not Path(DB_DIR).exists():
    print(
        f"⚠️  Database not found at '{DB_DIR}'.\n"
        "   Build the index first:\n"
        "     • Start the server and POST to /index, OR\n"
        "     • Run: python script_opt_ollama.py\n"
        "   Then re-run this evaluation script."
    )
    sys.exit(1)

# Make sure the embed model is loaded before the eval loop starts
ensure_embed_model_loaded()


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval helper — mirrors main.py retrieve_chunks_with_meta() pattern
# ─────────────────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int) -> list[tuple]:
    """
    Embed query and search LanceDB.
    Returns (text, source, page, dist) tuples — the format agents.py expects.
    """
    query_vector = embed_query(query)
    hits = rag_rust.lancedb_search(DB_DIR, TABLE_NAME, query_vector, top_k)
    return [(text, source, page, dist) for text, source, page, dist in hits]


# ─────────────────────────────────────────────────────────────────────────────
# LLM judge helpers
# Uses backend_chat() so it automatically respects API_TYPE (Ollama / OpenRouter)
# ─────────────────────────────────────────────────────────────────────────────
def judge_answer(question: str, ground_truth: str, rag_answer: str) -> float:
    """LLM judge: Answer Correctness (0-1)."""
    prompt = f"""
You are an expert academic evaluator.
Question: {question}
Ground Truth Answer: {ground_truth}
RAG System Answer: {rag_answer}

Did the RAG system correctly identify the core facts from the Ground Truth?
If the Ground Truth says "BIBREF19" and the RAG says "The text doesn't mention it", the score is 0.

Score the RAG answer from 0 to 10 based on accuracy.
Return ONLY an integer number between 0 and 10. No other text.
"""
    response, _ = backend_chat([{"role": "user", "content": prompt}])
    try:
        return min(int("".join(filter(str.isdigit, response))), 10) / 10.0
    except Exception:
        return 0.0


def judge_retrieval(question: str, ground_truth: str, retrieved_chunks: list) -> float:
    """LLM judge: Context Recall (0-1)."""
    combined = "\n---\n".join(str(c) for c in retrieved_chunks)
    prompt = f"""
You are an expert academic evaluator.
Question: {question}
Ground Truth Answer: {ground_truth}

Retrieved Context from Database:
{combined}

Does the Retrieved Context contain the necessary facts to deduce the Ground Truth Answer?
Score from 0 to 10. If the context is completely irrelevant or missing the facts, score 0.
Return ONLY an integer number between 0 and 10. No other text.
"""
    response, _ = backend_chat([{"role": "user", "content": prompt}])
    try:
        return min(int("".join(filter(str.isdigit, response))), 10) / 10.0
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────
print("Loading QASper dataset...")
dataset = load_dataset("allenai/qasper", split="validation", trust_remote_code=True)
first_item = dataset[0]

questions    = first_item["qas"]["question"]
answers_data = first_item["qas"]["answers"]

print(f"\n--- Evaluating MULTI-AGENT System on: {first_item['title']} ---")

total_score            = 0.0
retrieval_total_score  = 0.0
generation_total_score = 0.0

# Build agent chat functions once for the whole eval run (no per-request model override)
chat_refiner, chat_generator, chat_evaluator, chat_selector = build_agent_chat_fns(
    payload_chat_model=None
)

for i in range(len(questions)):
    query = questions[i]

    # ── Extract ground truth ──────────────────────────────────────────────
    ground_truth = "Unknown"
    if answers_data[i] and answers_data[i]["answer"]:
        ans_dict = answers_data[i]["answer"][0]
        if ans_dict.get("free_form_answer"):
            ground_truth = ans_dict["free_form_answer"]
        elif ans_dict.get("extractive_spans"):
            ground_truth = ", ".join(ans_dict["extractive_spans"])
        elif ans_dict.get("yes_no") is not None:
            ground_truth = str(ans_dict["yes_no"])

    print(f"\n📝 Question {i + 1}: {query}")
    print(f"✅ Ground Truth: {ground_truth}")

    # ── Initial state ─────────────────────────────────────────────────────
    state = {
        "query":        query,
        "attempts":     0,
        "should_retry": True,
        "trace":        [],
    }

    # ── Wire up agents ────────────────────────────────────────────────────
    proxy = UserProxy(
        refiner=QueryRefiner(chat_fn=chat_refiner),
        retriever=Retriever(
            retrieve_fn=retrieve,
            top_k=EVAL_TOP_K,
            top_n=TOP_N_RETRIEVAL,
        ),
        selector=(
            DynamicPassageSelector(
                chat_fn=chat_selector,
                max_passages=TOP_K_MAX,
                min_passages=TOP_K_MIN,
            )
            if DPS_ENABLED else None
        ),
        generator=Generator(chat_fn=chat_generator),
        evaluator=Evaluator(
            chat_fn=chat_evaluator,
            min_score=EVAL_MIN_SCORE,
            max_attempts=EVAL_MAX_ATTEMPTS,
        ),
    )

    # ── Run pipeline ──────────────────────────────────────────────────────
    print("🤖 Agents working (Refine → Retrieve → [Select] → Generate → Evaluate)...")
    start_time = time.time()
    state = proxy.run(state)
    elapsed = time.time() - start_time

    print(
        f"🤖 RAG Answer ({elapsed:.2f}s, Attempts: {state['attempts']}): "
        f"{state.get('answer', '')}"
    )
    print(f"🔄 Refined Query: {state.get('refined_query', 'n/a')}")
    if state.get("dps_selected_indices"):
        print(
            f"🎯 DPS selected passages {state['dps_selected_indices']} "
            f"from {state.get('dps_n_candidates', '?')} candidates"
        )

    # ── Judge retrieval (Context Recall) ──────────────────────────────────
    retrieval_score = judge_retrieval(query, ground_truth, state.get("chunks", []))
    retrieval_total_score += retrieval_score
    print(f"🔎 Context Recall (LanceDB Score): {retrieval_score:.2f} / 1.0")

    # ── Judge generation (Answer Correctness) ─────────────────────────────
    gen_score = judge_answer(query, ground_truth, state.get("answer", ""))
    generation_total_score += gen_score
    total_score += gen_score
    print(f"⚖️  Answer Correctness Score:       {gen_score:.2f} / 1.0")
    print("-" * 60)

n = len(questions)
print(f"\n🏆 Multi-Agent Average Answer Correctness : {total_score / n:.2f} / 1.0")
print(f"🏆 Multi-Agent Average Generation Score   : {generation_total_score / n:.2f} / 1.0")
print(f"🏆 Multi-Agent Average Context Recall     : {retrieval_total_score / n:.2f} / 1.0")