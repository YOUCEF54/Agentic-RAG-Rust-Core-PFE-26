"""Benchmarking and profiling utilities for the Hybrid RAG pipeline.

Covers:
- Profiling loop (PDF read → chunk → embed → DB insert → search) with CSV output
- Retrieval benchmark (throughput over repeated queries)
- Pure embedding benchmark (chunks/sec)
"""
from __future__ import annotations

import csv
import io
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

import rag_rust

# --- Platform/Env ---
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# --- Config (mirrors script_opt.py; override via env vars as needed) ---
def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got: {value!r}") from exc


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {value!r}") from exc


DB_DIR = os.getenv("DB_DIR", "lancedb")
TABLE_NAME = "pdf_chunks"
PDF_DIR = os.getenv("PDF_FOLDER", "pdfs")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME") or "BAAI/bge-small-en-v1.5"
TOP_K = _get_env_int("TOP_K", 3)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- Profiling config ---
NUM_RUNS = 100
CSV_FILE = "profile_results.csv"
CSV_HEADER = [
    "run",
    "pdf_read_ms",
    "chunking_ms",
    "model_embedding_ms",
    "db_insert_ms",
    "search_ms",
]

PROFILE_QUESTION = (
    "Question: According to the abstract, what specific type of 'framework' "
    "does this paper propose to support knowledge management and decision-making?"
)

# --- Retrieval benchmark config ---
BENCHMARK_ITERS = 20
BENCHMARK_QUERIES = [
    "Summarize the main topic.",
    "What are the key conclusions?",
    "List important definitions.",
]

# --- Embedding benchmark config ---
EMBED_BENCH_COUNT = 200
EMBED_BENCH_TEXT = "This is a test chunk for benchmarking embedding speed. " * 12


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_pdf_paths() -> list[Path]:
    pdf_dir = Path(PDF_DIR)
    return sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []


def embed_texts(texts: list[str]) -> list[list[float]]:
    prefixed = [f"passage: {t}" for t in texts]
    return rag_rust.embed_texts_rust(prefixed)


def retrieve(query: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
    prefixed_query = f"{BGE_QUERY_PREFIX}{query}"
    query_embedding = embed_texts([prefixed_query])[0]
    return rag_rust.lancedb_search(DB_DIR, TABLE_NAME, query_embedding, top_k)


# ---------------------------------------------------------------------------
# Profiling loop
# ---------------------------------------------------------------------------

def run_profiling(all_pdf_paths: list[str], run_llm: bool = False, chat_fn=None) -> None:
    """Run NUM_RUNS profiling iterations, writing per-run timings to CSV_FILE."""
    csv_path = Path(CSV_FILE)
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)

    profiling_context = ""

    for run in range(1, NUM_RUNS + 1):
        print(f"\n{'-' * 50}")
        print(f"Run {run}/{NUM_RUNS}")

        # 1) PDF read & extract
        t0 = time.perf_counter()
        pages = rag_rust.load_pdf_pages_many(all_pdf_paths)
        pages = [p for p in pages if p and p.strip()]
        pdf_read_ms = (time.perf_counter() - t0) * 1000
        print(f"  PDF Read & Extract:    {pdf_read_ms:.4f} ms  ({len(all_pdf_paths)} file(s), {len(pages)} pages)")

        # 2) Text chunking
        t0 = time.perf_counter()
        chunks: list[str] = []
        for page_text in pages:
            chunks.extend(
                c for c in rag_rust.smart_chunker(page_text, CHUNK_SIZE, CHUNK_OVERLAP) if c
            )
        chunking_ms = (time.perf_counter() - t0) * 1000
        print(f"  Text Chunking:         {chunking_ms:.4f} ms  ({len(chunks)} chunks)")

        # 3) Embedding
        t0 = time.perf_counter()
        embeddings = embed_texts(chunks)
        embedding_ms = (time.perf_counter() - t0) * 1000
        print(f"  Model/Embedding:       {embedding_ms:.4f} ms")

        # 4) DB insertion
        t0 = time.perf_counter()
        rag_rust.lancedb_create_or_open(DB_DIR, TABLE_NAME, chunks, embeddings, True)
        db_insert_ms = (time.perf_counter() - t0) * 1000
        print(f"  DB Init & Insertion:   {db_insert_ms:.4f} ms  ({len(chunks)} rows)")

        # 5) Vector search
        t0 = time.perf_counter()
        query_vec = embed_texts([f"{BGE_QUERY_PREFIX}{PROFILE_QUESTION}"])[0]
        results = rag_rust.lancedb_search(DB_DIR, TABLE_NAME, query_vec, TOP_K)
        profiling_context = "\n\n".join(text for text, _ in results)
        search_ms = (time.perf_counter() - t0) * 1000
        print(f"  Search:                {search_ms:.4f} ms")

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                run,
                f"{pdf_read_ms:.4f}",
                f"{chunking_ms:.4f}",
                f"{embedding_ms:.4f}",
                f"{db_insert_ms:.4f}",
                f"{search_ms:.4f}",
            ])

    if run_llm and chat_fn is not None:
        print(f"\n{'-' * 50}")
        print("Running LLM query after profiling (single call, not profiled)...")
        try:
            response_text, model_used = chat_fn([
                {
                    "role": "user",
                    "content": (
                        "Use the following context to answer the question.\n\n"
                        f"Context:\n{profiling_context}\n\nQuestion: {PROFILE_QUESTION}"
                    ),
                }
            ])
            print(f"\nAnswer: {response_text}")
            if model_used:
                print(f"Model used: {model_used}")
        except Exception as exc:
            print(f"LLM call failed: {exc}")

    print(f"\nProfiling complete. Results saved to '{CSV_FILE}'.")


# ---------------------------------------------------------------------------
# Retrieval benchmark
# ---------------------------------------------------------------------------

def run_benchmark() -> None:
    """Measure retrieval throughput over BENCHMARK_ITERS × BENCHMARK_QUERIES."""
    total_queries = BENCHMARK_ITERS * len(BENCHMARK_QUERIES)
    bench_start = time.perf_counter()
    for _ in range(BENCHMARK_ITERS):
        for q in BENCHMARK_QUERIES:
            retrieve(q)
    bench_total = time.perf_counter() - bench_start
    print(f"Benchmark: {bench_total * 1000:.2f}ms for {total_queries} queries")
    print(f"Avg per query: {bench_total / total_queries:.4f}s")


# ---------------------------------------------------------------------------
# Embedding benchmark
# ---------------------------------------------------------------------------

def run_embed_bench() -> None:
    """Measure raw embedding throughput (chunks/sec)."""
    test_texts = [EMBED_BENCH_TEXT] * EMBED_BENCH_COUNT
    t0 = time.perf_counter()
    _ = embed_texts(test_texts)
    dt = time.perf_counter() - t0
    print(
        f"Embed benchmark: {dt * 1000:.1f}ms for {EMBED_BENCH_COUNT} chunks "
        f"-> {EMBED_BENCH_COUNT / dt:.2f} chunks/sec"
    )


# ---------------------------------------------------------------------------
# Entry point – runs all three suites when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_pdf_paths = [str(p) for p in get_pdf_paths()]
    if not all_pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {PDF_DIR}")

    print(f"Found {len(all_pdf_paths)} PDF(s): {[Path(p).name for p in all_pdf_paths]}")
    print("Loading embed model...")
    rag_rust.load_embed_model()

    print("\n=== Embedding benchmark ===")
    run_embed_bench()

    print("\n=== Profiling loop ===")
    run_profiling(all_pdf_paths)

    print("\n=== Retrieval benchmark ===")
    run_benchmark()