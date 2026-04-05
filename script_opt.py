"""Hybrid RAG — Rust modules called from Python.

Original functionality preserved:
  - PDF hash caching (skip re-embed if PDFs unchanged)
  - Benchmark mode (BENCHMARK_ITERS x BENCHMARK_QUERIES)
  - LLM chat via OpenRouter (RUN_LLM flag)
  - log_run_info, all config flags

Optimizations added vs previous version:
  - embed_texts_rust() replaces SentenceTransformer — uses fastembed ONNX
    runtime in Rust (now with BGE Small EN v1.5)
  - lancedb_search no longer calls open_table on every run — Table handle
    is now cached in lib.rs via a global OnceCell, invalidated on each insert
  - RUN_PROFILING flag (default True) to toggle the 100-run loop
    CSV columns: run, pdf_read_ms, chunking_ms, model_embedding_ms,
                 db_insert_ms, search_ms
"""
import os
import sys
import io
import time
import csv
import hashlib
import json
from pathlib import Path
import torch
import requests
import rag_rust

from agents import Generator, Evaluator, QueryRefiner, Retriever, UserProxy
os.environ["OMP_NUM_THREADS"] = "8"        # i7-6700HQ has 8 threads
os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
os.environ["ONNXRUNTIME_FLAGS"] = "0"

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── Config ─────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY      = os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")
OPENROUTER_BASE_URL     = "https://openrouter.ai/api/v1"
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_TITLE        = os.getenv("OPENROUTER_TITLE", "Agentic-RAG-Rust-Core-PFE-26")
OPENROUTER_TIMEOUT      = 60
OPENROUTER_CHAT_MODEL   = os.getenv("OPENROUTER_CHAT_MODEL") or os.getenv("MODEL", "openrouter/free")
EMBED_MODEL_NAME        = os.getenv("EMBED_MODEL_NAME")
CHAT_TEMPERATURE        = float(os.getenv("CHAT_TEMPERATURE"))
TOP_K                   = 3

# Storage settings
DB_DIR     = os.getenv("DB_DIR", "lancedb")
TABLE_NAME = "pdf_chunks"
REBUILD_DB = False

# Cache settings
CACHE_FILE = ".rag_cache.json"

# PDF ingestion settings
PDF_DIR       = os.getenv("PDF_FOLDER", "pdfs")
PDF_PATHS     = []
MAX_PAGES     = None
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100

# Runtime options
RUN_LLM         = True
RUN_BENCHMARK   = False
BENCHMARK_ITERS = 20
BENCHMARK_QUERIES = [
    "Summarize the main topic.",
    "What are the key conclusions?",
    "List important definitions.",
]

# ── Profiling options ──────────────────────────────────────────────────────────
RUN_PROFILING = False
NUM_RUNS      = 100
CSV_FILE      = "profile_results.csv"
CSV_HEADER    = ["run", "pdf_read_ms", "chunking_ms", "model_embedding_ms",
                 "db_insert_ms", "search_ms"]

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

QUESTION = (
    "Question: According to the abstract, what specific type of 'framework' "
    "does this paper propose to support knowledge management and decision-making?"
)

start_time = time.perf_counter()


# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_pdf_hash() -> str:
    paths = []
    if PDF_PATHS:
        paths = [Path(p) for p in PDF_PATHS]
    else:
        pdf_dir = Path(PDF_DIR)
        if pdf_dir.exists():
            paths = sorted(pdf_dir.glob("*.pdf"))

    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    h.update(f"{CHUNK_SIZE}_{CHUNK_OVERLAP}_{EMBED_MODEL_NAME}".encode())
    return h.hexdigest()


def load_cache() -> dict:
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(pdf_hash: str, chunks: list) -> None:
    with open(CACHE_FILE, "w") as f:
        json.dump({"hash": pdf_hash, "chunks": chunks}, f)


def openrouter_headers():
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is not set.")
    key = key.strip().replace('"', "").replace("'", "")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_HTTP_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
    if OPENROUTER_TITLE:
        headers["X-Title"] = OPENROUTER_TITLE
    return headers

def openrouter_post(path, payload, retries=3):
    for attempt in range(retries):
        try:
            url = f"{OPENROUTER_BASE_URL}{path}"
            response = requests.post(
                url, json=payload, headers=openrouter_headers(), timeout=OPENROUTER_TIMEOUT
            )

            if not response.ok:
                try:
                    detail = response.json()
                except Exception:
                    detail = response.text
                if response.status_code == 401:
                    raise ValueError("OpenRouter auth failed (401).")
                raise RuntimeError(f"OpenRouter error {response.status_code}: {detail}")
        except requests.exceptions.ConnectionError:
                if attempt < retries - 1:
                    print(f"Connection error, retrying ({attempt+1}/{retries})...")
                    time.sleep(2)
                else:
                    raise
    return response.json()


def embed_texts(texts: list) -> list:
    """Embed via fastembed ONNX runtime in Rust — same engine as rust_only."""
    return rag_rust.embed_texts_rust(texts)


def chat_complete(messages):
    data = openrouter_post(
        "/chat/completions",
        {"model": OPENROUTER_CHAT_MODEL, "messages": messages, "temperature": CHAT_TEMPERATURE},
    )
    content = data["choices"][0]["message"]["content"]
    model_used = data.get("model") or data.get("choices", [{}])[0].get("model")
    return content, model_used


def load_pdf_texts() -> list:
    paths = []
    if PDF_PATHS:
        paths = [Path(p) for p in PDF_PATHS]
    else:
        pdf_dir = Path(PDF_DIR)
        if pdf_dir.exists():
            paths = list(pdf_dir.glob("*.pdf"))
    if not paths:
        raise FileNotFoundError("No PDFs found.")
    try:
        pages = rag_rust.load_pdf_pages_many([str(p) for p in paths])
    except Exception as exc:
        raise RuntimeError("Failed to load PDFs via Rust.") from exc
    if MAX_PAGES is not None:
        pages = pages[:MAX_PAGES]
    return [t for t in pages if t and t.strip()]


def chunk_texts(pages: list) -> list:
    chunks = []
    for page_text in pages:
        if page_text and page_text.strip():
            chunks.extend(
                c for c in rag_rust.smart_chunker(page_text, CHUNK_SIZE, CHUNK_OVERLAP) if c
            )
    return chunks


def log_run_info():
    print("=== Run Info ===")
    print(f"Chat model   : {OPENROUTER_CHAT_MODEL}")
    print(f"Embed model  : {EMBED_MODEL_NAME}")
    print(f"Chunk size   : {CHUNK_SIZE} | overlap: {CHUNK_OVERLAP}")
    print("Embed engine : fastembed ONNX (Rust)")
    print("================")


def build_or_open_table(chunks: list, needs_rebuild: bool) -> None:
    if not needs_rebuild:
        print("DB is up-to-date, skipping embed + insert.")
        return

    embed_start = time.perf_counter()
    embeddings = embed_texts(chunks)
    embed_time = time.perf_counter() - embed_start
    print(f"Embedding time: {embed_time*1000:.2f}ms ({len(chunks)/embed_time:.2f} chunks/s)")

    rag_rust.lancedb_create_or_open(DB_DIR, TABLE_NAME, chunks, embeddings, True)


def retrieve(query: str, top_k: int = TOP_K):
    prefixed_query = f"{BGE_QUERY_PREFIX}{query}"
    query_embedding = embed_texts([prefixed_query])[0]
    return rag_rust.lancedb_search(DB_DIR, TABLE_NAME, query_embedding, top_k)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # torch.set_num_threads(os.cpu_count())

    # Collect PDF paths — used by both profiling loop and original pipeline
    if PDF_PATHS:
        all_pdf_paths = sorted(str(Path(p)) for p in PDF_PATHS)
    else:
        all_pdf_paths = sorted(str(p) for p in Path(PDF_DIR).glob("*.pdf"))

    if not all_pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {PDF_DIR}")

    print(f"Found {len(all_pdf_paths)} PDF(s): {[Path(p).name for p in all_pdf_paths]}")

    # Pre-load fastembed model once (triggers OnceCell in Rust, mirrors rust_only)
    print("Loading fastembed model (once)...")

    rag_rust.load_embed_model()
    

    # ══════════════════════════════════════════════════════════════════════════
    # PROFILING LOOP — 100 runs, CSV format identical to rust_only / python_only
    # ══════════════════════════════════════════════════════════════════════════
    if RUN_PROFILING:
        csv_path = Path(CSV_FILE)
        if not csv_path.exists():
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(CSV_HEADER)

        profiling_context = ""

        for run in range(1, NUM_RUNS + 1):
            print(f"\n{'─' * 50}")
            print(f"Run {run}/{NUM_RUNS}")

            # 1. PDF read & extract — Rust Rayon (par_iter over files + pages) ─
            t0 = time.perf_counter()
            pages = rag_rust.load_pdf_pages_many(all_pdf_paths)
            pages = [p for p in pages if p and p.strip()]
            pdf_read_ms = (time.perf_counter() - t0) * 1000
            print(f"  PDF Read & Extract:    {pdf_read_ms:.4f} ms  ({len(all_pdf_paths)} file(s), {len(pages)} pages)")

            # 2. Text chunking — Rust smart_chunker ───────────────────────────
            t0 = time.perf_counter()
            chunks = []
            for page_text in pages:
                chunks.extend(
                    c for c in rag_rust.smart_chunker(page_text, CHUNK_SIZE, CHUNK_OVERLAP) if c
                )
            chunking_ms = (time.perf_counter() - t0) * 1000
            print(f"  Text Chunking:         {chunking_ms:.4f} ms  ({len(chunks)} chunks)")

            # 3. Embedding — fastembed ONNX via Rust (same engine as rust_only) ─
            t0 = time.perf_counter()
            embeddings = embed_texts(chunks)
            embedding_ms = (time.perf_counter() - t0) * 1000
            print(f"  Model/Embedding:       {embedding_ms:.4f} ms")

            # 4. DB insertion — Rust LanceDB bindings (table cache invalidated) ─
            t0 = time.perf_counter()
            rag_rust.lancedb_create_or_open(DB_DIR, TABLE_NAME, chunks, embeddings, True)
            db_insert_ms = (time.perf_counter() - t0) * 1000
            print(f"  DB Init & Insertion:   {db_insert_ms:.4f} ms  ({len(chunks)} rows)")

            # 5. Vector search — Rust LanceDB (cached Table handle, no open_table)
            t0 = time.perf_counter()
            query_vec = embed_texts([f"{BGE_QUERY_PREFIX}{QUESTION}"])[0]
            
            results = rag_rust.lancedb_search(DB_DIR, TABLE_NAME, query_vec, TOP_K)
            profiling_context = "\n\n".join(text for text, _ in results)
            search_ms = (time.perf_counter() - t0) * 1000
            print(f"  Search:                {search_ms:.4f} ms")

            # Append to CSV
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    run,
                    f"{pdf_read_ms:.4f}",
                    f"{chunking_ms:.4f}",
                    f"{embedding_ms:.4f}",
                    f"{db_insert_ms:.4f}",
                    f"{search_ms:.4f}",
                ])

        # Single LLM call after profiling (not profiled, mirrors other impls)
        print(f"\n{'─' * 50}")
        print("Running LLM query after profiling (single call, not profiled)...")
        try:
            response_text, model_used = chat_complete([
                {"role": "user", "content": (
                    f"Use the following context to answer the question.\n\n"
                    f"Context:\n{profiling_context}\n\nQuestion: {QUESTION}"
                )}
            ])
            print(f"\nAnswer: {response_text}")
            if model_used:
                print(f"Model used: {model_used}")
        except Exception as exc:
            print(f"LLM call failed: {exc}")

        print(f"\nProfiling complete. Results saved to '{CSV_FILE}'.")

    # ══════════════════════════════════════════════════════════════════════════
    # ORIGINAL PIPELINE (hash-cached, full features)
    # ══════════════════════════════════════════════════════════════════════════

    # 1. Hash PDFs to detect changes
    current_hash = compute_pdf_hash()
    cache = load_cache()
    db_exists = Path(DB_DIR).exists()
    needs_rebuild = REBUILD_DB or not db_exists or cache.get("hash") != current_hash

    # 2. Load + chunk
    if needs_rebuild:
        pdf_start = time.perf_counter()
        pages = load_pdf_texts()
        print(f"Loaded {len(pages)} pages in {(time.perf_counter()-pdf_start)*1000:.2f}ms")

        chunk_start = time.perf_counter()
        dataset = chunk_texts(pages)
        print(f"Chunked into {len(dataset)} chunks in {(time.perf_counter()-chunk_start)*1000:.2f}ms")
        if dataset:
            avg_len = sum(len(c) for c in dataset) / len(dataset)
            print(f"Avg chunk length: {avg_len:.1f} chars")

        save_cache(current_hash, dataset)
    else:
        dataset = cache["chunks"]
        print(f"Cache hit: {len(dataset)} chunks loaded instantly, skipping PDF+embed.")

    # 3. Embed + insert only when needed
    build_start = time.perf_counter()
    build_or_open_table(dataset, needs_rebuild)
    print(f"DB build/open time: {(time.perf_counter()-build_start)*1000:.2f}ms")

    log_run_info()

    # 4. Query
    # input_query = "what is Samyama?"
    input_query = input("Ask your question...: ")

    if not input_query:
        input_query = "What is this document about?"

    state = {
        "query" : input_query,
        "answer" : "",
        "score" : 0.0,
        "attempts" : 0,
        "chunks" : [],
        "should_retry" : False,
        "model_used" : "",
        "refined_query" : "" 
        }

    # refiner = QueryRefiner(chat_complete)
    # state = refiner.run(state)

    # query_start = time.perf_counter()
    # retriever_agent = Retriever(retrieve)
    # state = retriever_agent.run(state)
    # retrieved_knowledge = state["chunks"]
    # print(f"Retrieval time: {(time.perf_counter()-query_start)*1000:.2f}ms")

    # print("Retrieved knowledge:")
    # for text, distance in retrieved_knowledge:
    #     print(f" - (distance: {distance:.4f}) {text}")


    # state['chunks'] = retrieved_knowledge
    # generator_agent = Generator(chat_complete)
    # state = generator_agent.run(state)

    # print(f"\nChatbot response:\n{state['answer']}")
    # if state['model_used']:
    #     print(f"Chat model used: {state['model_used']}")

    # evaluator_agent = Evaluator(chat_fn=chat_complete, min_score=0.75)
    # state = evaluator_agent.run(state)
    proxy = UserProxy(
    refiner=QueryRefiner(chat_fn=chat_complete),
    retriever=Retriever(retrieve_fn=retrieve, top_k=int(os.getenv("TOP_K"))),
    generator=Generator(chat_fn=chat_complete),
    evaluator=Evaluator(chat_fn=chat_complete, min_score=0.75),
    )
    state = proxy.run(state)

    print(f"\nChatbot response:\n{state['answer']}")
    print(f"Score: {state['score']} | Attempts: {state['attempts']}")
    print(f"score: {state['score']}| retry: {state['should_retry']}")
    # # 5. LLM
    # if RUN_LLM:
    #     instruction_prompt = (
    #         "You are a helpful chatbot.\n"
    #         "Use only the following pieces of context to answer the question. "
    #         "Don't make up any new information:\n"
    #         f"{context_text}\n"
    #     )
    #     llm_start = time.perf_counter()
    #     response_text, model_used = chat_complete([
    #         {"role": "system", "content": instruction_prompt},
    #         {"role": "user", "content": input_query},
    #     ])
    #     print(f"\nChatbot response:\n{response_text}")
    #     if model_used:
    #         print(f"Chat model used: {model_used}")
    #     print(f"LLM time: {(time.perf_counter()-llm_start)*1000:.2f}ms")

    # print(f"\nExecution time: {(time.perf_counter()-start_time)*1000:.2f}ms")

    # 6. Benchmark
    if RUN_BENCHMARK:
        total_queries = BENCHMARK_ITERS * len(BENCHMARK_QUERIES)
        bench_start = time.perf_counter()
        for _ in range(BENCHMARK_ITERS):
            for q in BENCHMARK_QUERIES:
                retrieve(q)
        bench_total = time.perf_counter() - bench_start
        print(f"Benchmark: {bench_total*1000:.2f}ms for {total_queries} queries")
        print(f"Avg per query: {bench_total/total_queries:.4f}s")
