"""Minimal RAG demo using PDFs from the local `pdfs` folder."""
import os
import time
import hashlib
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import rag_rust
import requests
import sys
import io
import torch

torch.set_num_threads(os.cpu_count())  # only matters on CPU, ignored on GPU

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", "Agentic-RAG-Rust-Core-PFE-26")
OPENROUTER_TIMEOUT = 60
OPENROUTER_CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "openrouter/free")
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHAT_TEMPERATURE = 0.2
EMBED_BATCH_SIZE = 64        # ← increased from 32
EMBED_NORMALIZE = True
TOP_K = 3

# Storage settings
DB_DIR = 'lancedb'
TABLE_NAME = 'pdf_chunks'
REBUILD_DB = False

# Cache settings
CACHE_FILE = '.rag_cache.json'  # ← stores hash + chunks

# PDF ingestion settings
PDF_DIR = 'pdfs'
PDF_PATHS = []
MAX_PAGES = None
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Runtime options
RUN_LLM = True
RUN_BENCHMARK = False
BENCHMARK_ITERS = 20
BENCHMARK_QUERIES = [
    "Summarize the main topic.",
    "What are the key conclusions?",
    "List important definitions.",
]

start_time = time.perf_counter()


# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_pdf_hash() -> str:
    """SHA-256 of all PDF files in the folder, combined."""
    paths = []
    if PDF_PATHS:
        paths = [Path(p) for p in PDF_PATHS]
    else:
        pdf_dir = Path(PDF_DIR)
        if pdf_dir.exists():
            paths = sorted(pdf_dir.glob('*.pdf'))  # sorted for determinism

    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    # Include config that affects chunking
    h.update(f"{CHUNK_SIZE}_{CHUNK_OVERLAP}_{EMBED_MODEL_NAME}".encode())
    return h.hexdigest()


def load_cache() -> dict:
    try:
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(pdf_hash: str, chunks: list[str]) -> None:
    with open(CACHE_FILE, 'w') as f:
        json.dump({'hash': pdf_hash, 'chunks': chunks}, f)


def openrouter_headers():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is not set.")
    key = key.strip().replace('"', '').replace("'", "")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_HTTP_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
    if OPENROUTER_TITLE:
        headers["X-Title"] = OPENROUTER_TITLE
    return headers


def openrouter_post(path, payload):
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
    return response.json()


_EMBEDDER = None

def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Embedding device: {device}")
        _EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    return _EMBEDDER

def embed_texts(texts):
    embedder = get_embedder()
    vectors = embedder.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=EMBED_NORMALIZE,
        show_progress_bar=len(texts) > 50,  # only show bar for large batches
    )
    return vectors.tolist()


def chat_complete(messages):
    data = openrouter_post(
        "/chat/completions",
        {"model": OPENROUTER_CHAT_MODEL, "messages": messages, "temperature": CHAT_TEMPERATURE},
    )
    content = data["choices"][0]["message"]["content"]
    model_used = data.get("model") or data.get("choices", [{}])[0].get("model")
    return content, model_used


def load_pdf_texts():
    paths = []
    if PDF_PATHS:
        paths = [Path(p) for p in PDF_PATHS]
    else:
        pdf_dir = Path(PDF_DIR)
        if pdf_dir.exists():
            paths = list(pdf_dir.glob('*.pdf'))
    if not paths:
        raise FileNotFoundError("No PDFs found.")
    try:
        pages = rag_rust.load_pdf_pages_many([str(p) for p in paths])
    except Exception as exc:
        raise RuntimeError("Failed to load PDFs via Rust.") from exc
    if MAX_PAGES is not None:
        pages = pages[:MAX_PAGES]
    return [t for t in pages if t and t.strip()]


def chunk_texts(pages: list[str]) -> list[str]:
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
    print(f"Batch size   : {EMBED_BATCH_SIZE}")
    print(f"Chunk size   : {CHUNK_SIZE} | overlap: {CHUNK_OVERLAP}")
    try:
        embedder = get_embedder()
        device = getattr(embedder, "device", None) or getattr(embedder, "_target_device", None)
        print(f"Device       : {device}")
        print(f"Embed dim    : {embedder.get_sentence_embedding_dimension()}")
    except Exception as exc:
        print(f"Embed info N/A: {exc}")
    print("================")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_or_open_table(chunks: list[str], needs_rebuild: bool) -> None:
    if not needs_rebuild:
        print("DB is up-to-date, skipping embed + insert.")
        return

    embed_start = time.perf_counter()
    embeddings = embed_texts(chunks)
    embed_time = time.perf_counter() - embed_start
    print(f'Embedding time: {embed_time*1000:.2f}ms ({len(chunks)/embed_time:.2f} chunks/s)')

    rag_rust.lancedb_create_or_open(DB_DIR, TABLE_NAME, chunks, embeddings, True)


def retrieve(query, top_n=3):
    query_embedding = embed_texts([query])[0]
    return rag_rust.lancedb_search(DB_DIR, TABLE_NAME, query_embedding, top_n)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Hash PDFs to detect changes
    current_hash = compute_pdf_hash()
    cache = load_cache()
    db_exists = Path(DB_DIR).exists()

    needs_rebuild = REBUILD_DB or not db_exists or cache.get('hash') != current_hash

    # 2. Load + chunk (fast Rust path, always needed for cache check)
    if needs_rebuild:
        pdf_start = time.perf_counter()
        pages = load_pdf_texts()
        print(f'Loaded {len(pages)} pages in {(time.perf_counter()-pdf_start)*1000:.2f}ms')

        chunk_start = time.perf_counter()
        dataset = chunk_texts(pages)
        print(f'Chunked into {len(dataset)} chunks in {(time.perf_counter()-chunk_start)*1000:.2f}ms')
        if dataset:
            avg_len = sum(len(c) for c in dataset) / len(dataset)
            print(f'Avg chunk length: {avg_len:.1f} chars')

        save_cache(current_hash, dataset)
    else:
        # Load chunks from cache — no PDF reading or chunking needed
        dataset = cache['chunks']
        print(f'Cache hit: {len(dataset)} chunks loaded instantly, skipping PDF+embed.')

    # 3. Embed + insert only when needed
    build_start = time.perf_counter()
    build_or_open_table(dataset, needs_rebuild)
    print(f'DB build/open time: {(time.perf_counter()-build_start)*1000:.2f}ms')

    log_run_info()

    # 4. Query
    input_query = "what is Samyama?"
    if not input_query:
        input_query = "What is this document about?"

    query_start = time.perf_counter()
    retrieved_knowledge = retrieve(input_query, top_n=TOP_K)
    print(f'Retrieval time: {(time.perf_counter()-query_start)*1000:.2f}ms')

    print('Retrieved knowledge:')
    for text, distance in retrieved_knowledge:
        print(f' - (distance: {distance:.4f}) {text}')

    context_text = '\n'.join([f' - {text}' for text, _ in retrieved_knowledge])

    # 5. LLM
    if RUN_LLM:
        instruction_prompt = (
            "You are a helpful chatbot.\n"
            "Use only the following pieces of context to answer the question. "
            "Don't make up any new information:\n"
            f"{context_text}\n"
        )
        llm_start = time.perf_counter()
        response_text, model_used = chat_complete([
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
        ])
        print(f'\nChatbot response:\n{response_text}')
        if model_used:
            print(f'Chat model used: {model_used}')
        print(f'LLM time: {(time.perf_counter()-llm_start)*1000:.2f}ms')

    print(f"\nExecution time: {(time.perf_counter()-start_time)*1000:.2f}ms")

    # 6. Benchmark
    if RUN_BENCHMARK:
        total_queries = BENCHMARK_ITERS * len(BENCHMARK_QUERIES)
        bench_start = time.perf_counter()
        for _ in range(BENCHMARK_ITERS):
            for q in BENCHMARK_QUERIES:
                retrieve(q)
        bench_total = time.perf_counter() - bench_start
        print(f'Benchmark: {bench_total*1000:.2f}ms for {total_queries} queries')
        print(f'Avg per query: {bench_total/total_queries:.4f}s')
