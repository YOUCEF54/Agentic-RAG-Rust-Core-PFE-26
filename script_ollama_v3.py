"""Hybrid RAG (Python orchestrator + Rust core).

Features:
- PDF hash caching (skip re-embed if PDFs unchanged)
- Markdown-based semantic chunking via pdf_oxide
- Ollama chat pipeline
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List
from dataclasses import dataclass, field

import requests
from dotenv import load_dotenv

from agents import Evaluator, Generator, QueryRefiner, Retriever, UserProxy, DynamicPassageSelector
import rag_rust

# --- Platform/Env ---
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)

# --- Helpers ---

def get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got: {value!r}") from exc


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {value!r}") from exc

# --- Config ---
OLLAMA_TIMEOUT = get_env_int("OLLAMA_TIMEOUT", 300)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api")

# Chat Models
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "mistral")
#OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "phi4-mini:3.8b")

# Embedding Models
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME") or "zembed-1"
#EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME") or "BAAI/bge-small-en-v1.5"

CHAT_TEMPERATURE = get_env_float("CHAT_TEMPERATURE", 0.2)
TOP_K = get_env_int("TOP_K", 3)

# --- DPS Config ---
DPS_ENABLED     = True
TOP_N_RETRIEVAL = 15    # candidates fetched from vector DB
TOP_K_MAX       = 8     # hard ceiling — DPS can never select more than this
TOP_K_MIN       = 1     # hard floor
SELECTOR_MODEL  = os.getenv("SELECTOR_MODEL", "qwen2.5:3b")  # needs basic reasoning

DB_DIR = os.getenv("DB_DIR", "lancedb")
TABLE_NAME = "pdf_chunks"
REBUILD_DB = True

CACHE_FILE = ".rag_cache.json"

PDF_DIR = os.getenv("PDF_FOLDER", "pdfs")
PDF_PATHS: list[str] = []
MAX_PAGES = None
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_BATCH_SIZE = 4

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# --- PDF paths ---

def get_pdf_paths() -> list[Path]:
    if PDF_PATHS:
        return [Path(p) for p in PDF_PATHS]
    pdf_dir = Path(PDF_DIR)
    return sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []

# --- Cache ---

def compute_pdf_hash() -> str:
    paths = get_pdf_paths()
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    h.update(f"{CHUNK_SIZE}_{CHUNK_OVERLAP}_{EMBED_MODEL_NAME}".encode())
    return h.hexdigest()

def save_cache(pdf_hash: str, chunks: list[Chunk]) -> None:
    serialized = [
        {"text": c.text, "source": c.source, "page": c.page, "chunk_idx": c.chunk_idx}
        for c in chunks
    ]
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"hash": pdf_hash, "chunks": serialized}, f)

def load_cache() -> dict:
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Deserialize back to Chunk objects
        data["chunks"] = [
            Chunk(text=c["text"], source=c["source"], page=c["page"], chunk_idx=c["chunk_idx"])
            for c in data.get("chunks", [])
        ]
        return data
    except Exception:
        return {}

# --- Ollama ---

def ollama_post(path: str, payload: dict, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            url = f"{OLLAMA_BASE_URL}{path}"
            response = requests.post(
                url, json=payload, timeout=OLLAMA_TIMEOUT
            )
            if response.ok:
                return response.json()
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise RuntimeError(f"Ollama error {response.status_code}: {detail}")
        except (requests.exceptions.RequestException, RuntimeError) as e:
            if attempt < retries - 1:
                print(f"Ollama error/connection issue, retrying ({attempt+1}/{retries})... Error: {e}")
                time.sleep(3)
            else:
                raise
    return {}

def ollama_chat(model: str, messages: list[dict]) -> tuple[str, str | None]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": CHAT_TEMPERATURE}
    }
    data = ollama_post("/chat", payload)
    return data.get("message", {}).get("content", ""), data.get("model")

REFINER_MODEL   = os.getenv("REFINER_MODEL",   "qwen2.5:0.5b")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "qwen2.5:1.5b")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "qwen2.5:1.5b")

chat_refiner   = lambda msgs: ollama_chat(REFINER_MODEL,   msgs)
chat_generator = lambda msgs: ollama_chat(GENERATOR_MODEL, msgs)
chat_evaluator = lambda msgs: ollama_chat(EVALUATOR_MODEL, msgs)
chat_selector  = lambda msgs: ollama_chat(SELECTOR_MODEL,  msgs)

# --- Embedding ---

_EMBED_MODEL_READY = False

def ensure_embed_model_loaded() -> None:
    global _EMBED_MODEL_READY
    if not _EMBED_MODEL_READY:
        rag_rust.load_embed_model()
        _EMBED_MODEL_READY = True

# --- Active: z-embed Implementation ---
def embed_texts(texts: list[str], _batch_size) -> list[list[float]]:
    """Embed indexed passages via Rust core (z-embed does not require passage prefixes)."""
    ensure_embed_model_loaded()
    return rag_rust.embed_texts_rust(texts, _batch_size)

def embed_query(query: str) -> list[float]:
    """Embed query via Rust core (z-embed does not require query prefixes)."""
    ensure_embed_model_loaded()
    return rag_rust.embed_texts_rust([query], 1)[0]


# --- Commented: BGE Small Implementation ---
# BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
#
# def embed_texts_bge(texts: List[str]) -> List[List[float]]:
#     """Embed indexed passages via Rust fastembed (BGE expects 'passage:' prefix)."""
#     # refresh_hardware_config_if_needed(force=False)
#     ensure_embed_model_loaded()
#     prefixed_texts = [f"passage: {t}" for t in texts]
#     return rag_rust.embed_texts_rust(prefixed_texts, EMBED_BATCH_SIZE)
#
# def embed_query_bge(query: str) -> List[float]:
#     """Embed query via Rust fastembed with BGE query prefix."""
#     ensure_embed_model_loaded()
#     prefixed_query = f"{BGE_QUERY_PREFIX}{query}"
#     return rag_rust.embed_texts_rust([prefixed_query], 1)[0]


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_idx: int
    char_len: int = field(init=False)
    def __post_init__(self): self.char_len = len(self.text)

def load_and_chunk_pdfium(paths: list[str]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for pdf_path in paths:
        source_name = Path(pdf_path).name
        pages = rag_rust.load_pdf_pages_pdfium_many([pdf_path])
        print(f"  {source_name}: {len(pages)} pages")
        for page_idx, text in enumerate(pages, start=1):
            if not text or not text.strip():
                continue
            page_chunks = rag_rust.sliding_window_chunker(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for chunk_idx, chunk_text in enumerate(page_chunks):
                if len(chunk_text.strip()) > 30:
                    all_chunks.append(Chunk(
                        text=chunk_text,
                        source=source_name,
                        page=page_idx,
                        chunk_idx=chunk_idx,
                    ))
    return all_chunks

def build_or_open_table(chunks: list[Chunk], needs_rebuild: bool) -> None:
    if not needs_rebuild:
        print("DB is up-to-date, skipping.")
        return

    texts   = [c.text   for c in chunks]
    sources = [c.source for c in chunks]
    pages   = [c.page   for c in chunks]

    print(f"Embedding {len(texts)} chunks...")
    t0 = time.perf_counter()
    embeddings = embed_texts(texts, 4)
    embed_time = time.perf_counter() - t0
    print(f"Embedding time: {embed_time*1000:.2f}ms ({len(texts)/embed_time:.2f} chunks/s)")

    # Construct the vector DB after generating embeddings
    rag_rust.lancedb_create_or_open(
        DB_DIR, TABLE_NAME, texts, sources, pages, embeddings, True
    )

# --- Retrieval ---

# retrieve now returns Chunk-like tuples with provenance
def retrieve(query: str, top_k: int = TOP_K, source_filter: str | None = None):
    qvec = embed_query(query)   
    if source_filter:
        return rag_rust.lancedb_search_filtered(DB_DIR, TABLE_NAME, qvec, top_k, source_filter)
    return rag_rust.lancedb_search(DB_DIR, TABLE_NAME, qvec, top_k)

# --- Logging ---

def log_run_info() -> None:
    print("=== Run Info ===")
    print(f"Chat model   : {OLLAMA_CHAT_MODEL}")
    print(f"Embed model  : {EMBED_MODEL_NAME}")
    print(f"Chunking     : PDFium + Sliding Window")
    print("Embed engine : ZeroEntropy API (Rust)")
    print("================")

# --- Main ---

if __name__ == "__main__":
    all_pdf_paths = [str(p) for p in get_pdf_paths()]
    if not all_pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {PDF_DIR}")

    print(f"Found {len(all_pdf_paths)} PDF(s): {[Path(p).name for p in all_pdf_paths]}")
    print("Loading embed model (once)...")
    ensure_embed_model_loaded()

    current_hash = compute_pdf_hash()
    cache = load_cache()
    db_exists = Path(DB_DIR).exists()
    needs_rebuild = REBUILD_DB or not db_exists or cache.get("hash") != current_hash

    if needs_rebuild:
        t0 = time.perf_counter()
        dataset = load_and_chunk_pdfium(all_pdf_paths)
        print(f"Loaded+chunked into {len(dataset)} chunks in {(time.perf_counter()-t0)*1000:.2f}ms")
        if dataset:
            avg_len = sum(c.char_len for c in dataset) / len(dataset)
            print(f"Avg chunk length: {avg_len:.1f} chars")

        save_cache(current_hash, dataset)
    else:
        dataset = cache.get("chunks", [])
        print(f"Cache hit: {len(dataset)} chunks loaded instantly, skipping PDF+embed.")

    t0 = time.perf_counter()
    build_or_open_table(dataset, needs_rebuild)
    print(f"DB build/open time: {(time.perf_counter()-t0)*1000:.2f}ms")

    log_run_info()

    input_query = input("Ask your question...: ") or "What is this document about?"

    state = {
        "query": input_query,
        "answer": "",
        "score": 0.0,
        "attempts": 0,
        "chunks": [],
        "should_retry": False,
        "model_used": "",
        "refined_query": "",
        "trace": []
    }

    print("\n=== Agent Traces (Live) ===", flush=True)
    def terminal_emit(item):
        agent_name = item.get("agent", "Unknown")
        msg = item.get("message", "")
        data = item.get("data")
        print(f"[{agent_name}] {msg}", flush=True)
        if data:
            print(f"    Data: {data}", flush=True)
    
    state["emit"] = terminal_emit

    proxy = UserProxy(
        refiner=QueryRefiner(chat_fn=chat_refiner),
        retriever=Retriever(retrieve_fn=retrieve, top_k=TOP_K, top_n=TOP_N_RETRIEVAL),
        selector=DynamicPassageSelector(
            chat_fn=chat_selector,
            max_passages=TOP_K_MAX,
            min_passages=TOP_K_MIN,
        ) if DPS_ENABLED else None,
        generator=Generator(chat_fn=chat_generator),
        evaluator=Evaluator(chat_fn=chat_evaluator, min_score=0.75),
    )
    state = proxy.run(state)

    print("\n=== Retriever Agent ===")
    print("Retrieved chunks:")
    print(f"\nDPS selected {len(state['chunks'])} of {state.get('dps_n_candidates', '?')} candidates")
    print(f"Selected indices: {state.get('dps_selected_indices', [])}")
    for i, (text, source, page, dist) in enumerate(state["chunks"]):
        print(f"{i+1} - [{source} p.{page}] dist={dist:.4f}: {text[:120]}...")

    print("================")
    print(f"\nChatbot response:\n{state['answer']}")
    print(f"Score: {state['score']} | Attempts: {state['attempts']}")
    print(f"score: {state['score']} | retry: {state['should_retry']}")