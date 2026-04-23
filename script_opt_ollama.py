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
from typing import Iterable
from dataclasses import dataclass, field # added

import requests
from dotenv import load_dotenv

from agents import Evaluator, Generator, QueryRefiner, Retriever, UserProxy, DynamicPassageSelector # added new agent (DynamicPassageSelector) 

import rag_rust

# FIXED — return Chunk objects

@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_idx: int
    char_len: int = field(init=False)
    def __post_init__(self): self.char_len = len(self.text)


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
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api")
OLLAMA_TIMEOUT = get_env_int("OLLAMA_TIMEOUT", 300)
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "mistral:7b")
EMBED_MODE = os.getenv("EMBED_MODE")
if EMBED_MODE :
    EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME") or "zembed-1"
else:
    EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME") or "BAAI/bge-small-en-v1.5"

CHAT_TEMPERATURE = get_env_float("CHAT_TEMPERATURE", 0.2)
TOP_K = get_env_int("TOP_K", 3)

# added
# --- DPS Config ---
DPS_ENABLED     = True
TOP_N_RETRIEVAL = 15    # candidates fetched from vector DB
TOP_K_MAX       = 8     # hard ceiling — DPS can never select more than this
TOP_K_MIN       = 1     # hard floor
SELECTOR_MODEL  = os.getenv("SELECTOR_MODEL", "qwen2.5:3b")  # needs basic reasoning


DB_DIR = os.getenv("DB_DIR", "lancedb")
TABLE_NAME = "pdf_chunks"
REBUILD_DB = False

CACHE_FILE = ".rag_cache.json"

PDF_DIR = os.getenv("PDF_FOLDER", "pdfs")
PDF_PATHS: list[str] = []
MAX_PAGES = None
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_BATCH_SIZE = 4

# added
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


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


def load_cache() -> list[str]:
    """Load cached processed filenames (backward-compatible with legacy cache formats)."""
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    if isinstance(data, list):
        return [str(x) for x in data if str(x).strip()]

    if isinstance(data, dict):
        processed = data.get("processed_files", [])
        if isinstance(processed, list):
            return [str(x) for x in processed if str(x).strip()]

    return []
    
def save_cache(processed_files: list[str]) -> None:
    # Save the list of filenames we have already embedded
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"processed_files": sorted(set(processed_files))}, f, indent=2)



# def save_cache(pdf_hash: str, chunks: list[str]) -> None:
#     with open(CACHE_FILE, "w", encoding="utf-8") as f:
#         json.dump({"hash": pdf_hash, "chunks": chunks}, f)


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
            print("Test: (response) : ", response)
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise RuntimeError(f"Ollama error {response.status_code}: {detail}")
        except requests.exceptions.ConnectionError:
            if attempt < retries - 1:
                print(f"Connection error, retrying ({attempt+1}/{retries})...")
                time.sleep(2)
            else:
                raise
    return {}


# def chat_complete(messages: list[dict]) -> tuple[str, str | None]:
#     payload = {
#         "model": OLLAMA_CHAT_MODEL,
#         "messages": messages,
#         "stream": False,
#         "options": {
#             "temperature": CHAT_TEMPERATURE
#         }
#     }
#     data = ollama_post("/chat", payload)
#     content = data.get("message", {}).get("content", "")
#     model_used = data.get("model")
#     return content, model_used

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
SELECTOR_MODEL  = os.getenv("SELECTOR_MODEL",  "qwen2.5:3b")

chat_refiner   = lambda msgs: ollama_chat(REFINER_MODEL,   msgs)
chat_generator = lambda msgs: ollama_chat(GENERATOR_MODEL, msgs)
chat_evaluator = lambda msgs: ollama_chat(EVALUATOR_MODEL, msgs)
chat_selector  = lambda msgs: ollama_chat(SELECTOR_MODEL,  msgs)

# --- Embedding ---

## zembed
def embed_texts_zembed(texts: list[str]) -> list[list[float]]:
    return rag_rust.embed_texts_rust_zembed(texts, EMBED_BATCH_SIZE)

## local
def embed_passages_local(texts: list[str]) -> list[list[float]]:
    """For indexing — uses passage: prefix."""
    prefixed = [f"passage: {t}" for t in texts]
    return rag_rust.embed_texts_rust_local(prefixed, EMBED_BATCH_SIZE)

def embed_query_local(text: str) -> list[float]:
    """For retrieval — uses BGE query prefix, no passage: prefix."""
    prefixed = f"{BGE_QUERY_PREFIX}{text}"
    return rag_rust.embed_texts_rust_local([prefixed], EMBED_BATCH_SIZE)[0]

# --- Chunking (PDFium + Sliding Window) ---
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

# --- DB ---

# def build_or_open_table(chunks: list[str], needs_rebuild: bool, embed_mode : bool = os.getenv("EMBED_MODE")) -> None:
#     if not needs_rebuild:
#         print("DB is up-to-date, skipping.")
#         return

#     texts   = [c.text   for c in chunks]
#     sources = [c.source for c in chunks]
#     pages   = [c.page   for c in chunks]

#     print(f"Embedding {len(chunks)} chunks...")

#     t0 = time.perf_counter()
#     if embed_mode :
#         embeddings = embed_texts_zembed(texts)
#     else :
#         embeddings = embed_passages_local(texts)
#     embed_time = time.perf_counter() - t0
#     print(f"Embedding time: {embed_time*1000:.2f}ms ({len(chunks)/embed_time:.2f} chunks/s)")

#     rag_rust.lancedb_create_or_open(
#         DB_DIR, TABLE_NAME, texts, sources, pages, embeddings, True
#     )

def build_or_open_table(chunks: list[Chunk], overwrite: bool) -> None:
    if not chunks:
        return

    texts   = [c.text   for c in chunks]
    sources = [c.source for c in chunks]
    pages   = [c.page   for c in chunks]

    print(f"Embedding {len(texts)} chunks...")
    t0 = time.perf_counter()
    if EMBED_MODE :
        embeddings = embed_texts_zembed(texts)
    else :
        embeddings = embed_passages_local(texts)
    embed_time = time.perf_counter() - t0
    print(f"Embedding time: {embed_time*1000:.2f}ms ({len(texts)/embed_time:.2f} chunks/s)")

    # Pass the overwrite flag down to Rust
    rag_rust.lancedb_create_or_open(
        DB_DIR, TABLE_NAME, texts, sources, pages, embeddings, overwrite
    )


# --- Retrieval ---

def retrieve(query: str, top_k: int = TOP_K, source_filter: str | None = None):
    qvec = embed_query_local(query)   
    if source_filter:
        return rag_rust.lancedb_search_filtered(DB_DIR, TABLE_NAME, qvec, top_k, source_filter)
    return rag_rust.lancedb_search(DB_DIR, TABLE_NAME, qvec, top_k)


# --- Logging ---

def log_run_info() -> None:
    print("=== Run Info ===")
    print(f"Chat model   : {OLLAMA_CHAT_MODEL}")
    print(f"Embed model  : {EMBED_MODEL_NAME}")
    print(f"Chunking     : PDFium + Sliding Window")
    print(f"Embed engine : {'ZeroEntropy API (Rust)' if EMBED_MODE else 'ONNX (Rust)'}")
    print("================")


# --- Main ---


if __name__ == "__main__":
    all_pdf_paths = [str(p) for p in get_pdf_paths()]
    if not all_pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {PDF_DIR}")

    print(f"Found {len(all_pdf_paths)} PDF(s): {[Path(p).name for p in all_pdf_paths]}")
    print("Loading embed model (once)...")
    rag_rust.load_embed_model_zembed() if EMBED_MODE else rag_rust.load_embed_model_local()

    current_hash = compute_pdf_hash()
    processed_files = load_cache()
    db_exists = Path(DB_DIR).exists()
    # needs_rebuild = REBUILD_DB or not db_exists or cache.get("hash") != current_hash

    # if needs_rebuild:
    #     t0 = time.perf_counter()
    #     dataset = load_and_chunk_pdfium(all_pdf_paths)
    #     print(f"Loaded+chunked into {len(dataset)} chunks in {(time.perf_counter()-t0)*1000:.2f}ms")
    #     if dataset:
    #         avg_len = sum(len(c) for c in dataset) / len(dataset)
    #         print(f"Avg chunk length: {avg_len:.1f} chars")

    #     save_cache(current_hash, dataset)
    # else:
    #     dataset = cache.get("chunks", [])
    #     print(f"Cache hit: {len(dataset)} chunks loaded instantly, skipping PDF+embed.")
        # Step 1: Determine what needs processing
    if REBUILD_DB or not db_exists:
        print("Force rebuild requested or DB missing. Processing ALL files...")
        needs_process = all_pdf_paths
        overwrite_db = True
        processed_files = [] # Clear the cache memory
    else:
        # Step 2: Filter out files we've already done
        needs_process = [p for p in all_pdf_paths if Path(p).name not in processed_files]
        overwrite_db = False # We will APPEND

    
    # Step 3: Process and embed only what is necessary
    if needs_process:
        print(f"Processing {len(needs_process)} NEW PDF(s)...")
        t0 = time.perf_counter()
        dataset = load_and_chunk_pdfium(needs_process)
        print(f"Loaded+chunked into {len(dataset)} chunks in {(time.perf_counter()-t0)*1000:.2f}ms")
        
        if dataset:
            avg_len = sum(c.char_len for c in dataset) / len(dataset)
            print(f"Avg chunk length: {avg_len:.1f} chars")
            
            build_or_open_table(dataset, overwrite=overwrite_db)
            
        # Update our tracking list and save to disk
        processed_files.extend([Path(p).name for p in needs_process])
        save_cache(list(set(processed_files))) # set() ensures no accidental duplicates
    else:
        print("DB is up-to-date. All PDFs are already embedded, skipping.")

    log_run_info()

    # t0 = time.perf_counter()
    # build_or_open_table(dataset, needs_rebuild)
    # print(f"DB build/open time: {(time.perf_counter()-t0)*1000:.2f}ms")

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

    print("Retrieved chunks:")
    print(f"\nDPS selected {len(state['chunks'])} of {state.get('dps_n_candidates', '?')} candidates")
    print(f"Selected indices: {state.get('dps_selected_indices', [])}")
    for i, (text, source, page, dist) in enumerate(state["chunks"]):
        print(f"{i+1} - [{source} p.{page}] dist={dist:.4f}: {text[:120]}...")

    print("================")
    print(f"\nChatbot response:\n{state['answer']}")
    print(f"Score: {state['score']} | Attempts: {state['attempts']}")
    print(f"score: {state['score']} | retry: {state['should_retry']}")
