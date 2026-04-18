"""Hybrid RAG (Python orchestrator + Rust core).

Features:
- PDF hash caching (skip re-embed if PDFs unchanged)
- Markdown-based semantic chunking via pdf_oxide
- OpenRouter chat pipeline
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

import requests
from dotenv import load_dotenv

from agents import Evaluator, Generator, QueryRefiner, Retriever, UserProxy
import rag_rust

# --- Platform/Env ---
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", "Agentic-RAG-Rust-Core-PFE-26")
OPENROUTER_TIMEOUT = 60
OPENROUTER_CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL") or os.getenv("MODEL", "openrouter/free")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME") or "BAAI/bge-small-en-v1.5"
CHAT_TEMPERATURE = get_env_float("CHAT_TEMPERATURE", 0.2)
TOP_K = get_env_int("TOP_K", 3)

DB_DIR = os.getenv("DB_DIR", "lancedb")
TABLE_NAME = "pdf_chunks"
REBUILD_DB = True

CACHE_FILE = ".rag_cache.json"

PDF_DIR = os.getenv("PDF_FOLDER", "pdfs")
PDF_PATHS: list[str] = []
MAX_PAGES = None
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 4

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


# --- PDF paths ---

def get_pdf_paths() -> list[Path]:
    if PDF_PATHS:
        return [Path(p) for p in PDF_PATHS]
    pdf_dir = Path(PDF_DIR)
    print("Test (paths): ", sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else [])
    return sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []


# --- Cache ---

def compute_pdf_hash() -> str:
    paths = get_pdf_paths()
    print("Test: (paths) : ", paths)
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    h.update(f"{CHUNK_SIZE}_{CHUNK_OVERLAP}_{EMBED_MODEL_NAME}".encode())
    return h.hexdigest()


def load_cache() -> dict:
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(pdf_hash: str, chunks: list[str]) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"hash": pdf_hash, "chunks": chunks}, f)


# --- OpenRouter ---

def openrouter_headers() -> dict:
    key = os.getenv("OPENROUTER_API_KEY")
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


def openrouter_post(path: str, payload: dict, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            url = f"{OPENROUTER_BASE_URL}{path}"
            response = requests.post(
                url, json=payload, headers=openrouter_headers(), timeout=OPENROUTER_TIMEOUT
            )
            if response.ok:
                return response.json()
            print("Test: (response) : ", response)
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
    return {}


def chat_complete(messages: list[dict]) -> tuple[str, str | None]:
    data = openrouter_post(
        "/chat/completions",
        {"model": OPENROUTER_CHAT_MODEL, "messages": messages, "temperature": CHAT_TEMPERATURE},
    )
    content = data["choices"][0]["message"]["content"]
    model_used = data.get("model") or data.get("choices", [{}])[0].get("model")
    return content, model_used


# --- Embedding ---

def embed_texts(texts: list[str]) -> list[list[float]]:
    prefixed = [f"passage: {t}" for t in texts]
    return rag_rust.embed_texts_rust(prefixed, EMBED_BATCH_SIZE)


# --- Chunking (PDFium + Sliding Window) ---
def load_and_chunk_pdfium(paths: list[str]) -> list[str]:
    pages_text = rag_rust.load_pdf_pages_pdfium_many(paths)
    print(f"Loaded {len(pages_text)} pages via PDFium")

    all_chunks: list[str] = []
    for text in pages_text:
        if text and text.strip():
            chunks = rag_rust.sliding_window_chunker(text, CHUNK_SIZE, CHUNK_OVERLAP)
            all_chunks.extend(chunks)

    return [c for c in all_chunks if len(c.strip()) > 30]

# --- DB ---

def build_or_open_table(chunks: list[str], needs_rebuild: bool) -> None:
    if not needs_rebuild:
        print("DB is up-to-date, skipping.")
        return

    print(f"Embedding {len(chunks)} chunks...")
    t0 = time.perf_counter()
    embeddings = embed_texts(chunks)
    embed_time = time.perf_counter() - t0
    print(f"Embedding time: {embed_time*1000:.2f}ms ({len(chunks)/embed_time:.2f} chunks/s)")

    rag_rust.lancedb_create_or_open(DB_DIR, TABLE_NAME, chunks, embeddings, True)


# --- Retrieval ---

def retrieve(query: str, top_k: int = TOP_K):
    prefixed_query = f"{BGE_QUERY_PREFIX}{query}"
    query_embedding = embed_texts([prefixed_query])[0]
    return rag_rust.lancedb_search(DB_DIR, TABLE_NAME, query_embedding, top_k)


# --- Logging ---

def log_run_info() -> None:
    print("=== Run Info ===")
    print(f"Chat model   : {OPENROUTER_CHAT_MODEL}")
    print(f"Embed model  : {EMBED_MODEL_NAME}")
    print(f"Chunking     : PDFium + Sliding Window")
    print("Embed engine : ONNX (Rust)")
    print("================")


# --- Main ---

if __name__ == "__main__":
    all_pdf_paths = [str(p) for p in get_pdf_paths()]
    if not all_pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {PDF_DIR}")

    print(f"Found {len(all_pdf_paths)} PDF(s): {[Path(p).name for p in all_pdf_paths]}")
    print("Loading embed model (once)...")
    rag_rust.load_embed_model()

    current_hash = compute_pdf_hash()
    cache = load_cache()
    db_exists = Path(DB_DIR).exists()
    needs_rebuild = REBUILD_DB or not db_exists or cache.get("hash") != current_hash

    if needs_rebuild:
        t0 = time.perf_counter()
        dataset = load_and_chunk_pdfium(all_pdf_paths)
        print(f"Loaded+chunked into {len(dataset)} chunks in {(time.perf_counter()-t0)*1000:.2f}ms")
        if dataset:
            avg_len = sum(len(c) for c in dataset) / len(dataset)
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
    }

    proxy = UserProxy(
        refiner=QueryRefiner(chat_fn=chat_complete),
        retriever=Retriever(retrieve_fn=retrieve, top_k=TOP_K),
        generator=Generator(chat_fn=chat_complete),
        evaluator=Evaluator(chat_fn=chat_complete, min_score=0.75),
    )
    state = proxy.run(state)

    print("=== Retriever Agent ===")
    print("Retrieved chunks:")
    for i, e in enumerate(state["chunks"]):
        print(f"{i+1} - chunk: {e[0]}, similarity d: {e[1]}")
    print("================")
    print(f"\nChatbot response:\n{state['answer']}")
    print(f"Score: {state['score']} | Attempts: {state['attempts']}")
    print(f"score: {state['score']} | retry: {state['should_retry']}")