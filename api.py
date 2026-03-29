"""API server for the Rust/Python RAG stack.

Run:
  uvicorn api:app --reload
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import rag_rust

APP_NAME = "Agentic-RAG-Rust-Core-PFE-26"

# Storage / DB
PDF_DIR = Path("data/pdfs")
DB_DIR = "lancedb"
TABLE_NAME = "pdf_chunks"

# Embeddings
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_BATCH_SIZE = 32
EMBED_NORMALIZE = True

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", APP_NAME)
OPENROUTER_TIMEOUT = 60
OPENROUTER_CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "openrouter/free")
CHAT_TEMPERATURE = 0.2

app = FastAPI(title=APP_NAME)

_EMBEDDER: Optional[SentenceTransformer] = None
_INDEX_READY = False
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

# CORS (frontend integration)
_cors_allow_all = os.getenv("CORS_ALLOW_ALL", "false").lower() in {"1", "true", "yes"}
_cors_origins = os.getenv("CORS_ORIGINS", "")
if _cors_allow_all:
  allow_origins = ["*"]
else:
  allow_origins = [o.strip() for o in _cors_origins.split(",") if o.strip()] or [
    "http://localhost:3000",
    "https://agentic-rag-rust-core-frontend-pfe.vercel.app/",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
  ]

app.add_middleware(
  CORSMiddleware,
  allow_origins=allow_origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


class IndexRequest(BaseModel):
  rebuild: bool = True
  max_pages: Optional[int] = None


class QueryRequest(BaseModel):
  question: str
  top_k: int = 3
  chat_model: Optional[str] = None
  use_llm: bool = True


class QueryResponse(BaseModel):
  answer: Optional[str]
  model_used: Optional[str]
  retrieved: List[dict]


def ensure_dirs() -> None:
  PDF_DIR.mkdir(parents=True, exist_ok=True)


def get_embedder() -> SentenceTransformer:
  global _EMBEDDER
  if _EMBEDDER is None:
    if HF_TOKEN:
      _EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME, token=HF_TOKEN)
    else:
      _EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME)
  return _EMBEDDER


def embed_texts(texts: List[str]) -> List[List[float]]:
  embedder = get_embedder()
  vectors = embedder.encode(
    texts,
    batch_size=EMBED_BATCH_SIZE,
    normalize_embeddings=EMBED_NORMALIZE,
  )
  return vectors.tolist()


def load_pdf_texts(max_pages: Optional[int]) -> List[str]:
  paths = list(PDF_DIR.glob("*.pdf"))
  if not paths:
    raise HTTPException(status_code=400, detail="No PDFs found in data/pdfs.")

  try:
    pages = rag_rust.load_pdf_pages_many([str(p) for p in paths])
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Rust PDF loader failed: {exc}") from exc

  if max_pages is not None:
    pages = pages[:max_pages]

  return [text for text in pages if text and text.strip()]


def chunk_text(text: str) -> List[str]:
  if not text or not text.strip():
    return []
  return [
    chunk
    for chunk in rag_rust.smart_chunker(text, CHUNK_SIZE, CHUNK_OVERLAP)
    if chunk
  ]


def build_index(rebuild: bool, max_pages: Optional[int]) -> dict:
  global _INDEX_READY
  pages = load_pdf_texts(max_pages)

  chunks: List[str] = []
  for page_text in pages:
    chunks.extend(chunk_text(page_text))

  if not chunks:
    raise HTTPException(status_code=400, detail="No text extracted from PDFs.")

  embeddings = embed_texts(chunks)

  try:
    rag_rust.lancedb_create_or_open(
      DB_DIR,
      TABLE_NAME,
      chunks,
      embeddings,
      rebuild,
    )
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"LanceDB error: {exc}") from exc

  _INDEX_READY = True
  return {
    "pages": len(pages),
    "chunks": len(chunks),
    "rebuild": rebuild,
  }


def openrouter_headers() -> dict:
  if not OPENROUTER_API_KEY:
    raise HTTPException(status_code=400, detail="OPENROUTER_API_KEY is not set.")
  key = OPENROUTER_API_KEY.strip().replace('"', "").replace("'", "")
  headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json",
  }
  if OPENROUTER_HTTP_REFERER:
    headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
  if OPENROUTER_TITLE:
    headers["X-Title"] = OPENROUTER_TITLE
  return headers


def openrouter_chat(messages: List[dict], model_override: Optional[str]) -> tuple[str, Optional[str]]:
  payload = {
    "model": model_override or OPENROUTER_CHAT_MODEL,
    "messages": messages,
    "temperature": CHAT_TEMPERATURE,
  }
  resp = requests.post(
    f"{OPENROUTER_BASE_URL}/chat/completions",
    json=payload,
    headers=openrouter_headers(),
    timeout=OPENROUTER_TIMEOUT,
  )
  if not resp.ok:
    raise HTTPException(status_code=resp.status_code, detail=resp.text)
  data = resp.json()
  content = data["choices"][0]["message"]["content"]
  model_used = data.get("model") or data.get("choices", [{}])[0].get("model")
  return content, model_used


@app.get("/health")
def health():
  return {"status": "ok"}


@app.post("/documents")
def upload_documents(files: List[UploadFile] = File(...)):
  ensure_dirs()
  saved = []
  for f in files:
    if not f.filename.lower().endswith(".pdf"):
      raise HTTPException(status_code=400, detail=f"Not a PDF: {f.filename}")
    target = PDF_DIR / f.filename
    with target.open("wb") as out:
      out.write(f.file.read())
    saved.append(f.filename)
  return {"saved": saved}


@app.get("/documents")
def list_documents():
  ensure_dirs()
  return {"files": [p.name for p in PDF_DIR.glob("*.pdf")]}


@app.post("/index")
def build_index_endpoint(payload: IndexRequest):
  start = time.perf_counter()
  stats = build_index(payload.rebuild, payload.max_pages)
  end = time.perf_counter()
  stats["build_ms"] = (end - start) * 1000
  return stats


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest):
  if not _INDEX_READY:
    raise HTTPException(status_code=400, detail="Index not built. Call /index first.")

  query_vector = embed_texts([payload.question])[0]
  try:
    hits = rag_rust.lancedb_search(DB_DIR, TABLE_NAME, query_vector, payload.top_k)
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"LanceDB search failed: {exc}") from exc

  retrieved = [{"text": text, "distance": dist} for text, dist in hits]

  if not payload.use_llm:
    return QueryResponse(answer=None, model_used=None, retrieved=retrieved)

  context_text = "\n".join([f"- {row['text']}" for row in retrieved])
  system_prompt = (
    "You are a helpful chatbot.\n"
    "Use only the following pieces of context to answer the question. "
    "Don't make up any new information:\n"
    f"{context_text}"
  )
  answer, model_used = openrouter_chat(
    [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": payload.question},
    ],
    payload.chat_model,
  )

  return QueryResponse(
    answer=answer,
    model_used=model_used,
    retrieved=retrieved,
  )
