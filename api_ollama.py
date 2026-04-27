"""API server for the Rust/Python RAG stack.

Run:
  uvicorn api:app --reload

Embedding engines (controlled by EMBED_MODE env var):
  - EMBED_MODE is set (any truthy value)  → ZeroEntropy zembed API (Rust)
  - EMBED_MODE is unset / empty           → Local ONNX (BAAI/bge-small-en-v1.5, Rust fastembed)

LLM backend:
  Calls a local Ollama instance (http://localhost:11434) via its OpenAI-compatible
  /v1/chat/completions endpoint.  Set OLLAMA_BASE_URL and OLLAMA_CHAT_MODEL in
  your .env to override the defaults.
"""
from __future__ import annotations

import hashlib
import json
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAX_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# Désactive les flags de télémétrie ONNX qui peuvent ralentir l'initialisation
os.environ["ONNXRUNTIME_FLAGS"] = "0"
# ----------------------------------------
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Bootstrap the pyo3 extension on Windows when the repo contains the built DLL
# but the Python importable .pyd isn't present yet.
_RAG_RUST_PYD = Path(__file__).with_name("rag_rust.pyd")
if not _RAG_RUST_PYD.exists():
  for _dll in (
    Path("rag_rust/target/release/rag_rust.dll"),
    Path("rag_rust/target/maturin/rag_rust.dll"),
  ):
    if _dll.exists():
      try:
        shutil.copyfile(_dll, _RAG_RUST_PYD)
      except Exception:
        pass
      break

import rag_rust
from agents import Evaluator, Generator, QueryRefiner, Retriever, UserProxy, DynamicPassageSelector
from get_hardware_config import load_hardware_config, run_hardware_calibration
from dotenv import load_dotenv

APP_NAME = "Agentic-RAG-Rust-Core-PFE-26"
load_dotenv()
# Storage / DB
PDF_DIR = Path("data/pdfs")
META_PATH = Path("data/metadata.json")
DB_DIR = "lancedb"
TABLE_NAME = "pdf_chunks"

# ── Embedding engine selection ────────────────────────────────────────────────
def _truthy_env(name: str) -> bool:
  """Parse env vars like '1', 'true', 'yes', 'on', 'zembed' as True."""
  val = os.getenv(name)
  if val is None:
    return False
  return val.strip().lower() in ("1", "true", "yes", "y", "on", "zembed")


# Set EMBED_MODE to a truthy value (e.g. "zembed" or "1") to use the
# ZeroEntropy zembed API. Leave it unset/empty to use the local ONNX engine.
EMBED_MODE = _truthy_env("EMBED_MODE")

EMBED_MODEL_NAME = os.getenv(
  "EMBED_MODEL_NAME",
  "zembed-1" if EMBED_MODE else "BAAI/bge-small-en-v1.5",
)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# ─────────────────────────────────────────────────────────────────────────────

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", "30"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "4"))
HARDWARE_CONFIG_PATH = os.getenv("HARDWARE_CONFIG_PATH", "hardware_config.json")

# ── Ollama (local LLM) ────────────────────────────────────────────────────────
# Ollama exposes an OpenAI-compatible endpoint at /v1/chat/completions.
# Override these in your .env file as needed.
#
#   OLLAMA_BASE_URL=http://localhost:11434   (default)
#   OLLAMA_CHAT_MODEL=llama3                (default)
#   OLLAMA_TIMEOUT=120                      (default, in seconds)
#
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
CHAT_TEMPERATURE = 0.2
# ─────────────────────────────────────────────────────────────────────────────

# DPS (Dynamic Passage Selection)
DPS_ENABLED     = True
TOP_N_RETRIEVAL = 15    # candidates fetched from vector DB
TOP_K_MAX       = 8     # hard ceiling for DPS
TOP_K_MIN       = 1     # hard floor for DPS

app = FastAPI(title=APP_NAME)

_INDEX_READY = False
_INDEX_STATUS = "idle"  # idle | building | ready | stale | error
_INDEX_INFO: Dict[str, Any] = {
  "last_build_at": None,
  "last_build_ms": None,
  "pages": None,
  "chunks": None,
  "last_error": None,
  "chunking": "pdfium_sliding_window",
  "embed_batch_size": EMBED_BATCH_SIZE,
  "hardware_config_mtime": None,
  "embed_mode": "zembed" if EMBED_MODE else "local",
  "embed_model": EMBED_MODEL_NAME,
}
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
_EMBED_MODEL_READY = False
_ACTIVE_EMBED_BATCH_SIZE = EMBED_BATCH_SIZE
_HARDWARE_CONFIG_MTIME: Optional[float] = None

# CORS (frontend integration)
_cors_allow_all = True
_cors_origins = os.getenv("CORS_ORIGINS", "")
if _cors_allow_all:
  allow_origins = ["*"]
else:
  allow_origins = [o.strip() for o in _cors_origins.split(",") if o.strip()] or [
    "http://localhost:3000",
    "https://agentic-rag-rust-core-frontend-pfe.vercel.app/",
    "https://agentic-rag-rust-core-frontend-pfe-26-27avjnf8b.vercel.app",
    "https://agentic-rag-rust-core-frontend-pfe-26.vercel.app",
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
  run_hardware_test: bool = False
  save_hardware_config: bool = True
  hardware_quick_test: bool = True
  hardware_max_runtime_seconds: float = 25.0


class QueryRequest(BaseModel):
  question: str
  top_k: int = 3
  chat_model: Optional[str] = None
  use_llm: bool = True
  mode: str = "agentic"  # "agentic" or "naive"
  return_trace: bool = True
  min_score: float = 0.7
  max_attempts: int = 3


class QueryResponse(BaseModel):
  answer: Optional[str]
  model_used: Optional[str]
  retrieved: List[dict]
  mode: Optional[str] = None
  refined_query: Optional[str] = None
  score: Optional[float] = None
  attempts: Optional[int] = None
  trace: Optional[List[dict]] = None
  models: Optional[Dict[str, Optional[str]]] = None


class DocumentMeta(BaseModel):
  filename: str
  size_bytes: int
  uploaded_at: Optional[str] = None
  updated_at: Optional[str] = None
  sha256: Optional[str] = None
  pages: Optional[int] = None


class HardwareCalibrationRequest(BaseModel):
  save_config: bool = True
  quick_mode: bool = True
  max_runtime_seconds: float = 25.0


def ensure_dirs() -> None:
  PDF_DIR.mkdir(parents=True, exist_ok=True)
  META_PATH.parent.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
  return datetime.now(timezone.utc).isoformat()


def load_metadata() -> Dict[str, Dict[str, Any]]:
  if not META_PATH.exists():
    return {}
  try:
    raw = META_PATH.read_text(encoding="utf-8").strip()
    if not raw:
      return {}
    return json.loads(raw)
  except Exception:
    return {}


def save_metadata(meta: Dict[str, Dict[str, Any]]) -> None:
  META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def file_stats_meta(path: Path, uploaded_at: Optional[str] = None, sha256: Optional[str] = None) -> Dict[str, Any]:
  stat = path.stat()
  updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
  return {
    "filename": path.name,
    "size_bytes": stat.st_size,
    "uploaded_at": uploaded_at,
    "updated_at": updated_at,
    "sha256": sha256,
    "pages": None,
  }


def update_pages_meta(page_counts: Dict[str, int]) -> None:
  if not page_counts:
    return
  meta = load_metadata()
  for filename, pages in page_counts.items():
    if filename not in meta:
      meta[filename] = {"filename": filename}
    meta[filename]["pages"] = pages
  save_metadata(meta)


def parse_chunk_meta(text: str) -> tuple[str, Optional[str], Optional[int]]:
  match = re.match(r"^\[source:\s*(.+?)\s*\|\s*page:\s*(\d+)\]\s*(.*)$", text, re.DOTALL)
  if not match:
    return text, None, None
  source = match.group(1).strip()
  page = int(match.group(2))
  clean_text = match.group(3).strip()
  return clean_text, source, page


def format_chunk_with_meta(filename: str, page: int, chunk: str) -> str:
  return f"[source: {filename} | page: {page}] {chunk}"


def sse_event(event: str, data: Any) -> str:
  payload = json.dumps(data, ensure_ascii=True)
  return f"event: {event}\ndata: {payload}\n\n"


def set_active_embed_batch_size(batch_size: int) -> int:
  global _ACTIVE_EMBED_BATCH_SIZE
  _ACTIVE_EMBED_BATCH_SIZE = max(1, int(batch_size))
  _INDEX_INFO["embed_batch_size"] = _ACTIVE_EMBED_BATCH_SIZE
  return _ACTIVE_EMBED_BATCH_SIZE


def refresh_hardware_config_if_needed(force: bool = False) -> Optional[Dict[str, Any]]:
  global _HARDWARE_CONFIG_MTIME
  config_path = Path(HARDWARE_CONFIG_PATH)
  if not config_path.exists():
    return None
  try:
    mtime = config_path.stat().st_mtime
  except Exception:
    return None
  if not force and _HARDWARE_CONFIG_MTIME is not None and mtime <= _HARDWARE_CONFIG_MTIME:
    return None
  config = load_hardware_config(HARDWARE_CONFIG_PATH)
  if not config:
    return None
  optimal_batch_size = config.get("optimal_batch_size")
  if optimal_batch_size is None:
    return None
  try:
    set_active_embed_batch_size(int(optimal_batch_size))
  except Exception:
    return None
  _HARDWARE_CONFIG_MTIME = mtime
  _INDEX_INFO["hardware_config_mtime"] = mtime
  return config


refresh_hardware_config_if_needed(force=True)


# ── Embedding helpers (dual-engine) ──────────────────────────────────────────

def ensure_embed_model_loaded() -> None:
  """Load the embedding model exactly once, routing to the correct engine."""
  global _EMBED_MODEL_READY
  if not _EMBED_MODEL_READY:
    if EMBED_MODE:
      rag_rust.load_embed_model_zembed()
    else:
      rag_rust.load_embed_model_local()
    _EMBED_MODEL_READY = True


def embed_texts(texts: List[str]) -> List[List[float]]:
  """Embed indexed passages, routing to zembed or local ONNX based on EMBED_MODE.

  - zembed  : calls rag_rust.embed_texts_rust_zembed (no prefix needed)
  - local   : prepends 'passage: ' prefix and calls rag_rust.embed_texts_rust_local
  """
  refresh_hardware_config_if_needed(force=False)
  ensure_embed_model_loaded()
  if EMBED_MODE:
    return rag_rust.embed_texts_rust_zembed(texts, _ACTIVE_EMBED_BATCH_SIZE)
  else:
    prefixed = [f"passage: {t}" for t in texts]
    return rag_rust.embed_texts_rust_local(prefixed, _ACTIVE_EMBED_BATCH_SIZE)


def embed_query(query: str) -> List[float]:
  """Embed a retrieval query, routing to zembed or local ONNX based on EMBED_MODE.

  - zembed  : calls rag_rust.embed_query_rust_zembed
  - local   : prepends BGE query prefix and calls rag_rust.embed_texts_rust_local
  """
  ensure_embed_model_loaded()
  if EMBED_MODE:
    return rag_rust.embed_query_rust_zembed(query)
  else:
    prefixed = f"{BGE_QUERY_PREFIX}{query}"
    return rag_rust.embed_texts_rust_local([prefixed], 1)[0]

# ─────────────────────────────────────────────────────────────────────────────


def load_and_chunk_pdfs(max_pages: Optional[int]) -> tuple[List[str], List[str], List[int], Dict[str, int]]:
  """Load PDFs via PDFium and chunk with sliding_window_chunker.

  Returns (chunk_texts, chunk_sources, chunk_pages, page_counts).
  Uses only exported Rust functions: load_pdf_pages_pdfium_many, sliding_window_chunker.
  """
  paths = sorted(PDF_DIR.glob("*.pdf"))
  if not paths:
    raise HTTPException(status_code=400, detail="No PDFs found in data/pdfs.")

  chunk_texts: List[str] = []
  chunk_sources: List[str] = []
  chunk_pages: List[int] = []
  page_counts: Dict[str, int] = {}

  for path in paths:
    try:
      file_pages = rag_rust.load_pdf_pages_pdfium_many([str(path)])
    except Exception as exc:
      raise HTTPException(status_code=500, detail=f"PDFium loader failed for {path.name}: {exc}") from exc

    page_counts[path.name] = len(file_pages)
    for page_idx, text in enumerate(file_pages, start=1):
      if max_pages is not None and sum(page_counts.values()) > max_pages:
        break
      if not text or not text.strip():
        continue
      page_chunks = rag_rust.sliding_window_chunker(text, CHUNK_SIZE, CHUNK_OVERLAP)
      for chunk_text in page_chunks:
        if len(chunk_text.strip()) > MIN_CHUNK_LEN:
          chunk_texts.append(chunk_text)
          chunk_sources.append(path.name)
          chunk_pages.append(page_idx)

  update_pages_meta(page_counts)
  return chunk_texts, chunk_sources, chunk_pages, page_counts


def build_index(rebuild: bool, max_pages: Optional[int]) -> dict:
  global _INDEX_READY

  chunk_texts, chunk_sources, chunk_pages, page_counts = load_and_chunk_pdfs(max_pages)

  if not chunk_texts:
    raise HTTPException(status_code=400, detail="No text extracted from PDFs.")

  embeddings = embed_texts(chunk_texts)

  try:
    rag_rust.lancedb_create_or_open(
      DB_DIR,
      TABLE_NAME,
      chunk_texts,
      chunk_sources,
      chunk_pages,
      embeddings,
      rebuild,
    )
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"LanceDB error: {exc}") from exc

  total_pages = sum(page_counts.values())
  _INDEX_READY = True
  return {
    "pages": total_pages,
    "chunks": len(chunk_texts),
    "rebuild": rebuild,
    "chunking": "pdfium_sliding_window",
    "embed_batch_size": _ACTIVE_EMBED_BATCH_SIZE,
    "embed_mode": "zembed" if EMBED_MODE else "local",
    "embed_model": EMBED_MODEL_NAME,
  }


def run_index(
  rebuild: bool,
  max_pages: Optional[int],
  run_hardware_test: bool = False,
  save_hardware_config: bool = True,
  hardware_quick_test: bool = True,
  hardware_max_runtime_seconds: float = 25.0,
) -> dict:
  global _INDEX_STATUS, _INDEX_INFO, _INDEX_READY
  _INDEX_STATUS = "building"
  _INDEX_READY = False
  _INDEX_INFO["last_error"] = None
  start = time.perf_counter()
  hardware_result: Optional[Dict[str, Any]] = None
  try:
    if run_hardware_test:
      hardware_result = run_hardware_calibration(
        save_to_file=save_hardware_config,
        config_path=HARDWARE_CONFIG_PATH,
        quick_mode=hardware_quick_test,
        max_runtime_seconds=hardware_max_runtime_seconds,
      )
      optimal_batch_size = hardware_result.get("optimal_batch_size")
      if optimal_batch_size is not None:
        set_active_embed_batch_size(int(optimal_batch_size))
      if save_hardware_config:
        refresh_hardware_config_if_needed(force=True)
    else:
      refresh_hardware_config_if_needed(force=False)
    stats = build_index(rebuild, max_pages)
    if hardware_result:
      stats["hardware_calibration"] = hardware_result
  except HTTPException as exc:
    _INDEX_STATUS = "error"
    _INDEX_INFO["last_error"] = exc.detail
    raise
  except Exception as exc:
    _INDEX_STATUS = "error"
    _INDEX_INFO["last_error"] = str(exc)
    raise
  end = time.perf_counter()
  build_ms = (end - start) * 1000
  stats["build_ms"] = build_ms
  _INDEX_INFO.update(
    {
      "last_build_at": utc_now_iso(),
      "last_build_ms": build_ms,
      "pages": stats.get("pages"),
      "chunks": stats.get("chunks"),
      "chunking": stats.get("chunking"),
      "embed_batch_size": stats.get("embed_batch_size"),
      "embed_mode": "zembed" if EMBED_MODE else "local",
      "embed_model": EMBED_MODEL_NAME,
    }
  )
  _INDEX_STATUS = "ready"
  _INDEX_READY = True
  return stats


# ── Ollama chat helper ────────────────────────────────────────────────────────

def ollama_chat(messages: List[dict], model_override: Optional[str] = None) -> tuple[str, Optional[str]]:
  """Send a chat request to the local Ollama instance.

  Uses Ollama's OpenAI-compatible endpoint so the message format is identical
  to what the rest of the code already builds.

  Args:
    messages: list of {"role": ..., "content": ...} dicts.
    model_override: if supplied, overrides OLLAMA_CHAT_MODEL for this call.

  Returns:
    (answer_text, model_name_used)

  Raises:
    HTTPException(502) if Ollama is unreachable.
    HTTPException(status_code) for non-2xx responses from Ollama.
  """
  model = model_override or OLLAMA_CHAT_MODEL
  url = f"{OLLAMA_BASE_URL.rstrip('/')}/v1/chat/completions"
  payload = {
    "model": model,
    "messages": messages,
    "temperature": CHAT_TEMPERATURE,
    "stream": False,
  }
  try:
    resp = requests.post(
      url,
      json=payload,
      headers={"Content-Type": "application/json"},
      timeout=OLLAMA_TIMEOUT,
    )
  except requests.exceptions.ConnectionError as exc:
    raise HTTPException(
      status_code=502,
      detail=f"Could not connect to Ollama at {OLLAMA_BASE_URL}. "
             "Make sure Ollama is running (`ollama serve`).",
    ) from exc
  except requests.exceptions.Timeout as exc:
    raise HTTPException(
      status_code=504,
      detail=f"Ollama request timed out after {OLLAMA_TIMEOUT}s.",
    ) from exc

  if not resp.ok:
    raise HTTPException(status_code=resp.status_code, detail=resp.text)

  data = resp.json()
  content: str = data["choices"][0]["message"]["content"]
  model_used: Optional[str] = data.get("model") or model
  return content, model_used

# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
  refresh_hardware_config_if_needed(force=False)
  return {
    "status": "ok",
    "chunking": "pdfium_sliding_window",
    "embed_batch_size": _ACTIVE_EMBED_BATCH_SIZE,
    "hardware_config_mtime": _INDEX_INFO.get("hardware_config_mtime"),
    "embed_mode": "zembed" if EMBED_MODE else "local",
    "embed_model": EMBED_MODEL_NAME,
    "llm_backend": "ollama",
    "ollama_base_url": OLLAMA_BASE_URL,
    "ollama_chat_model": OLLAMA_CHAT_MODEL,
  }


@app.post("/documents")
def upload_documents(files: List[UploadFile] = File(...)):
  ensure_dirs()
  meta = load_metadata()
  saved = []
  for f in files:
    if not f.filename.lower().endswith(".pdf"):
      raise HTTPException(status_code=400, detail=f"Not a PDF: {f.filename}")
    target = PDF_DIR / f.filename
    with target.open("wb") as out:
      file_bytes = f.file.read()
      out.write(file_bytes)
    saved.append(f.filename)
    meta[f.filename] = file_stats_meta(
      target,
      uploaded_at=utc_now_iso(),
      sha256=sha256_bytes(file_bytes),
    )
  save_metadata(meta)
  global _INDEX_READY, _INDEX_STATUS
  _INDEX_READY = False
  _INDEX_STATUS = "stale"
  return {"saved": saved, "needs_reindex": True}


@app.get("/documents")
def list_documents():
  ensure_dirs()
  meta = load_metadata()
  docs: List[Dict[str, Any]] = []
  files = sorted(PDF_DIR.glob("*.pdf"))
  for path in files:
    existing = meta.get(path.name, {})
    uploaded_at = existing.get("uploaded_at")
    sha256 = existing.get("sha256")
    if not uploaded_at or not sha256:
      meta[path.name] = file_stats_meta(path, uploaded_at=uploaded_at, sha256=sha256)
    else:
      meta[path.name].update(file_stats_meta(path, uploaded_at=uploaded_at, sha256=sha256))
    docs.append(meta[path.name])
  save_metadata(meta)
  return {"files": [p.name for p in files], "documents": docs}


def clear_index() -> None:
  global _INDEX_READY
  _INDEX_READY = False
  db_path = Path(DB_DIR)
  if db_path.exists():
    shutil.rmtree(db_path)
  global _INDEX_STATUS, _INDEX_INFO
  _INDEX_STATUS = "idle"
  _INDEX_INFO.update(
    {
      "pages": 0,
      "chunks": 0,
      "last_build_ms": None,
      "last_build_at": None,
      "last_error": None,
      "chunking": "pdfium_sliding_window",
      "embed_batch_size": _ACTIVE_EMBED_BATCH_SIZE,
      "embed_mode": "zembed" if EMBED_MODE else "local",
      "embed_model": EMBED_MODEL_NAME,
    }
  )


@app.delete("/documents/{filename}")
def delete_document(filename: str, rebuild_index: bool = True):
  ensure_dirs()
  target = PDF_DIR / filename
  if not target.exists():
    raise HTTPException(status_code=404, detail="File not found.")

  target.unlink()
  meta = load_metadata()
  if filename in meta:
    meta.pop(filename, None)
    save_metadata(meta)

  remaining = list(PDF_DIR.glob("*.pdf"))
  if not remaining:
    clear_index()
    return {"deleted": filename, "index_ready": False, "index_cleared": True}

  if rebuild_index:
    stats = run_index(rebuild=True, max_pages=None)
    return {"deleted": filename, "index_ready": True, "index": stats}

  _INDEX_READY = False
  global _INDEX_STATUS
  _INDEX_STATUS = "stale"
  return {"deleted": filename, "index_ready": False, "needs_reindex": True}


@app.post("/index")
def build_index_endpoint(payload: IndexRequest):
  return run_index(
    payload.rebuild,
    payload.max_pages,
    run_hardware_test=payload.run_hardware_test,
    save_hardware_config=payload.save_hardware_config,
    hardware_quick_test=payload.hardware_quick_test,
    hardware_max_runtime_seconds=payload.hardware_max_runtime_seconds,
  )


@app.get("/index/status")
def index_status():
  refresh_hardware_config_if_needed(force=False)
  return {
    "status": _INDEX_STATUS,
    "ready": _INDEX_READY,
    "info": _INDEX_INFO,
  }


@app.get("/hardware/config")
def hardware_config():
  refresh_hardware_config_if_needed(force=False)
  config = load_hardware_config(HARDWARE_CONFIG_PATH)
  return {
    "config": config,
    "active_embed_batch_size": _ACTIVE_EMBED_BATCH_SIZE,
    "config_path": HARDWARE_CONFIG_PATH,
    "hardware_config_mtime": _INDEX_INFO.get("hardware_config_mtime"),
  }


@app.post("/hardware/calibrate")
def hardware_calibrate(payload: HardwareCalibrationRequest = HardwareCalibrationRequest()):
  try:
    result = run_hardware_calibration(
      save_to_file=payload.save_config,
      config_path=HARDWARE_CONFIG_PATH,
      quick_mode=payload.quick_mode,
      max_runtime_seconds=payload.max_runtime_seconds,
    )
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Hardware calibration failed: {exc}") from exc
  optimal_batch_size = result.get("optimal_batch_size")
  if optimal_batch_size is not None:
    set_active_embed_batch_size(int(optimal_batch_size))
  if payload.save_config:
    refresh_hardware_config_if_needed(force=True)
  return {
    "hardware_calibration": result,
    "active_embed_batch_size": _ACTIVE_EMBED_BATCH_SIZE,
    "hardware_config_mtime": _INDEX_INFO.get("hardware_config_mtime"),
  }


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest):
  if not _INDEX_READY:
    raise HTTPException(status_code=400, detail="Index not built. Call /index first.")

  def retrieve_chunks_with_meta(query: str, top_k: int) -> tuple[List[tuple], List[Dict[str, Any]]]:
    query_vector = embed_query(query)
    try:
      hits = rag_rust.lancedb_search(DB_DIR, TABLE_NAME, query_vector, top_k)
    except Exception as exc:
      raise HTTPException(status_code=500, detail=f"LanceDB search failed: {exc}") from exc
    chunks: List[tuple] = []
    meta: List[Dict[str, Any]] = []
    for text, source, page, dist in hits:
      chunks.append((text, source, page, dist))
      meta.append(
        {
          "text": text,
          "distance": dist,
          "source": source,
          "page": page,
        }
      )
    return chunks, meta

  if not payload.use_llm:
    hits, meta = retrieve_chunks_with_meta(payload.question, payload.top_k)
    return QueryResponse(answer=None, model_used=None, retrieved=meta, mode="retrieval_only")

  if payload.mode.lower() == "naive":
    hits, meta = retrieve_chunks_with_meta(payload.question, payload.top_k)
    context_text = "\n".join([f"- {row['text']}" for row in meta])
    system_prompt = (
      "You are a helpful chatbot.\n"
      "Use only the following pieces of context to answer the question. "
      "Don't make up any new information:\n"
      f"{context_text}"
    )
    answer, model_used = ollama_chat(
      [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload.question},
      ],
      payload.chat_model,
    )
    return QueryResponse(
      answer=answer,
      model_used=model_used,
      retrieved=meta,
      mode="naive",
      models={"generator": model_used},
    )

  state: Dict[str, Any] = {
    "query": payload.question,
    "attempts": 0,
    "should_retry": True,
    "trace": [] if payload.return_trace else None,
  }

  def retrieve_for_agent(q: str, top_k: int) -> List[tuple]:
    hits, meta = retrieve_chunks_with_meta(q, top_k)
    state["retrieved_meta"] = meta
    return hits

  chat_fn = lambda messages: ollama_chat(messages, payload.chat_model)
  refiner = QueryRefiner(chat_fn=chat_fn)
  retriever = Retriever(retrieve_fn=lambda q, top_k: retrieve_for_agent(q, top_k), top_k=payload.top_k, top_n=TOP_N_RETRIEVAL)
  selector = DynamicPassageSelector(
    chat_fn=chat_fn,
    max_passages=TOP_K_MAX,
    min_passages=TOP_K_MIN,
  ) if DPS_ENABLED else None
  generator = Generator(chat_fn=chat_fn)
  evaluator = Evaluator(
    chat_fn=chat_fn,
    min_score=payload.min_score,
    max_attempts=payload.max_attempts,
  )
  proxy = UserProxy(refiner, retriever, selector, generator, evaluator)
  state = proxy.run(state)

  retrieved = state.get("retrieved_meta") or []

  return QueryResponse(
    answer=state.get("answer"),
    model_used=state.get("model_used"),
    retrieved=retrieved,
    mode="agentic",
    refined_query=state.get("refined_query"),
    score=state.get("score"),
    attempts=state.get("attempts"),
    trace=state.get("trace") if payload.return_trace else None,
    models=state.get("models"),
  )


@app.post("/query/stream")
def query_stream(payload: QueryRequest):
  if not _INDEX_READY:
    raise HTTPException(status_code=400, detail="Index not built. Call /index first.")

  def retrieve_chunks_with_meta(query: str, top_k: int) -> tuple[List[tuple], List[Dict[str, Any]]]:
    query_vector = embed_query(query)
    try:
      hits = rag_rust.lancedb_search(DB_DIR, TABLE_NAME, query_vector, top_k)
    except Exception as exc:
      raise HTTPException(status_code=500, detail=f"LanceDB search failed: {exc}") from exc
    chunks: List[tuple] = []
    meta: List[Dict[str, Any]] = []
    for text, source, page, dist in hits:
      chunks.append((text, source, page, dist))
      meta.append(
        {
          "text": text,
          "distance": dist,
          "source": source,
          "page": page,
        }
      )
    return chunks, meta

  def event_stream() -> Iterable[str]:
    yield sse_event("status", {"state": "started"})

    if not payload.use_llm:
      _, meta = retrieve_chunks_with_meta(payload.question, payload.top_k)
      yield sse_event("retrieved", {"items": meta})
      yield sse_event("final", {"mode": "retrieval_only", "retrieved": meta})
      return

    if payload.mode.lower() == "naive":
      chunks, meta = retrieve_chunks_with_meta(payload.question, payload.top_k)
      yield sse_event("retrieved", {"items": meta})
      context_text = "\n".join([f"- {row['text']}" for row in meta])
      system_prompt = (
        "You are a helpful chatbot.\n"
        "Use only the following pieces of context to answer the question. "
        "Don't make up any new information:\n"
        f"{context_text}"
      )
      answer, model_used = ollama_chat(
        [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": payload.question},
        ],
        payload.chat_model,
      )
      yield sse_event("answer", {"answer": answer, "model_used": model_used})
      yield sse_event(
        "final",
        {
          "mode": "naive",
          "answer": answer,
          "model_used": model_used,
          "retrieved": meta,
          "models": {"generator": model_used},
        },
      )
      return

    pending: List[Dict[str, Any]] = []

    def emit(item: Dict[str, Any]) -> None:
      pending.append(item)

    state: Dict[str, Any] = {
      "query": payload.question,
      "attempts": 0,
      "should_retry": True,
      "trace": [] if payload.return_trace else None,
      "emit": emit,
    }

    def retrieve_for_agent(q: str, top_k: int) -> List[tuple]:
      hits, meta = retrieve_chunks_with_meta(q, top_k)
      state["retrieved_meta"] = meta
      return hits

    chat_fn = lambda messages: ollama_chat(messages, payload.chat_model)
    refiner = QueryRefiner(chat_fn=chat_fn)
    retriever = Retriever(retrieve_fn=lambda q, top_k: retrieve_for_agent(q, top_k), top_k=payload.top_k, top_n=TOP_N_RETRIEVAL)
    selector = DynamicPassageSelector(
      chat_fn=chat_fn,
      max_passages=TOP_K_MAX,
      min_passages=TOP_K_MIN,
    ) if DPS_ENABLED else None
    generator = Generator(chat_fn=chat_fn)
    evaluator = Evaluator(
      chat_fn=chat_fn,
      min_score=payload.min_score,
      max_attempts=payload.max_attempts,
    )

    while state["should_retry"]:
      state = refiner.run(state)
      for item in pending:
        yield sse_event("trace", item)
      pending.clear()

      state = retriever.run(state)
      for item in pending:
        yield sse_event("trace", item)
      pending.clear()
      yield sse_event("retrieved", {"items": state.get("retrieved_meta") or []})

      if selector is not None:
        state = selector.run(state)
        for item in pending:
          yield sse_event("trace", item)
        pending.clear()

      state = generator.run(state)
      for item in pending:
        yield sse_event("trace", item)
      pending.clear()
      yield sse_event("answer", {"answer": state.get("answer"), "model_used": state.get("model_used")})

      state = evaluator.run(state)
      for item in pending:
        yield sse_event("trace", item)
      pending.clear()
      yield sse_event(
        "evaluation",
        {
          "score": state.get("score"),
          "summary": state.get("judge_summary"),
          "should_retry": state.get("should_retry"),
          "attempts": state.get("attempts"),
        },
      )

      if state.get("should_retry"):
        yield sse_event(
          "retry",
          {
            "attempts": state.get("attempts"),
            "score": state.get("score"),
          },
        )

    final_payload = QueryResponse(
      answer=state.get("answer"),
      model_used=state.get("model_used"),
      retrieved=state.get("retrieved_meta") or [],
      mode="agentic",
      refined_query=state.get("refined_query"),
      score=state.get("score"),
      attempts=state.get("attempts"),
      trace=state.get("trace") if payload.return_trace else None,
      models=state.get("models"),
    )
    yield sse_event("final", final_payload.model_dump())

  return StreamingResponse(event_stream(), media_type="text/event-stream")