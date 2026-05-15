"""Centralized runtime configuration for the modular app package.

This mirrors current defaults from the legacy `main.py` so migration can happen
incrementally without behavior drift.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


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


def is_truthy_env(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def normalize_api_type(value: Optional[str]) -> str:
    normalized = (value or "").strip().strip('"').strip("'").lower()
    aliases = {
        "openrouter": "open_router",
        "open-router": "open_router",
        "open_router": "open_router",
        "ollama": "ollama",
    }
    return aliases.get(normalized, "ollama")


APP_NAME = "Agentic-RAG-Rust-Core-PFE-26"

# Storage / DB
PDF_DIR = Path("data/pdfs")
META_PATH = Path("data/metadata.json")
DB_DIR = "lancedb"
TABLE_NAME = "pdf_chunks"
INDEXED_SHA_KEY = "indexed_sha256"

# Embeddings
EMBED_MODE = is_truthy_env("EMBED_MODE")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME") or ("zembed-1" if EMBED_MODE else "BAAI/bge-large-en-v1.5")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBED_BATCH_SIZE = get_env_int("EMBED_BATCH_SIZE", 4)

# Chunking
CHUNK_SIZE = get_env_int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = get_env_int("CHUNK_OVERLAP", 150)
MIN_CHUNK_LEN = get_env_int("MIN_CHUNK_LEN", 30)
HARDWARE_CONFIG_PATH = os.getenv("HARDWARE_CONFIG_PATH", "hardware_config.json")

# LLM backend
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api")
OLLAMA_TIMEOUT = get_env_int("OLLAMA_TIMEOUT", 300)
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "phi4-mini:3.8b")
CHAT_TEMPERATURE = get_env_float("CHAT_TEMPERATURE", 0.2)
API_TYPE = normalize_api_type(os.getenv("API_TYPE"))

# OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", APP_NAME)
OPENROUTER_TIMEOUT = get_env_int("OPENROUTER_TIMEOUT", 60)
OPENROUTER_CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "openrouter/free")

# Per-agent models
REFINER_MODEL = os.getenv("REFINER_MODEL", "qwen2.5:1.5b")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "mistral:7b")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "mistral:7b")
SELECTOR_MODEL = os.getenv("SELECTOR_MODEL", "mistral:7b")

# DPS
DPS_ENABLED = True
TOP_N_RETRIEVAL = get_env_int("TOP_N_RETRIEVAL", 15)
TOP_K_MAX = 8
TOP_K_MIN = 1

# Dartboard reranking
DARTBOARD_ENABLED = is_truthy_env("DARTBOARD_ENABLED")
DARTBOARD_SIGMA = get_env_float("DARTBOARD_SIGMA", 0.1)

# CRAG
CRAG_CORRECT_THRESHOLD = get_env_float("CRAG_CORRECT_THRESHOLD", 0.75)
CRAG_AMBIGUOUS_THRESHOLD = get_env_float("CRAG_AMBIGUOUS_THRESHOLD", 0.45)
CRAG_EXTERNAL_TOP_K = get_env_int("CRAG_EXTERNAL_TOP_K", 5)
CRAG_ENABLE_EXTERNAL_ROUTE = (
    True if os.getenv("CRAG_ENABLE_EXTERNAL_ROUTE") is None else is_truthy_env("CRAG_ENABLE_EXTERNAL_ROUTE")
)


def get_cors_origins() -> list[str]:
    cors_origins = os.getenv("CORS_ORIGINS", "")
    return [o.strip() for o in cors_origins.split(",") if o.strip()] or [
        "http://localhost:3000",
        "https://agentic-rag-rust-core-frontend-pfe.vercel.app/",
        "https://agentic-rag-rust-core-frontend-pfe-26-27avjnf8b.vercel.app",
        "https://agentic-rag-rust-core-frontend-pfe-26.vercel.app",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def get_cors_settings() -> dict:
    """Return safe CORS settings for browser clients.

    Notes:
    - `*` + credentials is not a reliable browser combo.
    - default favors explicit local/frontend origins.
    """
    # Keep legacy behavior unless explicitly disabled.
    env_credentials = os.getenv("CORS_ALLOW_CREDENTIALS")
    allow_credentials = True if env_credentials is None else is_truthy_env("CORS_ALLOW_CREDENTIALS")
    allow_all = is_truthy_env("CORS_ALLOW_ALL")

    if allow_all and not allow_credentials:
        origins = ["*"]
    else:
        origins = get_cors_origins()

    return {
        "allow_origins": origins,
        "allow_credentials": allow_credentials,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }
