"""In-memory runtime state shared by modular services/routes."""

from __future__ import annotations

from typing import Any, Dict

from app.core import config

INDEX_READY = False
INDEX_STATUS = "idle"  # idle | building | ready | stale | error
INDEX_INFO: Dict[str, Any] = {
    "last_build_at": None,
    "last_build_ms": None,
    "pages": None,
    "chunks": None,
    "last_error": None,
    "chunking": "pdfium_sliding_window",
    "embed_batch_size": config.EMBED_BATCH_SIZE,
    "embed_engine": "zembed_api" if config.EMBED_MODE else "onnx_local",
    "hardware_config_mtime": None,
    "llm_backend": config.API_TYPE,
}
ACTIVE_EMBED_BATCH_SIZE = config.EMBED_BATCH_SIZE


def mark_stale() -> None:
    global INDEX_READY, INDEX_STATUS
    INDEX_READY = False
    INDEX_STATUS = "stale"


def clear_index_state() -> None:
    global INDEX_READY, INDEX_STATUS
    INDEX_READY = False
    INDEX_STATUS = "idle"
    INDEX_INFO.update(
        {
            "pages": 0,
            "chunks": 0,
            "last_build_ms": None,
            "last_build_at": None,
            "last_error": None,
            "chunking": "pdfium_sliding_window",
            "embed_batch_size": ACTIVE_EMBED_BATCH_SIZE,
            "llm_backend": config.API_TYPE,
        }
    )

