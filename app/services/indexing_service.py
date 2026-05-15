"""Indexing workflow service extracted from legacy main.py."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from app.core import config, runtime_state
from app.infrastructure import embeddings, rust_bridge
from app.schemas.api import IndexRequest
from app.services.ingestion_service import ensure_dirs, update_pages_meta
from get_hardware_config import load_hardware_config, run_hardware_calibration

CHUNK_CONFIG = {
    "max_chars": 2000,
    "window_size": 5,
}

_HARDWARE_CONFIG_MTIME: Optional[float] = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_active_embed_batch_size(batch_size: int) -> int:
    runtime_state.ACTIVE_EMBED_BATCH_SIZE = max(1, int(batch_size))
    runtime_state.INDEX_INFO["embed_batch_size"] = runtime_state.ACTIVE_EMBED_BATCH_SIZE
    return runtime_state.ACTIVE_EMBED_BATCH_SIZE


def refresh_hardware_config_if_needed(force: bool = False) -> Optional[Dict[str, Any]]:
    global _HARDWARE_CONFIG_MTIME
    config_path = Path(config.HARDWARE_CONFIG_PATH)
    if not config_path.exists():
        runtime_state.INDEX_INFO["hardware_config_mtime"] = None
        return None

    mtime = config_path.stat().st_mtime
    runtime_state.INDEX_INFO["hardware_config_mtime"] = mtime
    if not force and _HARDWARE_CONFIG_MTIME is not None and mtime <= _HARDWARE_CONFIG_MTIME:
        return None

    loaded = load_hardware_config(config.HARDWARE_CONFIG_PATH)
    if loaded:
        optimal_batch = loaded.get("optimal_batch_size")
        if optimal_batch:
            set_active_embed_batch_size(int(optimal_batch))
    _HARDWARE_CONFIG_MTIME = mtime
    return loaded


def load_and_chunk_pdfs(max_pages: Optional[int]) -> Tuple[List[str], List[str], List[int], Dict[str, int]]:
    ensure_dirs()
    paths = sorted(config.PDF_DIR.glob("*.pdf"))
    if not paths:
        raise HTTPException(status_code=400, detail="No PDFs found in data/pdfs.")

    chunk_texts: List[str] = []
    chunk_sources: List[str] = []
    chunk_pages: List[int] = []
    page_counts: Dict[str, int] = {}

    pages_seen = 0
    for path in paths:
        try:
            file_pages = rust_bridge.load_pdf_pages_pdfium_many([str(path)])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"PDFium loader failed for {path.name}: {exc}") from exc

        page_counts[path.name] = len(file_pages)
        for page_idx, text in enumerate(file_pages, start=1):
            pages_seen += 1
            if max_pages is not None and pages_seen > max_pages:
                break
            if not text or not text.strip():
                continue

            page_chunks = rust_bridge.semantic_window_chunker_advanced(
                text=text,
                max_chars=CHUNK_CONFIG["max_chars"],
                window_size=CHUNK_CONFIG["window_size"],
            )
            for chunk_text in page_chunks:
                if len(chunk_text.strip()) > config.MIN_CHUNK_LEN:
                    chunk_texts.append(chunk_text)
                    chunk_sources.append(path.name)
                    chunk_pages.append(page_idx)
        if max_pages is not None and pages_seen > max_pages:
            break

    update_pages_meta(page_counts)
    return chunk_texts, chunk_sources, chunk_pages, page_counts


def _build_index(rebuild: bool, max_pages: Optional[int]) -> Dict[str, Any]:
    chunk_texts, chunk_sources, chunk_pages, page_counts = load_and_chunk_pdfs(max_pages)
    if not chunk_texts:
        raise HTTPException(status_code=400, detail="No text extracted from PDFs.")

    vecs = embeddings.embed_texts(chunk_texts, runtime_state.ACTIVE_EMBED_BATCH_SIZE)
    try:
        rust_bridge.lancedb_create_or_open(
            config.DB_DIR,
            config.TABLE_NAME,
            chunk_texts,
            chunk_sources,
            chunk_pages,
            vecs,
            rebuild,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LanceDB error: {exc}") from exc

    total_pages = sum(page_counts.values())
    runtime_state.INDEX_READY = True
    return {
        "pages": total_pages,
        "chunks": len(chunk_texts),
        "rebuild": rebuild,
        "chunking": "pdfium_sliding_window",
        "embed_batch_size": runtime_state.ACTIVE_EMBED_BATCH_SIZE,
    }


def build_index(payload: IndexRequest) -> Dict[str, Any]:
    runtime_state.INDEX_STATUS = "building"
    runtime_state.INDEX_READY = False
    runtime_state.INDEX_INFO["last_error"] = None

    start = time.perf_counter()
    hardware_result: Optional[Dict[str, Any]] = None
    try:
        if payload.run_hardware_test:
            hardware_result = run_hardware_calibration(
                save_to_file=payload.save_hardware_config,
                config_path=config.HARDWARE_CONFIG_PATH,
                quick_mode=payload.hardware_quick_test,
                max_runtime_seconds=payload.hardware_max_runtime_seconds,
            )
            optimal = hardware_result.get("optimal_batch_size")
            if optimal is not None:
                set_active_embed_batch_size(int(optimal))
            if payload.save_hardware_config:
                refresh_hardware_config_if_needed(force=True)
        else:
            refresh_hardware_config_if_needed(force=False)

        stats = _build_index(payload.rebuild, payload.max_pages)
        if hardware_result:
            stats["hardware_calibration"] = hardware_result
    except HTTPException as exc:
        runtime_state.INDEX_STATUS = "error"
        runtime_state.INDEX_INFO["last_error"] = exc.detail
        raise
    except Exception as exc:
        runtime_state.INDEX_STATUS = "error"
        runtime_state.INDEX_INFO["last_error"] = str(exc)
        raise HTTPException(status_code=500, detail=f"Index build failed: {exc}") from exc

    build_ms = (time.perf_counter() - start) * 1000
    stats["build_ms"] = build_ms
    runtime_state.INDEX_INFO.update(
        {
            "last_build_at": utc_now_iso(),
            "last_build_ms": build_ms,
            "pages": stats.get("pages"),
            "chunks": stats.get("chunks"),
            "chunking": stats.get("chunking"),
            "embed_batch_size": stats.get("embed_batch_size"),
            "llm_backend": config.API_TYPE,
        }
    )
    runtime_state.INDEX_STATUS = "ready"
    runtime_state.INDEX_READY = True
    return stats


def get_index_status() -> Dict[str, Any]:
    refresh_hardware_config_if_needed(force=False)
    return {
        "status": runtime_state.INDEX_STATUS,
        "ready": runtime_state.INDEX_READY,
        "info": runtime_state.INDEX_INFO,
    }
