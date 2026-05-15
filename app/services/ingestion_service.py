"""Ingestion/document management services.

This module is migrated from root `main.py` and is used by modular API routes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, UploadFile

from app.core import config, runtime_state


def ensure_dirs() -> None:
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    config.META_PATH.parent.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_metadata() -> Dict[str, Dict[str, Any]]:
    if not config.META_PATH.exists():
        return {}
    try:
        raw = config.META_PATH.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        return json.loads(raw)
    except Exception:
        return {}


def save_metadata(meta: Dict[str, Dict[str, Any]]) -> None:
    config.META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def upload_documents(files: List[UploadFile]) -> dict:
    ensure_dirs()
    meta = load_metadata()
    saved = []
    for item in files:
        if not item.filename or not item.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Not a PDF: {item.filename}")
        target = config.PDF_DIR / item.filename
        with target.open("wb") as out:
            file_bytes = item.file.read()
            out.write(file_bytes)
        saved.append(item.filename)
        meta[item.filename] = file_stats_meta(
            target,
            uploaded_at=utc_now_iso(),
            sha256=sha256_bytes(file_bytes),
        )
    save_metadata(meta)
    runtime_state.mark_stale()
    return {"saved": saved, "needs_reindex": True}


def list_documents() -> dict:
    ensure_dirs()
    meta = load_metadata()
    docs: List[Dict[str, Any]] = []
    files = sorted(config.PDF_DIR.glob("*.pdf"))
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


def _clear_index_storage() -> None:
    db_path = Path(config.DB_DIR)
    if db_path.exists():
        shutil.rmtree(db_path)
    runtime_state.clear_index_state()


def delete_document(filename: str, rebuild_index: bool = True) -> dict:
    ensure_dirs()
    target = config.PDF_DIR / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    target.unlink()
    meta = load_metadata()
    if filename in meta:
        meta.pop(filename, None)
        save_metadata(meta)

    remaining = list(config.PDF_DIR.glob("*.pdf"))
    if not remaining:
        _clear_index_storage()
        return {"deleted": filename, "index_ready": False, "index_cleared": True}

    # Index rebuild orchestration stays in legacy pipeline for now.
    if rebuild_index:
        runtime_state.mark_stale()
        return {
            "deleted": filename,
            "index_ready": False,
            "needs_reindex": True,
            "message": "Document removed. Rebuild index via /index.",
        }

    runtime_state.mark_stale()
    return {"deleted": filename, "index_ready": False, "needs_reindex": True}
