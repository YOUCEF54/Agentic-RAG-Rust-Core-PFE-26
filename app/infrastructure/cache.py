"""Simple JSON cache helpers for intermediate RAG artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_json_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = path.read_text(encoding="utf-8").strip()
        if not data:
            return {}
        return json.loads(data)
    except Exception:
        return {}


def save_json_cache(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

