"""Health/status service."""

from __future__ import annotations

from pathlib import Path

from app.core import config, runtime_state


def _refresh_hardware_config_mtime() -> None:
    target = Path(config.HARDWARE_CONFIG_PATH)
    if target.exists():
        runtime_state.INDEX_INFO["hardware_config_mtime"] = target.stat().st_mtime
    else:
        runtime_state.INDEX_INFO["hardware_config_mtime"] = None


def build_health_payload() -> dict:
    _refresh_hardware_config_mtime()
    return {
        "status": "ok",
        "api_type": config.API_TYPE,
        "chunking": "pdfium_sliding_window",
        "embed_batch_size": runtime_state.ACTIVE_EMBED_BATCH_SIZE,
        "hardware_config_mtime": runtime_state.INDEX_INFO.get("hardware_config_mtime"),
        "llm_default_model": (
            config.OPENROUTER_CHAT_MODEL if config.API_TYPE == "open_router" else config.OLLAMA_CHAT_MODEL
        ),
    }

