"""Hardware config/calibration services."""

from __future__ import annotations

from typing import Any, Dict

from app.core import config
from app.schemas.api import HardwareCalibrationRequest
from app.services.indexing_service import refresh_hardware_config_if_needed, set_active_embed_batch_size
from get_hardware_config import load_hardware_config, run_hardware_calibration


def get_hardware_config_payload() -> Dict[str, Any]:
    refresh_hardware_config_if_needed(force=False)
    loaded = load_hardware_config(config.HARDWARE_CONFIG_PATH)
    return {
        "config": loaded,
        "active_embed_batch_size": None if loaded is None else loaded.get("optimal_batch_size"),
        "config_path": config.HARDWARE_CONFIG_PATH,
        "exists": loaded is not None,
    }


def calibrate_hardware(payload: HardwareCalibrationRequest) -> Dict[str, Any]:
    result = run_hardware_calibration(
        save_to_file=payload.save_config,
        config_path=config.HARDWARE_CONFIG_PATH,
        quick_mode=payload.quick_mode,
        max_runtime_seconds=payload.max_runtime_seconds,
    )
    optimal = result.get("optimal_batch_size")
    if optimal is not None:
        set_active_embed_batch_size(int(optimal))
    refresh_hardware_config_if_needed(force=True)
    return {
        "hardware_calibration": result,
        "active_embed_batch_size": optimal,
        "config_path": config.HARDWARE_CONFIG_PATH,
    }

