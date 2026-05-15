"""Hardware routes."""

from fastapi import APIRouter

from app.schemas.api import HardwareCalibrationRequest
from app.services.hardware_service import calibrate_hardware, get_hardware_config_payload

router = APIRouter(tags=["hardware"])


@router.get("/hardware/config")
def hardware_config_route():
    return get_hardware_config_payload()


@router.post("/hardware/calibrate")
def hardware_calibrate_route(payload: HardwareCalibrationRequest = HardwareCalibrationRequest()):
    return calibrate_hardware(payload)

