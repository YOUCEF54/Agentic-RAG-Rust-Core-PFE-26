"""Health route for the modular API package."""

from fastapi import APIRouter

from app.services.health_service import build_health_payload

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return build_health_payload()
