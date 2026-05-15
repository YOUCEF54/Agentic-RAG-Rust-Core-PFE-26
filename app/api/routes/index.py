"""Index routes."""

from fastapi import APIRouter

from app.schemas.api import IndexRequest
from app.services.indexing_service import build_index, get_index_status

router = APIRouter(tags=["index"])


@router.post("/index")
def build_index_route(payload: IndexRequest):
    return build_index(payload)


@router.get("/index/status")
def index_status_route():
    return get_index_status()
