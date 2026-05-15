"""Streaming routes."""

from fastapi import APIRouter

from app.schemas.api import QueryRequest
from app.services.stream_service import run_query_stream

router = APIRouter(tags=["stream"])


@router.post("/query/stream")
def query_stream_route(payload: QueryRequest):
    return run_query_stream(payload)
