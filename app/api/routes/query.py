"""Query routes."""

from fastapi import APIRouter

from app.schemas.api import QueryRequest, QueryResponse
from app.services.rag_service import run_query

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query_route(payload: QueryRequest):
    return run_query(payload)
