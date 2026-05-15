"""Document management routes."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, File, UploadFile

from app.services.ingestion_service import delete_document, list_documents, upload_documents

router = APIRouter(tags=["documents"])


@router.post("/documents")
def upload_documents_route(files: List[UploadFile] = File(...)) -> dict:
    return upload_documents(files)


@router.get("/documents")
def list_documents_route() -> dict:
    return list_documents()


@router.delete("/documents/{filename}")
def delete_document_route(filename: str, rebuild_index: bool = True) -> dict:
    return delete_document(filename, rebuild_index=rebuild_index)

