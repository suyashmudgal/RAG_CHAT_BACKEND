"""GET /documents  &  DELETE /documents/{document_id}."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/documents")
async def list_documents():
    """Return metadata for every indexed document."""
    from services.deps import get_vector_store

    store = get_vector_store()
    return {"documents": store.list_documents()}


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chunks from the vector store."""
    from services.deps import get_vector_store

    store = get_vector_store()
    doc = store.get_document_info(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        store.delete_document(document_id)
        return {
            "message": f"Document '{doc['filename']}' deleted successfully",
            "document_id": document_id,
        }
    except Exception as exc:
        logger.error("Delete error for %s: %s", document_id, exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to delete document: {exc}"
        ) from exc
