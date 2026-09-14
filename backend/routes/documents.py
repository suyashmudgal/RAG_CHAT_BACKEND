"""GET /documents  &  DELETE /documents/{document_id}."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from auth.dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/documents")
async def list_documents(user: dict = Depends(get_current_user)):
    """Return metadata for documents owned by the authenticated user."""
    from services.deps import get_vector_store

    store = get_vector_store()
    pg_user_id: int = user["pg_id"]
    return {"documents": store.list_documents_for_user(pg_user_id)}


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    user: dict = Depends(get_current_user),
):
    """Delete a document and all its chunks from the vector store."""
    from services.deps import get_vector_store

    store = get_vector_store()
    doc = store.get_document_info(document_id)

    # Document must exist
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    # Ownership check — user_id must match
    pg_user_id: int = user["pg_id"]
    doc_owner = doc.get("user_id")
    if doc_owner is None or doc_owner != pg_user_id:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        store.delete_document(document_id)

        # Also remove from PostgreSQL
        try:
            from database.session import SessionLocal
            from models.database import Document as PgDocument

            session = SessionLocal()
            try:
                pg_doc = session.query(PgDocument).filter(
                    PgDocument.id == document_id,
                    PgDocument.user_id == pg_user_id,
                ).first()
                if pg_doc:
                    session.delete(pg_doc)
                    session.commit()
            finally:
                session.close()
        except Exception as db_exc:
            logger.warning("Failed to delete document from PostgreSQL: %s", db_exc)

        return {
            "message": f"Document '{doc['filename']}' deleted successfully",
            "document_id": document_id,
        }
    except Exception as exc:
        logger.error("Delete error for %s: %s", document_id, exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to delete document: {exc}"
        ) from exc
