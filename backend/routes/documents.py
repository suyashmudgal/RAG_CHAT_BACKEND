"""GET /documents, DELETE /documents/{document_id}, & GET /documents/{document_id}/download.

Provides user-isolated document management using Supabase PostgreSQL as metadata
source of truth and Supabase Storage as physical file source of truth.
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response

from auth.dependencies import get_current_user
from database.session import SessionLocal
from models.database import Document as PgDocument
from services.deps import get_storage_service, get_vector_store

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/documents")
async def list_documents(user: dict = Depends(get_current_user)):
    """Return metadata for documents owned strictly by the authenticated user."""
    pg_user_id: int = user["pg_id"]

    session = SessionLocal()
    try:
        pg_docs = (
            session.query(PgDocument)
            .filter(PgDocument.user_id == pg_user_id)
            .order_by(PgDocument.created_at.desc())
            .all()
        )
        docs_list = [
            {
                "document_id": doc.id,
                "id": doc.id,
                "filename": doc.filename,
                "storage_path": doc.storage_path,
                "file_size": doc.file_size,
                "status": doc.status,
                "chunk_count": doc.chunk_count,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                "user_id": doc.user_id,
            }
            for doc in pg_docs
        ]
        return {"documents": docs_list}
    except Exception as exc:
        logger.error("Failed to query documents from PostgreSQL: %s", exc, exc_info=True)
        # Fallback to vector store for resilience
        store = get_vector_store()
        return {"documents": store.list_documents_for_user(pg_user_id)}
    finally:
        session.close()


@router.get("/documents/{document_id}/download")
async def download_document(
    document_id: str,
    user: dict = Depends(get_current_user),
):
    """Securely download a document belonging to the authenticated user from Supabase Storage."""
    pg_user_id: int = user["pg_id"]
    storage_service = get_storage_service()

    session = SessionLocal()
    try:
        pg_doc = (
            session.query(PgDocument)
            .filter(PgDocument.id == document_id)
            .first()
        )
        if not pg_doc or pg_doc.user_id != pg_user_id:
            # Secure denial: never reveal existence to another user
            raise HTTPException(status_code=404, detail="Document not found")

        storage_path = pg_doc.storage_path
        filename = pg_doc.filename or "download"

        # Guess mime type
        content_type, _ = mimetypes.guess_type(filename)
        content_type = content_type or "application/octet-stream"

        file_bytes: bytes | None = None

        # 1. Try Supabase Storage if storage_path is formatted as storage path
        if storage_path and (storage_path.startswith("users/") or not Path(storage_path).is_file()):
            try:
                file_bytes = await storage_service.download_file(storage_path)
            except Exception as s_exc:
                logger.warning("Supabase storage download error for %s: %s", storage_path, s_exc)

        # 2. Fallback to local file path for pre-Phase 5 documents
        if file_bytes is None and storage_path:
            local_path = Path(storage_path)
            if local_path.is_file():
                file_bytes = local_path.read_bytes()

        # 3. If still None and storage_service has it
        if file_bytes is None and storage_path:
            try:
                file_bytes = await storage_service.download_file(storage_path)
            except Exception:
                pass

        if file_bytes is None:
            raise HTTPException(status_code=404, detail="File not found in storage")

        return Response(
            content=file_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(file_bytes)),
            },
        )
    finally:
        session.close()


@router.get("/documents/{document_id}/signed-url")
async def get_document_signed_url(
    document_id: str,
    user: dict = Depends(get_current_user),
):
    """Generate a short-lived signed URL for an owned document."""
    pg_user_id: int = user["pg_id"]
    storage_service = get_storage_service()

    session = SessionLocal()
    try:
        pg_doc = (
            session.query(PgDocument)
            .filter(PgDocument.id == document_id)
            .first()
        )
        if not pg_doc or pg_doc.user_id != pg_user_id:
            raise HTTPException(status_code=404, detail="Document not found")

        if not pg_doc.storage_path:
            raise HTTPException(status_code=400, detail="Document has no storage path")

        signed_url = await storage_service.create_signed_url(
            storage_path=pg_doc.storage_path, expires_in=300
        )
        return {"signed_url": signed_url, "expires_in": 300}
    finally:
        session.close()


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    user: dict = Depends(get_current_user),
):
    """Delete a document and all associated data (ChromaDB vectors, Storage object, PostgreSQL)."""
    pg_user_id: int = user["pg_id"]
    vector_store = get_vector_store()
    storage_service = get_storage_service()

    session = SessionLocal()
    try:
        pg_doc = (
            session.query(PgDocument)
            .filter(PgDocument.id == document_id)
            .first()
        )

        # Secure denial: document must exist AND belong to current user
        if not pg_doc:
            # Check legacy vector store metadata as fallback
            doc_info = vector_store.get_document_info(document_id)
            if not doc_info or doc_info.get("user_id") != pg_user_id:
                raise HTTPException(status_code=404, detail="Document not found")
            filename = doc_info.get("filename", "document")
            vector_store.delete_document(document_id)
            return {
                "message": f"Document '{filename}' deleted successfully",
                "document_id": document_id,
            }

        if pg_doc.user_id != pg_user_id:
            raise HTTPException(status_code=404, detail="Document not found")

        filename = pg_doc.filename
        storage_path = pg_doc.storage_path

        # 1. Delete ChromaDB vectors
        vector_store.delete_document(document_id)

        # 2. Delete physical file from Supabase Storage / local
        if storage_path:
            if storage_path.startswith("users/"):
                await storage_service.delete_file(storage_path)
            else:
                local_path = Path(storage_path)
                if local_path.is_file():
                    try:
                        local_path.unlink()
                    except Exception as err:
                        logger.warning("Could not delete local file %s: %s", local_path, err)
                # Also try storage_service delete in case it was migrated
                await storage_service.delete_file(storage_path)

        # 3. Delete PostgreSQL record
        session.delete(pg_doc)
        session.commit()

        logger.info("Document '%s' (%s) deleted by user %d", filename, document_id, pg_user_id)
        return {
            "message": f"Document '{filename}' deleted successfully",
            "document_id": document_id,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Delete error for %s: %s", document_id, exc, exc_info=True)
        session.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to delete document: {exc}"
        ) from exc
    finally:
        session.close()
