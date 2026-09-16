"""POST /upload — receive, validate, and process document files.

Uploads physical files to Supabase Storage under user-scoped paths:
    users/{pg_user_id}/documents/{document_id}/{safe_filename}
Processes document chunks into ChromaDB with ephemeral local processing,
and persists metadata into PostgreSQL.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from auth.dependencies import get_current_user
from config.settings import settings
from database.session import SessionLocal
from models.database import Document as PgDocument
from services.deps import get_document_processor, get_storage_service, get_vector_store
from services.storage_service import build_storage_path, sanitize_filename

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@router.post("/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    user: dict = Depends(get_current_user),
):
    """Upload one or more documents for indexing with Supabase Storage."""
    processor = get_document_processor()
    storage_service = get_storage_service()
    vector_store = get_vector_store()
    pg_user_id: int = user["pg_id"]

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    results: list[dict] = []

    for file in files:
        raw_filename = file.filename or "unknown"
        # Sanitize display filename and validate extension
        safe_name = sanitize_filename(raw_filename)
        ext = Path(safe_name).suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:
            # Also check raw_filename extension in case sanitization altered it
            raw_ext = Path(raw_filename).suffix.lower()
            if raw_ext in ALLOWED_EXTENSIONS:
                ext = raw_ext
            else:
                results.append(
                    {
                        "filename": raw_filename,
                        "status": "error",
                        "message": (
                            f"Unsupported file type: {raw_ext or ext}. "
                            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                        ),
                    }
                )
                continue

        try:
            # ── read bytes ──────────────────────────────────────────
            content = await file.read()

            if len(content) == 0:
                results.append(
                    {"filename": raw_filename, "status": "error", "message": "File is empty"}
                )
                continue

            if len(content) > max_bytes:
                results.append(
                    {
                        "filename": raw_filename,
                        "status": "error",
                        "message": (
                            f"File too large ({len(content) / 1024 / 1024:.1f} MB). "
                            f"Maximum: {settings.max_file_size_mb} MB"
                        ),
                    }
                )
                continue

            # ── deterministic storage path ──────────────────────────
            document_id = str(uuid.uuid4())
            storage_path = build_storage_path(pg_user_id, document_id, raw_filename)

            # ── 1. Upload to Supabase Storage ───────────────────────
            try:
                await storage_service.upload_file(
                    storage_path=storage_path,
                    content=content,
                    content_type=file.content_type,
                )
            except Exception as storage_exc:
                logger.error(
                    "Supabase storage upload failed for %s: %s",
                    raw_filename,
                    storage_exc,
                    exc_info=True,
                )
                results.append(
                    {
                        "filename": raw_filename,
                        "status": "error",
                        "message": f"Storage upload failed: {storage_exc}",
                    }
                )
                continue

            # ── 2. Ephemeral local processing for text extraction ────
            temp_file_path: Path | None = None
            processing_succeeded = False
            try:
                # Write to ephemeral temp file for LangChain loaders
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(content)
                    temp_file_path = Path(tmp.name)

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    processor.process_document,
                    temp_file_path,
                    raw_filename,
                    pg_user_id,
                    document_id,
                )
                processing_succeeded = True

                # ── 3. Record in PostgreSQL ─────────────────────────
                session = SessionLocal()
                try:
                    pg_doc = PgDocument(
                        id=document_id,
                        user_id=pg_user_id,
                        filename=raw_filename,
                        storage_path=storage_path,
                        file_size=len(content),
                        status=result.get("status", "processed"),
                        chunk_count=result.get("chunk_count", 0),
                    )
                    session.add(pg_doc)
                    session.commit()
                except Exception as db_exc:
                    logger.error(
                        "Failed to record document %s in PostgreSQL: %s",
                        document_id,
                        db_exc,
                        exc_info=True,
                    )
                    session.rollback()
                    # Clean up ChromaDB and Storage on DB failure
                    vector_store.delete_document(document_id)
                    await storage_service.delete_file(storage_path)
                    results.append(
                        {
                            "filename": raw_filename,
                            "status": "error",
                            "message": f"Database recording failed: {db_exc}",
                        }
                    )
                    continue
                finally:
                    session.close()

                # Enrich result with storage_path
                result["storage_path"] = storage_path
                results.append(result)

            except ValueError as val_exc:
                logger.warning("Validation error processing %s: %s", raw_filename, val_exc)
                # Cleanup physical file on processing error
                await storage_service.delete_file(storage_path)
                if processing_succeeded:
                    vector_store.delete_document(document_id)
                results.append(
                    {"filename": raw_filename, "status": "error", "message": str(val_exc)}
                )
            except Exception as proc_exc:
                logger.error(
                    "Processing error for %s: %s", raw_filename, proc_exc, exc_info=True
                )
                # Cleanup physical file on processing error
                await storage_service.delete_file(storage_path)
                if processing_succeeded:
                    vector_store.delete_document(document_id)
                results.append(
                    {
                        "filename": raw_filename,
                        "status": "error",
                        "message": f"Failed to process document: {proc_exc}",
                    }
                )
            finally:
                # Ephemeral local file cleanup
                if temp_file_path and temp_file_path.exists():
                    try:
                        temp_file_path.unlink()
                    except Exception as clean_exc:
                        logger.warning("Could not delete temporary file %s: %s", temp_file_path, clean_exc)

        except Exception as exc:
            logger.error("Upload error for %s: %s", raw_filename, exc, exc_info=True)
            results.append(
                {"filename": raw_filename, "status": "error", "message": f"Upload failed: {exc}"}
            )

    return {"results": results}
