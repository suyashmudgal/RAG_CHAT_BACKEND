"""POST /upload — receive, validate, and non-blocking process document files.

Uploads physical files to Supabase Storage under user-scoped paths:
    users/{pg_user_id}/documents/{document_id}/{safe_filename}
Initiates non-blocking background document processing (extraction, chunking,
embedding, Chroma indexing) while maintaining real-time progress state.

Supports ?sync=true for synchronous processing compatibility with regression tests.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, UploadFile

from auth.dependencies import get_current_user
from config.settings import settings
from database.session import SessionLocal
from models.database import Document as PgDocument
from services.deps import get_document_processor, get_storage_service, get_vector_store
from services.progress_service import ProcessingStage, progress_tracker
from services.storage_service import build_storage_path, sanitize_filename

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


async def _run_document_processing(
    document_id: str,
    raw_filename: str,
    content: bytes,
    ext: str,
    pg_user_id: int,
    storage_path: str,
    is_sync: bool = False,
) -> dict:
    """Execute document extraction, chunking, embedding, and vector indexing."""
    processor = get_document_processor()
    storage_service = get_storage_service()
    vector_store = get_vector_store()

    temp_file_path: Path | None = None
    processing_succeeded = False
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            temp_file_path = Path(tmp.name)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            processor.process_document,
            temp_file_path,
            raw_filename,
            pg_user_id,
            document_id,
        )
        processing_succeeded = True
        chunk_count = result.get("chunk_count", 0)

        # Update PostgreSQL record to COMPLETED
        session = SessionLocal()
        try:
            doc = session.query(PgDocument).filter(PgDocument.id == document_id).first()
            if doc:
                doc.status = "processed"
                doc.processing_stage = "COMPLETED"
                doc.progress = 100
                doc.chunk_count = chunk_count
                doc.processed_chunks = chunk_count
                doc.error_message = None
                session.commit()
        except Exception as db_err:
            logger.error("Failed to commit completed status to DB for %s: %s", document_id, db_err)
            session.rollback()
        finally:
            session.close()

        progress_tracker.mark_completed(document_id, chunk_count=chunk_count)
        return result

    except Exception as proc_exc:
        logger.error(
            "Processing error for %s (%s): %s", raw_filename, document_id, proc_exc, exc_info=True
        )
        clean_err = str(proc_exc).split("\n")[0]

        # 1. Rollback ChromaDB vectors
        try:
            vector_store.delete_document(document_id)
        except Exception as v_err:
            logger.warning("Error deleting ChromaDB vectors for %s: %s", document_id, v_err)

        # 2. Rollback Supabase Storage file
        try:
            await storage_service.delete_file(storage_path)
        except Exception as s_err:
            logger.warning("Error deleting storage file %s: %s", storage_path, s_err)

        # 3. Synchronous mode: clean up PostgreSQL record so no orphan remains (Phase 5 test expectations)
        if is_sync:
            session = SessionLocal()
            try:
                doc = session.query(PgDocument).filter(PgDocument.id == document_id).first()
                if doc:
                    session.delete(doc)
                    session.commit()
            except Exception as d_err:
                session.rollback()
                logger.warning("Error deleting DB record for %s: %s", document_id, d_err)
            finally:
                session.close()
            raise proc_exc

        # 4. Asynchronous / background mode: mark record as FAILED with safe error message
        session = SessionLocal()
        try:
            doc = session.query(PgDocument).filter(PgDocument.id == document_id).first()
            if doc:
                doc.status = "FAILED"
                doc.processing_stage = "FAILED"
                doc.error_message = clean_err
                session.commit()
        except Exception as d_err:
            session.rollback()
            logger.warning("Error updating DB record to FAILED for %s: %s", document_id, d_err)
        finally:
            session.close()

        progress_tracker.mark_failed(document_id, clean_err)
        return {"document_id": document_id, "status": "FAILED", "error": clean_err}

    finally:
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception as clean_exc:
                logger.warning("Could not delete temporary file %s: %s", temp_file_path, clean_exc)


@router.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    sync: bool = Query(False),
    user: dict = Depends(get_current_user),
):
    """Upload one or more documents for non-blocking indexing with Supabase Storage.

    Accepts ?sync=true to run extraction and indexing synchronously when required.
    """
    storage_service = get_storage_service()
    pg_user_id: int = user["pg_id"]

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    results: list[dict] = []

    for file in files:
        raw_filename = file.filename or "unknown"
        safe_name = sanitize_filename(raw_filename)
        ext = Path(safe_name).suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:
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

            document_id = str(uuid.uuid4())
            storage_path = build_storage_path(pg_user_id, document_id, raw_filename)

            # Register job in real-time tracker
            progress_tracker.register_job(
                document_id=document_id,
                user_id=pg_user_id,
                filename=raw_filename,
                initial_stage=ProcessingStage.QUEUED,
            )

            # ── 1. Upload to Supabase Storage ───────────────────────
            progress_tracker.update_stage(document_id, ProcessingStage.UPLOADING)
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
                progress_tracker.mark_failed(document_id, f"Storage upload failed: {storage_exc}")
                results.append(
                    {
                        "filename": raw_filename,
                        "status": "error",
                        "message": f"Storage upload failed: {storage_exc}",
                    }
                )
                continue

            # ── 2. Create PostgreSQL Document Record ────────────────
            progress_tracker.update_stage(
                document_id, ProcessingStage.UPLOADED, progress=25, sync_db=False
            )
            session = SessionLocal()
            try:
                pg_doc = PgDocument(
                    id=document_id,
                    user_id=pg_user_id,
                    filename=raw_filename,
                    storage_path=storage_path,
                    file_size=len(content),
                    status=ProcessingStage.UPLOADED.value,
                    processing_stage=ProcessingStage.UPLOADED.value,
                    progress=25,
                    chunk_count=0,
                    processed_chunks=0,
                )
                session.add(pg_doc)
                session.commit()
            except Exception as db_exc:
                logger.error("Failed to record document %s in DB: %s", document_id, db_exc)
                session.rollback()
                await storage_service.delete_file(storage_path)
                progress_tracker.mark_failed(document_id, f"Database recording failed: {db_exc}")
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

            # ── 3. Processing: Synchronous vs Asynchronous Background ────
            if sync:
                try:
                    sync_res = await _run_document_processing(
                        document_id=document_id,
                        raw_filename=raw_filename,
                        content=content,
                        ext=ext,
                        pg_user_id=pg_user_id,
                        storage_path=storage_path,
                        is_sync=True,
                    )
                    sync_res["storage_path"] = storage_path
                    results.append(sync_res)
                except ValueError as val_exc:
                    results.append(
                        {"filename": raw_filename, "status": "error", "message": str(val_exc)}
                    )
                except Exception as proc_exc:
                    results.append(
                        {
                            "filename": raw_filename,
                            "status": "error",
                            "message": f"Processing error: {proc_exc}",
                        }
                    )
            else:
                # Non-blocking background execution
                background_tasks.add_task(
                    _run_document_processing,
                    document_id,
                    raw_filename,
                    content,
                    ext,
                    pg_user_id,
                    storage_path,
                    False,
                )
                results.append(
                    {
                        "document_id": document_id,
                        "filename": raw_filename,
                        "status": "UPLOADED",
                        "processing_stage": "UPLOADED",
                        "progress": 25,
                        "chunk_count": 0,
                        "processed_chunks": 0,
                        "message": "File uploaded successfully. Processing in background.",
                        "storage_path": storage_path,
                    }
                )

        except Exception as exc:
            logger.error("Upload error for %s: %s", raw_filename, exc, exc_info=True)
            results.append(
                {"filename": raw_filename, "status": "error", "message": f"Upload failed: {exc}"}
            )

    return {"results": results}
