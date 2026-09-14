"""POST /upload — receive, validate, and process document files."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from auth.dependencies import get_current_user
from config.settings import settings
from database.session import SessionLocal
from models.database import Document as PgDocument

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@router.post("/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    user: dict = Depends(get_current_user),
):
    """Upload one or more documents for indexing."""
    from services.deps import get_document_processor

    processor = get_document_processor()
    pg_user_id: int = user["pg_id"]

    # Per-user upload directory
    upload_dir = Path(settings.upload_dir) / f"user_{pg_user_id}"
    upload_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = settings.max_file_size_mb * 1024 * 1024

    results: list[dict] = []

    for file in files:
        filename = file.filename or "unknown"
        try:
            # ── extension check ─────────────────────────────────────
            ext = Path(filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                results.append(
                    {
                        "filename": filename,
                        "status": "error",
                        "message": (
                            f"Unsupported file type: {ext}. "
                            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                        ),
                    }
                )
                continue

            # ── read bytes ──────────────────────────────────────────
            content = await file.read()

            if len(content) == 0:
                results.append(
                    {"filename": filename, "status": "error", "message": "File is empty"}
                )
                continue

            if len(content) > max_bytes:
                results.append(
                    {
                        "filename": filename,
                        "status": "error",
                        "message": (
                            f"File too large ({len(content) / 1024 / 1024:.1f} MB). "
                            f"Maximum: {settings.max_file_size_mb} MB"
                        ),
                    }
                )
                continue

            # ── save to disk ────────────────────────────────────────
            file_id = str(uuid.uuid4())
            file_path = upload_dir / f"{file_id}{ext}"
            file_path.write_bytes(content)

            # ── process (run CPU-heavy work in a thread) ────────────
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    processor.process_document,
                    file_path,
                    filename,
                    pg_user_id,
                )

                # Record in PostgreSQL
                try:
                    session = SessionLocal()
                    pg_doc = PgDocument(
                        id=result["document_id"],
                        user_id=pg_user_id,
                        filename=filename,
                        storage_path=str(file_path),
                        file_size=len(content),
                        status=result["status"],
                        chunk_count=result["chunk_count"],
                    )
                    session.add(pg_doc)
                    session.commit()
                except Exception as db_exc:
                    logger.error("Failed to record document in PostgreSQL: %s", db_exc)
                    session.rollback()
                finally:
                    session.close()

                results.append(result)

            except ValueError as exc:
                results.append(
                    {"filename": filename, "status": "error", "message": str(exc)}
                )
            except Exception as exc:
                logger.error("Processing error for %s: %s", filename, exc, exc_info=True)
                results.append(
                    {
                        "filename": filename,
                        "status": "error",
                        "message": f"Failed to process document: {exc}",
                    }
                )

        except Exception as exc:
            logger.error("Upload error for %s: %s", filename, exc, exc_info=True)
            results.append(
                {"filename": filename, "status": "error", "message": f"Upload failed: {exc}"}
            )

    return {"results": results}
