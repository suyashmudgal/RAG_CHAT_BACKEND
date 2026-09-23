"""Document processing progress service and state machine.

Maintains fine-grained processing stages:
QUEUED -> UPLOADING -> UPLOADED -> EXTRACTING -> PARSING -> CHUNKING -> EMBEDDING -> INDEXING -> FINALIZING -> COMPLETED (or FAILED)

Provides thread-safe in-memory caching for sub-millisecond polling responses,
with persistent synchronization to PostgreSQL.
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Any

from sqlalchemy.orm import Session

from database.session import SessionLocal
from models.database import Document as PgDocument

logger = logging.getLogger(__name__)


class ProcessingStage(str, Enum):
    """Document processing pipeline stages."""

    QUEUED = "QUEUED"
    UPLOADING = "UPLOADING"
    UPLOADED = "UPLOADED"
    EXTRACTING = "EXTRACTING"
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    FINALIZING = "FINALIZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


STAGE_DEFAULTS: dict[ProcessingStage, tuple[int, str]] = {
    ProcessingStage.QUEUED: (5, "Document queued for processing..."),
    ProcessingStage.UPLOADING: (15, "Uploading file to storage..."),
    ProcessingStage.UPLOADED: (25, "File uploaded to storage. Initializing parser..."),
    ProcessingStage.EXTRACTING: (35, "Extracting text from document..."),
    ProcessingStage.PARSING: (45, "Parsing and cleaning extracted content..."),
    ProcessingStage.CHUNKING: (55, "Chunking document into semantic passages..."),
    ProcessingStage.EMBEDDING: (60, "Generating dense vector embeddings..."),
    ProcessingStage.INDEXING: (90, "Indexing chunks in Chroma vector database..."),
    ProcessingStage.FINALIZING: (95, "Finalizing document metadata..."),
    ProcessingStage.COMPLETED: (100, "Document processing completed successfully"),
    ProcessingStage.FAILED: (0, "Document processing failed"),
}


class ProgressTracker:
    """Thread-safe document processing progress manager."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._completed_cache: dict[str, dict[str, Any]] = {}

    def evict_job(self, document_id: str) -> None:
        """Evict a document from in-memory tracking and status cache (e.g. on deletion)."""
        with self._lock:
            self._jobs.pop(document_id, None)
            self._completed_cache.pop(document_id, None)

    def register_job(
        self,
        document_id: str,
        user_id: int,
        filename: str,
        initial_stage: ProcessingStage = ProcessingStage.QUEUED,
    ) -> dict[str, Any]:
        """Register a new active document processing job."""
        default_progress, default_msg = STAGE_DEFAULTS.get(
            initial_stage, (0, "Initializing...")
        )
        job_data = {
            "document_id": document_id,
            "user_id": user_id,
            "filename": filename,
            "status": initial_stage.value,
            "processing_stage": initial_stage.value,
            "progress": default_progress,
            "message": default_msg,
            "chunk_count": 0,
            "processed_chunks": 0,
            "error": None,
        }
        with self._lock:
            self._jobs[document_id] = job_data
        return job_data

    def update_stage(
        self,
        document_id: str,
        stage: ProcessingStage,
        message: str | None = None,
        progress: int | None = None,
        chunk_count: int | None = None,
        processed_chunks: int | None = None,
        error: str | None = None,
        sync_db: bool = True,
    ) -> dict[str, Any]:
        """Advance job state machine, updating in-memory cache and PostgreSQL."""
        default_prog, default_msg = STAGE_DEFAULTS.get(stage, (0, ""))
        new_progress = progress if progress is not None else default_prog
        new_message = message if message is not None else default_msg

        with self._lock:
            job = self._jobs.get(document_id, {})
            job["document_id"] = document_id
            job["status"] = stage.value
            job["processing_stage"] = stage.value
            job["progress"] = new_progress
            job["message"] = new_message
            if chunk_count is not None:
                job["chunk_count"] = chunk_count
            if processed_chunks is not None:
                job["processed_chunks"] = processed_chunks
            if error is not None:
                job["error"] = error
            self._jobs[document_id] = job

        if sync_db:
            self._persist_to_db(
                document_id=document_id,
                stage=stage,
                progress=new_progress,
                chunk_count=chunk_count,
                processed_chunks=processed_chunks,
                error_message=error,
            )

        return job

    def update_embedding_progress(
        self,
        document_id: str,
        processed_chunks: int,
        total_chunks: int,
        sync_db: bool = False,
    ) -> dict[str, Any]:
        """Update granular progress during vector embedding (55% -> 85%)."""
        total = max(1, total_chunks)
        # Scale embedding range across 55% -> 85%
        ratio = min(1.0, max(0.0, processed_chunks / total))
        calculated_progress = 55 + int(30 * ratio)
        message = f"Generating vector embeddings ({processed_chunks}/{total_chunks} chunks)..."

        with self._lock:
            job = self._jobs.get(document_id, {})
            job["status"] = ProcessingStage.EMBEDDING.value
            job["processing_stage"] = ProcessingStage.EMBEDDING.value
            job["progress"] = calculated_progress
            job["message"] = message
            job["chunk_count"] = total_chunks
            job["processed_chunks"] = processed_chunks
            self._jobs[document_id] = job

        if sync_db:
            self._persist_to_db(
                document_id=document_id,
                stage=ProcessingStage.EMBEDDING,
                progress=calculated_progress,
                chunk_count=total_chunks,
                processed_chunks=processed_chunks,
            )

        return job

    def mark_completed(
        self, document_id: str, chunk_count: int | None = None
    ) -> dict[str, Any]:
        """Mark document processing as completed successfully."""
        res = self.update_stage(
            document_id=document_id,
            stage=ProcessingStage.COMPLETED,
            message="Document processing completed successfully",
            progress=100,
            chunk_count=chunk_count,
            processed_chunks=chunk_count,
            sync_db=True,
        )
        with self._lock:
            self._completed_cache[document_id] = dict(res)
        return res

    def mark_failed(self, document_id: str, error_message: str) -> dict[str, Any]:
        """Mark document processing as failed with safe sanitized error message."""
        # Sanitize message to never leak secrets, database credentials, or tracebacks
        clean_msg = str(error_message).split("\n")[0]
        if len(clean_msg) > 300:
            clean_msg = clean_msg[:297] + "..."

        res = self.update_stage(
            document_id=document_id,
            stage=ProcessingStage.FAILED,
            message=f"Processing failed: {clean_msg}",
            error=clean_msg,
            sync_db=True,
        )
        with self._lock:
            self._completed_cache[document_id] = dict(res)
        return res

    def get_status(
        self, document_id: str, user_id: int | None = None
    ) -> dict[str, Any] | None:
        """Retrieve current document processing status, enforcing user ownership.

        Returns None if document not found or belongs to another user.
        Uses two tiers of in-memory caching to guarantee sub-millisecond polling response.
        """
        # 1. Check in-memory active jobs
        with self._lock:
            active_job = self._jobs.get(document_id)
            if active_job:
                job_user_id = active_job.get("user_id")
                if user_id is not None and job_user_id is not None and job_user_id != user_id:
                    return None
                return dict(active_job)

            # 2. Check in-memory completed cache
            cached_job = self._completed_cache.get(document_id)
            if cached_job:
                cached_user_id = cached_job.get("user_id")
                if user_id is not None and cached_user_id is not None and cached_user_id != user_id:
                    return None
                return dict(cached_job)

        # 3. Check PostgreSQL source of truth using column projection
        session = SessionLocal()
        try:
            doc = (
                session.query(
                    PgDocument.id,
                    PgDocument.user_id,
                    PgDocument.filename,
                    PgDocument.status,
                    PgDocument.processing_stage,
                    PgDocument.progress,
                    PgDocument.chunk_count,
                    PgDocument.processed_chunks,
                    PgDocument.error_message,
                )
                .filter(PgDocument.id == document_id)
                .first()
            )
            if not doc:
                return None
            if user_id is not None and doc.user_id != user_id:
                return None

            # Map legacy status 'processed' to 'COMPLETED'
            raw_status = (doc.status or "processed").upper()
            stage = doc.processing_stage or ("COMPLETED" if raw_status in ("PROCESSED", "COMPLETED") else raw_status)
            progress = doc.progress if doc.progress is not None else (100 if stage == "COMPLETED" else 0)
            processed_chunks = doc.processed_chunks if doc.processed_chunks is not None else (doc.chunk_count or 0)

            msg = "Document ready" if stage == "COMPLETED" else f"Status: {stage}"
            if doc.error_message:
                msg = f"Processing failed: {doc.error_message}"

            status_result = {
                "document_id": doc.id,
                "user_id": doc.user_id,
                "filename": doc.filename,
                "status": stage,
                "processing_stage": stage,
                "progress": progress,
                "message": msg,
                "chunk_count": doc.chunk_count or 0,
                "processed_chunks": processed_chunks,
                "error": doc.error_message,
            }

            # Cache terminal status to avoid future DB hits during polling
            with self._lock:
                self._completed_cache[document_id] = dict(status_result)

            return status_result
        except Exception as exc:
            logger.error("Error reading document status from DB for %s: %s", document_id, exc)
            return None
        finally:
            session.close()

    def _persist_to_db(
        self,
        document_id: str,
        stage: ProcessingStage,
        progress: int,
        chunk_count: int | None = None,
        processed_chunks: int | None = None,
        error_message: str | None = None,
    ) -> None:
        """Persist state update to PostgreSQL."""
        session: Session = SessionLocal()
        try:
            doc = (
                session.query(PgDocument)
                .filter(PgDocument.id == document_id)
                .first()
            )
            if doc:
                doc.status = "processed" if stage == ProcessingStage.COMPLETED else stage.value
                doc.processing_stage = stage.value
                doc.progress = progress
                if chunk_count is not None:
                    doc.chunk_count = chunk_count
                if processed_chunks is not None:
                    doc.processed_chunks = processed_chunks
                if error_message is not None:
                    doc.error_message = error_message
                session.commit()
        except Exception as exc:
            session.rollback()
            logger.warning("Failed to sync progress to DB for %s: %s", document_id, exc)
        finally:
            session.close()


# Global tracker singleton
progress_tracker = ProgressTracker()
