"""Document processing orchestrator.

Coordinates the full pipeline:
    file → extract → chunk → embed → store
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings
from services.progress_service import ProcessingStage, progress_tracker
from services.text_chunker import DocumentChunker
from services.text_extractor import extract_text
from services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Extract, chunk, and index a single uploaded document."""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store
        self.chunker = DocumentChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def process_document(
        self,
        file_path: Path,
        filename: str,
        user_id: int | None = None,
        document_id: str | None = None,
        progress_callback: Any | None = None,
    ) -> dict:
        """Run the full pipeline and return a result dict with real-time stage updates."""
        document_id = document_id or str(uuid.uuid4())

        logger.info("Processing document: %s (id: %s)", filename, document_id)

        # 1. Extract stage
        if progress_callback:
            progress_callback(ProcessingStage.EXTRACTING)
        else:
            progress_tracker.update_stage(document_id, ProcessingStage.EXTRACTING)

        documents = extract_text(file_path)

        # 2. Parsing stage
        if progress_callback:
            progress_callback(ProcessingStage.PARSING)
        else:
            progress_tracker.update_stage(document_id, ProcessingStage.PARSING)

        for doc in documents:
            doc.metadata["filename"] = filename
            if user_id is not None:
                doc.metadata["user_id"] = str(user_id)

        # 3. Chunking stage
        if progress_callback:
            progress_callback(ProcessingStage.CHUNKING)
        else:
            progress_tracker.update_stage(document_id, ProcessingStage.CHUNKING)

        chunks = self.chunker.chunk_documents(documents)
        if not chunks:
            raise ValueError("No text chunks could be created from the document")

        # 4. Build document-level metadata
        doc_info = {
            "document_id": document_id,
            "filename": filename,
            "status": "processed",
            "chunk_count": len(chunks),
            "upload_time": datetime.now(timezone.utc).isoformat(),
            "file_size": file_path.stat().st_size,
        }
        if user_id is not None:
            doc_info["user_id"] = user_id

        # 5. Embedding & Indexing stage with batch callback
        if progress_callback:
            progress_callback(ProcessingStage.EMBEDDING, chunk_count=len(chunks))
        else:
            progress_tracker.update_stage(
                document_id, ProcessingStage.EMBEDDING, chunk_count=len(chunks)
            )

        def _on_embedding_progress(processed: int, total: int):
            progress_tracker.update_embedding_progress(document_id, processed, total)

        self.vector_store.add_documents(
            document_id=document_id,
            chunks=chunks,
            doc_info=doc_info,
            user_id=user_id,
            progress_callback=_on_embedding_progress,
        )

        # 6. Finalizing
        if progress_callback:
            progress_callback(ProcessingStage.FINALIZING, chunk_count=len(chunks))
        else:
            progress_tracker.update_stage(
                document_id, ProcessingStage.FINALIZING, chunk_count=len(chunks)
            )

        logger.info(
            "Successfully processed %s → %d chunks", filename, len(chunks)
        )
        return {
            "document_id": document_id,
            "filename": filename,
            "status": "processed",
            "chunk_count": len(chunks),
            "message": f"Successfully processed {filename}",
        }
