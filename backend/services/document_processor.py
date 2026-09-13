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

    def process_document(self, file_path: Path, filename: str) -> dict:
        """Run the full pipeline and return a result dict."""
        document_id = str(uuid.uuid4())

        logger.info("Processing document: %s", filename)

        # 1. Extract
        documents = extract_text(file_path)

        # 2. Tag every page-level Document with the filename
        for doc in documents:
            doc.metadata["filename"] = filename

        # 3. Chunk
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

        # 5. Store in vector DB (embeddings are generated automatically by Chroma)
        self.vector_store.add_documents(document_id, chunks, doc_info)

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
