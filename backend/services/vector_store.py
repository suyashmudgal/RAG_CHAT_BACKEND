"""ChromaDB vector store using LangChain's Chroma wrapper.

Stores document chunks with embeddings, supports CRUD operations,
and persists both vectors and document-level metadata to disk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from services.embedding_service import get_embeddings

logger = logging.getLogger(__name__)


class VectorStore:
    """Thin wrapper around LangChain ``Chroma`` with document management."""

    def __init__(self, persist_dir: str, embedding_model_name: str) -> None:
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self._persist_dir / "document_metadata.json"

        embeddings = get_embeddings(embedding_model_name)
        self.vectorstore = Chroma(
            collection_name="documents",
            embedding_function=embeddings,
            persist_directory=str(self._persist_dir),
        )

        self._doc_metadata: dict[str, dict[str, Any]] = self._load_metadata()
        self._chunk_count: int = self.vectorstore._collection.count()
        logger.info(
            "VectorStore ready – %d document(s), %d chunk(s)",
            len(self._doc_metadata),
            self._chunk_count,
        )

    # ── metadata persistence ────────────────────────────────────────────

    def _load_metadata(self) -> dict[str, dict[str, Any]]:
        if self._metadata_file.exists():
            try:
                return json.loads(self._metadata_file.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Could not read metadata file – starting fresh.")
                return {}
        return {}

    def _save_metadata(self) -> None:
        self._metadata_file.write_text(
            json.dumps(self._doc_metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── public CRUD ─────────────────────────────────────────────────────

    def add_documents(
        self,
        document_id: str,
        chunks: list[Document],
        doc_info: dict[str, Any],
    ) -> None:
        """Index chunked ``Document`` objects for a single upload."""
        # Stamp every chunk with the document_id
        for chunk in chunks:
            chunk.metadata["document_id"] = document_id

        ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        self.vectorstore.add_documents(chunks, ids=ids)
        self._chunk_count += len(chunks)

        self._doc_metadata[document_id] = doc_info
        self._save_metadata()
        logger.info("Indexed %d chunks for document %s", len(chunks), document_id)

    def similarity_search(
        self, query: str, top_k: int = 5
    ) -> list[tuple[Document, float]]:
        """Return the *top_k* most relevant chunks with scores."""
        if self._chunk_count <= 0:
            return []
        k = min(top_k, self._chunk_count)
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def get_chunks_by_page(
        self, page_number: int, document_id: str | None = None
    ) -> list[Document]:
        """Return all chunks belonging to a specific page number.

        Optionally filter by *document_id* when multiple documents exist.
        """
        collection = self.vectorstore._collection
        where_filter: dict = {"page_number": page_number}
        if document_id:
            where_filter = {
                "$and": [
                    {"page_number": page_number},
                    {"document_id": document_id},
                ]
            }

        try:
            results = collection.get(where=where_filter, include=["documents", "metadatas"])
            docs: list[Document] = []
            for text, meta in zip(results["documents"] or [], results["metadatas"] or []):
                docs.append(Document(page_content=text, metadata=meta))
            return docs
        except Exception as exc:
            logger.warning("get_chunks_by_page error: %s", exc)
            return []

    def get_all_metadata(self) -> dict[str, dict]:
        """Return the full document-metadata dictionary."""
        return dict(self._doc_metadata)

    def delete_document(self, document_id: str) -> None:
        """Remove every chunk belonging to *document_id*."""
        collection = self.vectorstore._collection
        try:
            results = collection.get(where={"document_id": document_id})
            if results["ids"]:
                self.vectorstore.delete(ids=results["ids"])
                self._chunk_count = self.vectorstore._collection.count()
                logger.info(
                    "Deleted %d chunk(s) for document %s",
                    len(results["ids"]),
                    document_id,
                )
        except Exception as exc:
            logger.warning("Error deleting chunks from ChromaDB: %s", exc)


        self._doc_metadata.pop(document_id, None)
        self._save_metadata()

    def list_documents(self) -> list[dict[str, Any]]:
        """Return metadata for every indexed document."""
        return list(self._doc_metadata.values())

    def get_document_info(self, document_id: str) -> dict[str, Any] | None:
        """Return metadata for a single document (or ``None``)."""
        return self._doc_metadata.get(document_id)
