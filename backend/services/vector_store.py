"""ChromaDB vector store using LangChain's Chroma wrapper.

Stores document chunks with embeddings, supports CRUD operations,
and persists both vectors and document-level metadata to disk.
"""

from __future__ import annotations

# Guard against Windows DLL conflict (PyArrow / PyTorch CRT collision 0xC0000005)
try:
    import pyarrow  # noqa: F401
except ImportError:
    pass

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
        user_id: int | None = None,
        progress_callback: Any | None = None,
        batch_size: int = 32,
    ) -> None:
        """Index chunked ``Document`` objects for a single upload in batches."""
        # Stamp every chunk with the document_id and user_id
        for chunk in chunks:
            chunk.metadata["document_id"] = document_id
            if user_id is not None:
                chunk.metadata["user_id"] = str(user_id)

        total_chunks = len(chunks)
        ids = [f"{document_id}_chunk_{i}" for i in range(total_chunks)]

        # Process in batches to vectorize embeddings efficiently and report progress
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            self.vectorstore.add_documents(batch_chunks, ids=batch_ids)

            processed_so_far = min(i + batch_size, total_chunks)
            if progress_callback is not None:
                try:
                    progress_callback(processed_so_far, total_chunks)
                except Exception as cb_err:
                    logger.warning("Progress callback error: %s", cb_err)

        self._chunk_count += total_chunks

        # Store user_id in document metadata
        if user_id is not None:
            doc_info["user_id"] = user_id
        self._doc_metadata[document_id] = doc_info
        self._save_metadata()
        logger.info("Indexed %d chunks for document %s", total_chunks, document_id)

    def similarity_search(
        self, query: str, top_k: int = 5
    ) -> list[tuple[Document, float]]:
        """Return the *top_k* most relevant chunks with scores (global, unfiltered)."""
        if self._chunk_count <= 0:
            return []
        k = min(top_k, self._chunk_count)
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def similarity_search_for_user(
        self, query: str, user_id: int, top_k: int = 5
    ) -> list[tuple[Document, float]]:
        """Return the *top_k* most relevant chunks belonging to *user_id* without collection scanning."""
        if self._chunk_count <= 0:
            return []

        try:
            return self.vectorstore.similarity_search_with_score(
                query, k=top_k, filter={"user_id": str(user_id)}
            )
        except Exception as exc:
            logger.warning("similarity_search_for_user query error: %s", exc)
            return []

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

    def get_chunks_by_page_for_user(
        self, page_number: int, user_id: int, document_id: str | None = None
    ) -> list[Document]:
        """Return chunks for a page filtered by user ownership."""
        collection = self.vectorstore._collection
        conditions: list[dict] = [
            {"page_number": page_number},
            {"user_id": str(user_id)},
        ]
        if document_id:
            conditions.append({"document_id": document_id})

        where_filter: dict = {"$and": conditions} if len(conditions) > 1 else conditions[0]

        try:
            results = collection.get(where=where_filter, include=["documents", "metadatas"])
            docs: list[Document] = []
            for text, meta in zip(results["documents"] or [], results["metadatas"] or []):
                docs.append(Document(page_content=text, metadata=meta))
            return docs
        except Exception as exc:
            logger.warning("get_chunks_by_page_for_user error: %s", exc)
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

    def list_documents_for_user(self, user_id: int) -> list[dict[str, Any]]:
        """Return metadata only for documents owned by *user_id*."""
        return [
            doc for doc in self._doc_metadata.values()
            if doc.get("user_id") == user_id
        ]

    def get_document_info(self, document_id: str) -> dict[str, Any] | None:
        """Return metadata for a single document (or ``None``)."""
        return self._doc_metadata.get(document_id)
