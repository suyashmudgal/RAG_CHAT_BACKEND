"""Lazy singletons for shared services (avoids circular imports)."""

from __future__ import annotations

from functools import lru_cache

from config.settings import settings


@lru_cache()
def get_vector_store():
    """Return the singleton :class:`VectorStore`."""
    from services.vector_store import VectorStore

    return VectorStore(
        persist_dir=settings.chroma_persist_dir,
        embedding_model_name=settings.embedding_model,
    )


@lru_cache()
def get_document_processor():
    """Return the singleton :class:`DocumentProcessor`."""
    from services.document_processor import DocumentProcessor

    return DocumentProcessor(vector_store=get_vector_store())


@lru_cache()
def get_chat_service():
    """Return the singleton :class:`ChatService`."""
    from services.chat_service import ChatService

    return ChatService(vector_store=get_vector_store())
