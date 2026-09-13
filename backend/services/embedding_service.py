"""Embedding service using LangChain's HuggingFaceEmbeddings (local, free).

The model (~90 MB) is downloaded automatically on the first run and
cached locally.  No API key is required.
"""

from __future__ import annotations

import logging

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """Return a cached ``HuggingFaceEmbeddings`` instance (singleton)."""
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s (first run downloads ~90 MB)…", model_name)
        _embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded successfully.")
    return _embeddings
