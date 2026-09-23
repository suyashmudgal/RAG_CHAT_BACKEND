"""Embedding service using LangChain's HuggingFaceEmbeddings (local, free).

The model (~90 MB) is downloaded automatically on the first run and
cached locally.  No API key is required.
"""

from __future__ import annotations

# Guard against Windows DLL conflict (PyArrow / PyTorch CRT collision 0xC0000005)
try:
    import pyarrow  # noqa: F401
except ImportError:
    pass

import logging
import os
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """Return a cached ``HuggingFaceEmbeddings`` instance (singleton)."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    logger.info("Loading embedding model: %s (first run downloads ~90 MB)…", model_name)
    try:
        # Fast path: use local cache to prevent redundant online HF Hub HTTP roundtrips
        try:
            instance = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu", "local_files_only": True},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
            )
            _embeddings = instance
            logger.info("Embedding model loaded successfully from local cache.")
            return _embeddings
        except Exception:
            pass

        # Fallback: allow downloading model if not yet cached locally
        instance = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )
        _embeddings = instance
        logger.info("Embedding model loaded successfully.")
        return _embeddings
    except Exception as exc:
        hf_cache_dir = os.environ.get(
            "HF_HOME",
            str(Path.home() / ".cache" / "huggingface" / "hub"),
        )
        logger.error(
            "Failed to load embedding model '%s': %s\n"
            "Troubleshooting tips:\n"
            "  1. Check network connectivity if the model needs to be downloaded.\n"
            "  2. Check for corrupted model cache at: %s\n"
            "  3. Check disk permissions and available space.\n",
            model_name,
            exc,
            hf_cache_dir,
            exc_info=True,
        )
        _embeddings = None
        raise RuntimeError(
            f"Could not load embedding model '{model_name}'. Underlying error: {exc}"
        ) from exc

