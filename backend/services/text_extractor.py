"""Text extraction using LangChain document loaders.

Supports PDF (page-aware), DOCX, and TXT.
Returns lists of LangChain ``Document`` objects with enriched metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ── PDF ─────────────────────────────────────────────────────────────────────


def extract_pdf(file_path: Path) -> list[Document]:
    """Load a PDF with text extraction and visual/chart understanding."""
    from langchain_community.document_loaders import PyPDFLoader
    from services.visual_extractor import visual_extractor

    try:
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
    except Exception as exc:
        raise ValueError(f"Corrupted or invalid PDF file: {exc}") from exc

    # Filter empty pages and enrich metadata
    result: list[Document] = []
    for doc in docs:
        if not doc.page_content.strip():
            continue
        doc.metadata["page_number"] = doc.metadata.get("page", 0) + 1  # 1-indexed
        doc.metadata["source_type"] = "pdf"
        doc.metadata["content_type"] = "text"
        result.append(doc)

    # Visual extraction (charts, diagrams, visual tables, scanned pages)
    try:
        visual_docs = visual_extractor.extract_visuals_from_pdf(file_path, filename=file_path.name)
        if visual_docs:
            logger.info("Extracted %d visual knowledge document(s) from %s", len(visual_docs), file_path.name)
            result.extend(visual_docs)
    except Exception as vis_err:
        logger.warning("Visual extraction encountered an error on %s: %s", file_path.name, vis_err)

    if not result:
        raise ValueError("PDF contains no extractable text or visual content")

    return result


# ── DOCX ────────────────────────────────────────────────────────────────────


def extract_docx(file_path: Path) -> list[Document]:
    """Load a DOCX with Docx2txtLoader."""
    from langchain_community.document_loaders import Docx2txtLoader

    try:
        loader = Docx2txtLoader(str(file_path))
        docs = loader.load()
    except Exception as exc:
        raise ValueError(f"Corrupted or invalid DOCX file: {exc}") from exc

    result: list[Document] = []
    for doc in docs:
        if not doc.page_content.strip():
            continue
        doc.metadata["page_number"] = 0  # DOCX has no page numbers
        doc.metadata["source_type"] = "docx"
        result.append(doc)

    if not result:
        raise ValueError("DOCX file contains no extractable text")

    return result


# ── TXT ─────────────────────────────────────────────────────────────────────


def extract_txt(file_path: Path) -> list[Document]:
    """Load a plain-text file (tries several encodings)."""
    from langchain_community.document_loaders import TextLoader

    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            loader = TextLoader(str(file_path), encoding=encoding)
            docs = loader.load()
            if docs and docs[0].page_content.strip():
                for doc in docs:
                    doc.metadata["page_number"] = 0  # TXT has no page numbers
                    doc.metadata["source_type"] = "txt"
                return docs
        except Exception:
            continue

    raise ValueError("Could not decode TXT file with any supported encoding")


# ── Dispatcher ──────────────────────────────────────────────────────────────

_EXTRACTORS = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".txt": extract_txt,
}


def extract_text(file_path: Path) -> list[Document]:
    """Extract text from a file based on its extension.

    Returns a list of LangChain ``Document`` objects.
    """
    suffix = file_path.suffix.lower()
    extractor = _EXTRACTORS.get(suffix)
    if extractor is None:
        raise ValueError(f"Unsupported file type: {suffix}")

    logger.info("Extracting text from %s (type: %s)", file_path.name, suffix)
    return extractor(file_path)
