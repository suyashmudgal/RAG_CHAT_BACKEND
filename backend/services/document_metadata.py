"""Document metadata extraction and query classification.

Extracts structured metadata during ingestion and provides lightweight
query routing so metadata/page/summary questions bypass pure vector search.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ── Query categories ────────────────────────────────────────────────────────

METADATA = "METADATA"
PAGE_SPECIFIC = "PAGE_SPECIFIC"
SUMMARY = "SUMMARY"
CONTENT = "CONTENT"
MIXED = "MIXED"


# ── Metadata extraction ────────────────────────────────────────────────────


def extract_document_metadata(
    file_path: Path,
    filename: str,
    documents: list[Document],
) -> dict[str, Any]:
    """Build rich metadata from the uploaded file and its extracted documents.

    *documents* are the page-level ``Document`` objects produced by
    ``text_extractor.extract_text`` (before chunking).
    """
    ext = file_path.suffix.lower()
    meta: dict[str, Any] = {
        "filename": filename,
        "file_type": ext.lstrip("."),
        "file_size": file_path.stat().st_size,
    }

    if ext == ".pdf":
        _enrich_pdf_metadata(file_path, documents, meta)
    else:
        # DOCX / TXT — no reliable page count
        meta["total_pages"] = None
        meta["page_content_map"] = {}

    return meta


def _enrich_pdf_metadata(
    file_path: Path,
    documents: list[Document],
    meta: dict[str, Any],
) -> None:
    """Add PDF-specific metadata using pypdf and the extracted page documents."""

    # --- page count from extracted documents (most reliable) ----------------
    page_numbers = [
        int(doc.metadata.get("page_number", 0))
        for doc in documents
        if int(doc.metadata.get("page_number", 0)) > 0
    ]
    meta["total_pages"] = max(page_numbers) if page_numbers else len(documents)

    # --- page content map (first 300 chars per page for quick lookup) -------
    page_map: dict[str, str] = {}
    for doc in documents:
        pn = int(doc.metadata.get("page_number", 0))
        if pn > 0:
            page_map[str(pn)] = doc.page_content[:300].strip()
    meta["page_content_map"] = page_map

    # --- PDF info dict (title, author, dates) via pypdf --------------------
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))

        # pypdf exposes total physical pages — use this if higher than text pages
        physical_pages = len(reader.pages)
        if physical_pages > meta["total_pages"]:
            meta["total_pages"] = physical_pages

        info = reader.metadata
        if info:
            title = info.get("/Title") or ""
            if isinstance(title, str) and title.strip():
                meta["title"] = title.strip()

            author = info.get("/Author") or ""
            if isinstance(author, str) and author.strip():
                meta["author"] = author.strip()

            creation = info.get("/CreationDate") or ""
            if isinstance(creation, str) and creation.strip():
                meta["creation_date"] = creation.strip()

    except Exception as exc:
        logger.warning("Could not read PDF metadata via pypdf: %s", exc)


# ── Query classification ───────────────────────────────────────────────────

# Patterns compiled once at import time for speed
_META_PATTERNS = [
    re.compile(r"\bhow many pages\b", re.I),
    re.compile(r"\bpage count\b", re.I),
    re.compile(r"\bnumber of pages\b", re.I),
    re.compile(r"\btotal pages\b", re.I),
    re.compile(r"\bfile\s*(?:name|type|size|format)\b", re.I),
    re.compile(r"\bwho (?:is|are) the author", re.I),
    re.compile(r"\bwho wrote\b", re.I),
    re.compile(r"\bwho are the authors\b", re.I),
    re.compile(r"\btitle of (?:the|this)\b", re.I),
    re.compile(r"\bwhat is the title\b", re.I),
    re.compile(r"\bwhen was (?:it|this|the) (?:published|created|written)\b", re.I),
    re.compile(r"\bdocument (?:type|format|properties)\b", re.I),
]

_PAGE_PATTERNS = [
    re.compile(r"\b(?:what(?:'s| is| does)?|what is written) (?:on|in|at) page\s*\d+", re.I),
    re.compile(r"\bpage\s*\d+\s*(?:say|contain|discuss|show|have|talk)", re.I),
    re.compile(r"\bcontent (?:of|on|in) page\s*\d+", re.I),
    re.compile(r"\bread page\s*\d+", re.I),
    re.compile(r"\bwritten on page\s*\d+", re.I),
]

_SUMMARY_PATTERNS = [
    re.compile(r"\bsummar(?:y|ize|ise)\b", re.I),
    re.compile(r"\bwhat is (?:this|the) (?:paper|document|pdf|file) about\b", re.I),
    re.compile(r"\boverview\b", re.I),
    re.compile(r"\bgive (?:me )?(?:a |an )?(?:brief |short )?(?:summary|overview)\b", re.I),
    re.compile(r"\bmain (?:contribution|idea|point|finding|result)s?\b", re.I),
    re.compile(r"\bwhat (?:does|did) (?:this|the) (?:paper|document) (?:propose|present|introduce)\b", re.I),
]


def classify_query(question: str) -> str:
    """Classify a user question into a routing category.

    Returns one of: METADATA, PAGE_SPECIFIC, SUMMARY, CONTENT, MIXED.
    """
    has_meta = any(p.search(question) for p in _META_PATTERNS)
    has_page = any(p.search(question) for p in _PAGE_PATTERNS)
    has_summary = any(p.search(question) for p in _SUMMARY_PATTERNS)

    flags = sum([has_meta, has_page, has_summary])

    if flags >= 2:
        return MIXED
    if has_meta:
        return METADATA
    if has_page:
        return PAGE_SPECIFIC
    if has_summary:
        return SUMMARY
    return CONTENT


def extract_page_number_from_query(question: str) -> int | None:
    """Pull the target page number from a page-specific question."""
    m = re.search(r"\bpage\s*(\d+)\b", question, re.I)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None


# ── Metadata context building ──────────────────────────────────────────────


def build_metadata_context(doc_metadata_list: list[dict[str, Any]]) -> str:
    """Format all document metadata into a context string for the LLM."""
    if not doc_metadata_list:
        return ""

    parts: list[str] = []
    for i, meta in enumerate(doc_metadata_list, 1):
        lines = [f"Document {i}: {meta.get('filename', 'Unknown')}"]

        file_type = meta.get("file_type")
        if file_type:
            lines.append(f"  File type: {file_type.upper()}")

        total_pages = meta.get("total_pages")
        if total_pages is not None:
            lines.append(f"  Total pages: {total_pages}")

        file_size = meta.get("file_size")
        if file_size:
            if file_size < 1024:
                sz = f"{file_size} bytes"
            elif file_size < 1048576:
                sz = f"{file_size / 1024:.1f} KB"
            else:
                sz = f"{file_size / 1048576:.1f} MB"
            lines.append(f"  File size: {sz}")

        title = meta.get("title")
        if title:
            lines.append(f"  Title: {title}")

        author = meta.get("author")
        if author:
            lines.append(f"  Author(s): {author}")

        creation = meta.get("creation_date")
        if creation:
            lines.append(f"  Creation date: {creation}")

        chunk_count = meta.get("chunk_count")
        if chunk_count:
            lines.append(f"  Indexed chunks: {chunk_count}")

        parts.append("\n".join(lines))

    return "\n\n".join(parts)
