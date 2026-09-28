"""Text and structural extraction for PDFs, DOCX, and TXT documents.

Extracts structured text, layout blocks, headings, numbered questions/options,
tables, formulas, and visual content into normalized LangChain Document objects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from langchain_core.documents import Document

from services.document_representation import ElementType
from services.visual_extractor import visual_extractor

logger = logging.getLogger(__name__)


def _format_table_to_markdown(table_data: list[list[Any]], filename: str, page_num: int) -> str:
    """Format extracted 2D table data into a clean Markdown table with semantic row breakdown."""
    if not table_data or len(table_data) < 1:
        return ""

    headers = [str(c or "").strip() for c in table_data[0]]
    if not any(headers):
        headers = [f"Col {i+1}" for i in range(len(headers))]

    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join([":---"] * len(headers)) + " |"

    row_lines = []
    semantic_rows = []
    for r_idx, row in enumerate(table_data[1:]):
        cells = [str(c or "").strip() for c in row]
        if len(cells) < len(headers):
            cells.extend([""] * (len(headers) - len(cells)))
        cells = cells[: len(headers)]
        row_lines.append("| " + " | ".join(cells) + " |")

        row_pairs = [f"{h} = {c}" for h, c in zip(headers, cells) if h and c]
        if row_pairs:
            semantic_rows.append(f"- Row {r_idx + 1}: " + ", ".join(row_pairs))

    lines = [
        "[TABLE CONTENT]",
        f"Document: {filename}",
        f"Page: {page_num}",
        "Element Type: TABLE",
        "",
        header_line,
        sep_line,
        *row_lines,
    ]
    if semantic_rows:
        lines.append("")
        lines.append("Semantic Row Values:")
        lines.extend(semantic_rows)

    return "\n".join(lines)


# ── PDF Extraction ──────────────────────────────────────────────────────────


def extract_pdf(file_path: Path) -> list[Document]:
    """Extract layout-aware text, tables, and visual knowledge from a PDF using PyMuPDF.

    Preserves paragraph boundaries, multi-column order, headings, tables, formulas,
    and numbered questions/options. Falls back to PyPDFLoader if fitz encounters issues.
    """
    filename = file_path.name
    results: list[Document] = []

    try:
        doc = fitz.open(str(file_path))
        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1  # 1-indexed

            # 1. Extract structured tables using fitz's table finder
            table_bboxes = []
            try:
                tabs = page.find_tables()
                if tabs and tabs.tables:
                    for t_idx, t in enumerate(tabs.tables):
                        table_bboxes.append(t.bbox)
                        t_data = t.extract()
                        if t_data and len(t_data) >= 2:
                            md_table = _format_table_to_markdown(t_data, filename, page_num)
                            results.append(
                                Document(
                                    page_content=md_table,
                                    metadata={
                                        "page_number": page_num,
                                        "source_type": "pdf",
                                        "content_type": "table",
                                        "element_type": ElementType.TABLE.value,
                                        "filename": filename,
                                        "table_index": t_idx,
                                    },
                                )
                            )
            except Exception as tab_err:
                logger.debug("Table detection error on page %d: %s", page_num, tab_err)

            # 2. Extract text blocks (ordered top-to-bottom, preserving layout and reading order)
            try:
                blocks = page.get_text("blocks")
                # Filter out blocks that fall inside extracted table bounding boxes
                text_blocks = []
                for b in blocks:
                    if b[6] != 0:  # 0 is text block, 1 is image
                        continue
                    b_bbox = fitz.Rect(b[0], b[1], b[2], b[3])
                    # Check if heavily overlapping any detected table
                    is_in_table = False
                    for tb in table_bboxes:
                        t_rect = fitz.Rect(tb)
                        if t_rect.contains(b_bbox) or (t_rect.intersect(b_bbox).get_area() > 0.5 * b_bbox.get_area()):
                            is_in_table = True
                            break
                    if not is_in_table:
                        txt = b[4].strip()
                        if txt:
                            text_blocks.append(txt)

                if text_blocks:
                    # Join text blocks with paragraph separators to preserve document structure
                    page_text = "\n\n".join(text_blocks)
                    results.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "page_number": page_num,
                                "source_type": "pdf",
                                "content_type": "text",
                                "element_type": ElementType.TEXT.value,
                                "filename": filename,
                            },
                        )
                    )
            except Exception as block_err:
                logger.warning("Block text extraction error on page %d: %s", page_num, block_err)

    except Exception as exc:
        logger.warning("PyMuPDF extraction failed for %s: %s, falling back to PyPDFLoader", filename, exc)
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(str(file_path))
        pypdf_docs = loader.load()
        for doc in pypdf_docs:
            if doc.page_content.strip():
                doc.metadata["page_number"] = doc.metadata.get("page", 0) + 1
                doc.metadata["source_type"] = "pdf"
                doc.metadata["content_type"] = "text"
                doc.metadata["element_type"] = ElementType.TEXT.value
                doc.metadata["filename"] = filename
                results.append(doc)

    # 3. Extract visual charts, graphs, diagrams, and scanned content
    try:
        visual_docs = visual_extractor.extract_visuals_from_pdf(file_path, filename=filename)
        if visual_docs:
            logger.info("Extracted %d visual knowledge document(s) from %s", len(visual_docs), filename)
            results.extend(visual_docs)
    except Exception as vis_err:
        logger.warning("Visual extraction encountered an error on %s: %s", filename, vis_err)

    if not results:
        raise ValueError("PDF contains no extractable text or visual content")

    return results


# ── DOCX Extraction ─────────────────────────────────────────────────────────


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
        doc.metadata["page_number"] = 0
        doc.metadata["source_type"] = "docx"
        doc.metadata["content_type"] = "text"
        doc.metadata["element_type"] = ElementType.TEXT.value
        result.append(doc)

    if not result:
        raise ValueError("DOCX file contains no extractable text")

    return result


# ── TXT Extraction ──────────────────────────────────────────────────────────


def extract_txt(file_path: Path) -> list[Document]:
    """Load a plain-text file with encoding fallbacks."""
    from langchain_community.document_loaders import TextLoader

    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            loader = TextLoader(str(file_path), encoding=encoding)
            docs = loader.load()
            if docs and docs[0].page_content.strip():
                for doc in docs:
                    doc.metadata["page_number"] = 0
                    doc.metadata["source_type"] = "txt"
                    doc.metadata["content_type"] = "text"
                    doc.metadata["element_type"] = ElementType.TEXT.value
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
    """Extract text and structured elements based on file extension."""
    suffix = file_path.suffix.lower()
    extractor = _EXTRACTORS.get(suffix)
    if extractor is None:
        raise ValueError(f"Unsupported file type: {suffix}")

    logger.info("Extracting structured content from %s (type: %s)", file_path.name, suffix)
    return extractor(file_path)
