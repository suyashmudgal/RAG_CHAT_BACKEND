"""Universal Normalized Document Representation for RAG.

Defines the normalized internal structure for document elements, pages,
and whole documents across text, tables, charts, images, and formulas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ElementType(str, Enum):
    """Categorized type of an extracted document element."""

    TEXT = "TEXT"
    TABLE = "TABLE"
    CHART = "CHART"
    GRAPH = "GRAPH"
    IMAGE = "IMAGE"
    DIAGRAM = "DIAGRAM"
    FORMULA = "FORMULA"
    LIST = "LIST"
    HEADING = "HEADING"
    CAPTION = "CAPTION"
    OCR = "OCR"


@dataclass
class DocumentElement:
    """A single atomic structured element extracted from a document page."""

    element_id: str
    element_type: ElementType
    content: str
    page_number: int
    document_id: str
    user_id: str | None = None
    source_type: str = "pdf"
    bbox: list[float] | None = None  # [x0, y0, x1, y1]
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert element to serializable dictionary."""
        return {
            "element_id": self.element_id,
            "element_type": self.element_type.value,
            "content": self.content,
            "page_number": self.page_number,
            "document_id": self.document_id,
            "user_id": self.user_id,
            "source_type": self.source_type,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class DocumentPage:
    """A normalized representation of a single document page."""

    page_number: int
    document_id: str
    user_id: str | None = None
    elements: list[DocumentElement] = field(default_factory=list)
    has_visuals: bool = False
    has_tables: bool = False
    is_scanned: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_element(self, element: DocumentElement) -> None:
        """Add an element to the page and update page-level flags."""
        self.elements.append(element)
        if element.element_type in [
            ElementType.CHART,
            ElementType.GRAPH,
            ElementType.IMAGE,
            ElementType.DIAGRAM,
        ]:
            self.has_visuals = True
        elif element.element_type == ElementType.TABLE:
            self.has_tables = True
        elif element.element_type == ElementType.OCR:
            self.is_scanned = True


@dataclass
class NormalizedDocument:
    """Universal normalized document container holding pages and structured elements."""

    document_id: str
    filename: str
    user_id: str | None = None
    source_type: str = "pdf"
    pages: list[DocumentPage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def all_elements(self) -> list[DocumentElement]:
        elements: list[DocumentElement] = []
        for page in self.pages:
            elements.extend(page.elements)
        return elements

    def get_diagnostics(self) -> dict[str, Any]:
        """Return diagnostic metrics of extracted elements for development & auditing."""
        all_elems = self.all_elements
        type_counts: dict[str, int] = {}
        for elem in all_elems:
            k = elem.element_type.value
            type_counts[k] = type_counts.get(k, 0) + 1

        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "total_pages": self.total_pages,
            "total_elements": len(all_elems),
            "element_breakdown": type_counts,
            "pages_with_visuals": sum(1 for p in self.pages if p.has_visuals),
            "pages_with_tables": sum(1 for p in self.pages if p.has_tables),
            "is_scanned_pdf": any(p.is_scanned for p in self.pages),
        }
