"""Universal chunking strategies for text, tables, charts, and diagrams.

Applies element-aware chunking:
- Tables: Preserved intact to protect row/column relationships
- Charts & Graphs: Preserved intact to keep categories and contextual totals together
- Diagrams & Visual OCR: Preserved intact for structural coherence
- Text & Formulas: Semantic recursive splitting preserving paragraphs, headings, and lists
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.document_representation import ElementType

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Universal chunker supporting element-aware splitting."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks respecting element types and structure.

        - TABLES: kept intact (up to 3000 chars)
        - CHARTS / GRAPHS / VISUALS: kept intact (up to 3000 chars)
        - TEXT: recursively split preserving paragraphs and headings
        """
        all_chunks: list[Document] = []

        for doc in documents:
            c_type = doc.metadata.get("content_type")
            e_type = doc.metadata.get("element_type")

            # Element-aware preservation: tables, charts, diagrams, and visual chunks
            is_structured = (
                c_type in ["visual", "table"]
                or e_type in [
                    ElementType.TABLE.value,
                    ElementType.CHART.value,
                    ElementType.GRAPH.value,
                    ElementType.DIAGRAM.value,
                    ElementType.IMAGE.value,
                    ElementType.OCR.value,
                ]
            )

            if is_structured and len(doc.page_content) <= 3000:
                # Keep intact to preserve tabular and chart semantic integrity
                all_chunks.append(doc)
            else:
                chunks = self.text_splitter.split_documents([doc])
                for chk in chunks:
                    # Inherit element_type
                    if "element_type" not in chk.metadata:
                        chk.metadata["element_type"] = e_type or ElementType.TEXT.value
                all_chunks.extend(chunks)

        for idx, chunk in enumerate(all_chunks):
            chunk.metadata["chunk_index"] = idx
            doc_id = chunk.metadata.get("document_id", "doc")
            chunk.metadata["chunk_id"] = f"{doc_id}_chunk_{idx}"

        logger.info(
            "Created %d element-aware chunk(s) from %d document(s)",
            len(all_chunks),
            len(documents),
        )
        return all_chunks
