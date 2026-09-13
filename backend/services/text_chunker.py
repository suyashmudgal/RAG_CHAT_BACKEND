"""Text chunking using LangChain's RecursiveCharacterTextSplitter.

Wraps the splitter in a thin class so the chunking strategy is easy to swap.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Split LangChain ``Document`` objects into smaller, overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split *documents* and add a ``chunk_index`` to each piece's metadata."""
        chunks = self.splitter.split_documents(documents)

        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx

        logger.info("Created %d chunk(s) from %d document(s)", len(chunks), len(documents))
        return chunks
