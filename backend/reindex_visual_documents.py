"""Re-indexing and migration utility for PDF visual content.

Scans existing uploaded PDF documents and extracts visual knowledge
(charts, diagrams, tables, scanned content) without deleting existing text vectors.
Maintains strict user isolation by stamping all visual chunks with user_id and document_id.

Usage:
    python reindex_visual_documents.py [--dry-run] [--document-id ID]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import tempfile
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reindex_visual")

from database.session import SessionLocal
from models.database import Document as PgDocument
from services.deps import get_storage_service, get_vector_store
from services.text_chunker import DocumentChunker
from services.visual_extractor import visual_extractor


async def reindex_visual_documents(dry_run: bool = False, target_doc_id: str | None = None) -> None:
    """Scan existing documents and index visual chunks for PDFs lacking visual vectors."""
    vector_store = get_vector_store()
    storage_service = get_storage_service()
    chunker = DocumentChunker()

    session = SessionLocal()
    try:
        query = session.query(PgDocument).filter(PgDocument.filename.ilike("%.pdf"))
        if target_doc_id:
            query = query.filter(PgDocument.id == target_doc_id)

        pdf_docs = query.all()
        logger.info("Found %d PDF document record(s) to inspect.", len(pdf_docs))

        processed_count = 0
        visual_chunks_added_total = 0

        for doc in pdf_docs:
            doc_id = doc.id
            user_id = doc.user_id
            filename = doc.filename

            # Check if Chroma already has visual chunks for this document
            try:
                existing_chunks = vector_store.vectorstore._collection.get(
                    where={"$and": [{"document_id": doc_id}, {"content_type": "visual"}]}
                )
                if existing_chunks and existing_chunks.get("ids"):
                    logger.info("Doc %s ('%s') already has %d visual chunks. Skipping.", doc_id, filename, len(existing_chunks["ids"]))
                    continue
            except Exception as check_err:
                logger.debug("Check existing visual vectors error for %s: %s", doc_id, check_err)

            # Retrieve file content
            content: bytes | None = None
            storage_path = doc.storage_path or ""

            if storage_path.startswith("users/"):
                try:
                    content = await storage_service.download_file(storage_path)
                except Exception as dl_err:
                    logger.warning("Could not download %s from storage: %s", storage_path, dl_err)

            if not content:
                # Check local uploads directory
                possible_paths = [
                    Path("uploads") / f"{doc_id}.pdf",
                    Path("uploads") / f"user_{user_id}" / f"{doc_id}.pdf",
                    Path("uploads") / filename,
                    Path(f"C:/Users/suyas/Downloads/{filename}"),
                ]
                for p in possible_paths:
                    if p.exists() and p.is_file():
                        content = p.read_bytes()
                        break

            if not content:
                logger.warning("File content for doc %s ('%s') could not be located.", doc_id, filename)
                continue

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)

            try:
                # Extract visual knowledge
                visual_docs = visual_extractor.extract_visuals_from_pdf(tmp_path, filename=filename)
                if not visual_docs:
                    logger.info("No visual content found in doc %s ('%s').", doc_id, filename)
                    continue

                for v_doc in visual_docs:
                    v_doc.metadata["document_id"] = doc_id
                    v_doc.metadata["user_id"] = str(user_id)
                    v_doc.metadata["filename"] = filename

                v_chunks = chunker.chunk_documents(visual_docs)

                logger.info(
                    "Doc %s ('%s'): extracted %d visual chunk(s).",
                    doc_id,
                    filename,
                    len(v_chunks),
                )

                if dry_run:
                    logger.info("[DRY RUN] Would index %d visual chunks for %s.", len(v_chunks), doc_id)
                    continue

                # Add to vector store with unique visual chunk IDs to prevent overwriting text chunks
                v_ids = [f"{doc_id}_visual_{i}" for i in range(len(v_chunks))]
                for c in v_chunks:
                    c.metadata["document_id"] = doc_id
                    c.metadata["user_id"] = str(user_id)

                vector_store.vectorstore.add_documents(v_chunks, ids=v_ids)
                vector_store._chunk_count += len(v_chunks)

                # Update metadata in vector store
                if doc_id in vector_store._doc_metadata:
                    vector_store._doc_metadata[doc_id]["has_visual_chunks"] = True
                    vector_store._doc_metadata[doc_id]["visual_chunk_count"] = len(v_chunks)
                    vector_store._save_metadata()

                # Update DB chunk count
                doc.chunk_count = (doc.chunk_count or 0) + len(v_chunks)
                session.commit()

                processed_count += 1
                visual_chunks_added_total += len(v_chunks)
                logger.info("Successfully indexed %d visual chunk(s) for doc %s.", len(v_chunks), doc_id)

            except Exception as proc_err:
                logger.error("Error processing visual content for %s: %s", doc_id, proc_err, exc_info=True)
            finally:
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

        logger.info(
            "Re-indexing complete! Updated %d document(s), added %d visual chunk(s) total.",
            processed_count,
            visual_chunks_added_total,
        )
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-index visual content for existing PDF documents")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report without writing to vector store")
    parser.add_argument("--document-id", type=str, default=None, help="Target specific document ID")
    args = parser.parse_args()

    asyncio.run(reindex_visual_documents(dry_run=args.dry_run, target_doc_id=args.document_id))
