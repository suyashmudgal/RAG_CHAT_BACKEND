"""Migration utility: Migrate legacy local uploaded files to Supabase Storage.

Scans PostgreSQL `documents` table for records with local filesystem paths,
uploads physical files to deterministic user-scoped Supabase Storage paths:
    users/{user_id}/documents/{document_id}/{safe_filename}
and updates the `storage_path` in PostgreSQL.

Usage:
    python migrate_local_to_supabase.py [--remove-local]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("storage_migration")

from database.session import SessionLocal
from models.database import Document as PgDocument
from services.deps import get_storage_service
from services.storage_service import build_storage_path


async def migrate_documents(remove_local: bool = False) -> None:
    """Migrate local documents to Supabase Storage."""
    storage_service = get_storage_service()
    await storage_service.ensure_bucket()

    session = SessionLocal()
    try:
        docs = session.query(PgDocument).all()
        logger.info("Found %d total document record(s) in PostgreSQL", len(docs))

        migrated_count = 0
        skipped_count = 0
        missing_count = 0

        for doc in docs:
            current_path_str = doc.storage_path or ""
            # Already using Supabase Storage path
            if current_path_str.startswith("users/"):
                skipped_count += 1
                continue

            local_path = Path(current_path_str)
            if not local_path.is_file():
                logger.warning(
                    "Local file for doc %s ('%s') not found at: %s",
                    doc.id,
                    doc.filename,
                    current_path_str,
                )
                missing_count += 1
                continue

            # Read local file
            content = local_path.read_bytes()
            new_storage_path = build_storage_path(
                user_id=doc.user_id,
                document_id=doc.id,
                filename=doc.filename,
            )

            logger.info(
                "Migrating doc %s ('%s') -> %s (%d bytes)",
                doc.id,
                doc.filename,
                new_storage_path,
                len(content),
            )

            # Upload to Supabase Storage
            await storage_service.upload_file(
                storage_path=new_storage_path,
                content=content,
            )

            # Update PostgreSQL record
            doc.storage_path = new_storage_path
            session.commit()
            migrated_count += 1

            if remove_local:
                try:
                    local_path.unlink()
                    logger.info("Removed local copy: %s", local_path)
                except Exception as del_err:
                    logger.warning("Could not delete local file %s: %s", local_path, del_err)

        logger.info(
            "Migration complete! Migrated: %d, Already in Storage: %d, Missing local files: %d",
            migrated_count,
            skipped_count,
            missing_count,
        )
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate local uploads to Supabase Storage")
    parser.add_argument(
        "--remove-local",
        action="store_true",
        help="Remove original local files after successful upload to Supabase Storage",
    )
    args = parser.parse_args()
    asyncio.run(migrate_documents(remove_local=args.remove_local))
