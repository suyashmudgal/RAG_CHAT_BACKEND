"""SQLite database setup for user authentication."""

from __future__ import annotations

import logging
import aiosqlite

from config.settings import settings

logger = logging.getLogger(__name__)

_db_path: str = settings.auth_db_path
_initialized = False


async def init_db() -> None:
    """Create the users table if it does not exist."""
    global _initialized
    async with aiosqlite.connect(_db_path) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT    NOT NULL,
                email         TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        await db.commit()
    _initialized = True
    logger.info("✅ Auth database ready (%s)", _db_path)


async def get_db() -> aiosqlite.Connection:
    """Return a new database connection (caller must close it)."""
    global _initialized
    if not _initialized:
        await init_db()
    db = await aiosqlite.connect(_db_path)
    db.row_factory = aiosqlite.Row
    return db
