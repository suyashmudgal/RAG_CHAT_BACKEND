"""Synchronize SQLite auth users with PostgreSQL users table.

Ensures every authenticated user has a corresponding PostgreSQL record.
Uses email as the unique key to prevent duplicates.
"""

from __future__ import annotations

import logging

from database.session import SessionLocal
from models.database import User as PgUser

logger = logging.getLogger(__name__)


def ensure_pg_user(sqlite_user: dict) -> PgUser:
    """Get or create a PostgreSQL User record matching the SQLite auth user.

    Args:
        sqlite_user: Dict with keys ``id``, ``name``, ``email``,
                     ``password_hash``, ``created_at`` from SQLite.

    Returns:
        The PostgreSQL :class:`User` ORM instance.
    """
    email = sqlite_user["email"].strip().lower()
    session = SessionLocal()
    try:
        pg_user = session.query(PgUser).filter(PgUser.email == email).first()
        if pg_user is not None:
            return pg_user

        # Create new PG user from SQLite data
        pg_user = PgUser(
            name=sqlite_user["name"],
            email=email,
            password_hash=sqlite_user.get("password_hash", "synced"),
        )
        session.add(pg_user)
        session.commit()
        session.refresh(pg_user)
        logger.info("Synced SQLite user to PostgreSQL: %s (pg_id=%d)", email, pg_user.id)
        return pg_user
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
