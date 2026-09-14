"""Database engine and session management for PostgreSQL (Supabase)."""

from __future__ import annotations

import logging
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from config.settings import settings

logger = logging.getLogger(__name__)

Base = declarative_base()

_engine = None
_SessionFactory = None


def get_engine():
    """Return the cached SQLAlchemy engine or create one."""
    global _engine
    if _engine is None:
        url = settings.sync_database_url
        if not url:
            raise ValueError("DATABASE_URL is not configured in settings.")
        _engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=5,
            max_overflow=10,
            connect_args={"connect_timeout": 10},
        )
    return _engine


def get_session_factory():
    """Return the cached session factory."""
    global _SessionFactory
    if _SessionFactory is None:
        engine = get_engine()
        _SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _SessionFactory


def SessionLocal() -> Session:
    """Create and return a new Session instance."""
    factory = get_session_factory()
    return factory()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ping_db() -> bool:
    """Test PostgreSQL connection without exposing credentials."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("PostgreSQL connection check failed: %s", exc)
        return False
