"""FastAPI dependencies for authentication."""

from __future__ import annotations

import logging
import threading
import time

from fastapi import Cookie, HTTPException, status

from auth.database import get_db
from auth.utils import decode_token

logger = logging.getLogger(__name__)

# In-memory user resolution cache: user_id -> (expiry_monotonic, user_dict)
_user_cache: dict[int, tuple[float, dict]] = {}
_cache_lock = threading.Lock()
_CACHE_TTL_SECONDS = 120.0


def invalidate_user_cache(user_id: int | None = None) -> None:
    """Invalidate cached user entries.

    If ``user_id`` is supplied, only that user is evicted.
    Otherwise, the entire user cache is cleared.
    """
    with _cache_lock:
        if user_id is not None:
            _user_cache.pop(user_id, None)
        else:
            _user_cache.clear()


async def get_current_user(token: str | None = Cookie(None)):
    """Extract and validate the JWT from the ``token`` cookie.

    Returns a user dict containing both SQLite fields and ``pg_id``
    (the PostgreSQL user ID for ownership operations).
    Uses in-memory caching to eliminate redundant remote PostgreSQL user lookups.
    Raises 401 if not authenticated.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    payload = decode_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    user_id_raw = payload.get("sub")
    if user_id_raw is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    try:
        user_id = int(user_id_raw)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user ID in token",
        )

    # 1. Fast path: check in-memory cache
    now = time.monotonic()
    with _cache_lock:
        entry = _user_cache.get(user_id)
        if entry is not None:
            expiry, cached_user = entry
            if now < expiry:
                return dict(cached_user)
            # Expired
            _user_cache.pop(user_id, None)

    # 2. Cache miss: fetch from SQLite
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = await cursor.fetchone()
    finally:
        await db.close()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    user_dict = dict(user)

    # 3. Sync to PostgreSQL and attach pg_id
    try:
        from services.user_sync import ensure_pg_user

        pg_user = ensure_pg_user(user_dict)
        user_dict["pg_id"] = pg_user.id
        if pg_user.created_at:
            user_dict["created_at"] = (
                pg_user.created_at.isoformat()
                if hasattr(pg_user.created_at, "isoformat")
                else str(pg_user.created_at)
            )
    except Exception as exc:
        logger.error("Failed to sync user to PostgreSQL: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during user sync",
        )

    # 4. Cache resolved user dictionary
    with _cache_lock:
        _user_cache[user_id] = (now + _CACHE_TTL_SECONDS, dict(user_dict))

    return user_dict
