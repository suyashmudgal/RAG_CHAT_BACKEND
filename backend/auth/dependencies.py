"""FastAPI dependencies for authentication."""

from __future__ import annotations

import logging

from fastapi import Cookie, HTTPException, status

from auth.database import get_db
from auth.utils import decode_token

logger = logging.getLogger(__name__)


async def get_current_user(token: str | None = Cookie(None)):
    """Extract and validate the JWT from the ``token`` cookie.

    Returns a user dict containing both SQLite fields and ``pg_id``
    (the PostgreSQL user ID for ownership operations).
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

    # Sync to PostgreSQL and attach pg_id
    try:
        from services.user_sync import ensure_pg_user

        pg_user = ensure_pg_user(user_dict)
        user_dict["pg_id"] = pg_user.id
    except Exception as exc:
        logger.error("Failed to sync user to PostgreSQL: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during user sync",
        )

    return user_dict
