"""Password hashing and JWT token helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from config.settings import settings

_SECRET = settings.jwt_secret_key
_ALGORITHM = "HS256"
_EXPIRY_HOURS = settings.jwt_expiry_hours


# ── Password ────────────────────────────────────────────────────────────────


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of the plaintext password."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Return *True* if *plain* matches the bcrypt *hashed* value."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── JWT ─────────────────────────────────────────────────────────────────────


def create_token(user_id: int, email: str) -> str:
    """Create a signed JWT containing the user ID and email."""
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=_EXPIRY_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, _SECRET, algorithm=_ALGORITHM)


def decode_token(token: str) -> dict | None:
    """Decode and verify a JWT.  Return the payload dict, or *None* on failure."""
    try:
        return jwt.decode(token, _SECRET, algorithms=[_ALGORITHM])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None
