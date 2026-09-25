"""Authentication API endpoints."""

from __future__ import annotations

import logging
import re
from urllib.parse import quote_plus, urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse

from auth.database import get_db
from auth.dependencies import get_current_user, invalidate_user_cache
from auth.models import (
    AuthConfigResponse,
    AuthResponse,
    ChangePasswordRequest,
    GoogleAuthRequest,
    SignInRequest,
    SignUpRequest,
    UpdateProfileRequest,
    UserResponse,
)
from auth.utils import create_token, hash_password, verify_password
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Auth"])

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Cookie settings
_COOKIE_KEY = "token"
_COOKIE_MAX_AGE = 72 * 60 * 60  # 72 hours in seconds


def _set_token_cookie(response: Response, token: str) -> None:
    """Set the JWT as an httpOnly cookie on the response."""
    response.set_cookie(
        key=_COOKIE_KEY,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,  # Set True in production with HTTPS
        max_age=_COOKIE_MAX_AGE,
        path="/",
    )


# ── Sign Up ─────────────────────────────────────────────────────────────────


@router.post("/signup", response_model=AuthResponse)
async def signup(body: SignUpRequest, response: Response):
    """Create a new user account."""
    email = body.email.strip().lower()

    if not _EMAIL_RE.match(email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email address",
        )

    db = await get_db()
    try:
        # Check if user already exists
        cursor = await db.execute("SELECT id FROM users WHERE email = ?", (email,))
        if await cursor.fetchone():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email already exists",
            )

        # Create user
        pw_hash = hash_password(body.password)
        cursor = await db.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (body.name.strip(), email, pw_hash),
        )
        await db.commit()
        user_id = cursor.lastrowid

        # Fetch the created user
        cursor = await db.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = dict(await cursor.fetchone())
    finally:
        await db.close()

    token = create_token(user["id"], user["email"])
    _set_token_cookie(response, token)

    logger.info("New user signed up: %s", email)
    return AuthResponse(
        user=UserResponse(
            id=user["id"],
            name=user["name"],
            email=user["email"],
            created_at=user["created_at"],
        ),
        message="Account created successfully",
    )


# ── Sign In ─────────────────────────────────────────────────────────────────


@router.post("/signin", response_model=AuthResponse)
async def signin(body: SignInRequest, response: Response):
    """Authenticate with email and password."""
    email = body.email.strip().lower()

    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = await cursor.fetchone()
    finally:
        await db.close()

    if user is None or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    user = dict(user)
    token = create_token(user["id"], user["email"])
    _set_token_cookie(response, token)

    logger.info("User signed in: %s", email)
    return AuthResponse(
        user=UserResponse(
            id=user["id"],
            name=user["name"],
            email=user["email"],
            created_at=user["created_at"],
        ),
        message="Signed in successfully",
    )


# ── Sign Out ────────────────────────────────────────────────────────────────


@router.post("/signout")
async def signout(response: Response):
    """Clear the auth cookie."""
    response.delete_cookie(key=_COOKIE_KEY, path="/")
    return {"message": "Signed out successfully"}


# ── Current User & Profile Settings ─────────────────────────────────────────


@router.get("/me", response_model=UserResponse)
async def me(user: dict = Depends(get_current_user)):
    """Return the currently authenticated user's profile.

    Fast in-memory resolution from authenticated user context.
    """
    is_oauth = str(user.get("password_hash", "")).startswith("oauth:")
    return UserResponse(
        id=user["id"],
        name=user["name"],
        email=user["email"],
        created_at=str(user.get("created_at", "")),
        is_oauth=is_oauth,
    )


@router.patch("/me", response_model=UserResponse)
async def update_profile(
    request: Request,
    body: UpdateProfileRequest,
    user: dict = Depends(get_current_user),
):
    """Update current authenticated user's profile name.

    Email is strictly read-only; attempts to change email are rejected with 400.
    Updates both SQLite and PostgreSQL.
    """
    # Enforce read-only email: reject if body contains a different email
    try:
        raw_json = await request.json()
        if "email" in raw_json and raw_json["email"].strip().lower() != user["email"].strip().lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email cannot be changed",
            )
    except HTTPException:
        raise
    except Exception:
        pass

    new_name = body.name.strip()
    if not new_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Name cannot be empty",
        )

    # 1. Update SQLite
    db = await get_db()
    try:
        await db.execute(
            "UPDATE users SET name = ? WHERE id = ?",
            (new_name, user["id"]),
        )
        await db.commit()
    finally:
        await db.close()

    # 2. Update PostgreSQL
    created_at = user["created_at"]
    try:
        from database.session import SessionLocal
        from models.database import User as PgUser
        from sqlalchemy import func

        pg_session = SessionLocal()
        try:
            pg_user = pg_session.query(PgUser).filter(PgUser.id == user["pg_id"]).first()
            if pg_user:
                pg_user.name = new_name
                pg_user.updated_at = func.now()
                pg_session.commit()
                if hasattr(pg_user.created_at, "isoformat"):
                    created_at = pg_user.created_at.isoformat()
                elif pg_user.created_at:
                    created_at = str(pg_user.created_at)
        except Exception:
            pg_session.rollback()
            raise
        finally:
            pg_session.close()
    except Exception as exc:
        logger.error("Failed to update user name in PostgreSQL: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile",
        ) from exc

    is_oauth = str(user.get("password_hash", "")).startswith("oauth:")
    invalidate_user_cache(user["id"])
    logger.info("Updated profile name for user %s to '%s'", user["email"], new_name)
    return UserResponse(
        id=user["id"],
        name=new_name,
        email=user["email"],
        created_at=created_at,
        is_oauth=is_oauth,
    )


@router.post("/change-password")
async def change_password(
    body: ChangePasswordRequest,
    user: dict = Depends(get_current_user),
):
    """Change password for current authenticated user.

    Verifies current password, hashes new password with bcrypt, and updates both SQLite and PostgreSQL.
    Rejects password changes for Google OAuth accounts.
    """
    # Fetch authoritative fresh user record from SQLite
    db_auth = await get_db()
    try:
        cursor = await db_auth.execute("SELECT * FROM users WHERE id = ?", (user["id"],))
        fresh_user = await cursor.fetchone()
    finally:
        await db_auth.close()

    if fresh_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    fresh_dict = dict(fresh_user)
    pw_hash = fresh_dict.get("password_hash", "")
    if str(pw_hash).startswith("oauth:"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password changes are not supported for accounts authenticated via Google OAuth. Please manage your password through Google.",
        )

    # Verify current password
    if not verify_password(body.current_password, pw_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password",
        )

    # Validate new password
    new_pw = body.new_password
    if len(new_pw.strip()) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be at least 6 characters long and cannot be only whitespace",
        )

    new_hash = hash_password(new_pw)

    # 1. Update SQLite
    db = await get_db()
    try:
        await db.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (new_hash, user["id"]),
        )
        await db.commit()
    finally:
        await db.close()

    # 2. Update PostgreSQL
    try:
        from database.session import SessionLocal
        from models.database import User as PgUser
        from sqlalchemy import func

        pg_session = SessionLocal()
        try:
            pg_user = pg_session.query(PgUser).filter(PgUser.id == user["pg_id"]).first()
            if pg_user:
                pg_user.password_hash = new_hash
                pg_user.updated_at = func.now()
                pg_session.commit()
        except Exception:
            pg_session.rollback()
            raise
        finally:
            pg_session.close()
    except Exception as exc:
        logger.error("Failed to update password in PostgreSQL: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update password in database",
        ) from exc

    invalidate_user_cache(user["id"])
    logger.info("Password changed successfully for user %s", user["email"])
    return {"message": "Password changed successfully"}


# ── Google Authentication ───────────────────────────────────────────────────


@router.get("/config", response_model=AuthConfigResponse)
async def auth_config():
    """Return public authentication configuration."""
    return AuthConfigResponse(
        google_client_id=settings.google_client_id,
        google_enabled=bool(settings.google_client_id),
    )


@router.post("/google", response_model=AuthResponse)
async def google_auth(body: GoogleAuthRequest, response: Response):
    """Authenticate with a Google ID token from Google Identity Services."""
    token_str = body.credential.strip()
    if not token_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google credential token is required",
        )

    # Validate token with Google's tokeninfo service
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://oauth2.googleapis.com/tokeninfo",
                params={"id_token": token_str},
            )
            if resp.status_code != 200:
                logger.warning("Google token verification failed: %s", resp.text)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired Google token",
                )
            payload = resp.json()
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to connect to Google OAuth service: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to verify Google token with Google servers",
        )

    # Audience check if configured
    if settings.google_client_id and payload.get("aud") != settings.google_client_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google token audience mismatch",
        )

    email = payload.get("email", "").strip().lower()
    email_verified = payload.get("email_verified")
    if not email or str(email_verified).lower() != "true":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unverified Google email address",
        )

    name = payload.get("name") or email.split("@")[0]

    # Find or create user in SQLite
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM users WHERE email = ?", (email,))
        user_row = await cursor.fetchone()
        if user_row:
            user = dict(user_row)
        else:
            cursor = await db.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (name, email, "oauth:google"),
            )
            await db.commit()
            user_id = cursor.lastrowid
            cursor = await db.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            user = dict(await cursor.fetchone())
    finally:
        await db.close()

    # Synchronize with Supabase PostgreSQL
    try:
        from services.user_sync import ensure_pg_user
        pg_user = ensure_pg_user(user)
        user["pg_id"] = pg_user.id
    except Exception as pg_exc:
        logger.error("Failed to sync GIS Google user to PostgreSQL: %s", pg_exc)

    jwt_token = create_token(user["id"], user["email"])
    _set_token_cookie(response, jwt_token)

    logger.info("Google user authenticated via GIS: %s", email)
    return AuthResponse(
        user=UserResponse(
            id=user["id"],
            name=user["name"],
            email=user["email"],
            created_at=str(user.get("created_at", "")),
            is_oauth=True,
        ),
        message="Google sign-in successful",
    )


@router.get("/google")
@router.get("/google/login")
async def google_login():
    """Redirect to Google OAuth consent screen."""
    logger.info("[GOOGLE OAUTH] Starting authorization")
    logger.info("[GOOGLE OAUTH] Redirect URI: %s", settings.google_redirect_uri)

    if not settings.google_client_id:
        logger.warning("[GOOGLE OAUTH] Authorization failed: Google client ID not configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google authentication is not configured on the server. Please set GOOGLE_CLIENT_ID in .env.",
        )

    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account",
    }
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
    return RedirectResponse(url=url)


@router.get("/google/callback")
async def google_callback(code: str | None = None, error: str | None = None):
    """Handle Google OAuth redirect callback."""
    logger.info("[GOOGLE OAUTH] Callback reached")
    frontend_app_url = f"{settings.frontend_url}/app"
    frontend_login_url = f"{settings.frontend_url}/login"

    has_code = bool(code)
    logger.info("[GOOGLE OAUTH] Authorization code received: %s", "YES" if has_code else "NO")

    if error or not code:
        err_msg = error or "Authorization code missing"
        logger.warning("[GOOGLE OAUTH] Callback error: %s (type: OAuthCallbackError)", err_msg)
        return RedirectResponse(url=f"{frontend_login_url}?error={quote_plus(err_msg)}")

    if not settings.google_client_id or not settings.google_client_secret:
        logger.error("[GOOGLE OAUTH] Callback aborted: Google OAuth credentials incomplete in configuration")
        return RedirectResponse(url=f"{frontend_login_url}?error=Google+OAuth+not+fully+configured")

    logger.info("[GOOGLE OAUTH] Exchanging authorization code")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            token_resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": settings.google_client_id,
                    "client_secret": settings.google_client_secret,
                    "redirect_uri": settings.google_redirect_uri,
                    "grant_type": "authorization_code",
                },
            )
            if token_resp.status_code != 200:
                logger.error(
                    "[GOOGLE OAUTH] Token exchange successful: NO (status: %s, safe_msg: %s)",
                    token_resp.status_code,
                    token_resp.json().get("error_description", "Token exchange failed") if token_resp.headers.get("content-type", "").startswith("application/json") else "Non-JSON response",
                )
                return RedirectResponse(url=f"{frontend_login_url}?error=Failed+to+exchange+Google+code")

            logger.info("[GOOGLE OAUTH] Token exchange successful: YES")
            tokens = token_resp.json()
            access_token = tokens.get("access_token")

            if not access_token:
                logger.error("[GOOGLE OAUTH] Token exchange missing access_token in payload")
                return RedirectResponse(url=f"{frontend_login_url}?error=Invalid+token+response+from+Google")

            userinfo_resp = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if userinfo_resp.status_code != 200:
                logger.error("[GOOGLE OAUTH] Google identity received: NO (status: %s)", userinfo_resp.status_code)
                return RedirectResponse(url=f"{frontend_login_url}?error=Failed+to+fetch+user+profile")

            profile = userinfo_resp.json()
            logger.info("[GOOGLE OAUTH] Google identity received: YES")
    except Exception as exc:
        logger.error(
            "[GOOGLE OAUTH] OAuth connection error: type=%s, safe_msg=%s",
            type(exc).__name__,
            str(exc),
        )
        return RedirectResponse(url=f"{frontend_login_url}?error=OAuth+connection+error")

    email = profile.get("email", "").strip().lower()
    has_email = bool(email)
    logger.info("[GOOGLE OAUTH] Google email present: %s", "YES" if has_email else "NO")

    if not has_email:
        logger.error("[GOOGLE OAUTH] Google profile did not include an email address")
        return RedirectResponse(url=f"{frontend_login_url}?error=No+email+provided+by+Google")

    name = profile.get("name") or email.split("@")[0]

    # Find or create user in SQLite
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM users WHERE email = ?", (email,))
        user_row = await cursor.fetchone()
        if user_row:
            logger.info("[GOOGLE OAUTH] Existing local user found: YES")
            logger.info("[GOOGLE OAUTH] Creating local user: NO")
            user = dict(user_row)
        else:
            logger.info("[GOOGLE OAUTH] Existing local user found: NO")
            logger.info("[GOOGLE OAUTH] Creating local user: YES")
            cursor = await db.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (name, email, "oauth:google"),
            )
            await db.commit()
            user_id = cursor.lastrowid
            cursor = await db.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            user = dict(await cursor.fetchone())
    finally:
        await db.close()

    # PostgreSQL synchronization
    try:
        from services.user_sync import ensure_pg_user
        pg_user = ensure_pg_user(user)
        user["pg_id"] = pg_user.id
        logger.info("[GOOGLE OAUTH] PostgreSQL sync: SUCCESS")
    except Exception as pg_exc:
        logger.error(
            "[GOOGLE OAUTH] PostgreSQL sync: FAILED (type=%s, safe_msg=%s)",
            type(pg_exc).__name__,
            str(pg_exc),
        )

    # JWT / session creation
    jwt_token = create_token(user["id"], user["email"])
    logger.info("[GOOGLE OAUTH] JWT/session created: YES")

    # Set httpOnly cookie & redirect to frontend
    redirect = RedirectResponse(url=frontend_app_url, status_code=status.HTTP_302_FOUND)
    _set_token_cookie(redirect, jwt_token)
    logger.info("[GOOGLE OAUTH] Auth cookie set: YES")
    logger.info("[GOOGLE OAUTH] Redirecting to frontend")
    return redirect
