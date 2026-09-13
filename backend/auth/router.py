"""Authentication API endpoints."""

from __future__ import annotations

import logging
import re
from urllib.parse import quote_plus, urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import RedirectResponse

from auth.database import get_db
from auth.dependencies import get_current_user
from auth.models import (
    AuthConfigResponse,
    AuthResponse,
    GoogleAuthRequest,
    SignInRequest,
    SignUpRequest,
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


# ── Current User ────────────────────────────────────────────────────────────


@router.get("/me", response_model=UserResponse)
async def me(user: dict = Depends(get_current_user)):
    """Return the currently authenticated user."""
    return UserResponse(
        id=user["id"],
        name=user["name"],
        email=user["email"],
        created_at=user["created_at"],
    )


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

    # Find or create user
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

    jwt_token = create_token(user["id"], user["email"])
    _set_token_cookie(response, jwt_token)

    logger.info("Google user authenticated: %s", email)
    return AuthResponse(
        user=UserResponse(
            id=user["id"],
            name=user["name"],
            email=user["email"],
            created_at=user["created_at"],
        ),
        message="Google sign-in successful",
    )


@router.get("/google/login")
async def google_login():
    """Redirect to Google OAuth consent screen."""
    if not settings.google_client_id:
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
    frontend_app_url = f"{settings.frontend_url}/app"
    frontend_login_url = f"{settings.frontend_url}/login"

    if error or not code:
        err_msg = error or "Authorization code missing"
        return RedirectResponse(url=f"{frontend_login_url}?error={quote_plus(err_msg)}")

    if not settings.google_client_id or not settings.google_client_secret:
        return RedirectResponse(url=f"{frontend_login_url}?error=Google+OAuth+not+fully+configured")

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
                logger.error("Failed to exchange Google OAuth code: %s", token_resp.text)
                return RedirectResponse(url=f"{frontend_login_url}?error=Failed+to+exchange+Google+code")
            tokens = token_resp.json()
            access_token = tokens.get("access_token")

            userinfo_resp = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if userinfo_resp.status_code != 200:
                logger.error("Failed to fetch Google userinfo: %s", userinfo_resp.text)
                return RedirectResponse(url=f"{frontend_login_url}?error=Failed+to+fetch+user+profile")
            profile = userinfo_resp.json()
    except Exception as exc:
        logger.error("Error during Google OAuth callback: %s", exc)
        return RedirectResponse(url=f"{frontend_login_url}?error=OAuth+connection+error")

    email = profile.get("email", "").strip().lower()
    name = profile.get("name") or email.split("@")[0]

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

    jwt_token = create_token(user["id"], user["email"])
    redirect = RedirectResponse(url=frontend_app_url, status_code=status.HTTP_302_FOUND)
    _set_token_cookie(redirect, jwt_token)
    return redirect
