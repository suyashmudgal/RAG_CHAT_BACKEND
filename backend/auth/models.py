"""Pydantic schemas for authentication requests and responses."""

from pydantic import BaseModel, EmailStr, Field


class SignUpRequest(BaseModel):
    """Body for POST /auth/signup."""

    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)


class SignInRequest(BaseModel):
    """Body for POST /auth/signin."""

    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=1, max_length=128)


class UserResponse(BaseModel):
    """Public user data returned by the API."""

    id: int
    name: str
    email: str
    created_at: str
    is_oauth: bool = False


class UpdateProfileRequest(BaseModel):
    """Body for PATCH /auth/me."""

    name: str = Field(..., min_length=1, max_length=100)


class ChangePasswordRequest(BaseModel):
    """Body for POST /auth/change-password."""

    current_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=6, max_length=128)


class AuthResponse(BaseModel):
    """Returned after successful sign-up or sign-in."""

    user: UserResponse
    message: str = "ok"


class GoogleAuthRequest(BaseModel):
    """Body for POST /auth/google."""

    credential: str = Field(..., description="Google ID Token from Google Identity Services")


class AuthConfigResponse(BaseModel):
    """Public authentication configuration."""

    google_client_id: str = ""
    google_enabled: bool = False

