"""Pydantic models for API request / response schemas."""

from typing import Optional

from pydantic import BaseModel, Field


# ── Upload ──────────────────────────────────────────────────────────────────

class UploadResult(BaseModel):
    """Result for a single uploaded file."""

    document_id: str = ""
    filename: str
    status: str
    chunk_count: int = 0
    message: str


class UploadResponse(BaseModel):
    """Response for the /upload endpoint."""

    results: list[UploadResult]


# ── Documents ───────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    """Metadata for an indexed document."""

    document_id: str
    filename: str
    status: str
    chunk_count: int
    upload_time: str
    file_size: Optional[int] = None


class DocumentListResponse(BaseModel):
    """Response for GET /documents."""

    documents: list[DocumentInfo]


class DeleteResponse(BaseModel):
    """Response for DELETE /documents/{id}."""

    message: str
    document_id: str


# ── Chat ────────────────────────────────────────────────────────────────────

class SourceCitation(BaseModel):
    """A single source reference attached to an answer."""

    filename: str
    page_number: Optional[int] = None
    chunk_preview: str = ""
    document_id: str = ""
    chunk_id: Optional[str] = None
    relevance_score: Optional[float] = None


class ChatRequest(BaseModel):
    """Request body for /chat and /chat/stream."""

    question: str = Field(..., min_length=1, max_length=10000)
    session_id: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    """Response for POST /chat."""

    answer: str
    sources: list[SourceCitation]
    session_id: str


# ── Errors ──────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str
    error_code: Optional[str] = None
