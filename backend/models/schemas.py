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
    session_id: Optional[str] = Field(None)
    conversation_id: Optional[str] = Field(None)


class ChatResponse(BaseModel):
    """Response for POST /chat."""

    answer: str
    sources: list[SourceCitation]
    conversation_id: str
    session_id: str


# ── Conversations ───────────────────────────────────────────────────────────

class ConversationCreate(BaseModel):
    """Request body for POST /conversations."""

    title: Optional[str] = Field("New Conversation", max_length=255)


class ConversationSummary(BaseModel):
    """Summary of a conversation for list view."""

    id: str
    title: str
    created_at: str
    updated_at: str


class MessageItem(BaseModel):
    """Single chat message item."""

    id: str
    role: str
    content: str
    created_at: str


class ConversationDetail(BaseModel):
    """Full detail of a conversation including messages."""

    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[MessageItem]


# ── Errors ──────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str
    error_code: Optional[str] = None
