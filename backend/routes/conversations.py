"""Conversations router — persistent chat history endpoints.

Endpoints:
- GET    /conversations                     — List conversations for current user (pinned first)
- POST   /conversations                     — Create a new conversation thread
- GET    /conversations/{conversation_id}   — Get conversation details & messages
- PATCH  /conversations/{conversation_id}   — Rename chat and/or pin/unpin chat
- POST   /conversations/{conversation_id}/pin   — Pin conversation (convenience)
- POST   /conversations/{conversation_id}/unpin — Unpin conversation (convenience)
- GET    /conversations/{conversation_id}/export/pdf — Export conversation as PDF
- DELETE /conversations/{conversation_id}   — Delete a conversation and cascade its messages
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from auth.dependencies import get_current_user
from database.session import SessionLocal, get_db
from models.database import Conversation, Message
from models.schemas import (
    ConversationCreate,
    ConversationDetail,
    ConversationSummary,
    ConversationUpdate,
    MessageItem,
)
from services.pdf_export_service import generate_conversation_pdf

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(
    limit: int = 50,
    offset: int = 0,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return conversations belonging to the authenticated user.

    Ordering:
    1. Pinned conversations first (is_pinned DESC)
    2. Then most recently updated conversations (updated_at DESC)

    Uses column projection and composite index for maximum query performance.
    Offloaded to threadpool to avoid blocking the event loop.
    """
    pg_user_id: int = user["pg_id"]

    def _query():
        return (
            db.query(
                Conversation.id,
                Conversation.title,
                Conversation.created_at,
                Conversation.updated_at,
                Conversation.is_pinned,
            )
            .filter(Conversation.user_id == pg_user_id)
            .order_by(Conversation.is_pinned.desc(), Conversation.updated_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )

    try:
        conversations = await asyncio.to_thread(_query)
        return [
            ConversationSummary(
                id=c.id,
                title=c.title,
                created_at=c.created_at.isoformat() if c.created_at else "",
                updated_at=c.updated_at.isoformat() if c.updated_at else "",
                is_pinned=bool(c.is_pinned),
            )
            for c in conversations
        ]
    except Exception as exc:
        logger.error("Failed to list conversations for user %d: %s", pg_user_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversations",
        ) from exc


@router.post("/conversations", response_model=ConversationSummary)
async def create_conversation(
    payload: Optional[ConversationCreate] = None,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Explicitly create a new conversation thread for the current user."""
    pg_user_id: int = user["pg_id"]
    title = (payload.title if payload and payload.title else "New Conversation").strip()
    is_custom_title = bool(title and title != "New Conversation")
    conversation_id = str(uuid.uuid4())

    def _insert():
        conv = Conversation(
            id=conversation_id,
            user_id=pg_user_id,
            title=title,
            is_pinned=False,
            is_custom_title=is_custom_title,
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
        return conv

    try:
        conv = await asyncio.to_thread(_insert)
        return ConversationSummary(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at.isoformat() if conv.created_at else "",
            updated_at=conv.updated_at.isoformat() if conv.updated_at else "",
            is_pinned=bool(conv.is_pinned),
        )
    except Exception as exc:
        db.rollback()
        logger.error("Failed to create conversation for user %d: %s", pg_user_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create conversation",
        ) from exc


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    limit: int = 100,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return a single conversation and recent historical messages.

    Enforces strict ownership: returns 404 if not found or owned by another user.
    Uses column projection and limit to avoid unbounded message payloads.
    """
    pg_user_id: int = user["pg_id"]

    def _fetch():
        conv = (
            db.query(
                Conversation.id,
                Conversation.title,
                Conversation.created_at,
                Conversation.updated_at,
                Conversation.is_pinned,
            )
            .filter(Conversation.id == conversation_id, Conversation.user_id == pg_user_id)
            .first()
        )
        if conv is None:
            return None, None

        messages = (
            db.query(
                Message.id,
                Message.role,
                Message.content,
                Message.created_at,
            )
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
            .all()
        )
        return conv, messages

    try:
        conv, messages = await asyncio.to_thread(_fetch)
        if conv is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )

        # Re-order to chronological ascending order for chat presentation
        messages.reverse()

        return ConversationDetail(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at.isoformat() if conv.created_at else "",
            updated_at=conv.updated_at.isoformat() if conv.updated_at else "",
            is_pinned=bool(conv.is_pinned),
            messages=[
                MessageItem(
                    id=m.id,
                    role=m.role,
                    content=m.content,
                    created_at=m.created_at.isoformat() if m.created_at else "",
                )
                for m in messages
            ],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to load conversation %s for user %d: %s", conversation_id, pg_user_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load conversation",
        ) from exc


@router.patch("/conversations/{conversation_id}", response_model=ConversationSummary)
async def update_conversation(
    conversation_id: str,
    payload: ConversationUpdate,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Rename conversation and/or pin/unpin conversation.

    Enforces ownership — returns 404 if not found or belongs to another user.
    When a manual rename occurs, sets is_custom_title=True so future messages
    never overwrite the user's custom title.
    """
    pg_user_id: int = user["pg_id"]

    def _update():
        conv = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.user_id == pg_user_id)
            .first()
        )
        if conv is None:
            return None

        # 1. Update title if supplied
        if payload.title is not None:
            clean_title = payload.title.strip()
            if not clean_title:
                raise ValueError("Title cannot be empty")
            conv.title = clean_title
            conv.is_custom_title = True

        # 2. Update is_pinned if supplied
        if payload.is_pinned is not None:
            conv.is_pinned = payload.is_pinned

        conv.updated_at = func.now()
        db.commit()
        db.refresh(conv)
        return conv

    try:
        conv = await asyncio.to_thread(_update)
        if conv is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )

        return ConversationSummary(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at.isoformat() if conv.created_at else "",
            updated_at=conv.updated_at.isoformat() if conv.updated_at else "",
            is_pinned=bool(conv.is_pinned),
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        ) from val_err
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        logger.error("Failed to update conversation %s for user %d: %s", conversation_id, pg_user_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update conversation",
        ) from exc


@router.post("/conversations/{conversation_id}/pin", response_model=ConversationSummary)
async def pin_conversation(
    conversation_id: str,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Pin a conversation to keep it at the top of the chat list."""
    return await update_conversation(
        conversation_id=conversation_id,
        payload=ConversationUpdate(is_pinned=True),
        user=user,
        db=db,
    )


@router.post("/conversations/{conversation_id}/unpin", response_model=ConversationSummary)
async def unpin_conversation(
    conversation_id: str,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Unpin a conversation."""
    return await update_conversation(
        conversation_id=conversation_id,
        payload=ConversationUpdate(is_pinned=False),
        user=user,
        db=db,
    )


@router.get("/conversations/{conversation_id}/export/pdf")
async def export_conversation_pdf(
    conversation_id: str,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Export conversation and messages as a formatted PDF file.

    Enforces strict ownership: 404 if not found or belongs to another user.
    Returns:
    - Content-Type: application/pdf
    - Content-Disposition: attachment; filename="conversation_{clean_title}.pdf"
    """
    pg_user_id: int = user["pg_id"]

    def _fetch_for_pdf():
        conv = (
            db.query(
                Conversation.id,
                Conversation.title,
                Conversation.created_at,
            )
            .filter(Conversation.id == conversation_id, Conversation.user_id == pg_user_id)
            .first()
        )
        if conv is None:
            return None, None

        messages = (
            db.query(
                Message.role,
                Message.content,
                Message.created_at,
            )
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .all()
        )
        return conv, messages

    try:
        conv, raw_messages = await asyncio.to_thread(_fetch_for_pdf)
        if conv is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )

        formatted_messages = [
            {
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else "",
            }
            for m in raw_messages
        ]

        created_str = conv.created_at.isoformat() if conv.created_at else ""
        pdf_bytes = await asyncio.to_thread(
            generate_conversation_pdf,
            title=conv.title,
            created_at=created_str,
            messages=formatted_messages,
        )

        # Sanitize title for Content-Disposition filename
        safe_title = re.sub(r"[^a-zA-Z0-9_\-]", "_", conv.title[:40]).strip("_")
        if not safe_title:
            safe_title = f"chat_{conversation_id[:8]}"
        filename = f"{safe_title}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(pdf_bytes)),
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to export conversation %s to PDF: %s", conversation_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export conversation to PDF",
        ) from exc


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a conversation and all its cascade messages.

    Enforces ownership — returns 404 if not found or belongs to another user.
    """
    pg_user_id: int = user["pg_id"]

    def _delete():
        conv = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.user_id == pg_user_id)
            .first()
        )
        if conv is None:
            return False

        db.delete(conv)
        db.commit()
        return True

    try:
        deleted = await asyncio.to_thread(_delete)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )

        return {
            "message": "Conversation deleted successfully",
            "id": conversation_id,
        }
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        logger.error("Failed to delete conversation %s for user %d: %s", conversation_id, pg_user_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete conversation",
        ) from exc
