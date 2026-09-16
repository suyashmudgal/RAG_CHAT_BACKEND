"""Conversations router — persistent chat history endpoints.

Endpoints:
- GET    /conversations                     — List conversations for current user
- POST   /conversations                     — Create a new conversation
- GET    /conversations/{conversation_id}   — Get conversation details & messages
- DELETE /conversations/{conversation_id}   — Delete a conversation and cascade its messages
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from auth.dependencies import get_current_user
from database.session import get_db
from models.database import Conversation, Message
from models.schemas import (
    ConversationCreate,
    ConversationDetail,
    ConversationSummary,
    MessageItem,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(
    limit: int = 50,
    offset: int = 0,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return conversations belonging to the authenticated user, ordered by updated_at DESC.

    Uses column projection and composite index for maximum query speed.
    """
    pg_user_id: int = user["pg_id"]
    try:
        conversations = (
            db.query(
                Conversation.id,
                Conversation.title,
                Conversation.created_at,
                Conversation.updated_at,
            )
            .filter(Conversation.user_id == pg_user_id)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
        return [
            ConversationSummary(
                id=c.id,
                title=c.title,
                created_at=c.created_at.isoformat() if c.created_at else "",
                updated_at=c.updated_at.isoformat() if c.updated_at else "",
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
    conversation_id = str(uuid.uuid4())

    try:
        conv = Conversation(
            id=conversation_id,
            user_id=pg_user_id,
            title=title,
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)

        return ConversationSummary(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at.isoformat() if conv.created_at else "",
            updated_at=conv.updated_at.isoformat() if conv.updated_at else "",
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
    try:
        conv = (
            db.query(
                Conversation.id,
                Conversation.title,
                Conversation.created_at,
                Conversation.updated_at,
            )
            .filter(Conversation.id == conversation_id, Conversation.user_id == pg_user_id)
            .first()
        )
        if conv is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )

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
        # Re-order to chronological ascending order for chat presentation
        messages.reverse()

        return ConversationDetail(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at.isoformat() if conv.created_at else "",
            updated_at=conv.updated_at.isoformat() if conv.updated_at else "",
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
    try:
        conv = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.user_id == pg_user_id)
            .first()
        )
        if conv is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )

        db.delete(conv)
        db.commit()

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
