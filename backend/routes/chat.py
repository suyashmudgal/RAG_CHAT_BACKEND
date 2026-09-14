"""POST /chat  &  POST /chat/stream — RAG-powered chat endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from auth.dependencies import get_current_user
from models.schemas import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat")
async def chat(request: ChatRequest, user: dict = Depends(get_current_user)):
    """Non-streaming RAG chat endpoint."""
    from services.deps import get_chat_service

    service = get_chat_service()
    try:
        return await service.get_answer(
            question=request.question,
            conversation_id=request.conversation_id,
            session_id=request.session_id,
            user_id=user["pg_id"],
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Chat error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while generating the response.",
        ) from exc


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, user: dict = Depends(get_current_user)):
    """Streaming RAG chat endpoint (Server-Sent Events)."""
    from services.deps import get_chat_service

    service = get_chat_service()
    try:
        return StreamingResponse(
            service.stream_answer(
                question=request.question,
                conversation_id=request.conversation_id,
                session_id=request.session_id,
                user_id=user["pg_id"],
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Stream error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while streaming the response.",
        ) from exc
