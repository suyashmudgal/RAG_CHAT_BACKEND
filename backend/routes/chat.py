"""POST /chat  &  POST /chat/stream — RAG-powered chat endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from models.schemas import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat")
async def chat(request: ChatRequest):
    """Non-streaming RAG chat endpoint."""
    from services.deps import get_chat_service

    service = get_chat_service()
    try:
        return await service.get_answer(request.question, request.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Chat error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while generating the response.",
        ) from exc


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming RAG chat endpoint (Server-Sent Events)."""
    from services.deps import get_chat_service

    service = get_chat_service()
    try:
        return StreamingResponse(
            service.stream_answer(request.question, request.session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Stream error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while streaming the response.",
        ) from exc
