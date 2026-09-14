"""Chat service — RAG retrieval, persistent PostgreSQL conversation memory, Groq LLM (streaming).

Uses:
- PostgreSQL (via SQLAlchemy) as the source of truth for conversations & messages
- LangChain ``ChatGroq`` for LLM calls (free tier)
- LangChain message types for history
- Custom SSE streaming for the React frontend
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator

from fastapi import HTTPException, status
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from sqlalchemy import func
from sqlalchemy.orm import Session

from config.settings import settings
from database.session import SessionLocal
from models.database import Conversation, Message
from services.citation_service import distance_to_similarity, select_citations
from services.vector_store import VectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a helpful document assistant. Your role is to answer questions based \
ONLY on the provided document context.

IMPORTANT RULES:
1. Only use information from the provided document context to answer.
2. If the context does not contain enough information, clearly state:
   "I couldn't find this information in the uploaded documents."
3. Never fabricate or hallucinate information that is not in the context.
4. Reference which document the information comes from when relevant.
5. Be concise but thorough.
6. For follow-up questions, use conversation history to understand references."""


class ChatService:
    """Orchestrates RAG retrieval, persistent conversation memory, and Groq LLM calls."""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store
        self._llm: ChatGroq | None = None

    # ── LLM initialisation ──────────────────────────────────────────────

    def _get_llm(self) -> ChatGroq:
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Please add it to your .env file. "
                "Get a free key at https://console.groq.com/keys"
            )
        if self._llm is None:
            self._llm = ChatGroq(
                model=settings.groq_model,
                api_key=settings.groq_api_key,
                temperature=0.3,
                max_tokens=2048,
            )
        return self._llm

    # ── Conversation & Memory Persistence ───────────────────────────────

    def resolve_conversation(
        self,
        db: Session,
        user_id: int,
        conversation_id: str | None = None,
        session_id: str | None = None,
        initial_title: str | None = None,
    ) -> Conversation:
        """Resolve or create an authenticated user's conversation in PostgreSQL.

        Enforces strict ownership: if a conversation exists belonging to another user,
        raises 404 to avoid leaking existence or data.
        """
        # 1. Explicit conversation_id supplied
        if conversation_id:
            conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
            if conv is not None:
                if conv.user_id != user_id:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Conversation not found",
                    )
                return conv
            # If not found, create with this ID
            conv = Conversation(
                id=conversation_id,
                user_id=user_id,
                title=initial_title or "New Conversation",
            )
            db.add(conv)
            db.commit()
            db.refresh(conv)
            return conv

        # 2. Compatibility mode: session_id supplied
        if session_id:
            # Check if session_id matches an existing conversation owned by user
            conv = (
                db.query(Conversation)
                .filter(Conversation.id == session_id, Conversation.user_id == user_id)
                .first()
            )
            if conv is not None:
                return conv

            # If someone else owns this exact ID, reject with 404
            other_conv = db.query(Conversation).filter(Conversation.id == session_id).first()
            if other_conv is not None and other_conv.user_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found",
                )

            # Generate deterministic UUID per (user_id, session_id)
            mapped_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}:{session_id}"))
            conv = (
                db.query(Conversation)
                .filter(Conversation.id == mapped_id, Conversation.user_id == user_id)
                .first()
            )
            if conv is not None:
                return conv

            conv = Conversation(
                id=mapped_id,
                user_id=user_id,
                title=initial_title or "New Conversation",
            )
            db.add(conv)
            db.commit()
            db.refresh(conv)
            return conv

        # 3. Neither supplied: create fresh conversation
        new_id = str(uuid.uuid4())
        conv = Conversation(
            id=new_id,
            user_id=user_id,
            title=initial_title or "New Conversation",
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
        return conv

    def _load_history_for_llm(
        self, db: Session, conversation_id: str, limit: int = 20
    ) -> list:
        """Fetch the most recent `limit` messages from PostgreSQL for LLM context."""
        recent_messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
            .all()
        )
        recent_messages.reverse()

        history = []
        for msg in recent_messages:
            if msg.role == "user":
                history.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                history.append(AIMessage(content=msg.content))
        return history

    def _save_message(
        self, db: Session, conversation_id: str, role: str, content: str
    ) -> Message:
        """Persist a message and bump the conversation updated_at timestamp."""
        msg = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=content,
        )
        db.add(msg)
        db.query(Conversation).filter(Conversation.id == conversation_id).update(
            {"updated_at": func.now()}
        )
        db.commit()
        return msg

    # ── Retrieval ───────────────────────────────────────────────────────

    def _retrieve_context(
        self, question: str, user_id: int | None = None
    ) -> tuple[str, list[tuple[Document, float]]]:
        """Search vector store and build prompt context + raw candidate list."""
        top_k = getattr(settings, "retrieval_top_k", settings.top_k_results)

        if user_id is not None:
            results = self.vector_store.similarity_search_for_user(
                question, user_id, top_k=top_k
            )
        else:
            results = self.vector_store.similarity_search(question, top_k=top_k)

        if not results:
            return "", []

        # Filter out extreme noise before sending to LLM context
        candidates: list[tuple[Document, float]] = []
        for doc, dist in results:
            sim = distance_to_similarity(dist)
            if sim >= 0.10:
                candidates.append((doc, dist))

        if not candidates:
            candidates = [results[0]]

        context_parts: list[str] = []
        for doc, _dist in candidates:
            filename = doc.metadata.get("filename", "Unknown")
            page = doc.metadata.get("page_number", 0)
            text = doc.page_content

            header = f"[From: {filename}"
            if page and int(page) > 0:
                header += f", Page {page}"
            header += "]"
            context_parts.append(f"{header}\n{text}")

        context = "\n\n---\n\n".join(context_parts)
        return context, results

    # ── Message building ────────────────────────────────────────────────

    def _build_messages(
        self, question: str, context: str, history: list
    ) -> list:
        """System prompt → history → current context + question."""
        msgs: list = [SystemMessage(content=SYSTEM_PROMPT)]
        msgs.extend(history)

        if context:
            user_content = (
                f"Document Context:\n{context}\n\nQuestion: {question}"
            )
        else:
            user_content = (
                "No relevant document context was found. "
                "If the question requires information from uploaded documents, "
                "let the user know that no relevant content was found.\n\n"
                f"Question: {question}"
            )

        msgs.append(HumanMessage(content=user_content))
        return msgs

    # ── Public API ──────────────────────────────────────────────────────

    async def get_answer(
        self,
        question: str,
        conversation_id: str | None = None,
        session_id: str | None = None,
        user_id: int | None = None,
    ) -> dict:
        """Non-streaming RAG answer with PostgreSQL persistence."""
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )

        db: Session = SessionLocal()
        try:
            # 1. Resolve or create conversation
            conv = self.resolve_conversation(
                db, user_id=user_id, conversation_id=conversation_id, session_id=session_id
            )

            # Auto-title conversation on first message
            if conv.title == "New Conversation" and question:
                clean_title = question.strip()
                if len(clean_title) > 60:
                    clean_title = clean_title[:57] + "..."
                conv.title = clean_title
                db.commit()

            # 2. Load previous history before adding this message
            history = self._load_history_for_llm(db, conv.id, limit=20)

            # 3. Save current user message to PostgreSQL
            self._save_message(db, conv.id, "user", question)

            # 4. RAG context retrieval
            context, raw_candidates = await asyncio.to_thread(
                self._retrieve_context, question, user_id
            )
            messages = self._build_messages(question, context, history)

            # 5. Invoke LLM
            llm = self._get_llm()
            try:
                response = await llm.ainvoke(messages)
                answer: str = response.content or ""
            except Exception as exc:
                raise self._handle_groq_error(exc) from exc

            # 6. Save assistant response to PostgreSQL
            self._save_message(db, conv.id, "assistant", answer)

            sources = select_citations(
                query=question,
                answer=answer,
                retrieved_chunks=raw_candidates,
                similarity_threshold=settings.similarity_threshold,
                margin=settings.citation_margin,
                max_citations=settings.max_citations,
            )

            return {
                "answer": answer,
                "sources": sources,
                "conversation_id": conv.id,
                "session_id": conv.id,
            }
        finally:
            db.close()

    async def stream_answer(
        self,
        question: str,
        conversation_id: str | None = None,
        session_id: str | None = None,
        user_id: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yield Server-Sent Events with atomic final assistant persistence."""
        if user_id is None:
            yield f"data: {json.dumps({'type': 'error', 'content': 'Authentication required'})}\n\n"
            return

        db: Session = SessionLocal()
        conv_id: str = ""
        try:
            # 1. Resolve or create conversation
            conv = self.resolve_conversation(
                db, user_id=user_id, conversation_id=conversation_id, session_id=session_id
            )
            conv_id = conv.id

            # Auto-title on first message
            if conv.title == "New Conversation" and question:
                clean_title = question.strip()
                if len(clean_title) > 60:
                    clean_title = clean_title[:57] + "..."
                conv.title = clean_title
                db.commit()

            # 2. Load previous history before adding this message
            history = self._load_history_for_llm(db, conv_id, limit=20)

            # 3. Save user message to PostgreSQL
            self._save_message(db, conv_id, "user", question)

            # 4. RAG context retrieval
            context, raw_candidates = await asyncio.to_thread(
                self._retrieve_context, question, user_id
            )
            messages = self._build_messages(question, context, history)

            # 5. Stream LLM tokens
            llm = self._get_llm()
            full_answer = ""

            async for chunk in llm.astream(messages):
                token = chunk.content
                if token:
                    full_answer += token
                    yield (
                        f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                    )

            # 6. Save accumulated assistant message to PostgreSQL ONCE
            self._save_message(db, conv_id, "assistant", full_answer)

            # 7. Citations & done
            sources = select_citations(
                query=question,
                answer=full_answer,
                retrieved_chunks=raw_candidates,
                similarity_threshold=settings.similarity_threshold,
                margin=settings.citation_margin,
                max_citations=settings.max_citations,
            )

            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': conv_id, 'session_id': conv_id})}\n\n"

        except HTTPException as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': exc.detail})}\n\n"
        except Exception as exc:
            err = self._handle_groq_error(exc)
            yield f"data: {json.dumps({'type': 'error', 'content': str(err)})}\n\n"
        finally:
            db.close()

    # ── Error helpers ───────────────────────────────────────────────────

    @staticmethod
    def _handle_groq_error(exc: Exception) -> ValueError:
        msg = str(exc)
        if "GROQ_API_KEY" in msg:
            return ValueError(msg)
        if "rate_limit" in msg.lower() or "429" in msg:
            return ValueError(
                "Groq API rate limit reached. Please wait a moment and try again."
            )
        if "authentication" in msg.lower() or "401" in msg or "invalid_api_key" in msg.lower():
            return ValueError(
                "Invalid Groq API key. Please check your GROQ_API_KEY."
            )
        logger.error("Groq API error: %s", exc, exc_info=True)
        return ValueError(f"LLM error: {msg}")
