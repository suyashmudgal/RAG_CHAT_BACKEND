"""Chat service — RAG retrieval, conversational memory, Groq LLM (streaming).

Uses:
- LangChain ``ChatGroq`` for LLM calls (free tier)
- LangChain message types for history
- Custom SSE streaming for the React frontend
"""

import asyncio
import json
import logging
from collections import defaultdict
from typing import AsyncGenerator

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config.settings import settings
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
    """Orchestrates RAG retrieval, conversation memory, and Groq LLM calls."""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store
        # session_id → list[HumanMessage | AIMessage]  (clean, without context)
        self.conversations: dict[str, list] = defaultdict(list)
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

    # ── Retrieval ───────────────────────────────────────────────────────

    def _retrieve_context(
        self, question: str
    ) -> tuple[str, list[tuple[Document, float]]]:
        """Search vector store and build prompt context + raw candidate list."""
        top_k = getattr(settings, "retrieval_top_k", settings.top_k_results)

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
        self, question: str, context: str, session_id: str
    ) -> list:
        """System prompt → history → current context + question."""
        msgs: list = [SystemMessage(content=SYSTEM_PROMPT)]

        # Last 10 Q/A exchanges (20 messages)
        history = self.conversations[session_id][-20:]
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

    def _save_to_history(
        self, session_id: str, question: str, answer: str
    ) -> None:
        """Store the *clean* Q/A pair (no RAG context) in memory."""
        self.conversations[session_id].append(
            HumanMessage(content=question)
        )
        self.conversations[session_id].append(AIMessage(content=answer))
        # Cap at ~20 latest exchanges
        if len(self.conversations[session_id]) > 40:
            self.conversations[session_id] = self.conversations[session_id][
                -20:
            ]

    # ── Public API ──────────────────────────────────────────────────────

    async def get_answer(self, question: str, session_id: str) -> dict:
        """Non-streaming RAG answer."""
        context, raw_candidates = await asyncio.to_thread(self._retrieve_context, question)
        messages = self._build_messages(question, context, session_id)

        llm = self._get_llm()
        try:
            response = await llm.ainvoke(messages)
            answer: str = response.content or ""
        except Exception as exc:
            raise self._handle_groq_error(exc) from exc

        sources = select_citations(
            query=question,
            answer=answer,
            retrieved_chunks=raw_candidates,
            similarity_threshold=settings.similarity_threshold,
            margin=settings.citation_margin,
            max_citations=settings.max_citations,
        )

        self._save_to_history(session_id, question, answer)
        return {"answer": answer, "sources": sources, "session_id": session_id}

    async def stream_answer(
        self, question: str, session_id: str
    ) -> AsyncGenerator[str, None]:
        """Yield Server-Sent Events: ``token`` → ``sources`` → ``done``."""
        context, raw_candidates = await asyncio.to_thread(self._retrieve_context, question)
        messages = self._build_messages(question, context, session_id)


        try:
            llm = self._get_llm()
            full_answer = ""

            async for chunk in llm.astream(messages):
                token = chunk.content
                if token:
                    full_answer += token
                    yield (
                        f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                    )

            self._save_to_history(session_id, question, full_answer)

            # Precision citation selection aligned with the generated answer
            sources = select_citations(
                query=question,
                answer=full_answer,
                retrieved_chunks=raw_candidates,
                similarity_threshold=settings.similarity_threshold,
                margin=settings.citation_margin,
                max_citations=settings.max_citations,
            )

            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as exc:
            err = self._handle_groq_error(exc)
            yield f"data: {json.dumps({'type': 'error', 'content': str(err)})}\n\n"

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
