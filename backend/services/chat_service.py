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
import re
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
from services.query_understanding import (
    AnswerType,
    QueryAnalysis,
    QueryIntent,
    QueryUnderstanding,
)
from services.vector_store import VectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an intelligent knowledge-grounded AI assistant and document analyst.
Your purpose is to answer questions using the uploaded documents as KNOWLEDGE, CONCEPTS, CONTEXT, and EVIDENCE, while reasoning, deriving, applying, and extending those concepts logically, creatively, and accurately.

CORE OPERATIONAL PRINCIPLES:

1. REASONING & CONCEPTUAL APPLICATION (NOT JUST LITERAL SEARCH):
   - You are NOT restricted to answering only if the exact verbatim words or answers exist in the documents.
   - When the document introduces concepts, definitions, rules, architectures, or formulas:
     * REASON and DERIVE the logical consequences (e.g., if the document explains addition, calculate 3 + 4 = 7; if the document defines gradient descent updating opposite to the gradient, deduce how parameters update when the gradient is positive).
     * APPLY the concepts to practical tasks (e.g., if the document specifies the Transformer architecture without code, provide a clean Python/PyTorch implementation based on those architectural specifications).
     * SYNTHESIZE concepts across multiple uploaded documents (e.g., combining a Python document and a Transformer paper to build a working implementation).
   - Clearly state the conceptual basis: e.g., "The uploaded document explains the concept of addition; applying that concept, 3 + 4 = 7." or "The paper describes the Transformer architecture but does not provide implementation code. Based on that architecture, here is a practical PyTorch example..."

2. GROUNDING VS. APPLICATION VS. GENERAL KNOWLEDGE DISTINCTION:
   - Always clearly distinguish:
     a) What the uploaded source explicitly says (with document citations).
     b) What you derive or calculate from those concepts.
     c) Practical implementations, code examples, or general best practices you supply.
   - NEVER pretend that the uploaded document contains code, experimental numbers, or specific facts that it does not contain.

3. GENERAL KNOWLEDGE HANDLING:
   - If the user asks a legitimate general conceptual, programming, or domain question (such as 'What is the difference between supervised and unsupervised learning?') that is not covered in the uploaded documents:
     * Answer thoroughly, accurately, and helpfully using general knowledge.
     * Do NOT fabricate document citations or falsely attribute general knowledge to the uploaded files.
     * Do NOT refuse with 'I could not find this information' for general knowledge questions.

4. CITATION RULES:
   - Explicitly cite the document name and page number when referencing concepts or facts from documents (e.g., [From: NIPS-2017-attention-is-all-you-need-Paper.pdf, Page 3]).
   - Do NOT attach citations to generated code blocks or external general knowledge as if they were written in the document.

5. ATS & RESUME EVALUATION:
   - When analyzing resumes or candidates, perform an evidence-based qualitative assessment (skills, experience, projects, formatting).
   - NEVER fabricate an arbitrary numeric ATS score (e.g., "92% ATS" or "Score: 85/100") unless explicitly in the text.

6. WHEN TO STATE INSUFFICIENT INFORMATION:
   - State that information is unavailable ONLY when:
     1) The user explicitly demands a specific document-internal fact, metric, or entity that is completely absent and cannot be inferred, calculated, or derived (e.g. exact dollar training cost in a research paper).
     2) The topic is entirely outside both the uploaded documents and valid general knowledge.
   - NEVER say "I couldn't find this information in the uploaded documents." simply because the exact question or answer is not verbatim in the text. When a concept exists in the document, reason from it and answer!"""


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
                is_pinned=False,
                is_custom_title=bool(initial_title and initial_title != "New Conversation"),
            )
            db.add(conv)
            db.commit()
            db.refresh(conv)
            return conv

        # 2. Compatibility mode: session_id supplied
        if session_id:
            conv = (
                db.query(Conversation)
                .filter(Conversation.id == session_id, Conversation.user_id == user_id)
                .first()
            )
            if conv is not None:
                return conv

            other_conv = db.query(Conversation).filter(Conversation.id == session_id).first()
            if other_conv is not None and other_conv.user_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found",
                )

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
                is_pinned=False,
                is_custom_title=bool(initial_title and initial_title != "New Conversation"),
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
            is_pinned=False,
            is_custom_title=bool(initial_title and initial_title != "New Conversation"),
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
        return conv

    def _load_history_for_llm(
        self, db: Session, conversation_id: str, limit: int = 6, max_char_per_msg: int = 1000
    ) -> list:
        """Fetch the most recent `limit` messages from PostgreSQL for LLM context, condensing large assistant turns."""
        recent_messages = (
            db.query(Message.role, Message.content)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
            .all()
        )
        recent_messages.reverse()

        history = []
        for role, content in recent_messages:
            trimmed = content or ""
            if len(trimmed) > max_char_per_msg:
                trimmed = trimmed[:max_char_per_msg] + "… [condensed]"
            if role == "user":
                history.append(HumanMessage(content=trimmed))
            elif role == "assistant":
                history.append(AIMessage(content=trimmed))
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
        self,
        question: str,
        user_id: int | None = None,
        history: list | None = None,
    ) -> tuple[str, list[tuple[Document, float]], QueryAnalysis]:
        """Search vector store adaptively and build prompt context + raw candidate list."""
        available_docs = (
            self.vector_store.list_documents_for_user(user_id)
            if user_id is not None
            else self.vector_store.list_documents()
        )

        analysis = QueryUnderstanding.analyze_query(
            question, history=history, available_documents=available_docs
        )

        candidates: list[tuple[Document, float]] = []

        # Collect distinct search queries: contextualized + conceptual + expansions
        search_queries = [analysis.contextualized_query]
        if analysis.conceptual_search_queries:
            search_queries.extend(analysis.conceptual_search_queries)
        if analysis.expanded_search_terms:
            search_queries.extend(analysis.expanded_search_terms)
        search_queries = list(dict.fromkeys(search_queries))

        if user_id is None or not available_docs:
            if user_id is not None:
                results = self.vector_store.similarity_search_for_user(
                    analysis.contextualized_query, user_id, top_k=settings.retrieval_top_k
                )
            else:
                results = self.vector_store.similarity_search(
                    analysis.contextualized_query, top_k=settings.retrieval_top_k
                )
            candidates = results
        elif analysis.is_comparison or analysis.is_multi_doc or analysis.intent == QueryIntent.MULTI_DOC_SYNTHESIS:
            # Multi-document comparison or synthesis: balanced retrieval across user documents
            target_ids = None
            if analysis.target_document_hints:
                target_ids = [
                    d["document_id"]
                    for d in available_docs
                    if d.get("filename") in analysis.target_document_hints
                ]
            candidates = self.vector_store.similarity_search_multi_doc_for_user(
                queries=search_queries,
                user_id=user_id,
                document_ids=target_ids if target_ids else None,
                top_k_per_doc=2,
                global_top_k=2,
            )
        elif analysis.target_document_hints:
            # Targeted to specific document(s)
            target_ids = [
                d["document_id"]
                for d in available_docs
                if d.get("filename") in analysis.target_document_hints
            ]
            candidates = self.vector_store.similarity_search_multi_doc_for_user(
                queries=search_queries,
                user_id=user_id,
                document_ids=target_ids,
                top_k_per_doc=3,
                global_top_k=2,
            )
        else:
            # Standard single doc / direct query / follow-up / conceptual query
            results = self.vector_store.similarity_search_for_user(
                analysis.contextualized_query, user_id, top_k=settings.retrieval_top_k
            )
            top_sim = distance_to_similarity(results[0][1]) if results else 0.0
            if (not results or top_sim < 0.20 or analysis.is_code_requested or analysis.is_reasoning_required) and len(search_queries) > 1:
                # Adaptive conceptual retrieval: supplement with conceptual query rewrites
                extra = self.vector_store.similarity_search_multi_doc_for_user(
                    queries=search_queries[1:4],
                    user_id=user_id,
                    top_k_per_doc=2,
                    global_top_k=2,
                )
                chunk_map: dict[str, tuple[Document, float]] = {}
                for doc, dist in results + extra:
                    ck = f"{doc.metadata.get('document_id', '')}_{doc.metadata.get('page_number', 0)}_{hash(doc.page_content[:100])}"
                    if ck not in chunk_map or dist < chunk_map[ck][1]:
                        chunk_map[ck] = (doc, dist)
                candidates = sorted(chunk_map.values(), key=lambda x: x[1])
            else:
                candidates = results

        if not candidates:
            return "", [], analysis

        # Keep top candidate chunks (cap at 8 to respect LLM context token window and Groq TPM)
        final_candidates = candidates[:8]

        context_parts: list[str] = []
        for doc, _dist in final_candidates:
            filename = doc.metadata.get("filename", "Unknown")
            page = doc.metadata.get("page_number", 0)
            text = doc.page_content

            header = f"[From: {filename}"
            if page and int(page) > 0:
                header += f", Page {page}"
            header += "]"
            context_parts.append(f"{header}\n{text}")

        context = "\n\n---\n\n".join(context_parts)
        return context, final_candidates, analysis

    # ── Message building ────────────────────────────────────────────────

    def _build_messages(
        self,
        question: str,
        context: str,
        history: list,
        analysis: QueryAnalysis | None = None,
    ) -> list:
        """System prompt → history → current context + guidance + question."""
        msgs: list = [SystemMessage(content=SYSTEM_PROMPT)]
        msgs.extend(history)

        prompt_notes: list[str] = []
        if analysis:
            if analysis.intent == QueryIntent.CONCEPT_REASONING or analysis.is_reasoning_required:
                prompt_notes.append(
                    "CONCEPTUAL REASONING INSTRUCTION: Apply the principles and concepts from the document to reason, calculate, or derive the answer. "
                    "State the conceptual basis from the document, then show the derivation clearly (e.g. 'The document explains addition; applying that concept, 3 + 4 = 7'). "
                    "Do NOT say you couldn't find the information simply because the exact calculation is not in the text."
                )
            if analysis.intent == QueryIntent.APPLICATION_IMPLEMENTATION or analysis.is_code_requested:
                prompt_notes.append(
                    "APPLICATION / CODE INSTRUCTION: The user is requesting a practical implementation or code. "
                    "Use the document for theoretical/architectural grounding. Explicitly clarify that the document itself does not contain the implementation code, "
                    "then provide a clean, complete, working implementation (e.g. Python / PyTorch) based on the documented concepts."
                )
            if analysis.intent == QueryIntent.MULTI_DOC_SYNTHESIS or analysis.is_multi_doc:
                prompt_notes.append(
                    "MULTI-DOCUMENT SYNTHESIS INSTRUCTION: Combine and synthesize knowledge across the different uploaded documents. "
                    "Connect the concepts from both sources clearly."
                )
            if analysis.is_ats_related:
                prompt_notes.append(
                    "ATS EVALUATION INSTRUCTION: Analyze ATS-relevant factors qualitatively. "
                    "Explicitly state that this is an evidence-based qualitative analysis. Do NOT fabricate a numeric ATS score."
                )
            if analysis.is_comparison:
                prompt_notes.append(
                    "COMPARISON INSTRUCTION: Compare the candidates/documents fairly and thoroughly across the evidence retrieved."
                )
            if analysis.intent == QueryIntent.GENERAL_KNOWLEDGE:
                prompt_notes.append(
                    "GENERAL KNOWLEDGE INSTRUCTION: Provide thorough, actionable guidance, connecting back to the uploaded documents where relevant."
                )

        notes_str = ("\n\n[Special Instructions:\n" + "\n".join(prompt_notes) + "]") if prompt_notes else ""

        if context:
            user_content = (
                f"Document Context:\n{context}{notes_str}\n\nQuestion: {question}"
            )
        else:
            user_content = (
                f"No relevant document context was found.{notes_str}\n\n"
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
            conv_id = conv.id

            # Auto-title conversation on first message ONLY if not manually renamed
            if not getattr(conv, "is_custom_title", False) and conv.title == "New Conversation" and question:
                clean_title = question.strip()
                if len(clean_title) > 60:
                    clean_title = clean_title[:57] + "..."
                conv.title = clean_title
                db.commit()

            # 2. Load previous history before adding this message
            history = self._load_history_for_llm(db, conv_id, limit=20)

            # 3. Save current user message to PostgreSQL
            self._save_message(db, conv_id, "user", question)

            # 4. RAG context retrieval with query understanding
            context, raw_candidates, analysis = await asyncio.to_thread(
                self._retrieve_context, question, user_id, history
            )
            messages = self._build_messages(question, context, history, analysis)

            # 5. Invoke LLM
            llm = self._get_llm()
            try:
                response = await llm.ainvoke(messages)
                answer: str = response.content or ""
            except Exception as exc:
                raise self._handle_groq_error(exc) from exc

            # 6. Save assistant response to PostgreSQL
            self._save_message(db, conv_id, "assistant", answer)

            max_cites = (
                max(6, settings.max_citations)
                if (analysis and analysis.is_comparison)
                else settings.max_citations
            )
            sources = select_citations(
                query=question,
                answer=answer,
                retrieved_chunks=raw_candidates,
                similarity_threshold=settings.similarity_threshold,
                margin=settings.citation_margin,
                max_citations=max_cites,
            )

            answer_type = self._classify_answer_type(analysis, answer, sources)

            return {
                "answer": answer,
                "sources": sources,
                "conversation_id": conv_id,
                "session_id": conv_id,
                "answer_type": answer_type,
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

            # Auto-title on first message ONLY if not manually renamed
            if not getattr(conv, "is_custom_title", False) and conv.title == "New Conversation" and question:
                clean_title = question.strip()
                if len(clean_title) > 60:
                    clean_title = clean_title[:57] + "..."
                conv.title = clean_title
                db.commit()

            # 2. Load previous history before adding this message
            history = self._load_history_for_llm(db, conv_id, limit=20)

            # 3. Save user message to PostgreSQL
            self._save_message(db, conv_id, "user", question)

            # 4. RAG context retrieval with query understanding
            context, raw_candidates, analysis = await asyncio.to_thread(
                self._retrieve_context, question, user_id, history
            )
            messages = self._build_messages(question, context, history, analysis)

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
            max_cites = (
                max(6, settings.max_citations)
                if (analysis and analysis.is_comparison)
                else settings.max_citations
            )
            sources = select_citations(
                query=question,
                answer=full_answer,
                retrieved_chunks=raw_candidates,
                similarity_threshold=settings.similarity_threshold,
                margin=settings.citation_margin,
                max_citations=max_cites,
            )

            answer_type = self._classify_answer_type(analysis, full_answer, sources)

            yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'answer_type': answer_type})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': conv_id, 'session_id': conv_id, 'answer_type': answer_type})}\n\n"

        except HTTPException as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': exc.detail})}\n\n"
        except Exception as exc:
            err = self._handle_groq_error(exc)
            yield f"data: {json.dumps({'type': 'error', 'content': str(err)})}\n\n"
        finally:
            db.close()

    # ── Answer Classification ───────────────────────────────────────────

    @staticmethod
    def _classify_answer_type(analysis: QueryAnalysis | None, answer: str, sources: list) -> str:
        """Classify answer into one of the 5 architectural answer types."""
        low_ans = answer.lower()
        low_clean = re.sub(r'[*_`]', '', low_ans)
        rejection_phrases = [
            "couldn't find this information",
            "could not find this information",
            "cannot find this information",
            "can't find this information",
            "not mentioned in the uploaded documents",
            "no relevant document context was found",
            "insufficient information in the uploaded documents",
            "not present in the uploaded documents",
            "do not contain any information",
            "does not contain any information",
            "do not contain information",
            "does not contain information",
            "no information about the monetary cost",
            "no information about the cost",
            "no information about",
            "does not mention the cost",
            "do not mention the cost",
            "does not report a dollar amount",
            "does not report a cost",
            "does not specify the cost",
            "do not specify the cost",
            "don’t have a specific dollar amount",
            "dont have a specific dollar amount",
            "cannot be answered or derived",
        ]
        opening = low_clean[:350]
        # 1. Check for genuine missing document-specific information / refusal
        if any(rp in opening for rp in rejection_phrases) and "```" not in answer and "applying that concept" not in low_clean and "derived" not in low_clean:
            return AnswerType.INSUFFICIENT_INFORMATION.value

        if any(rp in low_clean for rp in rejection_phrases) and len(sources) == 0:
            return AnswerType.INSUFFICIENT_INFORMATION.value

        # 2. General Knowledge: explicitly classified or answered without document citations
        if analysis and analysis.intent == QueryIntent.GENERAL_KNOWLEDGE:
            return AnswerType.GENERAL_KNOWLEDGE.value

        if len(sources) == 0 and not (analysis and (analysis.is_code_requested or analysis.is_reasoning_required)):
            return AnswerType.GENERAL_KNOWLEDGE.value

        # 3. Application / Implementation: practical code generated based on document specifications
        if (analysis and analysis.is_code_requested) or (
            len(sources) > 0 and ("```" in answer or "practical implementation" in low_clean) and not (analysis and analysis.is_reasoning_required)
        ):
            return AnswerType.APPLICATION_IMPLEMENTATION.value

        # 4. Derived from Document Concept: logical or mathematical derivation from document rules
        if (analysis and analysis.is_reasoning_required) or "applying that concept" in low_clean or "derived" in low_clean or "conceptually" in low_clean:
            return AnswerType.DERIVED_FROM_DOCUMENT_CONCEPT.value

        # 5. Directly Grounded: exact fact present in document
        if len(sources) > 0:
            return AnswerType.DIRECTLY_GROUNDED.value

        return AnswerType.GENERAL_KNOWLEDGE.value

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
