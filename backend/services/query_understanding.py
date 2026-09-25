"""Query understanding, intent classification, follow-up contextualization, and conceptual expansion."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Categorized intent of the user query."""

    CONCEPT_REASONING = "concept_reasoning"
    APPLICATION_IMPLEMENTATION = "application_implementation"
    MULTI_DOC_SYNTHESIS = "multi_doc_synthesis"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    SINGLE_DOC_DIRECT = "single_doc_direct"
    FOLLOW_UP = "follow_up"
    GENERAL_KNOWLEDGE = "general_knowledge"
    DOCUMENT_QA = "document_qa"


class AnswerType(str, Enum):
    """Grounding & reasoning classification of the generated response."""

    DIRECTLY_GROUNDED = "DIRECTLY_GROUNDED"
    DERIVED_FROM_DOCUMENT_CONCEPT = "DERIVED_FROM_DOCUMENT_CONCEPT"
    APPLICATION_IMPLEMENTATION = "APPLICATION_IMPLEMENTATION"
    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"


@dataclass
class QueryAnalysis:
    """Structured analysis of a user question."""

    original_query: str
    contextualized_query: str
    intent: QueryIntent
    is_ats_related: bool = False
    is_comparison: bool = False
    is_multi_doc: bool = False
    is_reasoning_required: bool = False
    is_code_requested: bool = False
    target_document_hints: list[str] = field(default_factory=list)
    expanded_search_terms: list[str] = field(default_factory=list)
    conceptual_search_queries: list[str] = field(default_factory=list)


# Pattern definitions
_MATH_CALCULATION_PATTERNS = [
    r"\b\d+\s*[\+\-\*\/\^]\s*\d+\b",
    r"\b(?:what\s+is\s+)?\d+\s*(?:plus|minus|times|divided\s+by|\+)\s*\d+\b",
    r"\b(?:calculate|compute|solve|sum\s+of|add)\s+\d+\b",
]

_CODE_IMPLEMENTATION_PATTERNS = [
    r"\b(?:give\s+me|write|show\s+me|provide|create|generate)\s+.*?\b(?:code|script|implementation|example)\b",
    r"\b(?:python|pytorch|tensorflow|keras|numpy|pandas)\s+(?:code|script|implementation|example)\b",
    r"\b(?:code\s+for|implementation\s+for|how\s+to\s+implement|how\s+to\s+code)\b",
    r"\b(?:in\s+python|using\s+python|in\s+pytorch|using\s+pytorch)\b",
]

_CONCEPT_REASONING_PATTERNS = [
    r"\b(?:how\s+would|what\s+happens\s+if|how\s+does\s+.*?change|why\s+does|how\s+will)\b",
    r"\bgradient\s+is\s+(?:positive|negative|zero)\b",
    r"\bparameter\s+(?:update|change|increase|decrease)\b",
    r"\b(?:apply|applying\s+the\s+concept|derive|derivation|deduce|infer)\b",
]

_COMPARISON_PATTERNS = [
    r"\bwhich\s+(?:resume|one|candidate|profile|person|paper|model|method|approach|is|has)\b",
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bbetter\b",
    r"\bstrongest\b",
    r"\bstronger\b",
    r"\bbest\b",
    r"\bmore\s+suitable\b",
    r"\bwho\s+has\s+more\b",
    r"\bwho\s+is\s+better\b",
    r"\brank\b",
    r"\branking\b",
    r"\bats\s+score\b",
    r"\bats\s+friendly\b",
    r"\bats-friendly\b",
    r"\bhigh\s+ats\b",
]

_SYNTHESIS_PATTERNS = [
    r"\bcommon\s+skills\b",
    r"\bskills\s+(?:are\s+)?missing\b",
    r"\bmissing\s+(?:skills|from)\b",
    r"\ball\s+(?:three|these|the)?\s*resumes\b",
    r"\bacross\s+(?:all|these|the)?\s*resumes\b",
    r"\bin\s+both\s+(?:resumes|documents|papers)\b",
    r"\bshared\s+by\b",
    r"\bwhat\s+do\s+they\s+have\s+in\s+common\b",
    r"\bcombine\s+(?:both|the|these|information|knowledge)\b",
    r"\busing\s+.*?\s+implement\s+.*?\s+described\s+in\b",
    r"\buse\s+.*?\s+to\s+implement\s+.*?\s+described\s+in\b",
]

_SUMMARY_PATTERNS = [
    r"\bsummar(?:y|ize|ise)\b",
    r"\boverview\b",
    r"\bbrief\s+me\b",
    r"\btell\s+me\s+about\s+(?:[a-zA-Z]+(?:'s)?\s+resume|[a-zA-Z]+)\b",
]

_FOLLOW_UP_PATTERNS = [
    r"^(?:what|how)\s+about\b",
    r"^and\s+(?:what|how|who|the)\b",
    r"^which\s+one\b",
    r"^who\s+(?:has|is)\b",
    r"^(?:now\s+)?(?:give|show|write)\s+me\s+code\s+(?:for\s+it|for\s+that)?\??$",
    r"^(?:code\s+for\s+it|implementation\s+for\s+it)\??$",
    r"^what\s+about\s+(?:certifications|projects|skills|experience|education)\??$",
    r"^(?:certifications|projects|skills|experience|education)\??$",
    r"^tell\s+me\s+more\b",
    r"^(?:why|how)\s+so\??$",
]

_GENERAL_KNOWLEDGE_PATTERNS = [
    r"\bwhat\s+skills\s+should\s+a\s+fresher\s+learn\b",
    r"\bhow\s+to\s+prepare\s+for\b",
    r"\bwhat\s+is\s+an?\s+ats\b",
    r"\bwhat\s+is\s+rag\b",
    r"\bindustry\s+standards?\b",
    r"\bhow\s+does\s+ats\s+work\b",
    r"\bwhat\s+are\s+the\s+best\s+practices\b",
    r"\broadmap\s+for\b",
    r"\bcareer\s+advice\b",
]

_ATS_PATTERNS = [
    r"\bats\b",
    r"\bats\s+score\b",
    r"\bats-friendly\b",
    r"\bats\s+friendly\b",
    r"\bats\s+compliance\b",
    r"\bapplicant\s+tracking\b",
]


class QueryUnderstanding:
    """Understands user query intent, resolves follow-up context, and generates conceptual search expansions."""

    @staticmethod
    def analyze_query(
        question: str,
        history: list | None = None,
        available_documents: list[dict[str, Any]] | None = None,
    ) -> QueryAnalysis:
        """Perform comprehensive analysis of user query."""
        clean_q = question.strip()
        low_q = clean_q.lower()

        # 1. Detect ATS-specific intent
        is_ats = any(re.search(pat, low_q) for pat in _ATS_PATTERNS)

        # 2. Check for follow-up query with conversation history
        is_follow_up = False
        contextualized = clean_q
        if history and len(history) > 0:
            if (
                any(re.search(pat, low_q) for pat in _FOLLOW_UP_PATTERNS)
                or len(clean_q.split()) <= 6
                or "for it" in low_q
                or "for that" in low_q
                or "about it" in low_q
            ):
                is_follow_up = True
                contextualized = QueryUnderstanding._resolve_follow_up(clean_q, history)

        low_ctx = contextualized.lower()

        # 3. Detect code requests and reasoning requirements
        is_code = any(re.search(pat, low_q) for pat in _CODE_IMPLEMENTATION_PATTERNS) or any(
            re.search(pat, low_ctx) for pat in _CODE_IMPLEMENTATION_PATTERNS
        )
        is_math = any(re.search(pat, low_q) for pat in _MATH_CALCULATION_PATTERNS)
        is_reasoning = is_math or any(re.search(pat, low_q) for pat in _CONCEPT_REASONING_PATTERNS) or any(
            re.search(pat, low_ctx) for pat in _CONCEPT_REASONING_PATTERNS
        )

        # 4. Detect target document hints from available document filenames
        target_doc_hints: list[str] = []
        if available_documents:
            for doc in available_documents:
                fn = doc.get("filename", "")
                base_name = re.sub(r"\.[a-zA-Z0-9]+$", "", fn).lower()
                parts = re.split(r"[_\-\s]+", base_name)
                for part in parts:
                    if len(part) >= 3 and (part in low_q or part in low_ctx):
                        if fn not in target_doc_hints:
                            target_doc_hints.append(fn)
                        break

            # Special topic hints (e.g. 'paper', 'transformer' -> NIPS paper, 'python' -> python doc)
            if "transformer" in low_q or "paper" in low_q or "attention" in low_q:
                for doc in available_documents:
                    fn = doc.get("filename", "")
                    if "attention" in fn.lower() or "transformer" in fn.lower() or "paper" in fn.lower():
                        if fn not in target_doc_hints:
                            target_doc_hints.append(fn)

            if "python" in low_q:
                for doc in available_documents:
                    fn = doc.get("filename", "")
                    if "python" in fn.lower():
                        if fn not in target_doc_hints:
                            target_doc_hints.append(fn)

        # 5. Classify primary intent
        is_comparison = any(re.search(pat, low_q) for pat in _COMPARISON_PATTERNS) or (
            is_follow_up and any(re.search(pat, low_ctx) for pat in _COMPARISON_PATTERNS)
        )
        is_synthesis = any(re.search(pat, low_q) for pat in _SYNTHESIS_PATTERNS) or (
            is_follow_up and any(re.search(pat, low_ctx) for pat in _SYNTHESIS_PATTERNS)
        )
        is_summary = any(re.search(pat, low_q) for pat in _SUMMARY_PATTERNS)
        is_general = any(re.search(pat, low_q) for pat in _GENERAL_KNOWLEDGE_PATTERNS)

        if is_code:
            intent = QueryIntent.APPLICATION_IMPLEMENTATION
        elif is_reasoning:
            intent = QueryIntent.CONCEPT_REASONING
        elif is_synthesis:
            intent = QueryIntent.MULTI_DOC_SYNTHESIS
        elif is_summary:
            intent = QueryIntent.SUMMARY
        elif is_general and not target_doc_hints and not is_comparison:
            intent = QueryIntent.GENERAL_KNOWLEDGE
        elif is_comparison or is_ats:
            intent = QueryIntent.COMPARISON
        elif is_follow_up:
            intent = QueryIntent.FOLLOW_UP
        elif target_doc_hints and not is_comparison:
            intent = QueryIntent.SINGLE_DOC_DIRECT
        else:
            intent = QueryIntent.DOCUMENT_QA

        # Multi-doc condition: comparison, multi-doc synthesis, or multiple targeted docs
        total_docs = len(available_documents or [])
        is_multi_doc = intent in (
            QueryIntent.COMPARISON,
            QueryIntent.MULTI_DOC_SYNTHESIS,
        ) or (len(target_doc_hints) == 0 and total_docs > 1 and (is_code or is_reasoning or is_synthesis)) or len(target_doc_hints) > 1

        # 6. Conceptual search query rewrites and expansions
        conceptual_queries, expansions = QueryUnderstanding._generate_conceptual_rewrites(
            contextualized, intent, is_ats, target_doc_hints
        )

        logger.info(
            "Query Understanding: intent=%s, is_reasoning=%s, is_code=%s, is_multi_doc=%s, targets=%s, conceptual_rewrites=%s",
            intent.value,
            is_reasoning,
            is_code,
            is_multi_doc,
            target_doc_hints,
            conceptual_queries,
        )

        return QueryAnalysis(
            original_query=clean_q,
            contextualized_query=contextualized,
            intent=intent,
            is_ats_related=is_ats,
            is_comparison=is_comparison or is_ats,
            is_multi_doc=is_multi_doc,
            is_reasoning_required=is_reasoning,
            is_code_requested=is_code,
            target_document_hints=target_doc_hints,
            expanded_search_terms=expansions,
            conceptual_search_queries=conceptual_queries,
        )

    @staticmethod
    def _resolve_follow_up(question: str, history: list) -> str:
        """Resolve pronouns and missing context in follow-up queries using recent conversation history."""
        recent_contexts: list[str] = []
        for msg in reversed(history[-6:]):
            content = getattr(msg, "content", "")
            if content and isinstance(content, str):
                clean = re.sub(r"Document Context:.*?\n\nQuestion:", "", content, flags=re.DOTALL)
                clean = clean.strip()
                if clean:
                    recent_contexts.append(clean[:400])

        combined_prior = " ".join(recent_contexts).lower()

        # 1. Technical Concepts
        if "transformer" in combined_prior or "attention" in combined_prior:
            concept = "Transformer architecture and self-attention"
            if re.search(r"\b(?:code\s+for\s+it|give\s+me\s+code\s+for\s+it|implement\s+it)\b", question.lower()) or "code" in question.lower():
                return f"Give me practical Python / PyTorch implementation code for the {concept}"
            return f"{question} (context: {concept})"

        if "gradient descent" in combined_prior or "gradient" in combined_prior:
            concept = "gradient descent parameter update optimization"
            return f"{question} (context: {concept})"

        if "addition" in combined_prior or "arithmetic" in combined_prior:
            concept = "arithmetic addition concept"
            return f"{question} (context: {concept})"

        # 2. Resume / Candidate Entities
        entities: list[str] = []
        if "resume" in combined_prior or "candidate" in combined_prior:
            entities.append("resumes")
        if "ai/ml" in combined_prior or "machine learning" in combined_prior or "aiml" in combined_prior:
            entities.append("AI/ML")
        if "data analyst" in combined_prior:
            entities.append("Data Analyst")
        if "aditya" in combined_prior:
            entities.append("Aditya")
        if "suyash" in combined_prior:
            entities.append("Suyash")

        entity_suffix = " ".join(entities) if entities else "the uploaded documents"

        q_lower = question.lower().strip("?. ")
        if q_lower in ("what about certifications", "certifications", "certifications?"):
            return f"What certifications, credentials, and courses do the candidates have across {entity_suffix}?"
        if q_lower in ("what about projects", "projects", "projects?"):
            return f"What technical projects and software implementations are featured in {entity_suffix}?"
        if q_lower in ("what about experience", "experience", "experience?"):
            return f"What work experience, employment, and internships are listed in {entity_suffix}?"
        if q_lower in ("what about skills", "skills", "technical skills", "skills?"):
            return f"What technical skills, programming languages, and tools are listed in {entity_suffix}?"
        if "which one" in q_lower or "better" in q_lower or "stronger" in q_lower:
            return f"{question} across {entity_suffix}"

        return f"{question} (context: {entity_suffix})"

    @staticmethod
    def _generate_conceptual_rewrites(
        query: str,
        intent: QueryIntent,
        is_ats: bool,
        target_doc_hints: list[str],
    ) -> tuple[list[str], list[str]]:
        """Rewrite user queries into theoretical conceptual search terms and domain expansions."""
        low = query.lower()
        conceptual_queries: list[str] = []
        expansions: list[str] = []

        # 1. Arithmetic / Mathematics Conceptual Rewriting
        # e.g., "What is 3 + 4?", "What is 37 + 42?"
        if any(re.search(pat, low) for pat in _MATH_CALCULATION_PATTERNS) or "add" in low or "sum" in low or "arithmetic" in low:
            conceptual_queries.append("definition and concept of addition arithmetic combining numbers operations sum")
            conceptual_queries.append("mathematical operations addition subtraction equations numbers")
            expansions.append("addition plus combine total sum value arithmetic numbers")

        # 2. Optimization / Machine Learning Reasoning
        # e.g., "If gradient is positive, how does parameter change?", "Gradient descent"
        if "gradient" in low or "descent" in low or "learning rate" in low or "parameter" in low:
            conceptual_queries.append("gradient descent parameter update learning rate loss reduction minimization direction")
            conceptual_queries.append("optimization algorithm gradient descent parameter adjustment loss function")
            expansions.append("gradient parameter update rule learning rate loss objective function derivative")

        # 3. Transformer / Neural Architecture Implementation & Concepts
        # e.g., "Give me Python/PyTorch code to implement a Transformer"
        if "transformer" in low or "attention" in low or "encoder" in low or "decoder" in low:
            conceptual_queries.append("Transformer architecture self-attention multi-head attention encoder decoder layers")
            conceptual_queries.append("Scaled Dot-Product Attention multi-head attention positional encoding architecture")
            expansions.append("self-attention multi-head attention feed-forward layers encoder decoder model architecture")

        # 4. Multi-Topic Synthesis (e.g. Python + Transformer)
        if ("python" in low or "code" in low) and ("transformer" in low or "model" in low or "paper" in low):
            conceptual_queries.append("Python programming language syntax functions classes modules")
            conceptual_queries.append("Transformer architecture self-attention multi-head attention")
            expansions.append("Python implementation modules functions PyTorch torch.nn Transformer")

        # 5. ATS / Resume domain expansions
        if is_ats or "ats" in low or "best" in low or "strongest" in low or "stronger" in low:
            expansions.append("technical skills programming languages tools frameworks experience projects education")

        if "ai/ml" in low or "machine learning" in low or "aiml" in low or "deep learning" in low:
            expansions.append("machine learning deep learning PyTorch TensorFlow NLP LLM computer vision Python models")

        if "data analyst" in low or "analytics" in low:
            expansions.append("SQL Python data analysis visualization Tableau PowerBI statistics pandas Excel metrics")

        if "project" in low or "projects" in low:
            expansions.append("projects technical implementations GitHub system architecture tools technologies used")

        if "skill" in low or "skills" in low or "technical" in low:
            expansions.append("technical skills programming languages frameworks libraries databases cloud developer tools")

        if "certif" in low or "course" in low:
            expansions.append("certifications certified courses credentials licenses training achievements")

        if "experience" in low or "intern" in low or "work" in low:
            expansions.append("work experience employment internship professional responsibilities achievements tenure")

        if "missing" in low or "fresher" in low:
            expansions.append("core fundamentals machine learning algorithms data structures math statistics deployment cloud")

        if intent == QueryIntent.SUMMARY:
            expansions.append("education technical skills work experience projects certifications summary profile")

        if intent == QueryIntent.COMPARISON and not expansions:
            expansions.append("skills experience projects education qualifications achievements")

        # Clean duplicates while preserving order
        clean_concepts = list(dict.fromkeys(conceptual_queries))
        clean_expansions = list(dict.fromkeys(expansions))

        return clean_concepts, clean_expansions
