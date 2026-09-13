"""Citation selection service for RAG answers.

Implements precision citation filtering:
1. Negative answer detection (returns [] when information is not found).
2. Distance-to-similarity conversion for ChromaDB L2 embeddings.
3. Configurable similarity threshold filtering.
4. Answer-source alignment via:
   - Explicit page/document citation parsing from generated answer.
   - Lexical & numeric fact overlap between generated answer and chunk text.
5. Relative margin filtering (keeps only top-tier supporting chunks).
6. Page-level deduplication (combining multiple chunks from same page into 1 citation).
7. Structured debugging logs.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Common stopwords to ignore during term overlap analysis
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "by", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "from", "up", "down", "of", "off", "over", "under",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "would", "should", "could", "ought",
    "i", "you", "he", "she", "it", "we", "they", "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "can", "will", "just",
    "paper", "model", "transformer", "document", "documents", "uploaded"
}

# Phrases indicating that the information could not be found
NEGATIVE_PHRASES = [
    "couldn't find",
    "could not find",
    "cannot find",
    "can't find",
    "not found in",
    "not mentioned in",
    "does not contain",
    "doesn't contain",
    "no information",
    "not provide this information",
    "not provided in",
    "unable to find",
]


def is_negative_answer(answer: str) -> bool:
    """Return True if the answer states that the information was not found."""
    low = answer.lower()
    return any(p in low for p in NEGATIVE_PHRASES)


def extract_cited_pages(answer: str) -> set[int]:
    """Find page numbers explicitly referenced in the LLM answer."""
    pages = set()
    # Matches: Page 5, page 5, Page 5, Page: 5, Page - 5, p. 5
    for m in re.finditer(r'(?:page|p\.)\s*[\:\-]?\s*(\d+)', answer, re.IGNORECASE):
        try:
            pages.add(int(m.group(1)))
        except ValueError:
            pass
    return pages


def extract_answer_terms(answer: str) -> set[str]:
    """Extract informative terms, numbers, and technical tokens from the answer."""
    clean_ans = re.sub(r'【[^】]*】', ' ', answer)
    tokens = re.findall(r'[a-zA-Z0-9_\-\.\=]+', clean_ans.lower())
    terms = set()
    for t in tokens:
        t_clean = t.strip(".-_")
        if not t_clean or len(t_clean) < 2 or t_clean in STOP_WORDS:
            continue
        terms.add(t_clean)
    return terms


def distance_to_similarity(dist: float) -> float:
    """Convert Chroma Euclidean distance to cosine similarity [0, 1]."""
    # For normalized embeddings: d^2 = 2 - 2*sim => sim = 1 - d^2 / 2
    sim = 1.0 - (dist * dist) / 2.0
    return max(0.0, min(1.0, sim))


def select_citations(
    query: str,
    answer: str,
    retrieved_chunks: list[tuple[Document, float]],
    similarity_threshold: float = 0.20,
    margin: float = 0.25,
    max_citations: int = 4,
) -> list[dict[str, Any]]:
    """Select the minimum sufficient, highly relevant citations supporting the answer."""
    total_retrieved = len(retrieved_chunks)

    # 1. Check for negative / "not found" answer
    if is_negative_answer(answer):
        logger.info(
            "Citation Selection:\n"
            "  Query: %s\n"
            "  Retrieved chunks: %d\n"
            "  Answer is negative ('not found') -> Final citations: 0",
            query,
            total_retrieved,
        )
        return []

    cited_pages = extract_cited_pages(answer)
    answer_terms = extract_answer_terms(answer)
    answer_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', answer))

    # 2. Score and filter chunks
    scored_chunks = []
    for doc, dist in retrieved_chunks:
        sim = distance_to_similarity(dist)
        if sim < similarity_threshold:
            continue

        page = doc.metadata.get("page_number", 0)
        content_lower = doc.page_content.lower()

        page_explicitly_cited = page in cited_pages if page else False

        # Term overlap
        overlap_count = sum(1 for term in answer_terms if term in content_lower)
        overlap_ratio = overlap_count / max(len(answer_terms), 1)

        # Number overlap (crucial for factual answers like 8, 64, 6)
        content_words = set(re.findall(r'\b\w+(?:\.\w+)?\b', content_lower))
        number_matches = sum(1 for num in answer_numbers if num in content_words)
        has_number_match = number_matches > 0 if answer_numbers else True


        # Combined alignment score
        alignment_score = sim * 0.5 + overlap_ratio * 0.3
        if page_explicitly_cited:
            alignment_score += 0.35
        if has_number_match and answer_numbers:
            alignment_score += 0.15

        scored_chunks.append({
            "doc": doc,
            "dist": dist,
            "sim": sim,
            "alignment_score": alignment_score,
            "page": page,
            "page_cited": page_explicitly_cited,
            "overlap_ratio": overlap_ratio,
            "has_number_match": has_number_match,
        })

    after_threshold_count = len(scored_chunks)

    if not scored_chunks:
        logger.info(
            "Citation Selection:\n"
            "  Query: %s\n"
            "  Retrieved chunks: %d\n"
            "  After threshold (sim >= %.2f): 0\n"
            "  Final citations: 0",
            query,
            total_retrieved,
            similarity_threshold,
        )
        return []

    # Sort by alignment score descending
    scored_chunks.sort(key=lambda x: x["alignment_score"], reverse=True)
    top_alignment = scored_chunks[0]["alignment_score"]

    # 3. Apply margin cutoff relative to top score
    selected_candidates = []
    for sc in scored_chunks:
        is_close_enough = (top_alignment - sc["alignment_score"]) <= margin
        if sc["page_cited"] or is_close_enough:
            if answer_numbers and not sc["has_number_match"] and not sc["page_cited"] and sc["overlap_ratio"] < 0.3:
                continue
            selected_candidates.append(sc)

    # 4. Deduplicate by (filename, page_number)
    deduped = {}
    for sc in selected_candidates:
        doc = sc["doc"]
        fn = doc.metadata.get("filename", "Unknown")
        p = sc["page"]
        key = (fn, p)
        if key not in deduped or sc["alignment_score"] > deduped[key]["alignment_score"]:
            deduped[key] = sc

    final_list = list(deduped.values())
    final_list.sort(key=lambda x: x["alignment_score"], reverse=True)
    final_list = final_list[:max_citations]

    results = []
    for sc in final_list:
        doc = sc["doc"]
        fn = doc.metadata.get("filename", "Unknown")
        p = sc["page"]
        text = doc.page_content.strip()
        chunk_id = doc.metadata.get("chunk_id") or f"{doc.metadata.get('document_id', '')}_chunk_{doc.metadata.get('chunk_index', '')}"
        results.append({
            "filename": fn,
            "page_number": int(p) if p and str(p).isdigit() and int(p) > 0 else None,
            "chunk_preview": text[:150] + ("…" if len(text) > 150 else ""),
            "document_id": doc.metadata.get("document_id", ""),
            "chunk_id": chunk_id,
            "relevance_score": round(sc["sim"], 2),
        })

    citation_details = [
        f"Page {r.get('page_number')} (score: {r.get('relevance_score')})"
        for r in results
    ]

    logger.info(
        "Citation Selection:\n"
        "  Query: %s\n"
        "  Retrieved chunks: %d\n"
        "  After threshold (sim >= %.2f): %d\n"
        "  After deduplication: %d\n"
        "  Final citations: %d\n"
        "  Sources: %s",
        query,
        total_retrieved,
        similarity_threshold,
        after_threshold_count,
        len(deduped),
        len(results),
        ", ".join(citation_details) if citation_details else "None",
    )

    return results
