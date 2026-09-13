import re
import math
from typing import Any

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
]

def is_negative_answer(answer: str) -> bool:
    low = answer.lower()
    return any(p in low for p in NEGATIVE_PHRASES)

def extract_cited_pages(answer: str) -> set[int]:
    """Find page numbers explicitly referenced in the LLM answer."""
    pages = set()
    # Matches: Page 5, page 5, Page 5, Page: 5, Page - 5
    for m in re.finditer(r'(?:page|p\.)\s*[\:\-]?\s*(\d+)', answer, re.IGNORECASE):
        try:
            pages.add(int(m.group(1)))
        except ValueError:
            pass
    return pages

def extract_answer_terms(answer: str) -> set[str]:
    """Extract informative terms, numbers, and technical tokens from the answer."""
    # Clean answer: remove citation brackets like 【...】
    clean_ans = re.sub(r'【[^】]*】', ' ', answer)
    # Find numbers, alphanumeric codes like d_k, d_model, h=8, etc.
    tokens = re.findall(r'[a-zA-Z0-9_\-\.\=]+', clean_ans.lower())
    terms = set()
    for t in tokens:
        t_clean = t.strip(".-_")
        if not t_clean or len(t_clean) < 2:
            continue
        if t_clean in STOP_WORDS:
            continue
        terms.add(t_clean)
    return terms

def distance_to_similarity(dist: float) -> float:
    """Convert Chroma Euclidean distance to cosine similarity [0, 1]."""
    # d^2 = 2 - 2*sim => sim = 1 - d^2 / 2
    sim = 1.0 - (dist * dist) / 2.0
    return max(0.0, min(1.0, sim))

def select_citations(
    answer: str,
    retrieved_chunks: list[tuple[Any, float]],
    similarity_threshold: float = 0.20,
    margin: float = 0.25,
    max_citations: int = 4,
) -> list[dict]:
    """
    Select the minimum sufficient, highly relevant citations supporting the answer.
    """
    if is_negative_answer(answer):
        return []

    cited_pages = extract_cited_pages(answer)
    answer_terms = extract_answer_terms(answer)
    # Also extract numbers explicitly from answer
    answer_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', answer))

    scored_chunks = []
    for doc, dist in retrieved_chunks:
        sim = distance_to_similarity(dist)
        if sim < similarity_threshold:
            continue

        page = doc.metadata.get("page_number", 0)
        content_lower = doc.page_content.lower()

        # Check explicit page mention
        page_explicitly_cited = page in cited_pages if page else False

        # Term overlap
        overlap_count = 0
        for term in answer_terms:
            if term in content_lower:
                overlap_count += 1
        overlap_ratio = overlap_count / max(len(answer_terms), 1)

        # Number overlap (crucial for factual answers like 8, 64, 6)
        number_matches = sum(1 for num in answer_numbers if re.search(r'\b' + re.escape(num) + r'\b', content_lower))
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

    if not scored_chunks:
        return []

    # Sort by alignment_score descending
    scored_chunks.sort(key=lambda x: x["alignment_score"], reverse=True)

    top_alignment = scored_chunks[0]["alignment_score"]

    # Filter: must be within margin of the top candidate
    # OR explicitly cited by the LLM
    selected_candidates = []
    for sc in scored_chunks:
        is_close_enough = (top_alignment - sc["alignment_score"]) <= margin
        if sc["page_cited"] or is_close_enough:
            # If answer contains specific numbers, require that the chunk contains relevant info
            if answer_numbers and not sc["has_number_match"] and not sc["page_cited"] and sc["overlap_ratio"] < 0.3:
                continue
            selected_candidates.append(sc)

    # Deduplicate by (filename, page_number)
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
        results.append({
            "filename": fn,
            "page_number": int(p) if p and str(p).isdigit() and int(p) > 0 else None,
            "chunk_preview": text[:150] + ("…" if len(text) > 150 else ""),
            "document_id": doc.metadata.get("document_id", ""),
            "relevance_score": round(sc["sim"], 2),
        })

    return results

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    from services.deps import get_vector_store

    store = get_vector_store()

    baseline_data = [
        (
            "How many attention heads does the base Transformer use?",
            "The base Transformer model uses **8 parallel attention heads** ( h = 8 )【NIPS-2017-attention-is-all-you-need-Paper.pdf, Page 5】."
        ),
        (
            "What are dk and dv in the base Transformer?",
            "In the base Transformer model (the “base” row in Table 3), the dimensionalities of the keys and values are both 64:\n- **(d_k = 64)**\n- **(d_v = 64)**\nThese values are listed in Table 3 on page 9 of the paper【NIPS-2017-attention-is-all-you-need-Paper.pdf, Page 9】."
        ),
        (
            "What is the exact dollar cost of training the Transformer?",
            "I couldn't find this information in the uploaded documents."
        ),
        (
            "How many encoder and decoder layers does the Transformer have?",
            "The Transformer uses **six identical layers in the encoder stack and six identical layers in the decoder stack**【NIPS‑2017‑attention‑is‑all‑you‑need‑Paper.pdf, Page 2】【NIPS‑2017‑attention‑is‑all‑you‑need‑Paper.pdf, Page 3】."
        ),
        (
            "Explain the difference between encoder self-attention and decoder masked self-attention.",
            "Encoder self-attention attends freely across all positions, described on page 5. Decoder masked self-attention is masked to prevent positions from attending to subsequent positions, described on page 3."
        )
    ]

    for q, ans in baseline_data:
        print("\n" + "="*60)
        print("QUESTION:", q)
        print("ANSWER:", ans[:100] + ("..." if len(ans) > 100 else ""))
        retrieved = store.similarity_search(q, top_k=6)
        sources = select_citations(ans, retrieved)
        print(f"SELECTED CITATIONS ({len(sources)}):")
        for s in sources:
            print(f"  -> Page {s['page_number']} | score={s['relevance_score']} | file={s['filename']}")
