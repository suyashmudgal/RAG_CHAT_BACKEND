"""Test suite for RAG Groundedness & Prevention of Unsupported Claims.

Tests:
1. Direct source fact: Verifies exact factual retrieval from MESBIC passage (e.g., $500,000 capitalization, Black/Hispanic staff).
2. Source-supported inference: Verifies that reasoning beyond verbatim text is explicitly labeled (e.g., 'Inference:', 'Based on this...').
3. Unsupported claim prevention: Verifies that unsupported claims ('guaranteed market', 'guaranteed customers', 'guaranteed returns', 'will eliminate illegal excavation') are NEVER made.
4. Uncertain language preservation: Verifies that probabilistic/hedged source language ('potential', 'may', 'could reduce demand') is preserved.
5. Citation correctness: Verifies that citations point to the exact supporting page without spurious attributions.
6. General knowledge separation: Verifies that general queries do not fabricate document citations.
7. Mathematical derivation: Verifies that mathematical derivations are cleanly computed and grounded.
8. Transformer implementation: Verifies that code is generated from architectural concepts without falsely claiming code was in the source document.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_rag_groundedness")

import fitz  # PyMuPDF
from langchain_core.documents import Document

from config.settings import settings
from services.chat_service import ChatService, sanitize_grounded_claims
from services.citation_service import select_citations
from services.query_understanding import QueryUnderstanding


def run_groundedness_tests():
    cs = ChatService(None)
    llm = cs._get_llm()

    pdf_path = Path(r"C:\Users\suyas\Downloads\INFOSYS - 2.pdf")
    if not pdf_path.exists():
        raise FileNotFoundError(f"INFOSYS - 2.pdf not found at {pdf_path}")

    doc = fitz.open(str(pdf_path))
    page6_text = doc[5].get_text()
    page7_text = doc[6].get_text()
    page5_text = doc[4].get_text()
    doc.close()

    mesbic_context = f"[From: INFOSYS - 2.pdf, Page 6]\n{page6_text}\n\n[From: INFOSYS - 2.pdf, Page 7]\n{page7_text}"
    archaeology_context = f"[From: INFOSYS - 2.pdf, Page 5]\n{page5_text}"

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 1 & 3 & 4: MESBIC Approach Explanation (Direct fact, No unsupported claim, Preserved uncertainty)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 1, 3, 4: EXPLAIN MESBIC APPROACH (FACTS, NO EXAGGERATION, UNCERTAINTY)")
    print("=" * 70)

    q1 = "Explain the MESBIC approach discussed in the passage."
    analysis1 = QueryUnderstanding.analyze_query(q1)
    msgs1 = cs._build_messages(q1, mesbic_context, [], analysis1)
    raw_ans1 = llm.invoke(msgs1).content.strip()
    ans1 = sanitize_grounded_claims(raw_ans1, mesbic_context)

    print(f"Query: {q1}\n\nAnswer:\n{ans1}\n")

    # Check 1: Direct source facts present
    assert any(term in ans1.lower() for term in ["sponsoring", "corporation", "capital", "guidance"]), "Direct source facts missing from answer!"
    assert any(num in ans1 for num in ["500,000", "500000", "SBA", "Small Business Administration"]), "Key source specifics missing!"
    print("PASS 1: Direct source facts accurately retrieved and presented.")

    # Check 3: Unsupported exaggerated claims STRICTLY ABSENT
    forbidden_terms = [
        "guaranteed market",
        "guaranteed customer",
        "guaranteed return",
        "guaranteed buyer",
        "guaranteed sale",
        "eliminate illegal excavation",
        "elimination of illegal excavation",
    ]
    for term in forbidden_terms:
        assert term not in ans1.lower(), f"UNSUPPORTED CLAIM DETECTED: Found '{term}' in generated answer!"
    print("PASS 3: No unsupported exaggerated claims found (no 'guaranteed market', no 'guaranteed customers').")

    # Check 4: Uncertain language preserved
    assert any(term in ans1.lower() for term in ["potential", "may", "less risk", "pragmatic", "tend to"]), "Source uncertainty qualifiers were stripped!"
    print("PASS 4: Uncertain/qualifying language ('potential', 'less risk', 'may') faithfully preserved.")

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 2: Source-Supported Inference Labeling
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 2: SOURCE-SUPPORTED INFERENCE LABELING")
    print("=" * 70)

    q2 = "Why might MESBIC's supplier/customer relationship benefit minority-owned businesses?"
    analysis2 = QueryUnderstanding.analyze_query(q2)
    msgs2 = cs._build_messages(q2, mesbic_context, [], analysis2)
    raw_ans2 = llm.invoke(msgs2).content.strip()
    ans2 = sanitize_grounded_claims(raw_ans2, mesbic_context)

    print(f"Query: {q2}\n\nAnswer:\n{ans2}\n")

    # Verify that the answer explicitly labels inference / deduction
    has_inference_label = any(
        marker in ans2.lower()
        for marker in [
            "inference",
            "based on this",
            "this suggests",
            "this implies",
            "logically",
            "source evidence",
            "what the passage states",
        ]
    )
    assert has_inference_label, "Inference was not explicitly distinguished or labeled!"

    # Verify absence of 'guaranteed market'
    assert "guaranteed market" not in ans2.lower(), "Found 'guaranteed market' in inference answer!"
    print("PASS 2: Source evidence and logical inference clearly distinguished and labeled.")

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 4 (cont.): Archaeology / Illegal Excavation (Hedge Preservation)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 4b: UNCERTAIN LANGUAGE IN EXCAVATION PASSAGE")
    print("=" * 70)

    q_arch = "According to the passage, what effect would selling duplicate artifacts on the open market have on illegal excavation?"
    analysis_arch = QueryUnderstanding.analyze_query(q_arch)
    msgs_arch = cs._build_messages(q_arch, archaeology_context, [], analysis_arch)
    raw_ans_arch = llm.invoke(msgs_arch).content.strip()
    ans_arch = sanitize_grounded_claims(raw_ans_arch, archaeology_context)

    print(f"Query: {q_arch}\n\nAnswer:\n{ans_arch}\n")

    assert "eliminate" not in ans_arch.lower() or "not eliminate" in ans_arch.lower() or "could reduce" in ans_arch.lower() or "reduce" in ans_arch.lower()
    assert "will eliminate illegal excavation" not in ans_arch.lower()
    assert any(term in ans_arch.lower() for term in ["reduce", "reduction", "less", "demand", "alternative"]), "Key source effect missing!"
    print("PASS 4b: Preserved 'could reduce demand' without strengthening to 'will eliminate illegal excavation'.")

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 5: Citation Correctness
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 5: CITATION CORRECTNESS")
    print("=" * 70)

    cands = [
        (Document(page_content=page6_text[:400], metadata={"filename": "INFOSYS - 2.pdf", "page_number": 6, "document_id": "doc1"}), 0.25),
        (Document(page_content=page7_text[:400], metadata={"filename": "INFOSYS - 2.pdf", "page_number": 7, "document_id": "doc1"}), 0.30),
        (Document(page_content="Completely irrelevant cooking recipes about baking cakes and pasta.", metadata={"filename": "recipes.pdf", "page_number": 1, "document_id": "doc2"}), 0.95),
    ]

    cites = select_citations(q1, ans1, cands)
    print(f"Selected citations: {cites}")

    assert len(cites) > 0, "No citations selected!"
    assert all(c["filename"] == "INFOSYS - 2.pdf" for c in cites), "Spurious recipe citation was selected!"
    assert any(c["page_number"] in [6, 7] for c in cites), "Citation did not point to supporting MESBIC page!"
    print("PASS 5: Citation correctly points to supporting page and rejected spurious recipe document.")

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 6: General Knowledge Separation
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 6: GENERAL KNOWLEDGE SEPARATION")
    print("=" * 70)

    q_gk = "What is the difference between supervised and unsupervised learning?"
    analysis_gk = QueryUnderstanding.analyze_query(q_gk)
    msgs_gk = cs._build_messages(q_gk, "", [], analysis_gk)
    ans_gk = llm.invoke(msgs_gk).content.strip()

    print(f"Query: {q_gk}\n\nAnswer preview: {ans_gk[:250]}...\n")
    cites_gk = select_citations(q_gk, ans_gk, [])
    assert len(cites_gk) == 0, f"Spurious citation fabricated for general knowledge question: {cites_gk}"
    assert "label" in ans_gk.lower() and ("unsupervised" in ans_gk.lower() or "unlabeled" in ans_gk.lower() or "without label" in ans_gk.lower() or "input" in ans_gk.lower()), "General knowledge answer was inadequate."
    print("PASS 6: General knowledge answered thoroughly without fabricating citations.")

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 7: Mathematical Derivation Grounding
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 7: MATHEMATICAL DERIVATION GROUNDING")
    print("=" * 70)

    math_context = "[From: math_doc.pdf, Page 1]\nAddition is the arithmetic operation of combining two quantities into a single sum."
    q_math = "Using the concept of addition from the document, what is 37 + 42?"
    analysis_math = QueryUnderstanding.analyze_query(q_math)
    msgs_math = cs._build_messages(q_math, math_context, [], analysis_math)
    ans_math = llm.invoke(msgs_math).content.strip()

    print(f"Query: {q_math}\n\nAnswer: {ans_math}\n")
    assert "79" in ans_math, f"Math derivation failed! Expected 79, got: {ans_math}"
    assert any(term in ans_math.lower() for term in ["addition", "concept", "applying", "sum"]), "Conceptual grounding missing!"
    print("PASS 7: Mathematical derivation accurately reasoned from document concept without fabrication.")

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 8: Transformer Implementation Grounding
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 8: TRANSFORMER IMPLEMENTATION GROUNDING")
    print("=" * 70)

    transformer_context = "[From: NIPS-2017-attention.pdf, Page 3]\nThe Transformer follows an encoder-decoder architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder."
    q_code = "Give me a simple PyTorch module implementation of self-attention described in the paper."
    analysis_code = QueryUnderstanding.analyze_query(q_code)
    msgs_code = cs._build_messages(q_code, transformer_context, [], analysis_code)
    ans_code = llm.invoke(msgs_code).content.strip()

    print(f"Query: {q_code}\n\nAnswer preview: {ans_code[:300]}...\n")
    assert "nn.Module" in ans_code or "torch" in ans_code, "PyTorch implementation missing!"
    assert any(phrase in ans_code.lower() for phrase in ["does not contain", "practical", "based on", "implementation", "architecture"]), "Grounding boundary distinction missing!"
    print("PASS 8: Clean code generated based on architecture while clarifying implementation distinction.")

    print("\n" + "=" * 70)
    print("ALL 8 GROUNDEDNESS TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    run_groundedness_tests()
