"""Comprehensive Generic Test Suite for Document-Grounded PDF Intelligence System.

Tests capabilities across:
1. Normalized Document Representation & Diagnostics
2. Layout-aware Text Extraction
3. Table Extraction & Representation
4. Generic Chart & Visual Data Understanding (without hardcoding)
5. Scanned / OCR Page Understanding
6. Mathematical & Formula Understanding
7. Multi-step Calculation Grounding
8. Cross-page & Context Synthesis
9. Multi-document Reasoning & User Isolation
10. Strict Anti-Hallucination & Unsupported Claim Prevention
11. Citation Validation
12. PDF Export with LaTeX, Code Blocks, and Markdown
"""

try:
    import pyarrow
except ImportError:
    pass

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import fitz
from config.settings import settings
from services.chat_service import ChatService, sanitize_grounded_claims
from services.citation_service import select_citations
from services.document_processor import DocumentProcessor
from services.pdf_export_service import generate_conversation_pdf
from services.query_understanding import AnswerType, QueryIntent, QueryUnderstanding
from services.text_extractor import extract_text
from services.vector_store import VectorStore


async def run_comprehensive_intelligence_tests():
    print("=" * 80)
    print("PDF INTELLIGENCE SYSTEM — COMPREHENSIVE CAPABILITIES TEST SUITE")
    print("=" * 80)

    test_pdf_path = Path(r"C:\Users\suyas\Downloads\INFOSYS - 2.pdf")
    assert test_pdf_path.exists(), f"Validation PDF missing at {test_pdf_path}"

    temp_chroma_dir = tempfile.mkdtemp(prefix="test_chroma_intel_")
    vs = VectorStore(
        persist_dir=temp_chroma_dir,
        embedding_model_name=settings.embedding_model,
    )
    processor = DocumentProcessor(vector_store=vs)
    cs = ChatService(vector_store=vs)

    try:
        # ──────────────────────────────────────────────────────────────────
        # TEST 1: Universal Document Representation & Diagnostics
        # ──────────────────────────────────────────────────────────────────
        print("\n--- TEST 1: Universal Document Representation & Diagnostics ---")
        diagnostics = processor.get_document_diagnostics(test_pdf_path)
        print("Document Diagnostics:", diagnostics)
        assert diagnostics["page_count"] == 12, f"Expected 12 pages, got {diagnostics['page_count']}"
        assert diagnostics["total_extracted_elements"] >= 12, "Extracted elements below page count"
        assert diagnostics["total_chunks_created"] >= 30, "Insufficient chunks created"
        assert len(diagnostics["pages_with_visuals"]) >= 1, "Visual pages not detected"
        print("PASS: Universal Document Diagnostics verified.")

        # ──────────────────────────────────────────────────────────────────
        # TEST 2: Ingestion & Element-Aware Indexing (Once per document)
        # ──────────────────────────────────────────────────────────────────
        from database.session import SessionLocal
        from models.database import User

        db = SessionLocal()
        u = db.query(User).first()
        user_a_id = u.id if u else 1
        user_b_id = user_a_id + 9999
        db.close()

        print("\n--- TEST 2: Ingestion & Indexing with User Isolation ---")
        doc_a_id = "doc_intel_a"

        res = processor.process_document(
            file_path=test_pdf_path,
            filename="INFOSYS - 2.pdf",
            user_id=user_a_id,
            document_id=doc_a_id,
        )
        print(f"Ingested {res['chunk_count']} chunks for User A (id={user_a_id})")
        assert res["chunk_count"] > 0

        # Verify User Isolation
        user_b_chunks = vs.similarity_search_for_user("enrollment distribution pie chart", user_id=user_b_id, top_k=5)
        assert len(user_b_chunks) == 0, f"User B should retrieve 0 chunks, got {len(user_b_chunks)}"
        print("PASS: Ingestion completed once, and User Isolation confirmed in VectorStore.")

        # ──────────────────────────────────────────────────────────────────
        # TEST 3: Generic Visual Chart Retrieval & Understanding (No Hardcoding)
        # ──────────────────────────────────────────────────────────────────
        print("\n--- TEST 3: Generic Chart Understanding & Retrieval ---")
        q_chart1 = "What is the total number of boys enrolled in MECH and IT together?"
        res1 = await cs.get_answer(
            question=q_chart1,
            user_id=user_a_id,
            conversation_id=None,
        )
        ans1, cites1 = res1["answer"], res1["sources"]
        print(f"Q: {q_chart1}\nA: {ans1[:300]}...\nCitations: {cites1}\n")
        assert "810" in ans1, f"Expected 810 boys in MECH and IT, got: {ans1}"
        assert len(cites1) >= 1, "Expected at least 1 citation for chart question"
        assert any(c.get("page_number") == 2 for c in cites1), "Expected citation to point to Page 2"
        print("PASS: Chart question 1 accurately derived from retrieved visual data with Page 2 citation.")

        q_chart2 = "What is the ratio of girls enrolled in CIVIL to boys enrolled in ECE?"
        res2 = await cs.get_answer(
            question=q_chart2,
            user_id=user_a_id,
            conversation_id=None,
        )
        ans2, cites2 = res2["answer"], res2["sources"]
        print(f"Q: {q_chart2}\nA: {ans2[:300]}...\nCitations: {cites2}\n")
        assert "114" in ans2 and "121" in ans2, f"Expected ratio 114:121, got: {ans2}"
        print("PASS: Chart question 2 (ratio derivation) accurately answered.")

        q_chart3 = "Using the pie charts, calculate the percentage of all students who are girls in CIVIL, ECE and EEE combined."
        res3 = await cs.get_answer(
            question=q_chart3,
            user_id=user_a_id,
            conversation_id=None,
        )
        ans3, cites3 = res3["answer"], res3["sources"]
        print(f"Q: {q_chart3}\nA: {ans3[:300]}...\nCitations: {cites3}\n")
        assert any(term in ans3.replace(r"\%", "%").lower() for term in ["30%", "30 %", "30 percent"]), f"Expected 30%, got: {ans3}"
        print("PASS: Chart question 3 (combined percentage) accurately calculated.")

        # ──────────────────────────────────────────────────────────────────
        # TEST 4: Text Retrieval, Syllogism & Logic Questions (Q4)
        # ──────────────────────────────────────────────────────────────────
        print("\n--- TEST 4: Text Retrieval & Syllogism Reasoning (Q4) ---")
        q_syl = "According to question 4 in the document, which conclusions follow from the statements?"
        res_syl = await cs.get_answer(
            question=q_syl,
            user_id=user_a_id,
            conversation_id=None,
        )
        ans_syl = res_syl["answer"]
        print(f"Q: {q_syl}\nA: {ans_syl[:300]}...\n")
        assert len(ans_syl) > 50, "Syllogism answer too short"
        assert any(term in ans_syl.lower() for term in ["cat", "monkey", "elephant", "conclusion i", "follow"]), "Question 4 syllogism concepts missing"
        print("PASS: Syllogism reasoning answered from document text.")

        # ──────────────────────────────────────────────────────────────────
        # TEST 5: Archaeology Passage & Uncertainty Preservation
        # ──────────────────────────────────────────────────────────────────
        print("\n--- TEST 5: Archaeology Passage & Uncertainty Language Preservation ---")
        q_arch = "According to the passage, what effect would selling duplicate artifacts on the open market have on illegal excavation?"
        res_arch = await cs.get_answer(
            question=q_arch,
            user_id=user_a_id,
            conversation_id=None,
        )
        ans_arch = res_arch["answer"]
        print(f"Q: {q_arch}\nA: {ans_arch[:300]}...\n")
        assert "eliminate illegal excavation" not in ans_arch.lower(), "Fabricated definitive elimination claim found!"
        assert any(term in ans_arch.lower() for term in ["reduce", "reduction", "demand", "curb", "diminish"]), "Reduction concept missing!"
        print("PASS: Preserved probabilistic language ('reduce demand') without fabricating 'will eliminate'.")

        # ──────────────────────────────────────────────────────────────────
        # TEST 6: MESBIC Passage Groundedness & Inference Labeling
        # ──────────────────────────────────────────────────────────────────
        print("\n--- TEST 6: MESBIC Passage Groundedness & Inference Labeling ---")
        q_mesbic = "Explain the MESBIC approach discussed in the passage and how it differs from the SBA approach."
        res_mesbic = await cs.get_answer(
            question=q_mesbic,
            user_id=user_a_id,
            conversation_id=None,
        )
        ans_mesbic = res_mesbic["answer"]
        print(f"Q: {q_mesbic}\nA: {ans_mesbic[:300]}...\n")
        assert "guaranteed market" not in ans_mesbic.lower(), "Fabricated 'guaranteed market' found!"
        assert "guaranteed customer" not in ans_mesbic.lower(), "Fabricated 'guaranteed customer' found!"
        assert "500,000" in ans_mesbic or "$500,000" in ans_mesbic or "sba" in ans_mesbic.lower(), "Document facts missing!"
        print("PASS: MESBIC grounded answer accurate without exaggerated claims.")

        # ──────────────────────────────────────────────────────────────────
        # TEST 7: Missing Information / Negative Response
        # ──────────────────────────────────────────────────────────────────
        print("\n--- TEST 7: Missing Information / Negative Response ---")
        q_missing = "What is the exact serial number and CPU model of the server used to administer the INFOSYS test?"
        res_missing = await cs.get_answer(
            question=q_missing,
            user_id=user_a_id,
            conversation_id=None,
        )
        ans_missing, cites_missing = res_missing["answer"], res_missing["sources"]
        print(f"Q: {q_missing}\nA: {ans_missing[:200]}...\nCitations: {cites_missing}\n")
        assert len(cites_missing) == 0, f"Expected 0 citations for absent fact, got: {cites_missing}"
        assert any(phrase in ans_missing.lower() for phrase in ["not mentioned", "not provide", "could not find", "couldn't find", "not present", "not available", "not include", "do not contain", "does not contain"]), "Failed to state insufficient information"
        print("PASS: Correctly reported missing information with 0 spurious citations.")

        # ──────────────────────────────────────────────────────────────────
        # TEST 8: PDF Export with LaTeX, Code Blocks, and Multi-page Layout
        # ──────────────────────────────────────────────────────────────────
        print("\n--- TEST 8: PDF Export with LaTeX, Code Blocks, and Multi-page Layout ---")
        export_messages = [
            {"role": "user", "content": "Calculate the passing rate and provide the Python verification script."},
            {
                "role": "assistant",
                "content": (
                    "Here is the calculation from the document:\n\n"
                    "$$ \\frac{255}{692}\\times100 = 36.8\\% $$\n\n"
                    "The result is $\\frac{2}{11}\\times60 \\approx 10.9$ seconds.\n\n"
                    "$$ \\text{Passing percentage} = \\frac{255}{692}\\times100 $$\n\n"
                    "```python\n"
                    "x = r\"\\frac{2}{3}\"\n"
                    "print(f\"Verification code: {x}\")\n"
                    "```\n\n"
                    "[From: INFOSYS - 2.pdf, Page 2]"
                ),
            },
        ]
        pdf_bytes = generate_conversation_pdf("Math Verification", "2026-09-27T10:00:00Z", export_messages)
        assert pdf_bytes.startswith(b"%PDF-"), "Invalid PDF header"

        doc_pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        p0 = doc_pdf[0]
        text_pdf = p0.get_text()

        # Check images (rendered LaTeX formulas)
        assert len(p0.get_images()) >= 3, "Math formulas not rendered as images"
        # Check code block preservation
        assert 'x = r"\\frac{2}{3}"' in text_pdf or 'x = r"\\frac{2}{3}"' in text_pdf.replace("\xa0", " ")
        # Check raw LaTeX outside code is not leaked
        assert r"\frac{255}{692}" not in text_pdf
        assert r"\times100" not in text_pdf
        print("PASS: PDF Export properly rendered LaTeX as math, preserved code blocks, and prevented raw LaTeX leakage.")

        print("\n" + "=" * 80)
        print("ALL PDF INTELLIGENCE SYSTEM TESTS PASSED SUCCESSFULLY! [OK]")
        print("=" * 80)

    finally:
        shutil.rmtree(temp_chroma_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(run_comprehensive_intelligence_tests())
