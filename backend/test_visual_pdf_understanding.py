"""Comprehensive test suite for PDF visual, chart, diagram, and scanned content understanding.

Tests:
1. INFOSYS aptitude pie chart Q1: Total boys enrolled in MECH and IT together (810)
2. INFOSYS aptitude pie chart Q2: Ratio of girls in CIVIL to boys in ECE (114:121 or 570:605)
3. INFOSYS aptitude pie chart Q3: 20% of ECE girls move to MECH -> new total MECH students (593)
4. INFOSYS aptitude pie chart Q4: Percentage of all students who are girls in CIVIL, ECE, EEE combined (30%)
5. Page-level visual detection & metadata preservation (page_number=2, content_type='visual')
6. Strict user isolation (User A vs User B)
7. Page citation accuracy ([Source: INFOSYS - 2.pdf, Page 2])
8. Normal text PDF (no images)
9. Scanned PDF page OCR
10. PDF with tabular data
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_visual_pdf")

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

from config.settings import settings
from services.chat_service import ChatService
from services.citation_service import select_citations
from services.document_processor import DocumentProcessor
from services.text_extractor import extract_text
from services.vector_store import VectorStore
from services.visual_extractor import visual_extractor


def create_scanned_pdf(file_path: Path, text: str = "This is a scanned invoice receipt document."):
    """Helper: Create a synthetic scanned PDF (text drawn onto image bitmap, no font stream)."""
    img = Image.new("RGB", (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((40, 50), text, fill=(0, 0, 0))
    draw.text((40, 100), "Invoice ID: INV-99824", fill=(0, 0, 0))
    draw.text((40, 150), "Total Amount: $4,500.00", fill=(0, 0, 0))

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    doc = fitz.open()
    page = doc.new_page(width=600, height=400)
    rect = fitz.Rect(0, 0, 600, 400)
    page.insert_image(rect, stream=img_bytes)
    doc.save(str(file_path))
    doc.close()


def create_normal_text_pdf(file_path: Path):
    """Helper: Create a pure text PDF with zero images."""
    doc = fitz.open()
    page = doc.new_page(width=600, height=400)
    page.insert_text(
        (50, 50),
        "Company Policy Document\n\nAll full-time employees are entitled to 25 days of annual paid leave.\n"
        "Health insurance coverage begins on the first day of employment.\n"
        "Remote work is permitted up to two days per week with manager approval.",
    )
    doc.save(str(file_path))
    doc.close()


def create_table_pdf(file_path: Path):
    """Helper: Create a PDF containing structured tabular text."""
    doc = fitz.open()
    page = doc.new_page(width=600, height=400)
    page.insert_text(
        (50, 50),
        "Quarterly Financial Summary 2024\n\n"
        "Quarter | Revenue ($M) | Expenses ($M) | Net Profit ($M)\n"
        "Q1      | 120          | 85            | 35\n"
        "Q2      | 145          | 92            | 53\n"
        "Q3      | 160          | 98            | 62\n"
        "Q4      | 190          | 110           | 80\n",
    )
    doc.save(str(file_path))
    doc.close()


def run_tests():
    test_chroma_dir = Path("test_chroma_visual_tmp")
    if test_chroma_dir.exists():
        shutil.rmtree(test_chroma_dir)

    vector_store = VectorStore(
        persist_dir=str(test_chroma_dir),
        embedding_model_name=settings.embedding_model,
    )
    processor = DocumentProcessor(vector_store)
    chat_service = ChatService(vector_store)

    infosys_pdf_path = Path(r"C:\Users\suyas\Downloads\INFOSYS - 2.pdf")
    if not infosys_pdf_path.exists():
        raise FileNotFoundError(f"Test PDF not found at: {infosys_pdf_path}")

    user_a = 101
    user_b = 999

    print("\n" + "=" * 70)
    print("STEP 1: INGEST INFOSYS APTITUDE PDF WITH VISUAL CHARTS")
    print("=" * 70)

    res_a = processor.process_document(
        file_path=infosys_pdf_path,
        filename="INFOSYS - 2.pdf",
        user_id=user_a,
        document_id="doc_infosys_101",
    )
    print(f"Processed INFOSYS - 2.pdf: {res_a['chunk_count']} chunks created.")

    # Verify visual chunk exists in vector store
    visual_chunks_a = vector_store.vectorstore._collection.get(
        where={"$and": [{"user_id": str(user_a)}, {"content_type": "visual"}]}
    )
    assert visual_chunks_a and len(visual_chunks_a["ids"]) > 0, "Visual chunk was not indexed!"
    print(f"PASS: Found {len(visual_chunks_a['ids'])} visual chunk(s) for user {user_a} in Chroma.")

    meta = visual_chunks_a["metadatas"][0]
    assert meta.get("page_number") == 2, f"Expected page_number 2, got {meta.get('page_number')}"
    assert meta.get("content_type") == "visual", f"Expected content_type visual, got {meta.get('content_type')}"
    print(f"PASS: Visual chunk metadata verified: page_number={meta['page_number']}, content_type={meta['content_type']}")

    print("\n" + "=" * 70)
    print("STEP 2: TEST STRICT USER ISOLATION")
    print("=" * 70)

    # User B searches for chart data
    query = "What is the total number of boys enrolled in MECH and IT together?"
    res_user_b = vector_store.similarity_search_for_user(query, user_id=user_b, top_k=5)
    assert len(res_user_b) == 0, f"USER ISOLATION BREACH! User B retrieved User A's chunks: {res_user_b}"
    print("PASS: User B cannot retrieve User A's visual chart knowledge.")

    # User A searches for chart data
    res_user_a = vector_store.similarity_search_for_user(query, user_id=user_a, top_k=3)
    assert len(res_user_a) > 0, "User A failed to retrieve their own visual chunk."
    top_doc = res_user_a[0][0]
    print(f"PASS: User A retrieved chunk from page {top_doc.metadata.get('page_number')}, type={top_doc.metadata.get('content_type')}")
    assert top_doc.metadata.get("page_number") == 2

    print("\n" + "=" * 70)
    print("STEP 3: TEST 4 INFOSYS APTITUDE CHART QUESTIONS")
    print("=" * 70)

    test_queries = [
        {
            "id": 1,
            "q": "What is the total number of boys enrolled in MECH and IT together?",
            "expected_num": "810",
            "desc": "Boys in MECH (380) + IT (430) = 810",
        },
        {
            "id": 2,
            "q": "What is the ratio of girls enrolled in CIVIL to boys enrolled in ECE?",
            "expected_num": "114",  # 114:121 or 570:605
            "expected_num_alt": "570",
            "desc": "Ratio 570 / 605 = 114:121",
        },
        {
            "id": 3,
            "q": "If 20% of the girls enrolled in ECE move to MECH, what is the new total number of MECH students?",
            "expected_num": "593",
            "desc": "560 + 20% of 165 = 560 + 33 = 593",
        },
        {
            "id": 4,
            "q": "Using the pie charts, calculate the percentage of all students who are girls in CIVIL, ECE and EEE combined.",
            "expected_num": "30",
            "desc": "(570 + 165 + 315) / 3500 = 1050 / 3500 = 30%",
        },
    ]

    for item in test_queries:
        print(f"\n--- Testing Question {item['id']}: '{item['q']}' ---")
        context, candidates, analysis = chat_service._retrieve_context(
            question=item["q"],
            user_id=user_a,
        )
        assert len(candidates) > 0, f"No candidates retrieved for Q{item['id']}"
        assert "VISUAL CONTENT" in context or "pie chart" in context.lower(), f"Visual context missing for Q{item['id']}"

        # Generate answer with ChatGroq
        llm = chat_service._get_llm()
        messages = chat_service._build_messages(
            question=item["q"],
            context=context,
            history=[],
            analysis=analysis,
        )
        response = llm.invoke(messages)
        answer_text = response.content.strip()

        print(f"Answer:\n{answer_text}\n")

        # Verify calculations
        has_num = item["expected_num"] in answer_text
        if not has_num and "expected_num_alt" in item:
            has_num = item["expected_num_alt"] in answer_text
        assert has_num, f"Calculation failed for Q{item['id']}! Expected {item['expected_num']} in answer."
        assert "chart information is missing" not in answer_text.lower(), f"False negative for Q{item['id']}!"
        print(f"PASS: Correct numerical value found for Q{item['id']}.")

        # Verify page-level citation
        citations = select_citations(item["q"], answer_text, candidates)
        print(f"Citations: {citations}")
        assert len(citations) > 0, f"No citation generated for Q{item['id']}!"
        assert any(c["page_number"] == 2 for c in citations), f"Citation did not point to Page 2! Got: {citations}"
        assert any(c["filename"] == "INFOSYS - 2.pdf" for c in citations), f"Citation filename incorrect: {citations}"
        print(f"PASS: Page citation verified -> [Source: INFOSYS - 2.pdf, Page 2]")

    print("\n" + "=" * 70)
    print("STEP 4: TEST SCANNED PDF EXTRACTION (OCR)")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as scan_tmp:
        scan_pdf_path = Path(scan_tmp.name)

    try:
        create_scanned_pdf(scan_pdf_path)
        scan_docs = extract_text(scan_pdf_path)
        assert len(scan_docs) > 0, "Failed to extract text from scanned PDF!"
        combined_scan_text = " ".join(d.page_content for d in scan_docs)
        print(f"Scanned PDF extracted text: {combined_scan_text[:150]}...")
        assert "Invoice" in combined_scan_text or "4,500" in combined_scan_text or "INV" in combined_scan_text
        print("PASS: Scanned PDF page correctly OCRed and indexed.")
    finally:
        if scan_pdf_path.exists():
            scan_pdf_path.unlink()

    print("\n" + "=" * 70)
    print("STEP 5: TEST NORMAL TEXT PDF (NO IMAGES)")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as text_tmp:
        text_pdf_path = Path(text_tmp.name)

    try:
        create_normal_text_pdf(text_pdf_path)
        text_docs = extract_text(text_pdf_path)
        assert len(text_docs) == 1, f"Expected 1 text document, got {len(text_docs)}"
        assert text_docs[0].metadata.get("content_type") == "text"
        assert "annual paid leave" in text_docs[0].page_content
        print("PASS: Normal text PDF loaded cleanly without spurious visual chunks.")
    finally:
        if text_pdf_path.exists():
            text_pdf_path.unlink()

    print("\n" + "=" * 70)
    print("STEP 6: TEST TABULAR PDF")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tbl_tmp:
        tbl_pdf_path = Path(tbl_tmp.name)

    try:
        create_table_pdf(tbl_pdf_path)
        tbl_docs = extract_text(tbl_pdf_path)
        assert len(tbl_docs) >= 1, "Failed to extract table PDF"
        assert "Net Profit" in tbl_docs[0].page_content
        print("PASS: Tabular PDF loaded cleanly.")
    finally:
        if tbl_pdf_path.exists():
            tbl_pdf_path.unlink()

    # Cleanup temp chroma
    if test_chroma_dir.exists():
        try:
            shutil.rmtree(test_chroma_dir, ignore_errors=True)
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
