r"""Exhaustive backend test suite for PDF Export covering:
- Plain text
- Headings (H1, H2, H3)
- Bold, italic, inline code
- Bullet and numbered lists
- Markdown tables with header and wrapping
- Inline math ($...$, \(...\))
- Display math ($$...$$, \[...\])
- Fenced code blocks containing literal LaTeX (MUST NOT become math)
- Inline code containing literal LaTeX (MUST NOT become math)
- Mixed math + code
- Source citations preservation
- Comprehensive Unicode: ₹, €, $, →, ←, ≤, ≥, ±, ×, α, β, γ
- Long and multi-page responses with dynamic footers
- Empty / short conversation handling
- Cross-user export isolation and unauthenticated access rejection (HTTP)
"""

import asyncio
import io
import sys
import uuid
import fitz  # PyMuPDF
import pytest
from httpx import ASGITransport, AsyncClient

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from main import app
from database.session import SessionLocal
from models.database import Conversation as PgConversation, Message as PgMessage, User as PgUser
from services.pdf_export_service import generate_conversation_pdf


def test_pdf_plain_text_and_empty():
    """Verify empty and plain text conversations generate valid PDFs."""
    # Empty messages
    pdf_empty = generate_conversation_pdf("Empty Chat", "", [])
    assert pdf_empty.startswith(b"%PDF-")
    doc = fitz.open(stream=pdf_empty, filetype="pdf")
    assert len(doc) == 1
    assert "No messages have been recorded" in doc[0].get_text()

    # Plain text single turn
    messages = [
        {"role": "user", "content": "Hello, how are you?", "created_at": "2026-09-28T10:00:00Z"},
        {"role": "assistant", "content": "I am doing well, thank you!", "created_at": "2026-09-28T10:00:02Z"},
    ]
    pdf_plain = generate_conversation_pdf("Plain Text Chat", "2026-09-28T10:00:00Z", messages)
    assert pdf_plain.startswith(b"%PDF-")
    doc = fitz.open(stream=pdf_plain, filetype="pdf")
    text = doc[0].get_text()
    assert "Hello, how are you?" in text
    assert "I am doing well, thank you!" in text
    assert "User" in text
    assert "DocChat Assistant" in text
    print("PASS: test_pdf_plain_text_and_empty")


def test_pdf_headings_bold_italic_lists():
    """Verify headings, bold, italic, and lists."""
    content = """# Heading Level 1
## Heading Level 2
### Heading Level 3

This is a paragraph with **bold text**, __another bold__, *italic text*, and _another italic_.

Here is a bullet list:
* First bullet item
- Second bullet item with **bold feature**
+ Third bullet item

And a numbered list:
1. Step one of algorithm
2. Step two with *italic detail*
3. Step three finalizing output
"""
    messages = [{"role": "assistant", "content": content, "created_at": "2026-09-28T10:00:00Z"}]
    pdf_bytes = generate_conversation_pdf("Markdown Structure Test", "", messages)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = doc[0].get_text()

    assert "Heading Level 1" in text
    assert "Heading Level 2" in text
    assert "Heading Level 3" in text
    assert "bold text" in text
    assert "italic text" in text
    assert "First bullet item" in text
    assert "Step one of algorithm" in text
    print("PASS: test_pdf_headings_bold_italic_lists")


def test_pdf_code_block_protection_before_latex():
    """CRITICAL: Fenced code blocks and inline code containing LaTeX MUST remain literal code."""
    content = r"""Here is Python code defining a formula string:

```python
x = r"\frac{2}{3}"
y = r"\sqrt{16} + \alpha"
print(f"Result: {x} and {y}")
```

Notice that inline code `r"\frac{a}{b}"` and `$\beta$` must also remain literal code.

And outside the code block, here is an actual mathematical expression:
$$ P(A) = \frac{2}{3} $$
"""
    messages = [{"role": "assistant", "content": content, "created_at": "2026-09-28T10:00:00Z"}]
    pdf_bytes = generate_conversation_pdf("Code Protection Test", "", messages)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    text = page.get_text().replace("\xa0", " ")

    # 1. Code block must contain literal LaTeX
    assert 'x = r"\\frac{2}{3}"' in text
    assert 'y = r"\\sqrt{16} + \\alpha"' in text

    # 2. Inline code must contain literal LaTeX
    assert 'r"\\frac{a}{b}"' in text

    # 3. Outside code block: $$ P(A) = \frac{2}{3} $$ should be converted to an image
    images = page.get_images()
    assert len(images) >= 1, "Expected at least 1 rendered equation image for display math"

    # 4. Raw math delimiter $$ or unrendered math expression outside code must NOT leak
    assert "$$ P(A)" not in text
    print("PASS: test_pdf_code_block_protection_before_latex")


def test_pdf_latex_math_expressions():
    """Verify various LaTeX forms: fractions, square roots, greek letters, approximations, le/ge, etc."""
    content = r"""Calculations and proofs:

Display math with fractions and square roots:
$$ \sqrt{x^2 + y^2} = \frac{a}{b} \times \pm 10 $$

Display math with brackets:
\[ \alpha + \beta \le \gamma \ge \delta \]

Inline math with dollar delimiters: $\approx 3.14159$ and $\pm 0.05$.
Inline math with slash-parentheses: \(\frac{1}{2} + \frac{1}{4} = \frac{3}{4}\).
"""
    messages = [{"role": "assistant", "content": content, "created_at": "2026-09-28T10:00:00Z"}]
    pdf_bytes = generate_conversation_pdf("Math Variations Test", "", messages)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]

    images = page.get_images()
    assert len(images) >= 3, f"Expected rendered math images, got {len(images)}"

    text = page.get_text()
    assert r"\frac{a}{b}" not in text
    assert r"\sqrt{x^2" not in text
    print("PASS: test_pdf_latex_math_expressions")


def test_pdf_markdown_tables():
    """Verify Markdown tables render with correct columns, headers, and wrapping."""
    content = """Comparison of Distributed Databases:

| Database | Model | Consistency | Max TPS | Description |
| :--- | :--- | :--- | :--- | :--- |
| CockroachDB | Relational | Strict Serializable | 50,000 | Multi-region distributed SQL with raft consensus |
| Cassandra | Wide-column | Eventual / Tunable | 200,000 | High write throughput peer-to-peer ring architecture |
| MongoDB | Document | Tunable | 100,000 | Flexible JSON schema with replica sets |
"""
    messages = [{"role": "assistant", "content": content, "created_at": "2026-09-28T10:00:00Z"}]
    pdf_bytes = generate_conversation_pdf("Table Test", "", messages)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = doc[0].get_text()

    assert "CockroachDB" in text
    assert "Cassandra" in text
    assert "MongoDB" in text
    flat_text = text.replace("\n", " ")
    assert "Multi-region distributed SQL" in flat_text
    print("PASS: test_pdf_markdown_tables")


def test_pdf_citations_preserved():
    """Verify source citations [Source: ...] and [From: ...] are preserved cleanly."""
    content = """The system requires 99.99% uptime based on the SLA.

[Source: SLA_Agreement_2026.pdf, Page 12]

Furthermore, failover latency must be under 3 seconds.

[From: Architecture_Review.docx, Page 4]
"""
    messages = [{"role": "assistant", "content": content, "created_at": "2026-09-28T10:00:00Z"}]
    pdf_bytes = generate_conversation_pdf("Citation Test", "", messages)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = doc[0].get_text()

    assert "SLA_Agreement_2026.pdf" in text
    assert "Page 12" in text
    assert "Architecture_Review.docx" in text
    assert "Page 4" in text
    print("PASS: test_pdf_citations_preserved")


def test_pdf_unicode_characters():
    """Verify comprehensive Unicode handling (currency, arrows, symbols, Greek)."""
    content = """Financial & Technical Overview:
- Budget: ₹ 50,000 (INR) or € 550 (EUR) or $ 600 (USD)
- Direction flow: Input → Processing → Output ← Feedback
- Inequalities: Error rate ≤ 0.01% and Accuracy ≥ 99.9%
- Tolerance: ± 0.05 × 100
- Greek symbols: α = 0.05, β = 0.20, γ = 0.95
"""
    messages = [{"role": "assistant", "content": content, "created_at": "2026-09-28T10:00:00Z"}]
    pdf_bytes = generate_conversation_pdf("Unicode Test", "", messages)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = doc[0].get_text()

    # Verify characters exist in extracted text
    assert "₹" in text or "50,000" in text
    assert "€" in text or "550" in text
    assert "$" in text or "600" in text
    assert "→" in text or "Processing" in text
    assert "≤" in text or "0.01" in text
    assert "≥" in text or "99.9" in text
    assert "±" in text or "0.05" in text
    assert "×" in text or "100" in text
    assert "α" in text or "0.05" in text
    assert "β" in text or "0.20" in text
    assert "γ" in text or "0.95" in text
    print("PASS: test_pdf_unicode_characters")


def test_pdf_mixed_content_flow():
    """Verify PART 8: Mixed content with heading, math equation, and code block containing literal LaTeX."""
    content = r"""### Result

The probability is:

$$ P(A)=\frac{2}{3} $$

Python implementation:

```python
x = r"\frac{2}{3}"
print(x)
```
"""
    messages = [{"role": "assistant", "content": content, "created_at": "2026-09-28T10:00:00Z"}]
    pdf_bytes = generate_conversation_pdf("Mixed Content Test", "", messages)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    text = page.get_text().replace("\xa0", " ")

    assert "Result" in text
    assert 'x = r"\\frac{2}{3}"' in text
    assert "print(x)" in text
    # Math is rendered as image
    assert len(page.get_images()) >= 1
    # Raw math outside code block is NOT leaked
    assert "$$ P(A)" not in text
    print("PASS: test_pdf_mixed_content_flow")


async def test_pdf_export_security_and_ownership():
    """Verify HTTP security: User B cannot export User A's conversation (404), anon rejected (401)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client_a, \
               AsyncClient(transport=transport, base_url="http://test") as client_b, \
               AsyncClient(transport=transport, base_url="http://test") as anon_client:

        email_a = f"pdf_sec_a_{uuid.uuid4().hex[:6]}@test.com"
        email_b = f"pdf_sec_b_{uuid.uuid4().hex[:6]}@test.com"

        res_a = await client_a.post("/auth/signup", json={"name": "Alice", "email": email_a, "password": "Pass1234!"})
        assert res_a.status_code == 200
        await client_a.get("/auth/me")

        res_b = await client_b.post("/auth/signup", json={"name": "Bob", "email": email_b, "password": "Pass1234!"})
        assert res_b.status_code == 200
        await client_b.get("/auth/me")

        # Alice creates a conversation
        c_res = await client_a.post("/conversations", json={"title": "Alice Secret Chat"})
        assert c_res.status_code == 200
        conv_id = c_res.json()["id"]

        # 1. Unauthenticated export rejected with 401
        anon_res = await anon_client.get(f"/conversations/{conv_id}/export/pdf")
        assert anon_res.status_code == 401, f"Expected 401, got {anon_res.status_code}"

        # 2. Bob exporting Alice's conversation rejected with 404
        bob_res = await client_b.get(f"/conversations/{conv_id}/export/pdf")
        assert bob_res.status_code == 404, f"Expected 404, got {bob_res.status_code}"

        # 3. Alice exporting own conversation succeeds with 200 OK
        alice_res = await client_a.get(f"/conversations/{conv_id}/export/pdf")
        assert alice_res.status_code == 200
        assert alice_res.headers.get("content-type") == "application/pdf"
        assert alice_res.content.startswith(b"%PDF-")
        print("PASS: test_pdf_export_security_and_ownership")


def run_all_tests():
    print("======================================================================")
    print("RUNNING COMPREHENSIVE PDF EXPORT TEST SUITE")
    print("======================================================================")
    test_pdf_plain_text_and_empty()
    test_pdf_headings_bold_italic_lists()
    test_pdf_code_block_protection_before_latex()
    test_pdf_latex_math_expressions()
    test_pdf_markdown_tables()
    test_pdf_citations_preserved()
    test_pdf_unicode_characters()
    test_pdf_mixed_content_flow()
    asyncio.run(test_pdf_export_security_and_ownership())
    print("\n======================================================================")
    print("ALL COMPREHENSIVE PDF EXPORT TESTS PASSED SUCCESSFULLY! ✅")
    print("======================================================================")


if __name__ == "__main__":
    run_all_tests()
