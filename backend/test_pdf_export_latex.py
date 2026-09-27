"""Test suite for PDF Export with LaTeX rendering, Markdown support, and Code Block preservation."""

import io
import fitz  # PyMuPDF
import pytest
from services.pdf_export_service import generate_conversation_pdf


def test_exact_user_latex_failure_case():
    """Verify that LaTeX math formulas render properly as math images while code blocks remain code."""
    title = "Math & Engineering Calculations"
    created_at = "2026-09-27T10:00:00Z"

    user_msg = "Can you calculate the passing percentage and time in seconds?"
    assistant_msg = r"""Here are the requested calculations:

$$ \frac{255}{692}\times100 = 36.8\% $$

The result is $\frac{2}{11}\times60 \approx 10.9$ seconds.

$$ \text{Passing percentage} = \frac{255}{692}\times100 $$

Here is a Python code block verifying the ratio:
```python
x = r"\frac{2}{3}"
print(f"Code string: {x}")
```

[From: INFOSYS - 2.pdf, Page 7]
"""

    messages = [
        {"role": "user", "content": user_msg, "created_at": "2026-09-27T10:00:01Z"},
        {"role": "assistant", "content": assistant_msg, "created_at": "2026-09-27T10:00:05Z"},
    ]

    pdf_bytes = generate_conversation_pdf(title, created_at, messages)

    # 1. Basic PDF validity
    assert pdf_bytes.startswith(b"%PDF-"), "Generated file does not have valid PDF magic bytes"
    assert len(pdf_bytes) > 2000, f"PDF file size too small: {len(pdf_bytes)} bytes"

    # 2. Inspect with PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    assert len(doc) >= 1, "PDF has no pages"
    page = doc[0]

    # 3. Verify embedded images (math formulas rendered as images)
    images = page.get_images()
    assert len(images) >= 3, f"Expected at least 3 rendered math images, got {len(images)}"

    text = page.get_text()

    # 4. Verify code block remained code verbatim
    assert 'x = r"\\frac{2}{3}"' in text or 'x = r"\\frac{2}{3}"' in text.replace("\xa0", " "), (
        f"Code block text was modified or not preserved verbatim: {text}"
    )

    # 5. Verify NO raw LaTeX delimiters or commands in the body text outside code blocks
    # Specifically, the math expressions that were converted to images should not appear as raw text
    assert r"\frac{255}{692}" not in text, "Raw LaTeX \\frac{255}{692} leaked into PDF text"
    assert r"\times100" not in text and r"\times 100" not in text, "Raw LaTeX \\times leaked into PDF text"
    assert r"\text{Passing" not in text, "Raw LaTeX \\text{Passing leaked into PDF text"
    assert r"\frac{2}{11}" not in text, "Raw LaTeX \\frac{2}{11} leaked into PDF text"

    # 6. Verify citation was preserved
    assert "INFOSYS - 2.pdf" in text, "Citation filename missing from PDF"
    assert "Page 7" in text, "Citation page missing from PDF"

    # 7. Verify header and metadata
    assert "Math & Engineering Calculations" in text
    assert "DocChat AI" in text
    assert "Page 1 of 1" in text

    print("PASS: Exact LaTeX failure case rendered properly with images, safe code block, and no raw LaTeX.")


def test_markdown_formatting_in_pdf():
    """Verify headers, lists, tables, bold, and italic in conversation PDF export."""
    title = "Markdown Feature Verification"
    created_at = "2026-09-27T10:15:00Z"

    assistant_content = r"""# Main Analysis Summary
## Performance Metrics
### Detailed Breakdown

Here is a list of features:
* **High Throughput**: 1000 requests/sec
* *Low Latency*: Under 50ms average
* `inline_code`: fast path execution

Numbered steps:
1. First step in evaluation
2. Second step with formula: $E = mc^2$

| Metric | Target | Actual |
| :--- | :--- | :--- |
| Latency | < 100ms | 45ms |
| Precision | 95% | 98.2% |

[Source: benchmark_report.pdf, Page 3]
"""

    messages = [
        {"role": "user", "content": "Provide a summary report", "created_at": "2026-09-27T10:15:00Z"},
        {"role": "assistant", "content": assistant_content, "created_at": "2026-09-27T10:15:04Z"},
    ]

    pdf_bytes = generate_conversation_pdf(title, created_at, messages)
    assert pdf_bytes.startswith(b"%PDF-")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = doc[0].get_text()

    assert "Main Analysis Summary" in text
    assert "Performance Metrics" in text
    assert "High Throughput" in text
    assert "Low Latency" in text
    assert "inline_code" in text
    assert "First step in evaluation" in text
    assert "Latency" in text
    assert "benchmark_report.pdf" in text

    print("PASS: Markdown elements (headings, lists, table, inline code, citations) rendered cleanly.")


def test_multi_page_long_conversation():
    """Verify that multi-page conversations render across pages with correct 'Page X of Y' footers."""
    title = "Long Multi-Turn Conversation"
    created_at = "2026-09-27T08:00:00Z"

    messages = []
    for i in range(12):
        messages.append({
            "role": "user",
            "content": f"Turn {i+1}: What is the equation for iteration {i+1}?",
            "created_at": f"2026-09-27T08:{i:02d}:00Z",
        })
        messages.append({
            "role": "assistant",
            "content": f"For iteration {i+1}, the formula is:\n\n$$ f(x_{i+1}) = x_{i+1}^2 + {i+1} $$\n\nThis yields $f(2) = {4 + i + 1}$.",
            "created_at": f"2026-09-27T08:{i:02d}:30Z",
        })

    pdf_bytes = generate_conversation_pdf(title, created_at, messages)
    assert pdf_bytes.startswith(b"%PDF-")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)
    assert page_count >= 2, f"Expected multiple pages for 24 messages, got {page_count}"

    # Verify footers on pages
    for idx, page in enumerate(doc):
        page_text = page.get_text()
        expected_footer = f"Page {idx + 1} of {page_count}"
        assert expected_footer in page_text, f"Missing '{expected_footer}' on page {idx+1}"

    print(f"PASS: Multi-page conversation rendered across {page_count} pages with dynamic page numbers.")


if __name__ == "__main__":
    test_exact_user_latex_failure_case()
    test_markdown_formatting_in_pdf()
    test_multi_page_long_conversation()
    print("\nALL PDF EXPORT & LATEX TESTS PASSED WITH 100% SUCCESS!")
