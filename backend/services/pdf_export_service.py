"""PDF Export Service — Server-side in-memory PDF generation for conversations.

Generates beautifully formatted, multi-page PDFs using ReportLab.
Features:
- Conversation title, timestamp, and message statistics header
- Distinct user & assistant message bubbles
- Citation highlighting and clean text wrapping
- Two-pass NumberedCanvas for dynamic "Page X of Y" footers
- Strict privacy: entirely in-memory (BytesIO), no disk persistence
"""

from __future__ import annotations

import html
import io
import logging
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

logger = logging.getLogger(__name__)


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to compute and render accurate total page counts in footers."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count: int) -> None:
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748B"))

        # Footer divider rule
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(40, 40, letter[0] - 40, 40)

        # Left footer: DocChat AI
        self.drawString(40, 26, "DocChat AI — Exported Conversation")

        # Right footer: Page X of Y
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 40, 26, page_text)
        self.restoreState()


def _sanitize_for_pdf(text: str) -> str:
    """Escape XML characters and format newlines for ReportLab Paragraph."""
    if not text:
        return ""
    escaped = html.escape(text.strip())
    return escaped.replace("\n", "<br/>")


def generate_conversation_pdf(
    title: str,
    created_at: str,
    messages: list[dict],
) -> bytes:
    """Generate a high-quality PDF representation of a conversation in-memory.

    Args:
        title: Conversation title
        created_at: ISO or formatted creation date
        messages: List of dicts with keys: ``role``, ``content``, and optional ``created_at``

    Returns:
        bytes: Raw PDF content
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=55,
    )

    content_width = letter[0] - 80  # 612 - 80 = 532 pt

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "ExportTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
    )
    meta_style = ParagraphStyle(
        "ExportMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#475569"),
    )
    user_header_style = ParagraphStyle(
        "UserHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1E3A8A"),  # Blue 900
    )
    user_body_style = ParagraphStyle(
        "UserBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#0F172A"),
    )
    assistant_header_style = ParagraphStyle(
        "AssistantHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#3730A3"),  # Indigo 800
    )
    assistant_body_style = ParagraphStyle(
        "AssistantBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
    )

    story = []

    # 1. Header Banner
    safe_title = html.escape(title or "Conversation")
    story.append(Paragraph(safe_title, title_style))
    story.append(Spacer(1, 6))

    now_utc = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")
    date_str = f"Exported on: {now_utc}  •  Total Messages: {len(messages)}"
    if created_at:
        try:
            # Parse created_at if it looks like an ISO string
            parsed_date = created_at.split("T")[0]
            date_str = f"Started: {parsed_date}  •  " + date_str
        except Exception:
            pass
    story.append(Paragraph(date_str, meta_style))
    story.append(Spacer(1, 10))

    story.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=colors.HexColor("#2563EB"),  # Brand Blue
            spaceBefore=0,
            spaceAfter=14,
        )
    )

    # 2. Messages
    if not messages:
        empty_msg = Paragraph(
            "<i>No messages have been recorded in this conversation yet.</i>",
            meta_style,
        )
        story.append(empty_msg)
    else:
        for idx, msg in enumerate(messages):
            role = (msg.get("role") or "").lower()
            content = msg.get("content") or ""
            msg_time = msg.get("created_at") or ""
            if "T" in msg_time:
                msg_time = msg_time.replace("T", " ")[:19]

            sanitized_body = _sanitize_for_pdf(content)

            if role == "user":
                header_text = f"User ({msg_time})" if msg_time else "User"
                bubble_data = [
                    [Paragraph(f"<b>{header_text}</b>", user_header_style)],
                    [Paragraph(sanitized_body, user_body_style)],
                ]
                bubble_table = Table(bubble_data, colWidths=[content_width])
                bubble_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
                            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E1")),
                            ("ROUNDEDCORNERS", [4, 4, 4, 4]),
                            ("TOPPADDING", (0, 0), (-1, -1), 6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                            ("LEFTPADDING", (0, 0), (-1, -1), 10),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ]
                    )
                )
            else:
                header_text = f"DocChat Assistant ({msg_time})" if msg_time else "DocChat Assistant"
                bubble_data = [
                    [Paragraph(f"<b>{header_text}</b>", assistant_header_style)],
                    [Paragraph(sanitized_body, assistant_body_style)],
                ]
                bubble_table = Table(bubble_data, colWidths=[content_width])
                bubble_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFFFF")),
                            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#E2E8F0")),
                            ("LINELEFT", (0, 0), (0, -1), 3.0, colors.HexColor("#4F46E5")),
                            ("ROUNDEDCORNERS", [4, 4, 4, 4]),
                            ("TOPPADDING", (0, 0), (-1, -1), 6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                            ("LEFTPADDING", (0, 0), (-1, -1), 10),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ]
                    )
                )

            story.append(bubble_table)
            story.append(Spacer(1, 8))

    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()
