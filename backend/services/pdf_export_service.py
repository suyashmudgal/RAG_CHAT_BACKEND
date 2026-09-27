"""PDF Export Service — Server-side in-memory PDF generation for conversations.

Generates beautifully formatted, multi-page PDFs using ReportLab.
Features:
- Full LaTeX mathematics support (inline $...$ and display $$...$$) rendered via matplotlib.mathtext
- Preserved Markdown: headings, bold, italic, lists, tables, inline code, and citations
- Safe fenced code blocks: code remains verbatim code, never converted to math
- Conversation title, timestamp, and message statistics header
- Distinct user & assistant message bubbles
- Two-pass NumberedCanvas for dynamic "Page X of Y" footers
- Strict privacy: entirely in-memory (BytesIO), all temporary render files cleaned up automatically
"""

from __future__ import annotations

import html
import io
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import Any

from matplotlib import mathtext
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

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


def render_latex_to_png(
    expr: str,
    tmp_dir: str,
    dpi: int = 180,
    is_display: bool = False,
    max_width_pt: float = 480.0,
) -> tuple[str, float, float] | None:
    """Render a LaTeX expression to a PNG image file and return (filepath, width_pt, height_pt).
    
    Uses matplotlib.mathtext for fast, deterministic, server-side math rendering without
    requiring external TeX binaries or browser engines.
    """
    try:
        clean = expr.strip()
        # Remove delimiters if present
        if clean.startswith("$$") and clean.endswith("$$"):
            clean = clean[2:-2].strip()
        elif clean.startswith("\\[") and clean.endswith("\\]"):
            clean = clean[2:-2].strip()
        elif clean.startswith("$") and clean.endswith("$"):
            clean = clean[1:-1].strip()
        elif clean.startswith("\\(") and clean.endswith("\\)"):
            clean = clean[2:-2].strip()

        if not clean:
            return None

        # Normalize unescaped % (TeX treats unescaped % as comments)
        clean = re.sub(r"(?<!\\)%", r"\%", clean)

        tex_str = f"${clean}$"

        fd, out_path = tempfile.mkstemp(suffix=".png", dir=tmp_dir)
        os.close(fd)

        mathtext.math_to_image(tex_str, out_path, dpi=dpi, format="png")

        with PILImage.open(out_path) as im:
            w_px, h_px = im.size

        w_pt = w_px * 72.0 / dpi
        h_pt = h_px * 72.0 / dpi

        # Scale down if exceeds max_width_pt
        if is_display and w_pt > max_width_pt:
            ratio = max_width_pt / w_pt
            w_pt = max_width_pt
            h_pt = h_pt * ratio

        return out_path, w_pt, h_pt
    except Exception as exc:
        logger.debug("Failed to render LaTeX formula '%s' with mathtext: %s", expr[:50], exc)
        return None


def format_inline_markdown_and_math(
    text: str,
    tmp_dir: str,
    dpi: int = 180,
) -> str:
    """Process inline math ($...$ or \\(...\\)) and inline markdown (**bold**, *italic*, `code`, citations)."""
    if not text:
        return ""

    img_placeholders: dict[str, str] = {}
    img_idx = 0

    def math_repl(match: re.Match) -> str:
        nonlocal img_idx
        math_content = match.group(1).strip()
        res = render_latex_to_png(math_content, tmp_dir, dpi=dpi, is_display=False)
        if res:
            out_path, w_pt, h_pt = res
            tag = f'<img src="{out_path}" width="{w_pt:.1f}" height="{h_pt:.1f}" valign="-3"/>'
            ph = f"__INLINE_IMG_{img_idx}__"
            img_placeholders[ph] = tag
            img_idx += 1
            return ph
        return match.group(0)

    # 1. Match inline math delimiters: $...$ (not $$) and \(...\)
    text = re.sub(r"(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)", math_repl, text)
    text = re.sub(r"\\\((.*?)\\\)", math_repl, text)

    # 2. Match unwrapped inline LaTeX commands (e.g. \frac{2}{11}\times60 or \sqrt{x})
    def unwrapped_math_repl(match: re.Match) -> str:
        nonlocal img_idx
        math_content = match.group(0).strip()
        res = render_latex_to_png(math_content, tmp_dir, dpi=dpi, is_display=False)
        if res:
            out_path, w_pt, h_pt = res
            tag = f'<img src="{out_path}" width="{w_pt:.1f}" height="{h_pt:.1f}" valign="-3"/>'
            ph = f"__INLINE_IMG_{img_idx}__"
            img_placeholders[ph] = tag
            img_idx += 1
            return ph
        return math_content

    text = re.sub(
        r"\\(?:frac\{[^}]+\}\{[^}]+\}|sqrt\{[^}]+\})(?:[^\s,.<>)]*)",
        unwrapped_math_repl,
        text,
    )

    # 3. Escape HTML
    text = html.escape(text)

    # 4. Restore image tags
    for ph, tag in img_placeholders.items():
        escaped_ph = html.escape(ph)
        text = text.replace(escaped_ph, tag)
        text = text.replace(ph, tag)

    # 5. Inline markdown formatting
    # Bold **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    # Italic *text* or _text_
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    # Inline code `code`
    text = re.sub(
        r"`([^`]+)`",
        r'<font face="Courier" color="#0F172A" backColor="#F1F5F9">&nbsp;\1&nbsp;</font>',
        text,
    )
    # Citations like [From: ...] or [Source: ...]
    text = re.sub(
        r"\[(From:[^\]]+|Source:[^\]]+)\]",
        r'<font color="#2563EB"><b>[\1]</b></font>',
        text,
    )

    return text.replace("\n", "<br/>")


def parse_message_to_flowables(
    content: str,
    tmp_dir: str,
    body_style: ParagraphStyle,
    content_width: float,
) -> list:
    """Parse a message's content into rich ReportLab flowables supporting Math, Code, Tables, and Markdown."""
    if not content or not content.strip():
        return [Paragraph("", body_style)]

    flowables = []

    # 1. Protect Fenced Code Blocks (Code must remain code!)
    code_blocks: dict[str, tuple[str, str]] = {}
    code_idx = 0

    def code_repl(match: re.Match) -> str:
        nonlocal code_idx
        lang = match.group(1).strip()
        code_text = match.group(2)
        ph = f"__CODE_BLOCK_{code_idx}__"
        code_blocks[ph] = (lang, code_text)
        code_idx += 1
        return f"\n\n{ph}\n\n"

    processed_content = re.sub(
        r"```([a-zA-Z0-9_\-\+]*)\r?\n(.*?)\r?\n```",
        code_repl,
        content,
        flags=re.DOTALL,
    )

    # 2. Extract Display Math ($$ ... $$ or \[ ... \])
    display_math_blocks: dict[str, str] = {}
    math_idx = 0

    def display_math_repl(match: re.Match) -> str:
        nonlocal math_idx
        m_expr = match.group(1).strip()
        ph = f"__DISPLAY_MATH_{math_idx}__"
        display_math_blocks[ph] = m_expr
        math_idx += 1
        return f"\n\n{ph}\n\n"

    processed_content = re.sub(r"\$\$(.*?)\$\$", display_math_repl, processed_content, flags=re.DOTALL)
    processed_content = re.sub(r"\\\[(.*?)\\\]", display_math_repl, processed_content, flags=re.DOTALL)

    # Standalone LaTeX lines (e.g. lines starting with \frac, \text, etc.)
    def standalone_math_line(match: re.Match) -> str:
        nonlocal math_idx
        line = match.group(0).strip()
        ph = f"__DISPLAY_MATH_{math_idx}__"
        display_math_blocks[ph] = line
        math_idx += 1
        return f"\n\n{ph}\n\n"

    processed_content = re.sub(
        r"^\s*(\\frac\{[^}]+\}\{[^}]+\}[^\n]*)\s*$",
        standalone_math_line,
        processed_content,
        flags=re.MULTILINE,
    )
    processed_content = re.sub(
        r"^\s*(\\text\{[^}]+\}[^\n]*)\s*$",
        standalone_math_line,
        processed_content,
        flags=re.MULTILINE,
    )

    # 3. Extract Markdown Tables
    table_blocks: dict[str, list[str]] = {}
    table_idx = 0

    def extract_tables(text: str) -> str:
        nonlocal table_idx
        lines = text.split("\n")
        new_lines = []
        table_acc = []

        for line in lines:
            if re.match(r"^\s*\|.*\|\s*$", line):
                table_acc.append(line)
            else:
                if len(table_acc) >= 2 and any(re.match(r"^\s*\|[-:| ]+\|\s*$", l) for l in table_acc):
                    ph = f"__TABLE_BLOCK_{table_idx}__"
                    table_blocks[ph] = list(table_acc)
                    table_idx += 1
                    new_lines.append(f"\n\n{ph}\n\n")
                elif table_acc:
                    new_lines.extend(table_acc)
                table_acc = []
                new_lines.append(line)

        if len(table_acc) >= 2 and any(re.match(r"^\s*\|[-:| ]+\|\s*$", l) for l in table_acc):
            ph = f"__TABLE_BLOCK_{table_idx}__"
            table_blocks[ph] = list(table_acc)
            table_idx += 1
            new_lines.append(f"\n\n{ph}\n\n")
        elif table_acc:
            new_lines.extend(table_acc)

        return "\n".join(new_lines)

    processed_content = extract_tables(processed_content)

    # 4. Process Blocks
    blocks = [b.strip() for b in re.split(r"\n\s*\n", processed_content) if b.strip()]

    inner_width = content_width - 24.0

    code_style = ParagraphStyle(
        "CodeText",
        fontName="Courier",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#0F172A"),
    )
    h1_style = ParagraphStyle(
        "H1Style",
        parent=body_style,
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=4,
        spaceAfter=4,
    )
    h2_style = ParagraphStyle(
        "H2Style",
        parent=body_style,
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=3,
        spaceAfter=3,
    )
    h3_style = ParagraphStyle(
        "H3Style",
        parent=body_style,
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#334155"),
        spaceBefore=2,
        spaceAfter=2,
    )

    for b in blocks:
        # Check Code Block
        if b in code_blocks:
            lang, raw_code = code_blocks[b]
            escaped_code = html.escape(raw_code.rstrip())
            formatted_code = escaped_code.replace(" ", "&nbsp;").replace("\n", "<br/>")

            code_p = Paragraph(formatted_code, code_style)
            code_table = Table([[code_p]], colWidths=[inner_width])
            code_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E1")),
                        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            flowables.append(Spacer(1, 3))
            flowables.append(code_table)
            flowables.append(Spacer(1, 3))
            continue

        # Check Display Math Block
        if b in display_math_blocks:
            m_expr = display_math_blocks[b]
            res = render_latex_to_png(
                m_expr,
                tmp_dir,
                dpi=180,
                is_display=True,
                max_width_pt=inner_width - 20,
            )
            if res:
                out_path, w_pt, h_pt = res
                img = RLImage(out_path, width=w_pt, height=h_pt)
                # Center using a 1-cell Table
                center_tbl = Table([[img]], colWidths=[inner_width])
                center_tbl.setStyle(
                    TableStyle(
                        [
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 0),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                flowables.append(Spacer(1, 3))
                flowables.append(center_tbl)
                flowables.append(Spacer(1, 3))
            else:
                clean_expr = html.escape(m_expr).replace("\n", "<br/>")
                p = Paragraph(f"<b><i>{clean_expr}</i></b>", body_style)
                flowables.append(p)
            continue

        # Check Table Block
        if b in table_blocks:
            tbl_lines = table_blocks[b]
            parsed_rows = []
            for tl in tbl_lines:
                if re.match(r"^\s*\|[-:| ]+\|\s*$", tl):
                    continue  # Separator row
                cells = [c.strip() for c in tl.strip().strip("|").split("|")]
                row_flowables = [
                    Paragraph(format_inline_markdown_and_math(c, tmp_dir), body_style)
                    for c in cells
                ]
                parsed_rows.append(row_flowables)

            if parsed_rows:
                num_cols = max(len(r) for r in parsed_rows)
                col_w = inner_width / max(num_cols, 1)
                tbl = Table(parsed_rows, colWidths=[col_w] * num_cols)
                tbl.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                            ("LEFTPADDING", (0, 0), (-1, -1), 6),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ]
                    )
                )
                flowables.append(Spacer(1, 4))
                flowables.append(tbl)
                flowables.append(Spacer(1, 4))
            continue

        # Headings
        if b.startswith("# "):
            h_text = format_inline_markdown_and_math(b[2:].strip(), tmp_dir)
            flowables.append(Paragraph(h_text, h1_style))
            flowables.append(Spacer(1, 2))
        elif b.startswith("## "):
            h_text = format_inline_markdown_and_math(b[3:].strip(), tmp_dir)
            flowables.append(Paragraph(h_text, h2_style))
            flowables.append(Spacer(1, 2))
        elif b.startswith("### "):
            h_text = format_inline_markdown_and_math(b[4:].strip(), tmp_dir)
            flowables.append(Paragraph(h_text, h3_style))
            flowables.append(Spacer(1, 2))
        else:
            # Check for bullet / numbered list lines within paragraph block
            lines = b.split("\n")
            is_list = any(re.match(r"^\s*(?:[\*\-\+]|\d+\.)\s+", l) for l in lines)
            if is_list:
                for line in lines:
                    line_s = line.strip()
                    if not line_s:
                        continue
                    m_bullet = re.match(r"^\s*([\*\-\+])\s+(.+)$", line_s)
                    m_num = re.match(r"^\s*(\d+\.)\s+(.+)$", line_s)
                    if m_bullet:
                        item_text = format_inline_markdown_and_math(m_bullet.group(2), tmp_dir)
                        formatted_line = f"&nbsp;&nbsp;&bull;&nbsp;&nbsp;{item_text}"
                        flowables.append(Paragraph(formatted_line, body_style))
                        flowables.append(Spacer(1, 1.5))
                    elif m_num:
                        item_text = format_inline_markdown_and_math(m_num.group(2), tmp_dir)
                        formatted_line = f"&nbsp;&nbsp;<b>{m_num.group(1)}</b>&nbsp;&nbsp;{item_text}"
                        flowables.append(Paragraph(formatted_line, body_style))
                        flowables.append(Spacer(1, 1.5))
                    else:
                        line_text = format_inline_markdown_and_math(line_s, tmp_dir)
                        flowables.append(Paragraph(line_text, body_style))
                        flowables.append(Spacer(1, 1.5))
            else:
                p_text = format_inline_markdown_and_math(b, tmp_dir)
                flowables.append(Paragraph(p_text, body_style))
                flowables.append(Spacer(1, 3))

    return flowables if flowables else [Paragraph("", body_style)]


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

    content_width = letter[0] - 80.0  # 612 - 80 = 532 pt

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
        fontSize=9.5,
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

    # 2. Messages with Temporary Directory for rendering math images
    with tempfile.TemporaryDirectory() as tmp_dir:
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

                if role == "user":
                    header_text = f"User ({msg_time})" if msg_time else "User"
                    msg_flowables = parse_message_to_flowables(
                        content=content,
                        tmp_dir=tmp_dir,
                        body_style=user_body_style,
                        content_width=content_width,
                    )
                    cell_content = [
                        Paragraph(f"<b>{header_text}</b>", user_header_style),
                        Spacer(1, 4),
                        *msg_flowables,
                    ]
                    bubble_table = Table([[cell_content]], colWidths=[content_width])
                    bubble_table.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
                                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E1")),
                                ("ROUNDEDCORNERS", [4, 4, 4, 4]),
                                ("TOPPADDING", (0, 0), (-1, -1), 8),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                            ]
                        )
                    )
                else:
                    header_text = f"DocChat Assistant ({msg_time})" if msg_time else "DocChat Assistant"
                    msg_flowables = parse_message_to_flowables(
                        content=content,
                        tmp_dir=tmp_dir,
                        body_style=assistant_body_style,
                        content_width=content_width,
                    )
                    cell_content = [
                        Paragraph(f"<b>{header_text}</b>", assistant_header_style),
                        Spacer(1, 4),
                        *msg_flowables,
                    ]
                    bubble_table = Table([[cell_content]], colWidths=[content_width])
                    bubble_table.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFFFF")),
                                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#E2E8F0")),
                                ("LINELEFT", (0, 0), (0, -1), 3.0, colors.HexColor("#4F46E5")),
                                ("ROUNDEDCORNERS", [4, 4, 4, 4]),
                                ("TOPPADDING", (0, 0), (-1, -1), 8),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                            ]
                        )
                    )

                story.append(bubble_table)
                story.append(Spacer(1, 8))

        doc.build(story, canvasmaker=NumberedCanvas)

    return buffer.getvalue()
