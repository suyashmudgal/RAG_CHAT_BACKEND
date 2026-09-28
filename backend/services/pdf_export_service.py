"""PDF Export Service — Server-side in-memory PDF generation for conversations.

Generates beautifully formatted, multi-page PDFs using ReportLab.
Features:
- Unicode support: full UTF-8 coverage (currency symbols ₹/€/$, arrows →/←, math ≤/≥/±/×, Greek letters α/β/γ) via bundled DejaVu fonts
- LaTeX mathematics support (inline $...$ and display $$...$$) rendered via matplotlib.mathtext
- Strict code block protection: code blocks and inline code remain verbatim code, never converted to math
- Markdown support: headings (with keepWithNext), bold, italic, bullet/numbered lists, tables (with repeatRows), blockquotes, citations
- Clean multi-page layout: flowable-based message structure preventing blank gaps or table clipping across pages
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

import matplotlib
from matplotlib import mathtext
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
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

# Cache font registration status
_FONTS_REGISTERED = False
_BODY_FONT = "Helvetica"
_MONO_FONT = "Courier"


def _ensure_unicode_fonts() -> tuple[str, str]:
    """Ensure DejaVu unicode fonts are registered with ReportLab.

    Returns:
        tuple[str, str]: (body_font_family_name, mono_font_name)
    """
    global _FONTS_REGISTERED, _BODY_FONT, _MONO_FONT

    if _FONTS_REGISTERED:
        return _BODY_FONT, _MONO_FONT

    try:
        font_dir = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
        if os.path.exists(font_dir):
            sans_regular = os.path.join(font_dir, "DejaVuSans.ttf")
            sans_bold = os.path.join(font_dir, "DejaVuSans-Bold.ttf")
            sans_italic = os.path.join(font_dir, "DejaVuSans-Oblique.ttf")
            sans_bold_italic = os.path.join(font_dir, "DejaVuSans-BoldOblique.ttf")
            mono_regular = os.path.join(font_dir, "DejaVuSansMono.ttf")
            mono_bold = os.path.join(font_dir, "DejaVuSansMono-Bold.ttf")

            if os.path.exists(sans_regular) and os.path.exists(sans_bold):
                pdfmetrics.registerFont(TTFont("DejaVuSans", sans_regular))
                pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", sans_bold))
                pdfmetrics.registerFont(
                    TTFont("DejaVuSans-Oblique", sans_italic if os.path.exists(sans_italic) else sans_regular)
                )
                pdfmetrics.registerFont(
                    TTFont("DejaVuSans-BoldOblique", sans_bold_italic if os.path.exists(sans_bold_italic) else sans_bold)
                )
                pdfmetrics.registerFontFamily(
                    "DejaVuSans",
                    normal="DejaVuSans",
                    bold="DejaVuSans-Bold",
                    italic="DejaVuSans-Oblique",
                    boldItalic="DejaVuSans-BoldOblique",
                )
                _BODY_FONT = "DejaVuSans"

            if os.path.exists(mono_regular):
                pdfmetrics.registerFont(TTFont("DejaVuSansMono", mono_regular))
                pdfmetrics.registerFont(
                    TTFont("DejaVuSansMono-Bold", mono_bold if os.path.exists(mono_bold) else mono_regular)
                )
                pdfmetrics.registerFontFamily(
                    "DejaVuSansMono",
                    normal="DejaVuSansMono",
                    bold="DejaVuSansMono-Bold",
                    italic="DejaVuSansMono",
                    boldItalic="DejaVuSansMono-Bold",
                )
                _MONO_FONT = "DejaVuSansMono"
    except Exception as exc:
        logger.warning("Could not register DejaVu fonts: %s. Using default fonts.", exc)
        _BODY_FONT = "Helvetica"
        _MONO_FONT = "Courier"
    finally:
        _FONTS_REGISTERED = True

    return _BODY_FONT, _MONO_FONT


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
        body_font, _ = _ensure_unicode_fonts()
        font_to_use = body_font if body_font in pdfmetrics.getRegisteredFontNames() else "Helvetica"
        self.setFont(font_to_use, 9)
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


def clean_latex_expr(expr: str) -> str:
    """Normalize and prepare LaTeX expression for matplotlib mathtext rendering."""
    clean = expr.strip()
    if clean.startswith("$$") and clean.endswith("$$"):
        clean = clean[2:-2].strip()
    elif clean.startswith("\\[") and clean.endswith("\\]"):
        clean = clean[2:-2].strip()
    elif clean.startswith("$") and clean.endswith("$"):
        clean = clean[1:-1].strip()
    elif clean.startswith("\\(") and clean.endswith("\\)"):
        clean = clean[2:-2].strip()

    if not clean:
        return ""

    # Mathtext supports \leq and \geq, but not \le or \ge
    clean = re.sub(r"\\le(?![a-zA-Z])", r"\\leq", clean)
    clean = re.sub(r"\\ge(?![a-zA-Z])", r"\\geq", clean)

    # Normalize unescaped % (TeX treats unescaped % as comments)
    clean = re.sub(r"(?<!\\)%", r"\%", clean)

    # Common unicode symbols inside math to TeX
    clean = clean.replace("≤", r"\leq ")
    clean = clean.replace("≥", r"\geq ")
    clean = clean.replace("±", r"\pm ")
    clean = clean.replace("×", r"\times ")
    clean = clean.replace("→", r"\rightarrow ")
    clean = clean.replace("←", r"\leftarrow ")
    clean = clean.replace("≈", r"\approx ")
    clean = clean.replace("≠", r"\neq ")
    clean = clean.replace("·", r"\cdot ")
    clean = clean.replace("•", r"\cdot ")

    return clean


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
        clean = clean_latex_expr(expr)
        if not clean:
            return None

        tex_str = f"${clean}$"

        fd, out_path = tempfile.mkstemp(suffix=".png", dir=tmp_dir)
        os.close(fd)

        try:
            mathtext.math_to_image(tex_str, out_path, dpi=dpi, format="png")
        except Exception:
            # Fallback attempt: if expression had \text, translate to \mathrm
            alt = re.sub(r"\\text\{([^}]+)\}", r"\\mathrm{\1}", clean)
            mathtext.math_to_image(f"${alt}$", out_path, dpi=dpi, format="png")

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
    mono_font: str = "DejaVuSansMono",
) -> str:
    """Process inline markdown, inline code, and inline math with strict code protection."""
    if not text:
        return ""

    # STEP 1: PROTECT INLINE CODE (CRITICAL: code must NEVER be parsed as math!)
    inline_code_map: dict[str, str] = {}
    code_token_idx = 0

    def protect_inline_code(match: re.Match) -> str:
        nonlocal code_token_idx
        token = f"__INLINE_CODE_TOKEN_{code_token_idx}__"
        inline_code_map[token] = match.group(1)
        code_token_idx += 1
        return token

    text = re.sub(r"`([^`\n]+)`", protect_inline_code, text)

    # STEP 2: PROCESS INLINE MATH ($...$ or \(...\)) OUTSIDE CODE
    img_placeholders: dict[str, str] = {}
    img_idx = 0

    def math_repl(match: re.Match) -> str:
        nonlocal img_idx
        math_content = (match.group(1) or "").strip()
        res = render_latex_to_png(math_content, tmp_dir, dpi=dpi, is_display=False)
        if res:
            out_path, w_pt, h_pt = res
            tag = f'<img src="{out_path}" width="{w_pt:.1f}" height="{h_pt:.1f}" valign="-3"/>'
            ph = f"__INLINE_IMG_{img_idx}__"
            img_placeholders[ph] = tag
            img_idx += 1
            return ph
        return match.group(0)

    # Inline math delimiters: $...$ (not $$) and \(...\)
    text = re.sub(r"(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)", math_repl, text)
    text = re.sub(r"\\\((.*?)\\\)", math_repl, text)

    # Standalone inline LaTeX commands outside code (e.g. \frac{2}{11}\times60 or \sqrt{x})
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

    # STEP 3: ESCAPE HTML FOR BODY TEXT
    text = html.escape(text)

    # STEP 4: RESTORE PROTECTED INLINE CODE WITH FORMATTING
    for token, raw_code in inline_code_map.items():
        escaped_code = html.escape(raw_code).replace(" ", "&nbsp;")
        code_tag = f'<font face="{mono_font}" color="#0F172A" backColor="#F1F5F9">&nbsp;{escaped_code}&nbsp;</font>'
        text = text.replace(token, code_tag)

    # STEP 5: RESTORE MATH IMAGE TAGS
    for ph, tag in img_placeholders.items():
        escaped_ph = html.escape(ph)
        text = text.replace(escaped_ph, tag)
        text = text.replace(ph, tag)

    # STEP 6: INLINE MARKDOWN FORMATTING
    # Bold **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    # Italic *text* or _text_
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"(?<![a-zA-Z0-9_])_([^_]+)_(?![a-zA-Z0-9_])", r"<i>\1</i>", text)
    # Citations like [From: ...] or [Source: ...]
    text = re.sub(
        r"\[((?:From|Source):[^\]]+)\]",
        r'<font color="#2563EB"><b>[\1]</b></font>',
        text,
        flags=re.IGNORECASE,
    )

    return text.replace("\n", "<br/>")


def parse_message_to_flowables(
    content: str,
    tmp_dir: str,
    body_style: ParagraphStyle,
    content_width: float,
    body_font: str = "DejaVuSans",
    mono_font: str = "DejaVuSansMono",
) -> list:
    """Parse a message's content into rich ReportLab flowables supporting Math, Code, Tables, and Markdown."""
    if not content or not content.strip():
        return [Paragraph("", body_style)]

    flowables = []

    # 1. Protect Fenced Code Blocks (Code must remain verbatim code!)
    code_blocks: dict[str, tuple[str, str]] = {}
    code_idx = 0

    def code_repl(match: re.Match) -> str:
        nonlocal code_idx
        lang = (match.group(1) or "").strip()
        code_text = match.group(2)
        ph = f"__FENCED_CODE_BLOCK_{code_idx}__"
        code_blocks[ph] = (lang, code_text)
        code_idx += 1
        return f"\n\n{ph}\n\n"

    # Match fenced code blocks (with or without trailing newline)
    processed_content = re.sub(
        r"```([a-zA-Z0-9_\-\+\#]*)[ \t]*\r?\n(.*?)\r?\n?```",
        code_repl,
        content,
        flags=re.DOTALL,
    )

    # 2. Extract Display Math ($$ ... $$ or \[ ... \])
    display_math_blocks: dict[str, str] = {}
    math_idx = 0

    def display_math_repl(match: re.Match) -> str:
        nonlocal math_idx
        m_expr = (match.group(1) if match.group(1) is not None else match.group(2)).strip()
        ph = f"__DISPLAY_MATH_BLOCK_{math_idx}__"
        display_math_blocks[ph] = m_expr
        math_idx += 1
        return f"\n\n{ph}\n\n"

    processed_content = re.sub(
        r"\$\$(.*?)\$\$|\\\[(.*?)\\\]",
        display_math_repl,
        processed_content,
        flags=re.DOTALL,
    )

    # Standalone LaTeX lines (lines starting with \frac, \sqrt, \text, \mathrm)
    def standalone_math_line(match: re.Match) -> str:
        nonlocal math_idx
        line = match.group(0).strip()
        ph = f"__DISPLAY_MATH_BLOCK_{math_idx}__"
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
        r"^\s*(\\(?:text|mathrm)\{[^}]+\}[^\n]*)\s*$",
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

    code_style = ParagraphStyle(
        "CodeText",
        fontName=mono_font,
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#0F172A"),
    )
    h1_style = ParagraphStyle(
        "H1Style",
        parent=body_style,
        fontName=f"{body_font}-Bold" if f"{body_font}-Bold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=6,
        spaceAfter=4,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "H2Style",
        parent=body_style,
        fontName=f"{body_font}-Bold" if f"{body_font}-Bold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=5,
        spaceAfter=3,
        keepWithNext=True,
    )
    h3_style = ParagraphStyle(
        "H3Style",
        parent=body_style,
        fontName=f"{body_font}-Bold" if f"{body_font}-Bold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#334155"),
        spaceBefore=4,
        spaceAfter=2,
        keepWithNext=True,
    )
    blockquote_style = ParagraphStyle(
        "BlockquoteStyle",
        parent=body_style,
        fontName=f"{body_font}-Oblique" if f"{body_font}-Oblique" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#475569"),
    )
    table_header_style = ParagraphStyle(
        "TblHeader",
        parent=body_style,
        fontName=f"{body_font}-Bold" if f"{body_font}-Bold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#0F172A"),
    )
    table_cell_style = ParagraphStyle(
        "TblCell",
        parent=body_style,
        fontName=body_font,
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#334155"),
    )

    for b in blocks:
        # Check Code Block
        if b in code_blocks:
            lang, raw_code = code_blocks[b]
            # Code is literal code! Never converted to math!
            escaped_code = html.escape(raw_code.rstrip())
            formatted_code = escaped_code.replace(" ", "&nbsp;").replace("\n", "<br/>")

            code_p = Paragraph(formatted_code, code_style)
            code_table = Table([[code_p]], colWidths=[content_width])
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
                max_width_pt=content_width - 20,
            )
            if res:
                out_path, w_pt, h_pt = res
                img = RLImage(out_path, width=w_pt, height=h_pt)
                center_tbl = Table([[img]], colWidths=[content_width])
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
                # Fallback without leaking raw LaTeX delimiters
                clean_expr = clean_latex_expr(m_expr)
                escaped_expr = html.escape(clean_expr).replace("\n", "<br/>")
                p = Paragraph(f"<b><i>{escaped_expr}</i></b>", body_style)
                flowables.append(p)
            continue

        # Check Table Block
        if b in table_blocks:
            tbl_lines = table_blocks[b]
            parsed_rows = []
            is_header = True
            for tl in tbl_lines:
                if re.match(r"^\s*\|[-:| ]+\|\s*$", tl):
                    is_header = False
                    continue  # Separator row
                cells = [c.strip() for c in tl.strip().strip("|").split("|")]
                if is_header and not parsed_rows:
                    row_flowables = [
                        Paragraph(
                            format_inline_markdown_and_math(c, tmp_dir, mono_font=mono_font),
                            table_header_style,
                        )
                        for c in cells
                    ]
                else:
                    row_flowables = [
                        Paragraph(
                            format_inline_markdown_and_math(c, tmp_dir, mono_font=mono_font),
                            table_cell_style,
                        )
                        for c in cells
                    ]
                parsed_rows.append(row_flowables)

            if parsed_rows:
                num_cols = max(len(r) for r in parsed_rows)
                # Normalize row length
                for r in parsed_rows:
                    while len(r) < num_cols:
                        r.append(Paragraph("", table_cell_style))

                col_w = content_width / max(num_cols, 1)
                tbl = Table(parsed_rows, colWidths=[col_w] * num_cols, repeatRows=1)
                tbl.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                            ("LEFTPADDING", (0, 0), (-1, -1), 6),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ]
                    )
                )
                flowables.append(Spacer(1, 4))
                flowables.append(tbl)
                flowables.append(Spacer(1, 4))
            continue

        # Check Blockquote
        if b.startswith(">"):
            quote_lines = [re.sub(r"^>\s?", "", l) for l in b.split("\n")]
            quote_text = format_inline_markdown_and_math(
                "\n".join(quote_lines), tmp_dir, mono_font=mono_font
            )
            quote_p = Paragraph(quote_text, blockquote_style)
            quote_table = Table([[quote_p]], colWidths=[content_width])
            quote_table.setStyle(
                TableStyle(
                    [
                        ("LINELEFT", (0, 0), (0, -1), 2.5, colors.HexColor("#94A3B8")),
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            flowables.append(Spacer(1, 2))
            flowables.append(quote_table)
            flowables.append(Spacer(1, 2))
            continue

        # Headings
        if b.startswith("# "):
            h_text = format_inline_markdown_and_math(b[2:].strip(), tmp_dir, mono_font=mono_font)
            flowables.append(Paragraph(h_text, h1_style))
            flowables.append(Spacer(1, 2))
        elif b.startswith("## "):
            h_text = format_inline_markdown_and_math(b[3:].strip(), tmp_dir, mono_font=mono_font)
            flowables.append(Paragraph(h_text, h2_style))
            flowables.append(Spacer(1, 2))
        elif b.startswith("### "):
            h_text = format_inline_markdown_and_math(b[4:].strip(), tmp_dir, mono_font=mono_font)
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
                        item_text = format_inline_markdown_and_math(
                            m_bullet.group(2), tmp_dir, mono_font=mono_font
                        )
                        formatted_line = f"&nbsp;&nbsp;&bull;&nbsp;&nbsp;{item_text}"
                        flowables.append(Paragraph(formatted_line, body_style))
                        flowables.append(Spacer(1, 1.5))
                    elif m_num:
                        item_text = format_inline_markdown_and_math(
                            m_num.group(2), tmp_dir, mono_font=mono_font
                        )
                        formatted_line = f"&nbsp;&nbsp;<b>{m_num.group(1)}</b>&nbsp;&nbsp;{item_text}"
                        flowables.append(Paragraph(formatted_line, body_style))
                        flowables.append(Spacer(1, 1.5))
                    else:
                        line_text = format_inline_markdown_and_math(line_s, tmp_dir, mono_font=mono_font)
                        flowables.append(Paragraph(line_text, body_style))
                        flowables.append(Spacer(1, 1.5))
            else:
                p_text = format_inline_markdown_and_math(b, tmp_dir, mono_font=mono_font)
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

    # Ensure Unicode font family is registered
    body_font, mono_font = _ensure_unicode_fonts()
    bold_font = f"{body_font}-Bold" if f"{body_font}-Bold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ExportTitle",
        parent=styles["Normal"],
        fontName=bold_font,
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
    )
    meta_style = ParagraphStyle(
        "ExportMeta",
        parent=styles["Normal"],
        fontName=body_font,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#475569"),
    )
    user_header_style = ParagraphStyle(
        "UserHeader",
        parent=styles["Normal"],
        fontName=bold_font,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1E3A8A"),  # Blue 900
        keepWithNext=True,
    )
    user_body_style = ParagraphStyle(
        "UserBody",
        parent=styles["Normal"],
        fontName=body_font,
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#0F172A"),
        leftIndent=10,
        rightIndent=10,
    )
    assistant_header_style = ParagraphStyle(
        "AssistantHeader",
        parent=styles["Normal"],
        fontName=bold_font,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#3730A3"),  # Indigo 800
        keepWithNext=True,
    )
    assistant_body_style = ParagraphStyle(
        "AssistantBody",
        parent=styles["Normal"],
        fontName=body_font,
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
        leftIndent=10,
        rightIndent=10,
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

                is_user = role == "user"
                header_title = "User" if is_user else "DocChat Assistant"
                header_style = user_header_style if is_user else assistant_header_style
                body_style = user_body_style if is_user else assistant_body_style
                badge_bg = colors.HexColor("#EFF6FF") if is_user else colors.HexColor("#EEF2FF")
                border_color = colors.HexColor("#3B82F6") if is_user else colors.HexColor("#6366F1")

                # Compact, un-splittable message header bar
                header_p = Paragraph(f"<b>{header_title}</b>", header_style)
                time_p = Paragraph(
                    f'<font color="#64748B">{msg_time}</font>' if msg_time else "",
                    meta_style,
                )
                header_bar = Table(
                    [[header_p, time_p]],
                    colWidths=[content_width - 150, 150],
                )
                header_bar.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), badge_bg),
                            ("LINELEFT", (0, 0), (0, -1), 3.0, border_color),
                            ("ROUNDEDCORNERS", [3, 3, 3, 3]),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                            ("LEFTPADDING", (0, 0), (-1, -1), 8),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ]
                    )
                )

                story.append(header_bar)
                story.append(Spacer(1, 4))

                # Parse message body into individual flowables (enables natural multi-page flow)
                msg_flowables = parse_message_to_flowables(
                    content=content,
                    tmp_dir=tmp_dir,
                    body_style=body_style,
                    content_width=content_width,
                    body_font=body_font,
                    mono_font=mono_font,
                )
                story.extend(msg_flowables)

                # Message divider
                if idx < len(messages) - 1:
                    story.append(
                        HRFlowable(
                            width="100%",
                            thickness=0.5,
                            color=colors.HexColor("#E2E8F0"),
                            spaceBefore=6,
                            spaceAfter=8,
                        )
                    )

        doc.build(story, canvasmaker=NumberedCanvas)

    return buffer.getvalue()
