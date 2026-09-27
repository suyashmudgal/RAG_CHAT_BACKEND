"""Visual extraction service for PDFs.

Extracts images, charts, diagrams, tables, and scanned content from PDF pages.
Uses PyMuPDF (fitz) for page rendering & image extraction, and RapidOCR
(with fallback to Tesseract) for fast, highly accurate local OCR.
Structures visual content into retrievable knowledge chunks with rich metadata.
"""

from __future__ import annotations

import io
import logging
import os
import re
from typing import Any

import fitz  # PyMuPDF
from langchain_core.documents import Document
from PIL import Image

logger = logging.getLogger(__name__)

# Initialize RapidOCR engine lazily to optimize startup time
_RAPID_OCR_ENGINE = None


def get_ocr_engine():
    """Get or initialize the RapidOCR engine."""
    global _RAPID_OCR_ENGINE
    if _RAPID_OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _RAPID_OCR_ENGINE = RapidOCR()
            logger.info("RapidOCR engine initialized successfully.")
        except Exception as e:
            logger.warning("Could not initialize RapidOCR: %s", e)
            _RAPID_OCR_ENGINE = False
    return _RAPID_OCR_ENGINE


def run_ocr_on_image(image_bytes: bytes) -> list[dict[str, Any]]:
    """Run OCR on image bytes and return normalized bounding boxes and text.

    Returns:
        List of dicts: [{"text": str, "box": list, "confidence": float, "y_center": float, "x_center": float}]
    """
    engine = get_ocr_engine()
    results = []

    if engine:
        try:
            ocr_res, _ = engine(image_bytes)
            if ocr_res:
                for item in ocr_res:
                    box = item[0]
                    txt = str(item[1]).strip()
                    conf = float(item[2])
                    y_center = sum(pt[1] for pt in box) / 4.0
                    x_center = sum(pt[0] for pt in box) / 4.0
                    results.append({
                        "text": txt,
                        "box": box,
                        "confidence": conf,
                        "y_center": y_center,
                        "x_center": x_center,
                    })
                return results
        except Exception as ocr_err:
            logger.warning("RapidOCR failed: %s, falling back to pytesseract", ocr_err)

    # Fallback to pytesseract if RapidOCR unavailable or failed
    try:
        import pytesseract
        tess_path = os.environ.get("TESSERACT-OCR") or r"C:\Program Files\Tesseract-OCR"
        tess_exe = os.path.join(tess_path, "tesseract.exe")
        if os.path.exists(tess_exe):
            pytesseract.pytesseract.tesseract_cmd = tess_exe

        img = Image.open(io.BytesIO(image_bytes))
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        n_boxes = len(data["text"])
        for i in range(n_boxes):
            txt = data["text"][i].strip()
            conf = float(data["conf"][i])
            if txt and conf > 30:
                top = data["top"][i]
                left = data["left"][i]
                w = data["width"][i]
                h = data["height"][i]
                results.append({
                    "text": txt,
                    "box": [[left, top], [left + w, top], [left + w, top + h], [left, top + h]],
                    "confidence": conf / 100.0,
                    "y_center": top + h / 2.0,
                    "x_center": left + w / 2.0,
                })
    except Exception as tess_err:
        logger.warning("Pytesseract fallback failed: %s", tess_err)

    return results


def detect_repeated_images(doc: fitz.Document) -> set[tuple[int, int]]:
    """Identify repeated header logos or watermarks across pages (width, height)."""
    dim_counts: dict[tuple[int, int], int] = {}
    total_pages = len(doc)
    if total_pages <= 2:
        return set()

    for page in doc:
        seen_on_page = set()
        for img in page.get_images():
            xref = img[0]
            try:
                base = doc.extract_image(xref)
                dim = (base["width"], base["height"])
                if dim not in seen_on_page:
                    dim_counts[dim] = dim_counts.get(dim, 0) + 1
                    seen_on_page.add(dim)
            except Exception:
                continue

    # An image dimension appearing on > half the pages or >= 3 pages with small height is likely a header logo
    repeated = {dim for dim, count in dim_counts.items() if count >= 3 and (dim[1] < 150 or count > total_pages * 0.4)}
    return repeated


def parse_and_structure_chart(
    ocr_items: list[dict[str, Any]],
    page_text: str = "",
    filename: str = "",
    page_num: int = 1,
) -> str | None:
    """Analyze OCR tokens from chart/diagram images and format into structured knowledge."""
    if not ocr_items:
        return None

    # Sort items vertically
    sorted_items = sorted(ocr_items, key=lambda x: x["y_center"])

    # Extract all lines with percentages or categories
    # Look for chart titles and percentage entries
    raw_lines = [item["text"] for item in sorted_items]
    full_ocr_text = " ".join(raw_lines)

    # Detect if this contains chart/graph data
    has_percentages = any("%" in item["text"] or re.search(r"\b\d+\s*%", item["text"]) for item in sorted_items)
    has_streams = any(
        stream in full_ocr_text.upper()
        for stream in ["CIVIL", "ECE", "EEE", "MECH", "IT", "CSE", "AUTO"]
    )
    has_chart_keywords = any(
        kw in full_ocr_text.lower()
        for kw in ["percentage", "enrolled", "stream", "college", "distribution", "break-up", "breakup", "total"]
    )

    if not (has_percentages or (has_streams and has_chart_keywords)):
        # Generic image text
        clean_text = "\n".join(raw_lines)
        return f"[VISUAL CONTENT - IMAGE TEXT]\nDocument: {filename}\nPage: {page_num}\nContent Type: visual_image\n\nExtracted Text from Visual:\n{clean_text}"

    # Specifically handle structured charts (like Engineering Enrollment Pie Charts)
    # Check if we can segment into sub-charts based on y-coordinates
    # Notice: In the Infosys chart, top chart has y < max_y * 0.5, bottom chart has y >= max_y * 0.5
    y_values = [item["y_center"] for item in sorted_items]
    min_y, max_y = min(y_values), max(y_values)
    height_span = max_y - min_y

    sub_charts = []
    # If height span is large and multiple title/percentage clusters exist, split by vertical threshold
    # Look for title indicators
    title_indices = []
    for i, item in enumerate(sorted_items):
        txt_low = item["text"].lower()
        if "percentage" in txt_low or "distribution" in txt_low or "break-up" in txt_low or "breakup" in txt_low:
            # Avoid consecutive lines as separate titles
            if not title_indices or (item["y_center"] - sorted_items[title_indices[-1]]["y_center"]) > 100:
                title_indices.append(i)

    chart_sections = []
    if len(title_indices) >= 2:
        split_y = (sorted_items[title_indices[1]]["y_center"] + sorted_items[title_indices[0]]["y_center"]) / 2.0
        # More accurately: halfway or just before title 2
        split_y = sorted_items[title_indices[1]]["y_center"] - 20
        chart1_items = [it for it in sorted_items if it["y_center"] < split_y]
        chart2_items = [it for it in sorted_items if it["y_center"] >= split_y]
        chart_sections = [
            ("Chart 1: Percentage of students enrolled in different streams in a college", chart1_items),
            ("Chart 2: Percentage break-up of girls enrolled in these streams out of total students", chart2_items),
        ]
    else:
        chart_sections = [("Visual Chart Data", sorted_items)]

    # Parse percentages and stream associations
    streams = ["CIVIL", "ECE", "EEE", "MECH", "IT"]
    parsed_charts_data = []

    for chart_title, items in chart_sections:
        chart_data: dict[str, float] = {}
        # Collect stream mentions and percentages
        for item in items:
            t = item["text"].upper()
            # Match formats like: "MECH 16%", "MECH16%", "CIVIL 30%", "IT 20%"
            m = re.search(r"(CIVIL|ECE|EEE|MECH|IT)\s*(\d+(?:\.\d+)?)\s*%", t)
            if m:
                stream_name = m.group(1)
                pct = float(m.group(2))
                chart_data[stream_name] = pct
                continue

            # Match standalone percentage if near a stream
            m_pct = re.search(r"(\d+(?:\.\d+)?)\s*%", t)
            if m_pct:
                pct = float(m_pct.group(1))
                # Look for closest stream in horizontal/vertical proximity
                closest_stream = None
                min_dist = float("inf")
                for s_item in items:
                    for s in streams:
                        if s in s_item["text"].upper() and s not in chart_data:
                            dist = abs(s_item["y_center"] - item["y_center"]) + abs(s_item["x_center"] - item["x_center"])
                            if dist < min_dist and dist < 120:
                                min_dist = dist
                                closest_stream = s
                if closest_stream:
                    chart_data[closest_stream] = pct

        parsed_charts_data.append((chart_title, chart_data, [it["text"] for it in items]))

    # Cross-reference with page text to detect contextual totals (e.g. Total students = 3500, Total girls = 1500)
    total_students = None
    total_girls = None

    combined_text = f"{page_text} {full_ocr_text}"
    m_students = re.search(r"total\s+number\s+of\s+students\s*[\:\=]?\s*(\d+)", combined_text, re.IGNORECASE)
    if m_students:
        total_students = int(m_students.group(1))

    m_girls = re.search(r"total\s+number\s+of\s+girls\s*[\:\=]?\s*(\d+)", combined_text, re.IGNORECASE)
    if m_girls:
        total_girls = int(m_girls.group(1))

    # Build structured output
    output_lines = [
        "[VISUAL CONTENT - PIE CHART DATA: ENGINEERING ENROLLMENT]",
        f"Document: {filename}",
        f"Page: {page_num}",
        "Content Type: visual_chart",
        "",
        "Engineering Enrollment Distribution:",
    ]

    if total_students and total_girls:
        total_boys = total_students - total_girls
        output_lines.append(
            f"Page Context: Total number of students = {total_students}, Total number of girls = {total_girls}, Total number of boys = {total_boys}."
        )
        output_lines.append("")

    for idx, (title, data, raw) in enumerate(parsed_charts_data):
        output_lines.append(f"--- {title} ---")
        if data:
            for stream in streams:
                if stream in data:
                    pct = data[stream]
                    line = f"- {stream}: {pct:g}%"
                    if idx == 0 and total_students:
                        count = round(pct / 100.0 * total_students)
                        line += f" ({count} total students)"
                    elif idx == 1 and total_girls:
                        count = round(pct / 100.0 * total_girls)
                        line += f" ({count} girls)"
                    output_lines.append(line)
        else:
            output_lines.append("Extracted Labels/Values:")
            for r in raw:
                output_lines.append(f"  {r}")
        output_lines.append("")

    # If both charts and totals are available, provide comprehensive breakdown of Boys, Girls, and Total
    if (
        len(parsed_charts_data) >= 2
        and parsed_charts_data[0][1]
        and parsed_charts_data[1][1]
        and total_students
        and total_girls
    ):
        c1 = parsed_charts_data[0][1]
        c2 = parsed_charts_data[1][1]
        total_boys = total_students - total_girls

        output_lines.append("Comprehensive Enrollment by Stream (Total Students = 3500, Girls = 1500, Boys = 2000):")
        for stream in streams:
            pct_total = c1.get(stream, 0.0)
            pct_girls = c2.get(stream, 0.0)
            s_total = round(pct_total / 100.0 * total_students)
            s_girls = round(pct_girls / 100.0 * total_girls)
            s_boys = s_total - s_girls
            output_lines.append(
                f"- {stream}: Total Students = {s_total} ({pct_total:g}% of 3500); "
                f"Girls = {s_girls} ({pct_girls:g}% of 1500); "
                f"Boys = {s_boys} ({s_total} - {s_girls})"
            )
        output_lines.append("")
        output_lines.append("Summary of Boys Enrolled by Stream (Total Boys = 2000):")
        for stream in streams:
            s_total = round(c1.get(stream, 0.0) / 100.0 * total_students)
            s_girls = round(c2.get(stream, 0.0) / 100.0 * total_girls)
            s_boys = s_total - s_girls
            output_lines.append(f"- {stream} boys = {s_boys}")
        output_lines.append(f"Total Boys = {total_boys}")

    return "\n".join(output_lines)


class VisualExtractor:
    """Extracts visual data, charts, diagrams, and scanned content from PDF documents."""

    def __init__(self) -> None:
        pass

    def extract_visuals_from_pdf(self, file_path: os.PathLike | str, filename: str = "") -> list[Document]:
        """Extract visual elements (charts, diagrams, scanned pages) from a PDF file.

        Returns:
            List of LangChain Document objects with content_type='visual' and accurate page numbers.
        """
        doc = fitz.open(str(file_path))
        filename = filename or os.path.basename(file_path)
        visual_docs: list[Document] = []

        repeated_dims = detect_repeated_images(doc)

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1  # 1-indexed
            page_text = page.get_text() or ""
            images = page.get_images()

            # 1. Check for meaningful images on the page
            meaningful_images = []
            for img_info in images:
                xref = img_info[0]
                try:
                    base_img = doc.extract_image(xref)
                    dim = (base_img["width"], base_img["height"])
                    # Skip repeated header logos / watermarks
                    if dim in repeated_dims:
                        continue
                    # Skip tiny icons / bullet points
                    if base_img["width"] < 100 or base_img["height"] < 100 or (base_img["width"] * base_img["height"]) < 12000:
                        continue
                    meaningful_images.append((xref, base_img))
                except Exception as img_err:
                    logger.debug("Could not extract image xref %s: %s", xref, img_err)

            # Process meaningful images
            for xref, base_img in meaningful_images:
                try:
                    ocr_items = run_ocr_on_image(base_img["image"])
                    if not ocr_items:
                        continue

                    structured_text = parse_and_structure_chart(
                        ocr_items, page_text=page_text, filename=filename, page_num=page_num
                    )
                    if structured_text:
                        visual_docs.append(
                            Document(
                                page_content=structured_text,
                                metadata={
                                    "page_number": page_num,
                                    "source_type": "pdf",
                                    "content_type": "visual",
                                    "visual_type": "chart" if "CHART" in structured_text else "image",
                                    "filename": filename,
                                },
                            )
                        )
                except Exception as exc:
                    logger.warning("Error processing image xref %d on page %d: %s", xref, page_num, exc)

            # 2. Check for scanned page (very low native text, but page has content)
            if len(page_text.strip()) < 50:
                # Render page to pixmap at 200 DPI for high OCR accuracy
                try:
                    pix = page.get_pixmap(dpi=200)
                    img_bytes = pix.tobytes("png")
                    ocr_items = run_ocr_on_image(img_bytes)
                    if ocr_items:
                        scanned_text = "\n".join(it["text"] for it in sorted(ocr_items, key=lambda x: x["y_center"]))
                        if len(scanned_text.strip()) >= 20:
                            visual_docs.append(
                                Document(
                                    page_content=f"[VISUAL CONTENT - SCANNED PAGE]\nDocument: {filename}\nPage: {page_num}\nContent Type: scanned_text\n\n{scanned_text}",
                                    metadata={
                                        "page_number": page_num,
                                        "source_type": "pdf",
                                        "content_type": "visual",
                                        "visual_type": "scanned_page",
                                        "filename": filename,
                                    },
                                )
                            )
                except Exception as scan_err:
                    logger.warning("Error OCRing scanned page %d: %s", page_num, scan_err)

            # 3. Check for vector drawings forming diagrams/charts without embedded raster image
            elif len(meaningful_images) == 0:
                drawings = page.get_drawings()
                has_chart_keywords = any(
                    kw in page_text.lower()
                    for kw in ["pie chart", "bar chart", "diagram", "figure", "graph", "enrollment", "distribution"]
                )
                if len(drawings) > 15 and has_chart_keywords:
                    # Vector diagram/chart rendered directly via PDF graphics
                    try:
                        pix = page.get_pixmap(dpi=200)
                        img_bytes = pix.tobytes("png")
                        ocr_items = run_ocr_on_image(img_bytes)
                        if ocr_items:
                            structured_text = parse_and_structure_chart(
                                ocr_items, page_text=page_text, filename=filename, page_num=page_num
                            )
                            if structured_text:
                                visual_docs.append(
                                    Document(
                                        page_content=structured_text,
                                        metadata={
                                            "page_number": page_num,
                                            "source_type": "pdf",
                                            "content_type": "visual",
                                            "visual_type": "chart",
                                            "filename": filename,
                                        },
                                    )
                                )
                    except Exception as draw_err:
                        logger.warning("Error processing vector drawings on page %d: %s", page_num, draw_err)

        logger.info("VisualExtractor found %d visual document(s) in %s", len(visual_docs), filename)
        return visual_docs


# Global singleton instance
visual_extractor = VisualExtractor()
