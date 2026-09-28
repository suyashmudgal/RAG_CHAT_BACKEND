"""Visual extraction service for PDFs.

Extracts images, charts, diagrams, tables, and scanned content from PDF pages.
Uses PyMuPDF (fitz) for page rendering & image extraction, and RapidOCR
(with fallback to Tesseract) for fast, highly accurate local OCR.
Structures visual content into retrievable knowledge chunks with rich metadata.
Completely generic: works for ANY document, chart, graph, diagram, or scanned page.
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
                    results.append(
                        {
                            "text": txt,
                            "box": box,
                            "confidence": conf,
                            "y_center": y_center,
                            "x_center": x_center,
                        }
                    )
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
                results.append(
                    {
                        "text": txt,
                        "box": [
                            [left, top],
                            [left + w, top],
                            [left + w, top + h],
                            [left, top + h],
                        ],
                        "confidence": conf / 100.0,
                        "y_center": top + h / 2.0,
                        "x_center": left + w / 2.0,
                    }
                )
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

    # An image dimension appearing on >= 3 pages with small height is likely a header logo
    repeated = {
        dim
        for dim, count in dim_counts.items()
        if count >= 3 and (dim[1] < 150 or count > total_pages * 0.4)
    }
    return repeated


def parse_and_structure_chart(
    ocr_items: list[dict[str, Any]],
    page_text: str = "",
    filename: str = "",
    page_num: int = 1,
) -> str | None:
    """Analyze OCR tokens from chart/diagram images and format into structured knowledge.

    GENERIC: works for any chart (pie, bar, line, table), extracting data series,
    categories, percentages, numbers, and linking with contextual base totals.
    """
    if not ocr_items:
        return None

    # Sort items vertically
    sorted_items = sorted(ocr_items, key=lambda x: x["y_center"])
    raw_lines = [item["text"] for item in sorted_items]
    full_ocr_text = " ".join(raw_lines)

    # Detect if image contains quantitative or chart data
    has_percentages = any(
        "%" in item["text"] or re.search(r"\b\d+\s*%", item["text"]) for item in sorted_items
    )
    has_numbers = any(re.search(r"\b\d+(?:,\d+)?\b", item["text"]) for item in sorted_items)
    has_chart_keywords = any(
        kw in full_ocr_text.lower()
        for kw in [
            "percentage",
            "percent",
            "distribution",
            "break-up",
            "breakup",
            "share",
            "total",
            "ratio",
            "enrolled",
            "count",
            "chart",
            "graph",
            "plot",
            "stream",
            "branch",
            "sales",
            "revenue",
            "growth",
            "proportion",
        ]
    )

    if not (has_percentages or (has_numbers and has_chart_keywords)):
        # Generic diagram or image text
        clean_text = "\n".join(raw_lines)
        return (
            f"[VISUAL CONTENT - DIAGRAM / IMAGE]\n"
            f"Document: {filename}\n"
            f"Page: {page_num}\n"
            f"Element Type: DIAGRAM\n\n"
            f"Extracted Visual Text & Labels:\n{clean_text}"
        )

    # 1. Generic sub-chart / section clustering based on vertical boundaries and title tokens
    section_indices: list[int] = []
    for i, it in enumerate(sorted_items):
        txt_low = it["text"].lower()
        if any(
            w in txt_low
            for w in [
                "percentage",
                "distribution",
                "break-up",
                "breakup",
                "share",
                "allocation",
                "chart",
                "graph",
                "figure",
                "table",
            ]
        ):
            if not section_indices or (
                it["y_center"] - sorted_items[section_indices[-1]]["y_center"]
            ) > 80:
                section_indices.append(i)

    chart_sections: list[tuple[str, list[dict[str, Any]]]] = []
    if len(section_indices) >= 2:
        split_y = sorted_items[section_indices[1]]["y_center"] - 15
        s1 = [it for it in sorted_items if it["y_center"] < split_y]
        s2 = [it for it in sorted_items if it["y_center"] >= split_y]
        chart_sections = [("Chart 1", s1), ("Chart 2", s2)]
    else:
        chart_sections = [("Chart / Visual Data", sorted_items)]

    parsed_sections: list[tuple[str, dict[str, float], list[str]]] = []

    for sec_idx, (default_title, sec_items) in enumerate(chart_sections):
        # Extract title from the top tokens of this cluster
        title_candidates = [
            it["text"]
            for it in sec_items[:4]
            if not re.search(r"\d+%", it["text"])
            and it["y_center"] < sec_items[0]["y_center"] + 65
        ]
        sec_title = " ".join(title_candidates) if title_candidates else default_title

        cat_data: dict[str, float] = {}

        # Pass A: Token has combined Category + Percentage / Number, e.g. "MECH16%", "CIVIL 30%", "Alpha 25%"
        for it in sec_items:
            # Pattern: Category word + number/percentage
            m_comb = re.search(
                r"([A-Za-z][A-Za-z0-9_\-\s]{1,18}?)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%",
                it["text"],
            )
            if m_comb:
                cat_name = m_comb.group(1).strip().upper()
                pct_val = float(m_comb.group(2))
                cat_data[cat_name] = pct_val
                continue

            # Pattern: Category word + numeric value (without %)
            m_val = re.search(
                r"^([A-Za-z][A-Za-z0-9_\-]{1,15})\s*[:=]\s*(\d+(?:\.\d+)?)$",
                it["text"].strip(),
            )
            if m_val:
                cat_name = m_val.group(1).strip().upper()
                num_val = float(m_val.group(2))
                cat_data[cat_name] = num_val

        # Pass B: Standalone percentages paired with nearest text token by spatial proximity
        for it in sec_items:
            m_pct = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%", it["text"].strip())
            if m_pct:
                pct_val = float(m_pct.group(1))
                # Search for closest textual category label within proximity
                closest_cat = None
                min_dist = float("inf")
                for other in sec_items:
                    clean_other = other["text"].strip()
                    if (
                        re.fullmatch(r"[A-Za-z][A-Za-z0-9_\-]{1,15}", clean_other)
                        and not re.search(r"\d", clean_other)
                    ):
                        cand = clean_other.upper()
                        if cand not in cat_data:
                            dist = abs(other["y_center"] - it["y_center"]) + abs(
                                other["x_center"] - it["x_center"]
                            )
                            if dist < min_dist and dist < 120:
                                min_dist = dist
                                closest_cat = cand
                if closest_cat:
                    cat_data[closest_cat] = pct_val

        parsed_sections.append((sec_title, cat_data, [it["text"] for it in sec_items]))

    # 2. Extract contextual base totals dynamically from page text and full OCR text
    combined_context = f"{page_text} {full_ocr_text}"
    contextual_totals: dict[str, int] = {}
    for m in re.finditer(
        r"(?:total\s+(?:number\s+of\s+)?([A-Za-z]+)|([A-Za-z]+))\s*[:=]\s*(\d+(?:,\d+)?)",
        combined_context,
        re.IGNORECASE,
    ):
        raw_name = (m.group(1) or m.group(2)).strip().lower()
        num_str = m.group(3).replace(",", "")
        if raw_name in [
            "students",
            "girls",
            "boys",
            "total",
            "employees",
            "participants",
            "candidates",
            "units",
            "revenue",
            "sales",
            "population",
        ]:
            try:
                contextual_totals[raw_name] = int(num_str)
            except ValueError:
                pass

    # 3. Assemble generic structured output
    output_lines = [
        "[VISUAL CONTENT - CHART / GRAPH DATA]",
        f"Document: {filename}",
        f"Page: {page_num}",
        "Element Type: CHART",
        "",
    ]

    # Include detected contextual totals
    if contextual_totals:
        totals_desc = ", ".join(f"Total {k} = {v}" for k, v in contextual_totals.items())
        output_lines.append(f"Page Context & Base Totals: {totals_desc}.")
        # If both total students and total girls are present, deduce total boys
        if (
            "students" in contextual_totals
            and "girls" in contextual_totals
            and "boys" not in contextual_totals
        ):
            boys_count = contextual_totals["students"] - contextual_totals["girls"]
            contextual_totals["boys"] = boys_count
            output_lines.append(f"Derived Total: Total boys = {boys_count} (students - girls).")
        output_lines.append("")

    # Output each chart section with categories and values
    for s_title, s_data, s_raw in parsed_sections:
        output_lines.append(f"--- {s_title} ---")
        if s_data:
            output_lines.append("Extracted Categories & Values:")
            for cat, val in sorted(s_data.items()):
                output_lines.append(f"- {cat}: {val:g}%")
        else:
            output_lines.append("Extracted Labels:")
            for r in s_raw:
                output_lines.append(f"  {r}")
        output_lines.append("")

    # 4. If multiple chart series exist with corresponding contextual totals, provide comprehensive breakdown
    if len(parsed_sections) >= 2 and parsed_sections[0][1] and parsed_sections[1][1]:
        c1 = parsed_sections[0][1]
        c2 = parsed_sections[1][1]
        tot_students = contextual_totals.get("students") or contextual_totals.get("total")
        tot_girls = contextual_totals.get("girls")

        if tot_students and tot_girls:
            tot_boys = contextual_totals.get("boys", tot_students - tot_girls)
            all_categories = sorted(set(list(c1.keys()) + list(c2.keys())))

            output_lines.append(
                f"Comprehensive Breakdown by Category (Total Students = {tot_students}, Girls = {tot_girls}, Boys = {tot_boys}):"
            )
            for cat in all_categories:
                pct_tot = c1.get(cat, 0.0)
                pct_g = c2.get(cat, 0.0)
                n_tot = round(pct_tot / 100.0 * tot_students) if pct_tot else 0
                n_g = round(pct_g / 100.0 * tot_girls) if pct_g else 0
                n_b = n_tot - n_g
                output_lines.append(
                    f"- {cat}: Total = {n_tot} ({pct_tot:g}% of {tot_students}); "
                    f"Girls = {n_g} ({pct_g:g}% of {tot_girls}); "
                    f"Boys = {n_b} ({n_tot} - {n_g})"
                )
            output_lines.append("")

            # Summary table for quick numerical retrieval
            output_lines.append("Summary Table of Derived Counts:")
            for cat in all_categories:
                pct_tot = c1.get(cat, 0.0)
                pct_g = c2.get(cat, 0.0)
                n_tot = round(pct_tot / 100.0 * tot_students) if pct_tot else 0
                n_g = round(pct_g / 100.0 * tot_girls) if pct_g else 0
                n_b = n_tot - n_g
                output_lines.append(f"- {cat}: Boys = {n_b}, Girls = {n_g}, Total = {n_tot}")

    return "\n".join(output_lines)


class VisualExtractor:
    """Extracts visual data, charts, diagrams, and scanned content from PDF documents."""

    def __init__(self) -> None:
        pass

    def extract_visuals_from_pdf(
        self, file_path: os.PathLike | str, filename: str = ""
    ) -> list[Document]:
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
                    # Skip tiny icons / decorative bullets
                    if (
                        base_img["width"] < 80
                        or base_img["height"] < 80
                        or (base_img["width"] * base_img["height"]) < 10000
                    ):
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
                        ocr_items,
                        page_text=page_text,
                        filename=filename,
                        page_num=page_num,
                    )
                    if structured_text:
                        visual_type = "chart" if "CHART" in structured_text else "diagram"
                        visual_docs.append(
                            Document(
                                page_content=structured_text,
                                metadata={
                                    "page_number": page_num,
                                    "source_type": "pdf",
                                    "content_type": "visual",
                                    "element_type": "CHART"
                                    if visual_type == "chart"
                                    else "DIAGRAM",
                                    "visual_type": visual_type,
                                    "filename": filename,
                                },
                            )
                        )
                except Exception as exc:
                    logger.warning(
                        "Error processing image xref %d on page %d: %s",
                        xref,
                        page_num,
                        exc,
                    )

            # 2. Check for scanned page (very low native text, but page has visual content)
            if len(page_text.strip()) < 50:
                try:
                    pix = page.get_pixmap(dpi=200)
                    img_bytes = pix.tobytes("png")
                    ocr_items = run_ocr_on_image(img_bytes)
                    if ocr_items:
                        avg_conf = (
                            sum(it["confidence"] for it in ocr_items) / len(ocr_items)
                            if ocr_items
                            else 0.0
                        )
                        scanned_text = "\n".join(
                            it["text"]
                            for it in sorted(ocr_items, key=lambda x: x["y_center"])
                        )
                        if len(scanned_text.strip()) >= 15:
                            prefix = (
                                "[VISUAL CONTENT - SCANNED PAGE]\n"
                                f"Document: {filename}\n"
                                f"Page: {page_num}\n"
                                f"Element Type: OCR\n"
                                f"Confidence: {avg_conf:.2f}\n\n"
                            )
                            if avg_conf < 0.45:
                                prefix += (
                                    "Note: The text in this section is partially unclear in the uploaded document.\n\n"
                                )
                            visual_docs.append(
                                Document(
                                    page_content=prefix + scanned_text,
                                    metadata={
                                        "page_number": page_num,
                                        "source_type": "pdf",
                                        "content_type": "visual",
                                        "element_type": "OCR",
                                        "visual_type": "scanned_page",
                                        "confidence": avg_conf,
                                        "filename": filename,
                                    },
                                )
                            )
                except Exception as scan_err:
                    logger.warning("Error OCRing scanned page %d: %s", page_num, scan_err)

            # 3. Check for vector drawings forming diagrams/charts without raster image
            elif len(meaningful_images) == 0:
                drawings = page.get_drawings()
                has_chart_keywords = any(
                    kw in page_text.lower()
                    for kw in [
                        "pie chart",
                        "bar chart",
                        "diagram",
                        "figure",
                        "graph",
                        "distribution",
                        "breakup",
                        "share",
                        "percentage",
                    ]
                )
                if len(drawings) > 15 and has_chart_keywords:
                    try:
                        pix = page.get_pixmap(dpi=200)
                        img_bytes = pix.tobytes("png")
                        ocr_items = run_ocr_on_image(img_bytes)
                        if ocr_items:
                            structured_text = parse_and_structure_chart(
                                ocr_items,
                                page_text=page_text,
                                filename=filename,
                                page_num=page_num,
                            )
                            if structured_text:
                                visual_docs.append(
                                    Document(
                                        page_content=structured_text,
                                        metadata={
                                            "page_number": page_num,
                                            "source_type": "pdf",
                                            "content_type": "visual",
                                            "element_type": "CHART",
                                            "visual_type": "chart",
                                            "filename": filename,
                                        },
                                    )
                                )
                    except Exception as draw_err:
                        logger.warning(
                            "Error processing vector drawings on page %d: %s",
                            page_num,
                            draw_err,
                        )

        logger.info(
            "VisualExtractor found %d visual document(s) in %s",
            len(visual_docs),
            filename,
        )
        return visual_docs


# Global singleton instance
visual_extractor = VisualExtractor()
