"""Reusable page-layout and chart geometry analysis."""

from __future__ import annotations

import html
from typing import Any

from core_pdf.impl.engine.layout.chart_geometry import detect_chart_regions
from core_pdf.impl.engine.layout.chart_model import build_chart_model, positioned_tokens
from core_pdf.impl.engine.structured import Block, BlockKind, Figure, Table
from core_pdf.impl.engine.structured.serialization import table_to_html, table_to_markdown


def layout_prediction(
    element: Any,
    page_width: float,
    page_height: float,
    page_number: int = 1,
) -> dict[str, Any] | None:
    """Return a normalized layout-detection record for one page element."""

    bbox = getattr(element, "bbox", None)
    if bbox is None or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = (float(value) for value in bbox)
    if x1 <= x0 or y1 <= y0:
        return None
    text = str(getattr(element, "text", "") or "")
    clipped_x0 = max(0.0, min(page_width, x0))
    clipped_x1 = max(0.0, min(page_width, x1))
    clipped_y0 = max(0.0, min(page_height, page_height - y1))
    clipped_y1 = max(0.0, min(page_height, page_height - y0))
    if clipped_x1 <= clipped_x0 or clipped_y1 <= clipped_y0:
        return None
    if isinstance(element, Table):
        kind = "table"
    elif isinstance(element, Figure):
        kind = element.kind
    else:
        kind = getattr(getattr(element, "kind", None), "value", "paragraph")
    normalized_top = max(0.0, min(1.0, (page_height - y1) / page_height))
    normalized_bottom = max(0.0, min(1.0, (page_height - y0) / page_height))
    if text and normalized_top <= 0.07:
        label = "page-header"
    elif text and normalized_bottom >= 0.93:
        label = "page-footer"
    elif (
        kind == BlockKind.HEADING
        or kind == "heading"
        or isinstance(element, Block)
        and any(line.bold for line in element.lines)
    ):
        label = "section_header"
    elif kind == "table":
        label = "table"
    elif kind in {"image", "figure"}:
        label = "picture"
    else:
        label = "text"
    content: dict[str, Any] | None = None
    if label == "table":
        content = {"type": "table", "html": table_to_html(element)}
    elif text:
        content = {"type": "text", "text": text}
    return {
        "bbox": [clipped_x0, clipped_y0, clipped_x1, clipped_y1],
        "score": float(getattr(element, "confidence", None) or 0.8),
        "label": label,
        "page": page_number,
        "content": content,
    }


def layout_item(element: Any, page_width: float, page_height: float) -> dict[str, Any]:
    """Return the normalized layout-page item for one structured element."""

    bbox = getattr(element, "bbox", None)
    x = y = w = h = 0.0
    if bbox is not None and len(bbox) == 4 and page_width > 0 and page_height > 0:
        x0, y0, x1, y1 = (float(value) for value in bbox)
        x = max(0.0, min(1.0, x0 / page_width))
        y = max(0.0, min(1.0, (page_height - y1) / page_height))
        w = max(0.0, min(1.0 - x, (x1 - x0) / page_width))
        h = max(0.0, min(1.0 - y, (y1 - y0) / page_height))

    if isinstance(element, Table):
        item_type = "table"
        html_output = table_to_html(element)
        value = "\n".join("\t".join(cell.text for cell in row) for row in element.rows)
        markdown = table_to_markdown(element)
    elif isinstance(element, Figure):
        item_type = "picture"
        html_output = ""
        value = ""
        markdown = ""
    else:
        item_type = getattr(getattr(element, "kind", None), "value", "text")
        if item_type == "heading":
            item_type = "section_header"
        value = str(getattr(element, "text", "") or "")
        html_output = ""
        markdown = value
    return {
        "type": item_type,
        "md": markdown,
        "html": html_output,
        "value": value,
        "bbox": {"x": x, "y": y, "w": w, "h": h, "label": item_type},
    }


def page_layout_items(page: Any) -> list[dict[str, Any]]:
    """Return layout items, omitting tables from layout-only evaluation."""

    width = float(page.width)
    height = float(page.height)
    return [
        layout_item(element, width, height)
        for element in page.elements
        if not isinstance(element, Table)
    ]


def page_plain_text(page: Any) -> str:
    """Return the native block text without table or figure serialization."""

    return "\n\n".join(element.text for element in page.elements if isinstance(element, Block))


def page_layout_predictions(page: Any, page_number: int) -> list[dict[str, Any]]:
    """Return normalized layout predictions for a structured page."""

    return [
        prediction
        for element in getattr(page, "elements", ())
        if not isinstance(element, Table)
        and (
            prediction := layout_prediction(
                element, float(page.width), float(page.height), page_number
            )
        )
        is not None
    ]


def chart_table_html(document: Any, extracted: Any | None = None) -> str:
    """Serialize generic positioned chart models as HTML tables."""

    tables: list[str] = []
    for page_index, page in enumerate(document.pages):
        program = page.get_page_program()
        structured_page = (
            extracted.pages[page_index]
            if extracted is not None and page_index < len(extracted.pages)
            else None
        )
        regions = detect_chart_regions(
            lines=program.products.lines,
            drawings=program.products.drawings,
            figures=getattr(structured_page, "figures", ()) if structured_page else (),
            page_width=float(page.width),
            page_height=float(page.height),
        )
        runs = [run for run in program.products.runs if run.text.strip()]
        if structured_page is not None and len(runs) < 6:
            runs.extend(
                element
                for element in structured_page.elements
                if getattr(element, "text", "") and getattr(element, "bbox", None) is not None
            )
        tokens = positioned_tokens(runs)
        for region in regions:
            model = build_chart_model(region, tokens, float(page.width), float(page.height))
            if not model.rows:
                continue
            tables.append(
                "<table><thead><tr><th>Label</th><th>Value</th></tr></thead><tbody>"
                + "".join(
                    f"<tr><td>{html.escape(row.label)}</td><td>{html.escape(row.value)}</td></tr>"
                    for row in model.rows
                )
                + "</tbody></table>"
            )
    return "\n".join(tables)


__all__ = (
    "chart_table_html",
    "layout_item",
    "layout_prediction",
    "page_plain_text",
    "page_layout_items",
    "page_layout_predictions",
)
