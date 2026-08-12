from __future__ import annotations

import re
from typing import Any

from core_pdf import PdfDocument
from core_pdf.impl.engine.layout.geometry import flip_rect_vertical

_DATE_ONLY = re.compile(r"^[0-3]?\d[/\-][0-3]?\d[/\-]\d{2,4}$")


def internal_overlap_ratio(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    overlap = width * height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    return overlap / first_area if first_area else 0.0


def internal_uniform(page: Any, box: tuple[float, float, float, float]) -> bool:
    raster = page.render().rasterize(scale=1.0, crop=box)
    pixels = memoryview(raster.pixels).cast("B")
    if not pixels or raster.channels <= 0:
        return False
    first = pixels[: raster.channels].tobytes()
    return all(
        pixels[index : index + raster.channels].tobytes() == first
        for index in range(0, len(pixels), raster.channels)
    )


def inspect(source: Any) -> dict[int, list[dict[str, object]]]:
    """Return x-ray-shaped bad-redaction findings from engine evidence."""
    output: dict[int, list[dict[str, object]]] = {}
    with PdfDocument.open(source) as document:
        for page in document.pages:
            for drawing in page.get_drawings():
                if drawing.fill is None or (drawing.fill_opacity or 0.0) < 1.0:
                    continue
                rectangle = drawing.rect
                if (
                    rectangle is None
                    or rectangle[2] - rectangle[0] < 5
                    or rectangle[3] - rectangle[1] < 5
                ):
                    continue
                covered: list[str] = []
                for run in page.chars:
                    if run.seqno > drawing.seqno + 1 and drawing.seqno != 0:
                        continue
                    for cluster in run.glyph_clusters:
                        if (
                            cluster.ink_bbox is not None
                            and internal_overlap_ratio(cluster.ink_bbox, rectangle) >= 0.5
                        ):
                            covered.append(cluster.text)
                text = "".join(covered).strip()
                if not text or not internal_uniform(page, rectangle):
                    continue
                output.setdefault(page.page_number, []).append(
                    {
                        "bbox": flip_rect_vertical(rectangle, float(page.height)),
                        "text": text,
                    }
                )
    if output and all(
        _DATE_ONLY.fullmatch(str(item["text"]).strip())
        for findings in output.values()
        for item in findings
    ):
        return {}
    return output


__all__ = ("inspect",)
