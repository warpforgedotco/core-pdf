# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, hypot
from statistics import median
from typing import Any

from core_pdf.impl.engine.extraction.ocr.types import (
    OcrImage,
    OcrTextResult,
)
from core_pdf.impl.engine.extraction.ocr.backend import TesseractCtypesBackend


VECTOR_TEXT_DPI = 900
VECTOR_TEXT_MIN_STROKES = 120
VECTOR_TEXT_MIN_SEGMENT_LENGTH = 0.02
VECTOR_TEXT_MAX_SEGMENT_LENGTH = 16.0
VECTOR_TEXT_MIN_LINE_WIDTH = 0.05
VECTOR_TEXT_MAX_LINE_WIDTH = 1.25
VECTOR_TEXT_COMPONENT_CELL = 3.0
VECTOR_TEXT_COMPONENT_PADDING = 0.7
VECTOR_TEXT_REGION_PADDING = 2.0
VECTOR_TEXT_GLYPH_PATH_MAX_WIDTH = 8.0
VECTOR_TEXT_GLYPH_PATH_MAX_HEIGHT = 8.0
VECTOR_TEXT_GLYPH_PATH_STRICT_MAX_DIMENSION = 6.0
VECTOR_TEXT_OCR_VARIABLES = {
    "load_freq_dawg": "0",
    "load_system_dawg": "0",
    "preserve_interword_spaces": "1",
}


@dataclass(frozen=True, slots=True)
class VectorStroke:
    x0: float
    y0: float
    x1: float
    y1: float
    line_width: float
    length: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            min(self.x0, self.x1),
            min(self.y0, self.y1),
            max(self.x0, self.x1),
            max(self.y0, self.y1),
        )


@dataclass(frozen=True, slots=True)
class VectorTextComponent:
    x0: float
    y0: float
    x1: float
    y1: float
    stroke_indexes: tuple[int, ...]

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def mid_y(self) -> float:
        return (self.y0 + self.y1) * 0.5


@dataclass(frozen=True, slots=True)
class VectorTextRegion:
    x0: float
    y0: float
    x1: float
    y1: float
    stroke_indexes: tuple[int, ...]
    component_count: int

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True, slots=True)
class VectorStrokeOcrRegion:
    image: OcrImage
    bbox: tuple[float, float, float, float]
    crop_x0: float = 0.0
    crop_y1: float = 0.0
    scale: float = 1.0


@dataclass(frozen=True, slots=True)
class VectorStrokeOcrLine:
    text: str
    bbox: tuple[float, float, float, float]
    confidence: int | None


@dataclass(frozen=True, slots=True)
class VectorStrokeOcrResult:
    text: str
    confidence: int | None
    lines: tuple[VectorStrokeOcrLine, ...] = ()


def page_has_vector_stroke_text_candidates(page: Any) -> bool:
    try:
        graphics = page.get_graphics()
    except Exception:
        return False
    return any(
        len(strokes) >= VECTOR_TEXT_MIN_STROKES
        for strokes in vector_stroke_candidate_sets(graphics)
    )


def vector_stroke_ocr_text_with_timeout(
    page: Any, timeout: float | None
) -> OcrTextResult:
    result = vector_stroke_ocr_result_with_timeout(page, timeout)
    return OcrTextResult(result.text, result.confidence)


def vector_stroke_ocr_result_with_timeout(
    page: Any, timeout: float | None
) -> VectorStrokeOcrResult:
    del page, timeout
    return VectorStrokeOcrResult("", None)


def vector_stroke_ocr_worker(
    regions: list[VectorStrokeOcrRegion],
    result_queue: Any | None = None,
) -> VectorStrokeOcrResult | None:
    result = VectorStrokeOcrResult("", None)
    try:
        backend = TesseractCtypesBackend.from_system()
        if backend is not None:
            result = vector_stroke_regions_to_text_result(backend, regions)
    except BaseException:
        result = VectorStrokeOcrResult("", None)
    if result_queue is None:
        return result
    try:
        result_queue.put_nowait(result)
    except BaseException:
        pass
    return None


def vector_stroke_images_to_text_result(
    backend: TesseractCtypesBackend, images: list[OcrImage]
) -> OcrTextResult:
    result = vector_stroke_regions_to_text_result(
        backend,
        [VectorStrokeOcrRegion(image, (0.0, 0.0, 0.0, 0.0)) for image in images],
    )
    return OcrTextResult(result.text, result.confidence)


def vector_stroke_regions_to_text_result(
    backend: TesseractCtypesBackend, regions: list[VectorStrokeOcrRegion]
) -> VectorStrokeOcrResult:
    lines: list[str] = []
    line_results: list[VectorStrokeOcrLine] = []
    confidences: list[int] = []
    seen: set[str] = set()
    for region in regions:
        result = best_vector_stroke_region_result(backend, region.image)
        for line_result in vector_stroke_result_lines(region, result):
            line = normalize_vector_stroke_ocr_line(line_result.text)
            if not should_keep_vector_stroke_ocr_line(line, line_result.confidence):
                continue
            key = vector_stroke_ocr_line_key(line)
            if key in seen:
                continue
            seen.add(key)
            lines.append(line)
            line_results.append(
                VectorStrokeOcrLine(line, line_result.bbox, line_result.confidence)
            )
            if line_result.confidence is not None:
                confidences.append(line_result.confidence)
    confidence = (
        int(round(sum(confidences) / len(confidences))) if confidences else None
    )
    return VectorStrokeOcrResult("\n".join(lines), confidence, tuple(line_results))


def best_vector_stroke_region_result(
    backend: TesseractCtypesBackend, image: OcrImage
) -> OcrTextResult:
    primary_layout = backend.image_to_iterator_layout(
        image,
        psm=7,
        resolution=image.resolution or VECTOR_TEXT_DPI,
        variables=VECTOR_TEXT_OCR_VARIABLES,
    )
    primary = vector_stroke_layout_text_result(primary_layout)
    if primary.text.strip():
        return primary
    fallback_layout = backend.image_to_iterator_layout(
        image,
        psm=13,
        resolution=image.resolution or VECTOR_TEXT_DPI,
        variables=VECTOR_TEXT_OCR_VARIABLES,
    )
    fallback = vector_stroke_layout_text_result(fallback_layout)
    if fallback.text.strip():
        return fallback
    return backend.image_to_text_result(
        image,
        psm=13,
        resolution=image.resolution or VECTOR_TEXT_DPI,
        variables=VECTOR_TEXT_OCR_VARIABLES,
    )


def vector_stroke_layout_text_result(layout: Any) -> OcrTextResult:
    if str(getattr(layout, "text", "")).strip():
        return OcrTextResult(
            layout.text,
            layout.confidence,
            line_rows=tuple(layout.textline_rows),
            word_rows=tuple(layout.word_rows),
            symbol_rows=tuple(layout.symbol_rows),
        )
    line_result = vector_stroke_rows_text_result(layout.textline_rows)
    if line_result.text.strip():
        return OcrTextResult(
            line_result.text,
            line_result.confidence,
            line_rows=tuple(layout.textline_rows),
            word_rows=tuple(layout.word_rows),
            symbol_rows=tuple(layout.symbol_rows),
        )
    if not layout.word_rows:
        return OcrTextResult("", None)
    words = sorted(
        layout.word_rows,
        key=lambda row: (
            int(row.get("block_num", 1)),
            int(row.get("par_num", 1)),
            int(row.get("line_num", 1)),
            int(row.get("left", 0)),
        ),
    )
    text = " ".join(str(row.get("text", "")).strip() for row in words).strip()
    confidences = [int(row["conf"]) for row in words if "conf" in row]
    confidence = (
        int(round(sum(confidences) / len(confidences))) if confidences else None
    )
    return OcrTextResult(
        text,
        confidence,
        word_rows=tuple(layout.word_rows),
        symbol_rows=tuple(layout.symbol_rows),
    )


def vector_stroke_rows_text_result(rows: list[dict[str, Any]]) -> OcrTextResult:
    lines = [
        str(row["text"]).strip() for row in rows if str(row.get("text", "")).strip()
    ]
    if not lines:
        return OcrTextResult("", None)
    confidences = [int(row["conf"]) for row in rows if "conf" in row]
    confidence = (
        int(round(sum(confidences) / len(confidences))) if confidences else None
    )
    return OcrTextResult("\n".join(lines), confidence, line_rows=tuple(rows))


def vector_stroke_result_lines(
    region: VectorStrokeOcrRegion, result: OcrTextResult
) -> list[VectorStrokeOcrLine]:
    lines: list[VectorStrokeOcrLine] = []
    for row in result.line_rows:
        line = vector_stroke_row_line(region, row)
        if line is not None:
            lines.append(line)
    if lines:
        return lines
    for rows in vector_stroke_word_row_lines(result.word_rows):
        line = vector_stroke_rows_line(region, rows)
        if line is not None:
            lines.append(line)
    if lines:
        return lines
    return [VectorStrokeOcrLine(result.text, region.bbox, result.confidence)]


def vector_stroke_word_row_lines(
    rows: tuple[dict[str, Any], ...],
) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        try:
            key = (
                int(row.get("block_num", 1)),
                int(row.get("par_num", 1)),
                int(row.get("line_num", 1)),
            )
        except TypeError, ValueError:
            continue
        grouped.setdefault(key, []).append(row)
    return [
        sorted(
            items,
            key=lambda row: (int(row.get("left", 0)), int(row.get("word_num", 0))),
        )
        for items in grouped.values()
    ]


def vector_stroke_row_line(
    region: VectorStrokeOcrRegion, row: dict[str, Any]
) -> VectorStrokeOcrLine | None:
    text = str(row.get("text", "")).strip()
    if not text:
        return None
    bbox = vector_stroke_row_bbox(region, row)
    if bbox is None:
        return None
    confidence = vector_stroke_row_confidence(row)
    return VectorStrokeOcrLine(text, bbox, confidence)


def vector_stroke_rows_line(
    region: VectorStrokeOcrRegion, rows: list[dict[str, Any]]
) -> VectorStrokeOcrLine | None:
    if not rows:
        return None
    text = " ".join(str(row.get("text", "")).strip() for row in rows).strip()
    if not text:
        return None
    boxes = [bbox for row in rows if (bbox := vector_stroke_row_bbox(region, row))]
    if not boxes:
        return None
    confidences = [
        confidence
        for row in rows
        if (confidence := vector_stroke_row_confidence(row)) is not None
    ]
    confidence = (
        int(round(sum(confidences) / len(confidences))) if confidences else None
    )
    return VectorStrokeOcrLine(text, union_float_boxes(boxes), confidence)


def vector_stroke_row_bbox(
    region: VectorStrokeOcrRegion, row: dict[str, Any]
) -> tuple[float, float, float, float] | None:
    if region.scale <= 0:
        return None
    try:
        left = float(row["left"])
        top = float(row["top"])
        width = float(row["width"])
        height = float(row["height"])
    except KeyError, TypeError, ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    x0 = region.crop_x0 + left / region.scale
    x1 = region.crop_x0 + (left + width) / region.scale
    y1 = region.crop_y1 - top / region.scale
    y0 = region.crop_y1 - (top + height) / region.scale
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def vector_stroke_row_confidence(row: dict[str, Any]) -> int | None:
    try:
        return int(round(float(row["conf"])))
    except KeyError, TypeError, ValueError:
        return None


def union_float_boxes(
    boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def normalize_vector_stroke_ocr_line(text: str) -> str:
    if not text:
        return ""
    line = " ".join(text.replace("\f", " ").split())
    return line.strip(" \t\r\n|")


def should_keep_vector_stroke_ocr_line(text: str, confidence: int | None) -> bool:
    if not text:
        return False
    alnum = sum(1 for ch in text if ch.isalnum() or ch == "_")
    if alnum == 0:
        return False
    if len(text) == 1:
        return (confidence if confidence is not None else 0) >= 70
    if alnum == 1 and len(text) <= 3:
        return (confidence if confidence is not None else 0) >= 50
    punctuation = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    if punctuation > max(4, alnum * 2):
        return False
    return (confidence if confidence is not None else 35) >= 8


def vector_stroke_ocr_line_key(text: str) -> str:
    return "".join(ch.casefold() for ch in text if ch.isalnum() or ch == "_")


def vector_stroke_ocr_images(page: Any) -> list[OcrImage]:
    return [region.image for region in vector_stroke_ocr_regions(page)]


def vector_stroke_ocr_regions(page: Any) -> list[VectorStrokeOcrRegion]:
    graphics = page.get_graphics()
    ocr_regions: list[VectorStrokeOcrRegion] = []
    for strokes in vector_stroke_candidate_sets(graphics):
        if len(strokes) < VECTOR_TEXT_MIN_STROKES:
            continue
        regions = vector_text_ocr_regions(vector_text_regions_from_strokes(strokes))
        for region in regions:
            if region.width <= 0.0 or region.height <= 0.0:
                continue
            image = rasterize_vector_text_region(strokes, region)
            crop_x0, crop_y1, scale = vector_text_region_crop_geometry(region)
            ocr_regions.append(
                VectorStrokeOcrRegion(
                    image,
                    (region.x0, region.y0, region.x1, region.y1),
                    crop_x0,
                    crop_y1,
                    scale,
                )
            )
    return ocr_regions


def vector_stroke_candidate_sets(graphics: Any) -> list[list[VectorStroke]]:
    drawing_sets = vector_stroke_candidate_sets_from_drawings(
        getattr(graphics, "drawings", ())
    )
    if drawing_sets:
        return drawing_sets
    strokes = candidate_vector_strokes(getattr(graphics, "lines", ()))
    return [strokes] if strokes else []


def vector_stroke_candidate_sets_from_drawings(
    drawings: Any,
) -> list[list[VectorStroke]]:
    glyph_strokes = candidate_vector_strokes_from_drawings(
        drawings,
        max_path_width=VECTOR_TEXT_GLYPH_PATH_MAX_WIDTH,
        max_path_height=VECTOR_TEXT_GLYPH_PATH_MAX_HEIGHT,
    )
    strict_glyph_strokes = candidate_vector_strokes_from_drawings(
        drawings,
        max_path_width=VECTOR_TEXT_GLYPH_PATH_MAX_WIDTH,
        max_path_height=VECTOR_TEXT_GLYPH_PATH_MAX_HEIGHT,
        max_path_dimension=VECTOR_TEXT_GLYPH_PATH_STRICT_MAX_DIMENSION,
    )
    sets: list[list[VectorStroke]] = []
    seen: set[tuple[tuple[float, float, float, float, float], ...]] = set()
    for strokes in (glyph_strokes, strict_glyph_strokes):
        if len(strokes) < VECTOR_TEXT_MIN_STROKES:
            continue
        key = vector_stroke_set_key(strokes)
        if key in seen:
            continue
        seen.add(key)
        sets.append(strokes)
    if sets:
        return sets
    all_strokes = candidate_vector_strokes_from_drawings(drawings)
    if len(all_strokes) >= VECTOR_TEXT_MIN_STROKES:
        sets.append(all_strokes)
    return sets


def vector_stroke_set_key(
    strokes: list[VectorStroke],
) -> tuple[tuple[float, float, float, float, float], ...]:
    return tuple(
        (
            round(stroke.x0, 3),
            round(stroke.y0, 3),
            round(stroke.x1, 3),
            round(stroke.y1, 3),
            round(stroke.line_width, 3),
        )
        for stroke in strokes
    )


def candidate_vector_strokes_from_drawings(
    drawings: Any,
    *,
    max_path_width: float | None = None,
    max_path_height: float | None = None,
    max_path_dimension: float | None = None,
) -> list[VectorStroke]:
    strokes: list[VectorStroke] = []
    for drawing in drawings:
        kind = getattr(drawing, "kind", None)
        if kind not in {"stroke", "fillstroke"}:
            continue
        if getattr(drawing, "stroke_pattern", None) is not None:
            continue
        stroke_opacity = getattr(drawing, "stroke_opacity", None)
        if stroke_opacity is not None:
            try:
                if float(stroke_opacity) <= 0.0:
                    continue
            except TypeError, ValueError:
                continue
        path = getattr(drawing, "path", None)
        if path is None:
            continue
        if not vector_drawing_path_is_small_enough(
            path,
            max_path_width=max_path_width,
            max_path_height=max_path_height,
            max_path_dimension=max_path_dimension,
        ):
            continue
        try:
            line_width = float(getattr(drawing, "line_width"))
        except AttributeError, TypeError, ValueError:
            continue
        if not (VECTOR_TEXT_MIN_LINE_WIDTH <= line_width <= VECTOR_TEXT_MAX_LINE_WIDTH):
            continue
        try:
            subpaths = path.stroke_subpaths()
        except Exception:
            continue
        for subpath in subpaths:
            for x0, y0, x1, y1 in subpath:
                length = hypot(x1 - x0, y1 - y0)
                if not (
                    VECTOR_TEXT_MIN_SEGMENT_LENGTH
                    <= length
                    <= VECTOR_TEXT_MAX_SEGMENT_LENGTH
                ):
                    continue
                strokes.append(VectorStroke(x0, y0, x1, y1, line_width, length))
    return strokes


def vector_drawing_path_is_small_enough(
    path: Any,
    *,
    max_path_width: float | None,
    max_path_height: float | None,
    max_path_dimension: float | None,
) -> bool:
    if (
        max_path_width is None
        and max_path_height is None
        and max_path_dimension is None
    ):
        return True
    try:
        bbox = path.bbox()
    except Exception:
        return False
    if bbox is None:
        return False
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if max_path_width is not None and width > max_path_width:
        return False
    if max_path_height is not None and height > max_path_height:
        return False
    return max_path_dimension is None or max(width, height) <= max_path_dimension


def candidate_vector_strokes(lines: Any) -> list[VectorStroke]:
    strokes: list[VectorStroke] = []
    for line in lines:
        try:
            x0 = float(line.x0)
            y0 = float(line.y0)
            x1 = float(line.x1)
            y1 = float(line.y1)
            line_width = float(line.line_width)
        except AttributeError, TypeError, ValueError:
            continue
        length = hypot(x1 - x0, y1 - y0)
        if not (
            VECTOR_TEXT_MIN_SEGMENT_LENGTH <= length <= VECTOR_TEXT_MAX_SEGMENT_LENGTH
        ):
            continue
        if not (VECTOR_TEXT_MIN_LINE_WIDTH <= line_width <= VECTOR_TEXT_MAX_LINE_WIDTH):
            continue
        strokes.append(VectorStroke(x0, y0, x1, y1, line_width, length))
    return strokes


def vector_text_regions_from_strokes(
    strokes: list[VectorStroke],
) -> list[VectorTextRegion]:
    components = vector_text_components(strokes)
    regions = merge_vector_text_components(components)
    regions.sort(key=lambda region: (-region.y1, region.x0))
    return regions


def vector_text_ocr_regions(
    regions: list[VectorTextRegion],
) -> list[VectorTextRegion]:
    return sorted(regions, key=lambda region: (-region.y1, region.x0))


def vector_text_components(strokes: list[VectorStroke]) -> list[VectorTextComponent]:
    if not strokes:
        return []
    parent = list(range(len(strokes)))
    rank = [0] * len(strokes)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    boxes: list[tuple[float, float, float, float]] = []
    grid: dict[tuple[int, int], list[int]] = {}
    for index, stroke in enumerate(strokes):
        pad = max(VECTOR_TEXT_COMPONENT_PADDING, stroke.line_width * 2.5)
        box = padded_box(stroke.bbox, pad)
        boxes.append(box)
        x_cell0, y_cell0, x_cell1, y_cell1 = box_cells(box, VECTOR_TEXT_COMPONENT_CELL)
        seen: set[int] = set()
        for x_cell in range(x_cell0, x_cell1 + 1):
            for y_cell in range(y_cell0, y_cell1 + 1):
                for other in grid.get((x_cell, y_cell), ()):
                    if other in seen:
                        continue
                    seen.add(other)
                    if boxes_overlap(box, boxes[other]):
                        union(index, other)
        for x_cell in range(x_cell0, x_cell1 + 1):
            for y_cell in range(y_cell0, y_cell1 + 1):
                grid.setdefault((x_cell, y_cell), []).append(index)

    groups: dict[int, list[int]] = {}
    for index in range(len(strokes)):
        groups.setdefault(find(index), []).append(index)

    components: list[VectorTextComponent] = []
    for indexes in groups.values():
        component = vector_component_from_indexes(strokes, indexes)
        if component is not None:
            components.append(component)
    components.sort(key=lambda component: (-component.mid_y, component.x0))
    return components


def vector_component_from_indexes(
    strokes: list[VectorStroke], indexes: list[int]
) -> VectorTextComponent | None:
    x_values: list[float] = []
    y_values: list[float] = []
    for index in indexes:
        stroke = strokes[index]
        x_values.extend((stroke.x0, stroke.x1))
        y_values.extend((stroke.y0, stroke.y1))
    if not x_values or not y_values:
        return None
    x0 = min(x_values)
    y0 = min(y_values)
    x1 = max(x_values)
    y1 = max(y_values)
    width = x1 - x0
    height = y1 - y0
    if len(indexes) < 2 and width < 2.0 and height < 2.0:
        return None
    if height < 0.5 or height > 24.0:
        return None
    if width < 0.5 or width > 180.0:
        return None
    return VectorTextComponent(x0, y0, x1, y1, tuple(indexes))


def merge_vector_text_components(
    components: list[VectorTextComponent],
) -> list[VectorTextRegion]:
    bands: list[list[VectorTextComponent]] = []
    band_mid_y: list[float] = []
    band_height: list[float] = []
    for component in components:
        for index, band in enumerate(bands):
            if abs(component.mid_y - band_mid_y[index]) <= max(
                2.5,
                min(component.height, band_height[index]) * 0.8,
            ):
                band.append(component)
                band_mid_y[index] = sum(item.mid_y for item in band) / len(band)
                band_height[index] = median(item.height for item in band)
                break
        else:
            bands.append([component])
            band_mid_y.append(component.mid_y)
            band_height.append(component.height)

    regions: list[VectorTextRegion] = []
    for band in bands:
        current: list[VectorTextComponent] = []
        for component in sorted(band, key=lambda item: item.x0):
            if not current:
                current = [component]
                continue
            gap = component.x0 - max(item.x1 for item in current)
            merged_height = median(item.height for item in (*current, component))
            if gap <= max(5.5, merged_height * 2.5):
                current.append(component)
                continue
            append_vector_region(regions, current)
            current = [component]
        if current:
            append_vector_region(regions, current)
    return regions


def append_vector_region(
    regions: list[VectorTextRegion], components: list[VectorTextComponent]
) -> None:
    if not components:
        return
    x0 = min(component.x0 for component in components)
    y0 = min(component.y0 for component in components)
    x1 = max(component.x1 for component in components)
    y1 = max(component.y1 for component in components)
    width = x1 - x0
    height = y1 - y0
    stroke_indexes = tuple(
        index for component in components for index in component.stroke_indexes
    )
    if len(stroke_indexes) < 3:
        return
    if width < 2.0 or height < 1.0 or height > 28.0:
        return
    if width / height < 0.25:
        return
    regions.append(
        VectorTextRegion(
            x0,
            y0,
            x1,
            y1,
            stroke_indexes,
            component_count=len(components),
        )
    )


def rasterize_vector_text_region(
    strokes: list[VectorStroke],
    region: VectorTextRegion,
    *,
    dpi: int = VECTOR_TEXT_DPI,
    padding: float = VECTOR_TEXT_REGION_PADDING,
) -> OcrImage:
    x0, y1, scale = vector_text_region_crop_geometry(
        region,
        dpi=dpi,
        padding=padding,
    )
    y0 = region.y0 - padding
    x1 = region.x1 + padding
    width = max(1, int(round((x1 - x0) * scale)))
    height = max(1, int(round((y1 - y0) * scale)))
    data = bytearray(bytes((255, 255, 255, 255)) * width * height)
    for index in region.stroke_indexes:
        draw_vector_stroke(data, width, height, scale, x0, y1, strokes[index])
    return OcrImage(
        bytes(data),
        width,
        height,
        4,
        width * 4,
        source="vector_stroke_region",
        resolution=dpi,
    )


def vector_text_region_crop_geometry(
    region: VectorTextRegion,
    *,
    dpi: int = VECTOR_TEXT_DPI,
    padding: float = VECTOR_TEXT_REGION_PADDING,
) -> tuple[float, float, float]:
    return (
        region.x0 - padding,
        region.y1 + padding,
        dpi / 72.0,
    )


def draw_vector_stroke(
    data: bytearray,
    image_width: int,
    image_height: int,
    scale: float,
    origin_x: float,
    origin_y: float,
    stroke: VectorStroke,
) -> None:
    x0 = (stroke.x0 - origin_x) * scale
    y0 = (origin_y - stroke.y0) * scale
    x1 = (stroke.x1 - origin_x) * scale
    y1 = (origin_y - stroke.y1) * scale
    dx = x1 - x0
    dy = y1 - y0
    length2 = dx * dx + dy * dy
    half_width = max(1.2, stroke.line_width * scale * 0.55)
    px0 = max(0, int(floor(min(x0, x1) - half_width - 1.0)))
    py0 = max(0, int(floor(min(y0, y1) - half_width - 1.0)))
    px1 = min(image_width, int(ceil(max(x0, x1) + half_width + 1.0)))
    py1 = min(image_height, int(ceil(max(y0, y1) + half_width + 1.0)))
    half_width2 = half_width * half_width
    if length2 <= 1e-9:
        for py in range(py0, py1):
            row = py * image_width * 4
            for px in range(px0, px1):
                if (px + 0.5 - x0) ** 2 + (py + 0.5 - y0) ** 2 <= half_width2:
                    data[row + px * 4 : row + px * 4 + 4] = b"\0\0\0\xff"
        return
    for py in range(py0, py1):
        row = py * image_width * 4
        for px in range(px0, px1):
            t = ((px + 0.5 - x0) * dx + (py + 0.5 - y0) * dy) / length2
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            closest_x = x0 + dx * t
            closest_y = y0 + dy * t
            if (px + 0.5 - closest_x) ** 2 + (py + 0.5 - closest_y) ** 2 <= half_width2:
                data[row + px * 4 : row + px * 4 + 4] = b"\0\0\0\xff"


def padded_box(
    box: tuple[float, float, float, float], padding: float
) -> tuple[float, float, float, float]:
    return (
        box[0] - padding,
        box[1] - padding,
        box[2] + padding,
        box[3] + padding,
    )


def box_cells(
    box: tuple[float, float, float, float], cell_size: float
) -> tuple[int, int, int, int]:
    return (
        int(floor(box[0] / cell_size)),
        int(floor(box[1] / cell_size)),
        int(floor(box[2] / cell_size)),
        int(floor(box[3] / cell_size)),
    )


def boxes_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return (
        left[0] <= right[2]
        and right[0] <= left[2]
        and left[1] <= right[3]
        and right[1] <= left[3]
    )
