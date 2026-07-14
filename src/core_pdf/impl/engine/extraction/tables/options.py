# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import (
    Any,
    Mapping,
    Protocol,
    Sequence,
    SupportsFloat,
    SupportsIndex,
    TypeAlias,
)

from core_pdf.impl.engine.extraction.common.ordering import LayoutAnalyzer
from core_pdf.impl.engine.extraction.tables.types import TableCacheKey


class MarkedTextLine(Protocol):
    mid_y: float
    x0: float
    y0: float
    y1: float

    def text(self) -> str: ...


FloatCoord: TypeAlias = str | bytes | bytearray | memoryview | SupportsFloat | SupportsIndex


def parse_float_coord(value: FloatCoord, *, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name} coordinate: {value!r}") from exc


def parse_bbox(
    area: str | Sequence[float] | Sequence[int],
) -> tuple[float, float, float, float]:
    if isinstance(area, str):
        parts = [p.strip() for p in area.split(",")]
        if len(parts) != 4:
            raise ValueError(f"invalid area specification: {area!r}")
        return (
            parse_float_coord(parts[0], name="left"),
            parse_float_coord(parts[1], name="top"),
            parse_float_coord(parts[2], name="right"),
            parse_float_coord(parts[3], name="bottom"),
        )

    if len(area) != 4:
        raise ValueError(f"invalid area specification: {area!r}")

    x0 = parse_float_coord(area[0], name="left")
    y0 = parse_float_coord(area[1], name="top")
    x1 = parse_float_coord(area[2], name="right")
    y1 = parse_float_coord(area[3], name="bottom")
    return (x0, y0, x1, y1)


def parse_table_areas(
    areas: Sequence[str] | Sequence[Sequence[float] | Sequence[int] | str] | None,
) -> list[tuple[float, float, float, float]]:
    if not areas:
        return []
    normalized: list[tuple[float, float, float, float]] = []
    for area in areas:
        x0, y0, x1, y1 = parse_bbox(area)
        normalized.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
    return normalized


def parse_columns(values: Sequence[str] | Sequence[float] | None) -> list[float]:
    if not values:
        return []
    columns: list[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            columns.append(float(value))
            continue
        if not isinstance(value, str):
            raise ValueError(f"invalid columns specification: {value!r}")
        for part in value.split(","):
            part = part.strip()
            if part:
                columns.append(float(part))
    return sorted(set(columns))


def normalize_text_markers(
    value: str | Sequence[str] | None,
) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        markers = [value]
    else:
        markers = [str(marker) for marker in value]
    return [marker for marker in markers if marker]


def find_marked_line(
    lines: Sequence[MarkedTextLine],
    anchors: list[str],
    *,
    reverse: bool = False,
) -> MarkedTextLine | None:
    if not anchors:
        return None

    ordered_lines = sorted(
        lines,
        key=lambda line: (line.mid_y, line.x0) if reverse else (-line.mid_y, line.x0),
    )

    for line in ordered_lines:
        text = line.text().strip()
        if any(anchor in text for anchor in anchors):
            return line
    return None


def intersect_bbox_with_regions(
    bbox: tuple[float, float, float, float],
    regions: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    if not regions:
        return [bbox]

    x0, y0, x1, y1 = bbox
    intersected: list[tuple[float, float, float, float]] = []

    for region in regions:
        region_x0, region_y0, region_x1, region_y1 = region
        ix0 = max(x0, region_x0)
        iy0 = max(y0, region_y0)
        ix1 = min(x1, region_x1)
        iy1 = min(y1, region_y1)

        if ix0 < ix1 and iy0 < iy1:
            intersected.append((ix0, iy0, ix1, iy1))

    return intersected


def derive_table_areas_from_markers(
    visible_runs: list[Any],
    *,
    header_text: list[str],
    footer_text: list[str],
    page_width: float,
    page_height: float,
    table_regions: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    if not (header_text or footer_text) or not visible_runs:
        return []

    page_left = 0.0

    if page_width <= 0:
        max_x = max((run.x1 for run in visible_runs), default=0.0)
        min_x = min((run.x0 for run in visible_runs), default=0.0)
        page_left = min_x
        page_width = max(0.0, max_x - min_x)
    if page_height <= 0:
        max_y = max((run.y1 for run in visible_runs), default=0.0)
        min_y = min((run.y0 for run in visible_runs), default=0.0)
        page_height = max(0.0, max_y - min_y)

    if page_width <= 0 or page_height <= 0:
        return []

    lines = LayoutAnalyzer.cluster_into_lines([run for run in visible_runs if run.text])
    if not lines:
        return []

    header_line = find_marked_line(lines, header_text)
    footer_line = find_marked_line(lines, footer_text, reverse=True)

    if (header_text and header_line is None) or (footer_text and footer_line is None):
        return []

    top = header_line.y0 if header_line is not None else page_height
    bottom = footer_line.y1 if footer_line is not None else 0.0
    if bottom >= top:
        return []

    table_area = (page_left, bottom, page_left + page_width, top)
    return intersect_bbox_with_regions(table_area, table_regions)


def derive_table_areas_from_regions(
    visible_runs: list[Any],
    regions: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    if not visible_runs or not regions:
        return []

    lines = LayoutAnalyzer.cluster_into_lines([run for run in visible_runs if run.text])
    if not lines:
        return regions

    ordered_lines = sorted(lines, key=lambda line: -line.mid_y)
    gaps = [
        ordered_lines[index].y0 - ordered_lines[index + 1].y1
        for index in range(len(ordered_lines) - 1)
        if ordered_lines[index].y0 >= ordered_lines[index + 1].y1
    ]
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0.0
    max_gap = max(25.0, median_gap * 1.75)

    derived: list[tuple[float, float, float, float]] = []
    for region in regions:
        x0, y0, x1, y1 = region
        candidate_lines = [line for line in ordered_lines if line.x1 > x0 and line.x0 < x1]
        seed_indices = [
            index for index, line in enumerate(candidate_lines) if line.y1 >= y0 and line.y0 <= y1
        ]
        if not seed_indices:
            derived.append(region)
            continue

        start = min(seed_indices)
        end = max(seed_indices)

        expanded = 0
        while start > 0:
            previous = candidate_lines[start - 1]
            current = candidate_lines[start]
            gap = previous.y0 - current.y1
            if gap > max_gap or expanded >= 3:
                break
            start -= 1
            expanded += 1

        expanded = 0
        while end + 1 < len(candidate_lines):
            current = candidate_lines[end]
            next_line = candidate_lines[end + 1]
            gap = current.y0 - next_line.y1
            if gap > max_gap or expanded >= 4:
                break
            end += 1
            expanded += 1

        selected = candidate_lines[start : end + 1]
        derived.append(
            (
                min(line.x0 for line in selected),
                min(line.y0 for line in selected),
                max(line.x1 for line in selected),
                max(line.y1 for line in selected),
            )
        )

    return derived


def table_cache_key(
    options: Mapping[str, Any],
) -> TableCacheKey:
    replace_text = options.get("replace_text")
    replace_key = None if replace_text is None else tuple(sorted(replace_text.items()))
    strip_text = options.get("strip_text")
    strip_key: object
    if isinstance(strip_text, (list, tuple)):
        strip_key = tuple(strip_text)
    else:
        strip_key = strip_text
    return (
        options.get("flavor"),
        options.get("detect_header", False),
        options.get("include_span_info", False),
        options.get("header_text", ()),
        options.get("footer_text", ()),
        options.get("columns"),
        options.get("row_tolerance", 2.0),
        options.get("column_tolerance", 8.0),
        options.get("edge_tolerance", 50.0),
        options.get("split_text", False),
        options.get("flag_size", False),
        options.get("copy_text", ()),
        options.get("shift_text", ()),
        strip_key,
        replace_key,
    )


__all__ = (
    "derive_table_areas_from_markers",
    "derive_table_areas_from_regions",
    "normalize_text_markers",
    "parse_columns",
    "parse_table_areas",
    "table_cache_key",
)
