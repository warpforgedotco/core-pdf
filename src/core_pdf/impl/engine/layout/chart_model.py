"""Generic positioned-text models for chart extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from core_pdf.impl.engine.layout.chart_geometry import ChartRegion

_NUMBER = re.compile(r"^[\(\[\-+−]?\s*(?:[$€£¥]\s*)?\d[\d,.]*(?:\s*[KkMmBb%])?\s*[\)\]]?$")


@dataclass(frozen=True, slots=True)
class ChartToken:
    """A normalized positioned text token from a PDF page."""

    text: str
    bbox: tuple[float, float, float, float]
    numeric: bool

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) * 0.5

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) * 0.5


@dataclass(frozen=True, slots=True)
class ChartRow:
    """A chart label associated with one or more nearby values."""

    label: str
    value: str
    position: float


@dataclass(frozen=True, slots=True)
class ChartModel:
    """A region and the positioned rows inferred from its text."""

    region: ChartRegion
    rows: tuple[ChartRow, ...]


def _run_bbox(run: Any) -> tuple[float, float, float, float] | None:
    values: Any
    if hasattr(run, "x0"):
        values = (run.x0, run.y0, run.x1, run.y1)
    else:
        values = getattr(run, "bbox", None)
    if values is None:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _is_number(text: str) -> bool:
    return bool(_NUMBER.fullmatch(text.replace("−", "-")))


def positioned_tokens(runs: Iterable[Any]) -> tuple[ChartToken, ...]:
    """Normalize native runs and structured blocks into deduplicated tokens."""

    tokens: list[ChartToken] = []
    seen: set[tuple[str, tuple[int | float, ...]]] = set()
    for run in runs:
        text = str(getattr(run, "text", "") or "")
        bbox = _run_bbox(run)
        if not text.strip() or bbox is None:
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            continue
        x0, y0, x1, y1 = bbox
        line_height = (y1 - y0) / len(lines)
        for index, line in enumerate(lines):
            line_bbox = (x0, y0 + index * line_height, x1, y0 + (index + 1) * line_height)
            key = (line, tuple(round(value, 3) for value in line_bbox))
            if key in seen:
                continue
            seen.add(key)
            tokens.append(ChartToken(line, line_bbox, _is_number(line)))
    tokens.sort(key=lambda token: (token.bbox[1], token.bbox[0]))
    merged: list[ChartToken] = []
    for token in tokens:
        if merged:
            previous = merged[-1]
            previous_height = previous.bbox[3] - previous.bbox[1]
            token_height = token.bbox[3] - token.bbox[1]
            gap = token.bbox[0] - previous.bbox[2]
            same_baseline = (
                abs(token.center_y - previous.center_y) <= max(previous_height, token_height) * 0.5
            )
            close = gap <= max(previous_height, token_height) * 1.5
            previous_width = previous.bbox[2] - previous.bbox[0]
            token_width = token.bbox[2] - token.bbox[0]
            fragmented = (
                previous_width <= previous_height * 0.5 and token_width <= token_height * 0.5
            )
            if (
                not previous.numeric
                and not token.numeric
                and same_baseline
                and close
                and fragmented
            ):
                separator = "" if gap <= max(previous_height, token_height) * 0.25 else " "
                merged[-1] = ChartToken(
                    previous.text + separator + token.text,
                    (
                        previous.bbox[0],
                        min(previous.bbox[1], token.bbox[1]),
                        max(previous.bbox[2], token.bbox[2]),
                        max(previous.bbox[3], token.bbox[3]),
                    ),
                    False,
                )
                continue
        merged.append(token)
    return tuple(merged)


def _inside(token: ChartToken, region: ChartRegion, padding: float) -> bool:
    x0, y0, x1, y1 = region.bbox
    token_x0, token_y0, token_x1, token_y1 = token.bbox
    return (
        token_x1 >= x0 - padding
        and token_x0 <= x1 + padding
        and token_y1 >= y0 - padding
        and token_y0 <= y1 + padding
    )


def build_chart_model(
    region: ChartRegion,
    tokens: Iterable[ChartToken],
    page_width: float,
    page_height: float,
) -> ChartModel:
    """Associate chart labels with values using the dominant label axis."""

    padding = min(page_width, page_height) * 0.08
    visible = [token for token in tokens if _inside(token, region, padding)]
    values = [token for token in visible if token.numeric]
    labels = [
        token
        for token in visible
        if not token.numeric
        and len(token.text) <= 32
        and not any(char in token.text for char in ".:;!?")
    ]
    if not values or len(labels) < 3:
        return ChartModel(region, ())

    def dominant_band(axis: str) -> list[ChartToken]:
        coordinate = (
            (lambda token: token.center_x) if axis == "x" else (lambda token: token.center_y)
        )
        best: list[ChartToken] = []
        for anchor in labels:
            band = [
                token for token in labels if abs(coordinate(token) - coordinate(anchor)) <= padding
            ]
            if len(band) > len(best):
                best = band
        return best

    horizontal_labels = dominant_band("y")
    vertical_labels = dominant_band("x")
    horizontal = len(horizontal_labels) >= len(vertical_labels)
    labels = horizontal_labels if horizontal else vertical_labels
    region_span = region.bbox[2] - region.bbox[0] if horizontal else region.bbox[3] - region.bbox[1]
    max_distance = max(region_span * 0.16, padding * 0.75)
    rows: list[ChartRow] = []
    for label in labels:
        label_position = label.center_x if horizontal else label.center_y
        candidates = sorted(
            (
                (
                    abs((value.center_x if horizontal else value.center_y) - label_position),
                    value,
                )
                for value in values
                if abs((value.center_x if horizontal else value.center_y) - label_position)
                <= max_distance
            ),
            key=lambda item: (item[0], -item[1].center_y),
        )
        if not candidates:
            continue
        nearest = candidates[0][0]
        value = next(item[1] for item in candidates if item[0] <= nearest + 1e-6)
        rows.append(ChartRow(label.text, value.text, label_position))

    rows.sort(key=lambda row: row.position)
    return ChartModel(region, tuple(rows) if len(rows) >= 3 else ())


__all__ = ("ChartModel", "ChartRow", "ChartToken", "build_chart_model", "positioned_tokens")
