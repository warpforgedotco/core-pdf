# SPDX-License-Identifier: AGPL-3.0-only

from collections.abc import Callable
from typing import Any

Rectangle = tuple[float, float, float, float]
Matrix6 = tuple[float, float, float, float, float, float]

def horizontal_glyph_geometry(
    offsets: list[float],
    advances: list[float],
    glyph_boxes: list[float],
    basis: Matrix6,
    font_ascent: float,
    font_descent: float,
    rise: float,
    font_scale: float,
    advance_scale: float,
    font_size: float,
    clip_primary: Rectangle | None,
    clip_page: Rectangle | None,
    visible: bool,
    want_bitmap: list[int],
    want_transform: bool = ...,
) -> tuple[
    list[Rectangle],
    list[Rectangle],
    list[Matrix6 | None],
    list[Rectangle],
    list[int],
    list[int],
    Rectangle | None,
    Rectangle | None,
]: ...

DECODED_GLYPH_FIELDS: tuple[str, ...]
OBSERVATION_FIELDS: tuple[str, ...]

class SlotLayout:
    cls: type
    count: int
    def __init__(self, cls: type, names: tuple[str, ...]) -> None: ...
    def build(self, values: tuple[Any, ...]) -> Any: ...

def capture_horizontal_glyphs(
    text: str,
    glyphs: tuple[Any, ...],
    glyph_layout: SlotLayout,
    observation_layout: SlotLayout,
    glyph_width: Callable[[int], float],
    glyph_bbox: Callable[[int], Rectangle | None] | None,
    font_size: float,
    char_space: float,
    word_space: float,
    horizontal_scale: float,
    want_render: bool,
    want_runs: bool,
    basis: Matrix6,
    font_ascent: float,
    font_descent: float,
    rise: float,
    font_scale: float,
    advance_scale: float,
    clip_primary: Rectangle | None,
    clip_page: Rectangle | None,
    visible: bool,
    style: object,
    seqno: int,
    font_name: str | None,
    cluster_start: int,
    confidence_of: Callable[[str, str, tuple[str, ...]], float],
    suspicious_multi: Callable[[str], bool],
    bitmap_labels: frozenset[str],
    out_glyphs: list[Any],
    out_clusters: list[Any],
) -> tuple[int, Rectangle | None, Rectangle | None, float | None] | None: ...
