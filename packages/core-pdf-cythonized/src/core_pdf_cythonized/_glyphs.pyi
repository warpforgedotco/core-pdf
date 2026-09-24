# SPDX-License-Identifier: AGPL-3.0-only

Rectangle = tuple[float, float, float, float]
Matrix6 = tuple[float, float, float, float, float, float]

def horizontal_glyph_geometry(
    offsets: list[float],
    advances: list[float],
    glyph_boxes: list[float],
    *,
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
]: ...
