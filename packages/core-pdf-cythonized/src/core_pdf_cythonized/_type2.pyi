# SPDX-License-Identifier: AGPL-3.0-only

Contours = list[list[tuple[float, float]]]
Rectangle = tuple[float, float, float, float]
Seac = tuple[int, int, float, float]

def type2_glyph_geometry(
    charstring: bytes,
    local_subrs: tuple[bytes, ...],
    global_subrs: tuple[bytes, ...],
    flatten: bool = ...,
    retain_contours: bool = ...,
) -> tuple[Contours, Rectangle | None, Seac | None, bool]: ...
