# SPDX-License-Identifier: AGPL-3.0-only

from collections.abc import Sequence

def glyph_bitmap_rows(
    contours: Sequence[Sequence[tuple[float, float]]], width: int, height: int
) -> tuple[int, ...] | None: ...
