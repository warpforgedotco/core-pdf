# SPDX-License-Identifier: AGPL-3.0-only

from collections.abc import Callable, Sequence
from typing import Any

def plumber_char_dicts(
    glyphs: Sequence[Any],
    marks: Sequence[Any],
    skipped: set[int],
    ligatures: dict[int, Any],
    embedded_unusable: Callable[[Any], bool],
    glyph_text: Callable[[Any], str],
    normalized_width: Callable[[Any, str | None], float],
    descent_of: Callable[[Any], float],
    font_name: Callable[[Any], Any],
    ctm: tuple[float, ...],
    page_number: int,
    media_x0_value: float,
    media_top_value: float,
    page_height_value: float,
    doctop_value: float,
    nfc: bool,
) -> list[dict[str, Any]] | None: ...
