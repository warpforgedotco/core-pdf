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
def no_char() -> object: ...
def pdfminer_chars(
    glyphs: Sequence[Any],
    glyph_matrices: dict[int, Any],
    strict_resources: bool,
    skipped: set[int],
    ligatures: dict[int, Any],
    embedded_unusable: Callable[[Any], bool],
    is_clipped: Callable[[dict[str, Any]], bool],
    glyph_text: Callable[[Any], str],
    projected_widths: Callable[[Any, int | None, str | None], tuple[float | None, float]],
    descent_of: Callable[[Any], float],
    font_name: Callable[[Any], Any],
    char_type: Callable[..., Any],
    empty_glyph_text: str,
) -> list[Any]: ...
