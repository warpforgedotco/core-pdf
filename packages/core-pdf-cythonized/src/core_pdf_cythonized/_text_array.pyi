# SPDX-License-Identifier: AGPL-3.0-only

from collections.abc import Callable
from typing import Any

class RunAccumulator:
    @property
    def output(self) -> list[Any]: ...
    def __init__(
        self, output: list[Any], word_gap_threshold: Callable[[float, float], float]
    ) -> None: ...
    def append(self, run: Any) -> None: ...
    def flush(self) -> None: ...

def is_visible_text(text: str) -> bool: ...
def text_advance(
    widths: Any,
    default_width: Any,
    glyphs: Any,
    font_size: float,
    char_space: float,
    word_space: float,
    horizontal_scale: float,
) -> float: ...
def show_text_array(
    state: Any, array: Any, decoder: Any, spec_context: tuple[Any, Any]
) -> None: ...
def show_string(state: Any, data: bytes, decoder: Any, spec_context: tuple[Any, Any]) -> None: ...
def show_decoded(
    state: Any, text: str, data: Any, glyphs: Any, decoder: Any, spec_context: tuple[Any, Any]
) -> None: ...
def show_text(
    state: Any, text: str, glyphs: Any, decoder: Any, adv_x: float, adv_y: float, glyph_paint: Any
) -> None: ...
