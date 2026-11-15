# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

class CharGapRules:
    def __init__(
        self,
        estimated_char_width: float | None,
        column_gap_threshold: float,
        is_table_like_line: bool,
        has_explicit_spaces: bool,
        is_all_caps_line: bool,
        is_tracked_glyph_line: bool,
        tracked_word_gap: float | None,
        explicit_spaces_control_glyph_gaps: bool,
    ) -> None: ...
    def separator(
        self,
        prev_run: Any,
        run: Any,
        prev_char: str,
        char: str,
        prev_box: tuple[float, float, float, float],
        box: tuple[float, float, float, float],
        prev_baseline: tuple[float, float, float, float] | None,
        baseline: tuple[float, float, float, float] | None,
        both_glyphs: bool,
    ) -> str | None: ...
