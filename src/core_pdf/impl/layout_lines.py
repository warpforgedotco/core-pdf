# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING

from core_pdf.impl.runs import TextRun
from core_pdf.impl.types import TextWord

if TYPE_CHECKING:
    from core_pdf.impl.runs import (
        LayoutLineText,
        LayoutLineTextSegment,
    )


class LayoutLine:
    __slots__ = (
        "runs",
        "x0",
        "y0",
        "x1",
        "y1",
        "is_vertical",
        "rotation_angle",
        "height",
        "is_all_caps_text",
    )

    runs: list[TextRun]
    x0: float
    y0: float
    x1: float
    y1: float
    is_vertical: bool
    rotation_angle: int
    is_all_caps_text: bool

    def __init__(self, runs: list[TextRun] | None = None) -> None:
        self.runs = run_list = runs if runs is not None else []
        if not run_list:
            self.x0 = self.y0 = self.x1 = self.y1 = 0.0
            self.is_vertical = False
            self.rotation_angle = 0
            self.height = 0.0
            self.is_all_caps_text = True
            return

        first_run = run_list[0]
        x0 = first_run.x0
        y0 = first_run.y0
        x1 = first_run.x1
        y1 = first_run.y1
        is_all_caps_text = not first_run.has_text or first_run.text_is_upper

        for run in islice(run_list, 1, None):
            run_x0 = run.x0
            run_y0 = run.y0
            run_x1 = run.x1
            run_y1 = run.y1
            if run_x0 < x0:
                x0 = run_x0
            if run_y0 < y0:
                y0 = run_y0
            if run_x1 > x1:
                x1 = run_x1
            if run_y1 > y1:
                y1 = run_y1
            if is_all_caps_text and run.has_text and not run.text_is_upper:
                is_all_caps_text = False

        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.is_vertical = first_run.is_vertical
        self.rotation_angle = first_run.rotation_angle
        self.height = y1 - y0
        self.is_all_caps_text = is_all_caps_text

    def reconstructed_text(self) -> LayoutLineText:
        from core_pdf.impl.layout_reconstruction import reconstruct_layout_line_text

        return reconstruct_layout_line_text(self.runs, is_all_caps_text=self.is_all_caps_text)

    def text_and_words(
        self, reconstructed: LayoutLineText | None = None
    ) -> tuple[str, tuple[TextWord, ...]]:
        if reconstructed is None:
            reconstructed = self.reconstructed_text()
        parts: list[str] = []
        words: list[TextWord] = []
        word = ""
        word_x0 = word_y0 = word_x1 = word_y1 = 0.0
        append_part = parts.append
        append_word = words.append

        def flush_word() -> None:
            nonlocal word, word_x0, word_y0, word_x1, word_y1
            if not word:
                return
            append_word(TextWord(word, (word_x0, word_y0, word_x1, word_y1)))
            word = ""

        def append_space() -> None:
            if parts and parts[-1] == " ":
                return
            flush_word()
            append_part(" ")

        for segment in reconstructed.segments:
            if segment.separator_before:
                append_space()
            text = segment.text
            text_length = len(text)
            for index, char in enumerate(text):
                if char.isspace():
                    # The bounding box is only ever used to grow a word, so a
                    # space does not need one computed and thrown away.
                    append_space()
                    continue
                # extend_word, inlined: this ran once per character of the
                # page, and a call plus five nonlocal cells is most of what it
                # cost. flush_word stays a closure -- it runs per word.
                bx0, by0, bx1, by1 = layout_line_segment_char_bbox(segment, index, text_length)
                if word:
                    if bx0 < word_x0:
                        word_x0 = bx0
                    if by0 < word_y0:
                        word_y0 = by0
                    if bx1 > word_x1:
                        word_x1 = bx1
                    if by1 > word_y1:
                        word_y1 = by1
                else:
                    word_x0, word_y0, word_x1, word_y1 = bx0, by0, bx1, by1
                word += char
                append_part(char)

        flush_word()
        return "".join(parts).rstrip(), tuple(words)


def layout_line_segment_char_bbox(
    segment: LayoutLineTextSegment,
    index: int,
    text_length: int,
) -> tuple[float, float, float, float]:
    if text_length <= 1:
        return segment.advance_bbox
    x0, y0, x1, y1 = segment.advance_bbox
    if segment.rotation_angle in (90, 270):
        step = (y1 - y0) / text_length
        char_y0 = y0 + step * index
        return (x0, char_y0, x1, char_y0 + step)
    step = (x1 - x0) / text_length
    char_x0 = x0 + step * index
    return (char_x0, y0, char_x0 + step, y1)
