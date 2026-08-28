# SPDX-License-Identifier: AGPL-3.0-only
"""Grouped layout lines and their cached text reconstruction.

A ``LayoutLine`` is what the line-grouping heuristics produce, not what capture
emits, so it lives here rather than with the capture records in ``impl/capture_model``.
"""

from __future__ import annotations

from itertools import islice
from typing import TypeAlias

from core_pdf.impl.capture_model.line_text import (
    LayoutLineText,
    LayoutLineTextSegment,
    LayoutWordSnapshot,
)
from core_pdf.impl.capture_model.runs import TextRun, internal_track_text_run

LayoutLineReconstructionKey: TypeAlias = tuple[
    bool | None,
    tuple[tuple[TextRun, int, tuple[float, ...]], ...],
]


def reconstruct_cached_layout_line_text(
    runs: list[TextRun],
    *,
    is_all_caps_text: bool | None = None,
    internal_key: LayoutLineReconstructionKey | None = None,
) -> LayoutLineText:
    """Reconstruct a line once for every revision of its constituent runs."""
    from core_pdf.impl.engine.layout.text_lines import reconstruct_layout_line_text

    key: LayoutLineReconstructionKey = (
        internal_key
        if internal_key is not None
        else (
            is_all_caps_text,
            tuple((run, run.internal_revision, tuple(run.coords)) for run in runs),
        )
    )
    first_run = runs[0] if runs else None
    shared_cache = first_run.internal_layout_reconstruction_cache if first_run is not None else None
    if shared_cache is not None and shared_cache[0] == key:
        return shared_cache[1]
    reconstructed = reconstruct_layout_line_text(runs, is_all_caps_text=is_all_caps_text)
    if first_run is not None:
        internal_track_text_run(first_run)
        object.__setattr__(first_run, "internal_layout_reconstruction_cache", (key, reconstructed))
    return reconstructed


class LayoutLine:
    __slots__ = (
        "internal_reconstructed_cache",
        "internal_reconstructed_cache_key",
        "runs",
        "x0",
        "y0",
        "x1",
        "y1",
        "is_vertical",
        "rotation_angle",
        "max_order",
        "max_depth",
        "min_order",
        "mid_y",
        "height",
        "max_font_size",
        "is_all_caps_text",
    )

    runs: list[TextRun]
    x0: float
    y0: float
    x1: float
    y1: float
    is_vertical: bool
    rotation_angle: int
    max_order: int
    max_depth: int
    min_order: int
    mid_y: float
    max_font_size: float
    is_all_caps_text: bool

    def __init__(
        self,
        runs: list[TextRun] | None = None,
        x0: float = 0.0,
        y0: float = 0.0,
        x1: float = 0.0,
        y1: float = 0.0,
        is_vertical: bool = False,
        rotation_angle: int = 0,
        max_order: int = -1,
        max_depth: int = -1,
        min_order: int = 999999,
        mid_y: float = 0.0,
        height: float = 0.0,
        max_font_size: float = 0.0,
        is_all_caps_text: bool = True,
    ) -> None:
        self.internal_reconstructed_cache: LayoutLineText | None = None
        self.internal_reconstructed_cache_key: LayoutLineReconstructionKey | None = None
        compute_from_runs = (
            runs is not None
            and len(runs) > 0
            and x0 == 0.0
            and y0 == 0.0
            and x1 == 0.0
            and y1 == 0.0
            and not is_vertical
            and rotation_angle == 0
            and max_order == -1
            and max_depth == -1
            and min_order == 999999
            and mid_y == 0.0
            and height == 0.0
            and max_font_size == 0.0
            and is_all_caps_text
        )
        if compute_from_runs and runs is not None:
            run_list = runs
            first_run = run_list[0]
            first_coords = first_run.coords
            x0 = first_coords[TextRun.X0]
            y0 = first_coords[TextRun.Y0]
            x1 = first_coords[TextRun.X1]
            y1 = first_coords[TextRun.Y1]
            max_order = first_run.order
            min_order = first_run.order
            max_depth = first_run.xobject_depth
            max_font_size = first_coords[TextRun.FONT_SIZE]
            is_all_caps_text = not first_run.has_text or first_run.text_is_upper

            text_run_x0 = TextRun.X0
            text_run_y0 = TextRun.Y0
            text_run_x1 = TextRun.X1
            text_run_y1 = TextRun.Y1
            text_run_font_size = TextRun.FONT_SIZE

            for run in islice(run_list, 1, None):
                coords = run.coords
                run_x0 = coords[text_run_x0]
                run_y0 = coords[text_run_y0]
                run_x1 = coords[text_run_x1]
                run_y1 = coords[text_run_y1]
                font_size = coords[text_run_font_size]
                if run_x0 < x0:
                    x0 = run_x0
                if run_y0 < y0:
                    y0 = run_y0
                if run_x1 > x1:
                    x1 = run_x1
                if run_y1 > y1:
                    y1 = run_y1
                if run.order > max_order:
                    max_order = run.order
                if run.order < min_order:
                    min_order = run.order
                if run.xobject_depth > max_depth:
                    max_depth = run.xobject_depth
                if font_size > max_font_size:
                    max_font_size = font_size
                if is_all_caps_text and run.has_text and not run.text_is_upper:
                    is_all_caps_text = False

            is_vertical = first_run.is_vertical
            rotation_angle = first_run.rotation_angle
            mid_y = (y0 + y1) * 0.5
            height = y1 - y0

        self.runs = runs if runs is not None else []
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.is_vertical = is_vertical
        self.rotation_angle = rotation_angle
        self.max_order = max_order
        self.max_depth = max_depth
        self.min_order = min_order
        self.mid_y = mid_y
        self.height = height
        self.max_font_size = max_font_size
        self.is_all_caps_text = is_all_caps_text

    def reconstruction_key(self) -> LayoutLineReconstructionKey:
        return (
            self.is_all_caps_text,
            tuple((run, run.internal_revision, tuple(run.coords)) for run in self.runs),
        )

    def reconstructed_text(self) -> LayoutLineText:
        key = self.reconstruction_key()
        cached = self.internal_reconstructed_cache
        if cached is not None and key == self.internal_reconstructed_cache_key:
            return cached
        reconstructed = reconstruct_cached_layout_line_text(
            self.runs,
            is_all_caps_text=self.is_all_caps_text,
            internal_key=key,
        )
        self.internal_reconstructed_cache = reconstructed
        self.internal_reconstructed_cache_key = key
        return reconstructed

    def internal_build_text_and_words(self) -> tuple[str, tuple[LayoutWordSnapshot, ...]]:
        reconstructed = self.reconstructed_text()
        parts: list[str] = []
        words: list[LayoutWordSnapshot] = []
        word = ""
        word_x0 = word_y0 = word_x1 = word_y1 = 0.0
        append_part = parts.append
        append_word = words.append

        def flush_word() -> None:
            nonlocal word, word_x0, word_y0, word_x1, word_y1
            if not word:
                return
            append_word(LayoutWordSnapshot(word, (word_x0, word_y0, word_x1, word_y1)))
            word = ""

        def extend_word(char: str, bbox: tuple[float, float, float, float]) -> None:
            nonlocal word, word_x0, word_y0, word_x1, word_y1
            if not word:
                word_x0, word_y0, word_x1, word_y1 = bbox
            else:
                word_x0 = min(word_x0, bbox[0])
                word_y0 = min(word_y0, bbox[1])
                word_x1 = max(word_x1, bbox[2])
                word_y1 = max(word_y1, bbox[3])
            word += char

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
                bbox = layout_line_segment_char_bbox(segment, index, text_length)
                if char.isspace():
                    append_space()
                    continue
                if word and char.isalnum() != word[-1].isalnum():
                    flush_word()
                extend_word(char, bbox)
                append_part(char)

        flush_word()
        return "".join(parts).rstrip(), tuple(words)

    def cached_text_and_words(self) -> tuple[str, tuple[LayoutWordSnapshot, ...]]:
        key = self.reconstruction_key()
        first_run = self.runs[0] if self.runs else None
        cache = first_run.internal_layout_words_cache if first_run is not None else None
        if cache is not None and cache[0] == key:
            return cache[1]
        result = self.internal_build_text_and_words()
        if first_run is not None:
            internal_track_text_run(first_run)
            object.__setattr__(first_run, "internal_layout_words_cache", (key, result))
        return result


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
