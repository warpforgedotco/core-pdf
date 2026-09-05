# SPDX-License-Identifier: AGPL-3.0-only
"""Normalize captured text and accumulate adjacent fragments into runs."""

from __future__ import annotations

from collections import deque
from itertools import chain

from core_pdf.impl._impl.model.glyphs import GlyphCluster
from core_pdf.impl._impl.model.runs import TextRun
from core_pdf.impl._impl.model.text import word_gap_threshold

internal_NO_SPACE_BEFORE = frozenset(".,;:!?)]}%")
internal_NO_SPACE_AFTER = frozenset("([{")
internal_NORMALIZE_TEXT_TABLE = dict.fromkeys(range(0xD800, 0xE000))


def normalize_extracted_text(text: str) -> str:
    """Drop lone surrogates, which cannot survive UTF-8 serialization."""
    return text if text.isascii() else text.translate(internal_NORMALIZE_TEXT_TABLE)


def is_garbage_text(text: str) -> bool:
    return all(ord(char) < 32 or 0xE000 <= ord(char) <= 0xF8FF for char in text)


def internal_gap_separator(left: str, right: str, gap: float, run: TextRun) -> str:
    if gap <= word_gap_threshold(run.space_width, run.font_size):
        return ""
    if not left or not right or left[-1].isspace() or right[0].isspace():
        return ""
    if right[0] in internal_NO_SPACE_BEFORE or left[-1] in internal_NO_SPACE_AFTER:
        return ""
    return " "


def internal_can_merge_cross_font_word(left: str, right: str) -> bool:
    return (
        bool(left and right)
        and (left[-1].isalnum() or left[-1] == "_")
        and (right[0].isalnum() or right[0] == "_")
    )


class internal_PendingRun:
    """Own a run and defer joining fragments until it is emitted.

    A deque avoids quadratic copying when a producer shows one glyph at a time.
    Cluster chunks retain emission order even when text is prepended for RTL.
    """

    __slots__ = ("run", "parts", "clusters", "head", "tail")

    def __init__(self, run: TextRun) -> None:
        self.run = run
        self.parts: deque[str] = deque((run.text,))
        self.clusters: list[tuple[GlyphCluster, ...]] = [run.glyph_clusters]
        self.head = run.text[:1]
        self.tail = run.text[-1:]

    def finish(self) -> TextRun:
        if len(self.parts) > 1:
            self.run.text = "".join(self.parts)
            self.run.glyph_clusters = tuple(chain.from_iterable(self.clusters))
        return self.run

    def merge(
        self,
        new_run: TextRun,
        gap: float,
        threshold: float,
        *,
        widen: str,
        reverse: bool = False,
    ) -> bool:
        if not -2.0 <= gap < threshold:
            return False
        if reverse:
            separator = internal_gap_separator(new_run.text, self.head, gap, self.run)
            added = new_run.text + separator
            self.parts.appendleft(added)
            if added:
                self.head = added[:1]
            if not self.tail:
                self.tail = added[-1:]
        else:
            separator = internal_gap_separator(self.tail, new_run.text, gap, self.run)
            added = separator + new_run.text
            self.parts.append(added)
            if not self.head:
                self.head = added[:1]
            if added:
                self.tail = added[-1:]
        self.clusters.append(new_run.glyph_clusters)
        self.run.union_ink_bbox(new_run.ink_bbox)
        edge = getattr(new_run, widen)
        keep = min if widen.endswith("0") else max
        setattr(self.run, widen, keep(edge, getattr(self.run, widen)))
        return True

    def try_append(self, new_run: TextRun) -> bool:
        pending = self.run
        font_size = pending.font_size
        space_width = pending.space_width
        rotation = pending.rotation_angle
        threshold = max(space_width * 0.45, 2.0)
        same_style = (
            rotation == new_run.rotation_angle
            and pending.visible == new_run.visible
            and not new_run.line_break_before
            and font_size == new_run.font_size
            and (
                pending.font_name == new_run.font_name
                or internal_can_merge_cross_font_word(self.tail, new_run.text)
                or internal_can_merge_cross_font_word(new_run.text, self.head)
            )
            and pending.fill_color == new_run.fill_color
        )
        if same_style and rotation == 90:
            if abs(new_run.y0 - pending.y1) > max(space_width * 0.5, font_size * 0.8, 2.0):
                same_style = False
        elif same_style and rotation == 0 and abs(pending.y0 - new_run.y0) > font_size * 0.5:
            same_style = False
        if not same_style:
            return False
        if rotation == 0:
            merged = self.merge(
                new_run, new_run.x0 - pending.x1, threshold, widen="x1"
            ) or self.merge(new_run, pending.x0 - new_run.x1, threshold, widen="x0", reverse=True)
        elif rotation == 90:
            merged = self.merge(new_run, new_run.y0 - pending.y1, threshold, widen="y1")
        else:
            merged = self.merge(
                new_run, pending.x0 - new_run.x1, threshold, widen="x0"
            ) or self.merge(new_run, new_run.x0 - pending.x1, threshold, widen="x1", reverse=True)
        if merged:
            pending.advance_bbox = (pending.x0, pending.y0, pending.x1, pending.y1)
        return merged


class RunAccumulator:
    """Accumulate adjacent runs and append completed runs to the capture output."""

    __slots__ = ("output", "pending")

    def __init__(self, output: list[TextRun]) -> None:
        self.output = output
        self.pending: internal_PendingRun | None = None

    def append(self, run: TextRun) -> None:
        if self.pending is not None and self.pending.try_append(run):
            return
        self.flush()
        self.pending = internal_PendingRun(run)

    def flush(self) -> None:
        if self.pending is not None:
            self.output.append(self.pending.finish())
            self.pending = None
