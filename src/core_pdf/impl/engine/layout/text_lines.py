# SPDX-License-Identifier: AGPL-3.0-only
"""Assemble glyph runs into layout lines, words, and their reconstructed text."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from statistics import median_low

from core_pdf.impl.engine.layout.models import TextRun
from core_pdf.impl.engine.layout.word_frequencies import word_rank

FOOTER_RE = re.compile(r"^\s*page\s*\d+\s*$", re.IGNORECASE)
LEADER_START_CHARS = "._~-–—"
SUPERSCRIPT_DIGIT_TRANSLATION = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
SUBSCRIPT_DIGIT_TRANSLATION = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SCRIPT_DIGITS = frozenset("⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉")
INLINE_MARKERS = frozenset({"™", "℠", "®", "©"})
FORMULA_MARKERS = frozenset("∂∑√∞∈θΦω")


@dataclass(frozen=True, slots=True)
class LayoutLineTextSegment:
    text: str
    separator_before: str
    spacing_decision: str
    bbox: tuple[float, float, float, float]
    advance_bbox: tuple[float, float, float, float]
    ink_bbox: tuple[float, float, float, float]
    baseline: tuple[float, float, float, float] | None
    writing_mode: str
    rotation_angle: int
    provenance: tuple[tuple[str, object], ...]
    confidence: float | None
    visible: bool


# Builder-only atoms are short-lived and never escape into the immutable layout result.
@dataclass(slots=True)
class LayoutLineTextAtom:
    text: str
    run: TextRun
    advance_bbox: tuple[float, float, float, float]
    ink_bbox: tuple[float, float, float, float]
    baseline: tuple[float, float, float, float] | None
    provenance: tuple[tuple[str, object], ...]
    confidence: float | None
    visible: bool
    has_glyph_geometry: bool


@dataclass(frozen=True, slots=True)
class LayoutLineText:
    text: str
    segments: tuple[LayoutLineTextSegment, ...]


EMPTY_LAYOUT_LINE_TEXT = LayoutLineText("", ())


def reconstruct_layout_line_text(
    runs: list[TextRun], *, is_all_caps_text: bool | None = None
) -> LayoutLineText:
    if not runs:
        return EMPTY_LAYOUT_LINE_TEXT
    if len(runs) == 1:
        run = runs[0]
        if run.glyph_clusters:
            return GlyphLineBuilder(runs).build()
        text = render_single_run_text(run)
        if not text:
            return EMPTY_LAYOUT_LINE_TEXT
        return LayoutLineText(text, (line_text_segment(run, text, "", "single_run"),))

    angle = runs[0].rotation_angle
    if angle == 0 and len(runs) <= 3:
        sorted_runs = (
            runs if runs_are_left_to_right(runs) else sorted(runs, key=lambda r: (r.x0, r.order))
        )
        return GlyphLineBuilder(sorted_runs).build()

    if angle == 0 and runs_are_left_to_right(runs):
        sorted_runs = runs
    elif runs_are_right_to_left(runs):
        sorted_runs = sorted(runs, key=lambda r: (r.order, r.stream_order))
    else:
        if angle == 90:
            sorted_runs = sorted(runs, key=lambda r: (r.y0, r.order))
        elif angle == 270:
            sorted_runs = sorted(runs, key=lambda r: (-r.y1, r.order))
        else:
            sorted_runs = sorted(runs, key=lambda r: (r.x0, r.order))
            if has_interleaved_horizontal_overlap(sorted_runs):
                sorted_runs = sorted(runs, key=lambda r: (r.order, r.stream_order))

    is_formula_like_line = formula_like_runs(sorted_runs)
    if angle == 0 and is_formula_like_line:
        sorted_runs = reorder_stacked_formula_numerators(sorted_runs)

    digit_text_runs = 0
    all_text_runs_upper = True if is_all_caps_text is None else is_all_caps_text
    has_explicit_spaces = False
    max_text_font_size = 0.0
    non_space_runs: list[TextRun] = []
    for run in sorted_runs:
        if run.has_text:
            non_space_runs.append(run)
            if run.stripped_text.isdigit():
                digit_text_runs += 1
            if is_all_caps_text is None and not run.text_is_upper:
                all_text_runs_upper = False
            font_size = run.font_size
            if font_size > max_text_font_size:
                max_text_font_size = font_size
        elif run.text_is_space:
            has_explicit_spaces = True

    alnum_text_runs = sum(
        1 for run in non_space_runs if any(ch.isalnum() for ch in run.stripped_text)
    )
    is_table_like_line = (len(sorted_runs) >= 4 and digit_text_runs >= 2) or (
        len(non_space_runs) >= 3 and alnum_text_runs >= 3 and len(sorted_runs) >= 5
    )
    is_all_caps_line = len(sorted_runs) >= 2 and all_text_runs_upper
    is_tracked_glyph_line = is_tracked_glyph_run_line(
        non_space_runs, has_explicit_spaces=has_explicit_spaces
    )
    is_formula_like_line = formula_like_runs(sorted_runs)

    if angle in {90, 270} and len(non_space_runs) >= 8:
        return reconstruct_rotated_table_line(sorted_runs)

    return GlyphLineBuilder(
        sorted_runs,
        is_table_like_line=is_table_like_line,
        is_all_caps_line=is_all_caps_line,
        has_explicit_spaces=has_explicit_spaces,
        is_tracked_glyph_line=is_tracked_glyph_line,
        is_formula_like_line=is_formula_like_line,
        suppress_tiny_page_footer=(len(sorted_runs) <= 3 and max_text_font_size <= 5.0),
    ).build()


def line_text_segment(
    run: TextRun,
    text: str,
    separator_before: str,
    spacing_decision: str,
) -> LayoutLineTextSegment:
    advance_bbox = GlyphLineBuilder.advance_bbox(run)
    ink_bbox = run.ink_bbox
    return LayoutLineTextSegment(
        text=text,
        separator_before=separator_before,
        spacing_decision=spacing_decision,
        bbox=(run.x0, run.y0, run.x1, run.y1),
        advance_bbox=advance_bbox,
        ink_bbox=ink_bbox,
        baseline=run.baseline,
        writing_mode="vertical" if run.is_vertical else "horizontal",
        rotation_angle=run.rotation_angle,
        provenance=run.provenance,
        confidence=run.confidence,
        visible=run.visible,
    )


def line_text_segment_from_atom(
    atom: LayoutLineTextAtom,
    separator_before: str,
    spacing_decision: str,
) -> LayoutLineTextSegment:
    run = atom.run
    bbox = atom.advance_bbox
    return LayoutLineTextSegment(
        text=atom.text,
        separator_before=separator_before,
        spacing_decision=spacing_decision,
        bbox=bbox,
        advance_bbox=atom.advance_bbox,
        ink_bbox=atom.ink_bbox,
        baseline=atom.baseline,
        writing_mode="vertical" if run.is_vertical else "horizontal",
        rotation_angle=run.rotation_angle,
        provenance=atom.provenance,
        confidence=atom.confidence,
        visible=atom.visible,
    )


def render_single_run_text(run: TextRun) -> str:
    text = run.text
    if not text:
        return ""
    if is_structural_list_marker_run(run):
        return ""
    if not text.isprintable() and any(is_private_use_or_control(ch) for ch in text):
        text = strip_private_use_chars(text)
        if not text:
            return ""
    if is_tiny_page_footer(text) and run.font_size <= 5.0:
        return ""
    return collapse_repeated_spaces(text)


def reconstruct_rotated_table_line(sorted_runs: list[TextRun]) -> LayoutLineText:
    parts: list[str] = []
    segments: list[LayoutLineTextSegment] = []
    previous_run: TextRun | None = None
    geometric_cell_spacing = (
        sum(len(run.stripped_text) >= 2 for run in sorted_runs if run.has_text) >= 6
    )
    for run in sorted_runs:
        text = run.text
        if text.isspace():
            if parts and parts[-1] != " ":
                parts.append(" ")
                segments.append(line_text_segment(run, " ", " ", "rotated_explicit_space"))
            previous_run = run
            continue
        if not run.has_text:
            continue
        if is_structural_list_marker_run(run):
            continue
        if not text.isprintable() and any(is_private_use_or_control(ch) for ch in text):
            text = strip_private_use_chars(text)
        text = text.strip()
        if text:
            separator = ""
            spacing_decision = "line_start"
            if parts:
                spacing_decision = "rotated_join"
                if (
                    parts[-1] != " "
                    and geometric_cell_spacing
                    and previous_run is not None
                    and rotated_table_run_gap(previous_run, run)
                    > max(1.0, min(previous_run.font_size, run.font_size) * 0.40)
                ):
                    parts.append(" ")
                    separator = " "
                    spacing_decision = "rotated_geometric_space"
            parts.append(text)
            segments.append(line_text_segment(run, text, separator, spacing_decision))
            previous_run = run
    combined = split_glued_numeric_label_boundaries("".join(parts))
    if not combined:
        return EMPTY_LAYOUT_LINE_TEXT
    return LayoutLineText(combined, tuple(segments))


def rotated_table_run_gap(previous: TextRun, current: TextRun) -> float:
    if current.rotation_angle == 270:
        return previous.y0 - current.y1
    return current.y0 - previous.y1


class GlyphLineBuilder:
    __slots__ = (
        "runs",
        "page_label_indexes",
        "is_table_like_line",
        "is_all_caps_line",
        "has_explicit_spaces",
        "is_tracked_glyph_line",
        "is_formula_like_line",
        "suppress_tiny_page_footer",
        "tracked_word_gap",
        "explicit_spaces_control_glyph_gaps",
        "explicit_space_count",
        "has_large_column_gap",
        "next_non_space_texts",
        "next_non_space_x0s",
        "estimated_char_width",
        "column_gap_threshold",
    )

    def __init__(
        self,
        runs: list[TextRun],
        *,
        is_table_like_line: bool = False,
        is_all_caps_line: bool = False,
        has_explicit_spaces: bool | None = None,
        is_tracked_glyph_line: bool = False,
        is_formula_like_line: bool = False,
        suppress_tiny_page_footer: bool = False,
    ) -> None:
        self.runs = runs
        self.page_label_indexes = trailing_tiny_page_label_run_indexes(runs)
        self.is_table_like_line = is_table_like_line
        self.is_all_caps_line = is_all_caps_line
        self.has_explicit_spaces = (
            any(run.text_is_space for run in runs)
            if has_explicit_spaces is None
            else has_explicit_spaces
        )
        self.is_tracked_glyph_line = is_tracked_glyph_line
        self.is_formula_like_line = is_formula_like_line
        self.suppress_tiny_page_footer = suppress_tiny_page_footer
        self.tracked_word_gap = (
            tracked_glyph_word_gap_threshold(runs) if is_tracked_glyph_line else None
        )
        non_space_runs = [run for run in runs if run.has_text]
        self.explicit_spaces_control_glyph_gaps = explicit_spaces_should_control_glyph_gaps(
            non_space_runs,
            explicit_space_count=sum(1 for run in runs if run.text_is_space),
        )
        self.explicit_space_count = 0
        self.has_large_column_gap = False
        self.next_non_space_texts: list[str] = []
        self.next_non_space_x0s: list[float] = []
        self.estimated_char_width = (
            None
            if is_table_like_line or is_tracked_glyph_line or self.has_explicit_spaces
            else estimated_char_width_for_suspect_line(runs)
        )
        self.column_gap_threshold = column_gap_threshold_for_runs(runs)
        if self.has_explicit_spaces:
            self.internal_prepare_explicit_space_context()

    def render(self) -> str:
        return self.build().text

    def build(self) -> LayoutLineText:
        parts: list[str] = []
        segments: list[LayoutLineTextSegment] = []
        append_part = parts.append
        append_segment = segments.append

        prev_run: TextRun | None = None
        prev_run_text = ""
        prev_last_char = ""
        prev_run_bbox: tuple[float, float, float, float] | None = None
        prev_atom: LayoutLineTextAtom | None = None
        recent_emitted_runs: list[tuple[tuple[float, float, float, float], str]] = []

        for index, run in enumerate(self.runs):
            text = self.normalized_text(run, index)
            if not text:
                continue
            stripped = text.strip()
            if not stripped and not text.isspace():
                continue

            if (
                prev_run is not None
                and prev_run_bbox is not None
                and self.is_duplicate_overlap(prev_run_bbox, prev_run_text, run, text)
            ):
                continue
            if self.is_recent_duplicate_overlap(recent_emitted_runs, run, text):
                continue

            if (
                run.text_is_space
                and prev_run is not None
                and prev_run_text
                and self.should_drop_explicit_space(index, run, prev_run, prev_last_char)
            ):
                continue

            emitted_run = False
            for atom in self.text_atoms(run, text):
                atom_text = atom.text
                if not atom_text:
                    continue
                if not atom_text.strip() and not atom_text.isspace():
                    continue

                separator_before = ""
                spacing_decision = "line_start" if prev_atom is None else "join"
                if prev_atom is not None:
                    separator_before, spacing_decision = self.atom_separator(
                        prev_atom,
                        atom,
                    )
                    if separator_before:
                        append_part(separator_before)
                if atom_text.isspace():
                    spacing_decision = "explicit_space"

                append_part(atom_text)
                append_segment(
                    line_text_segment_from_atom(
                        atom,
                        separator_before,
                        spacing_decision,
                    )
                )
                prev_atom = atom
                emitted_run = True

            if not emitted_run:
                continue
            prev_run = run
            prev_run_text = text
            prev_last_char = text[-1:]
            prev_run_bbox = self.advance_bbox(run)
            recent_emitted_runs.append((prev_run_bbox, text))
            if len(recent_emitted_runs) > 256:
                del recent_emitted_runs[:64]

        combined = "".join(parts)
        if self.suppress_tiny_page_footer and is_tiny_page_footer(combined):
            return EMPTY_LAYOUT_LINE_TEXT
        text = collapse_repeated_spaces(combined)
        text = repair_table_split_word_boundaries(text)
        if not text:
            return EMPTY_LAYOUT_LINE_TEXT
        return LayoutLineText(text, tuple(segments))

    def text_atoms(
        self,
        run: TextRun,
        text: str,
    ) -> tuple[LayoutLineTextAtom, ...]:
        clusters = run.glyph_clusters
        if (
            clusters
            and text == run.text
            and "".join(cluster.text for cluster in clusters) == run.text
        ):
            atoms = [
                LayoutLineTextAtom(
                    text=cluster.text,
                    run=run,
                    advance_bbox=cluster.advance_bbox,
                    ink_bbox=cluster.ink_bbox,
                    baseline=cluster.baseline,
                    provenance=(
                        *run.provenance,
                        *cluster.provenance,
                        ("glyph_cluster_id", cluster.cluster_id),
                        ("glyph_cluster_kind", cluster.kind),
                    ),
                    confidence=cluster.confidence,
                    visible=run.visible,
                    has_glyph_geometry=True,
                )
                for cluster in clusters
                if cluster.text
            ]
            if atoms:
                return tuple(atoms)
        return (
            LayoutLineTextAtom(
                text=text,
                run=run,
                advance_bbox=self.advance_bbox(run),
                ink_bbox=run.ink_bbox,
                baseline=run.baseline,
                provenance=run.provenance,
                confidence=run.confidence,
                visible=run.visible,
                has_glyph_geometry=False,
            ),
        )

    def internal_prepare_explicit_space_context(self) -> None:
        runs = self.runs
        self.explicit_space_count = sum(1 for run in runs if run.text_is_space)
        self.next_non_space_texts = [""] * len(runs)
        self.next_non_space_x0s = [0.0] * len(runs)
        next_text = ""
        next_x0 = 0.0
        for reverse_index in range(len(runs) - 1, -1, -1):
            self.next_non_space_texts[reverse_index] = next_text
            self.next_non_space_x0s[reverse_index] = next_x0
            run = runs[reverse_index]
            if run.has_text:
                next_text = run.text
                next_x0 = run.x0

        previous_run: TextRun | None = None
        for run in runs:
            if previous_run is not None:
                gap = run.x0 - previous_run.x1
                gap_threshold = max(run.space_width * 8.0, run.height * 3.0, 40.0)
                if gap > gap_threshold:
                    self.has_large_column_gap = True
                    break
            previous_run = run

    def normalized_text(self, run: TextRun, index: int) -> str:
        text = run.text
        if not text:
            return ""
        if index in self.page_label_indexes:
            if run.stripped_text.casefold() == "page":
                return ""
            text = run.stripped_text
        if is_structural_list_marker_run(run):
            return ""
        if not text.isprintable() and any(is_private_use_or_control(ch) for ch in text):
            text = strip_private_use_chars(text)
            if not text:
                return ""
        if is_tiny_page_footer(text) and run.font_size <= 5.0:
            return ""
        first_char = text[:1]
        if (first_char in LEADER_START_CHARS or first_char.isspace()) and is_decorative_leader(
            text
        ):
            return ""
        if self.is_trademark_marker_run(run, index):
            return "™"
        if self.is_superscript_like_numeric_run(run, index) or self.is_unit_exponent_run(
            run, index
        ):
            return text.translate(SUPERSCRIPT_DIGIT_TRANSLATION)
        if self.is_subscript_like_numeric_run(run, index) or (
            self.is_formula_like_line and self.is_formula_subscript_like_numeric_run(run, index)
        ):
            return text.translate(SUBSCRIPT_DIGIT_TRANSLATION)
        return text

    def is_trademark_marker_run(self, run: TextRun, index: int) -> bool:
        stripped = run.stripped_text
        if stripped != "TM" or run.rotation_angle != 0 or run.baseline is None:
            return False
        previous = self.previous_non_space_run(index)
        following = self.next_non_space_run(index)
        context_runs = [
            candidate
            for candidate in (previous, following)
            if candidate is not None
            and candidate.baseline is not None
            and candidate.rotation_angle == 0
            and candidate.stripped_text
            and candidate.stripped_text != stripped
        ]
        if not context_runs:
            return False
        context_font_size = max(candidate.font_size for candidate in context_runs)
        if context_font_size <= 0.0 or run.font_size >= context_font_size * 0.82:
            return False
        run_baseline = baseline_midpoint(run.baseline, 0)
        baseline_raise = max(
            run_baseline - baseline_midpoint(candidate.baseline, 0)
            for candidate in context_runs
            if candidate.baseline is not None
        )
        if baseline_raise < max(1.5, context_font_size * 0.18):
            return False
        attach_gap = max(run.space_width * 0.5, context_font_size * 0.25, 2.0)
        attached_prev = (
            previous is not None
            and bool(previous.stripped_text)
            and run.x0 - previous.x1 <= attach_gap
        )
        attached_next = (
            following is not None
            and bool(following.stripped_text)
            and following.x0 - run.x1 <= attach_gap
        )
        return attached_prev or attached_next

    def is_superscript_like_numeric_run(self, run: TextRun, index: int) -> bool:
        stripped = run.stripped_text
        if (
            not stripped
            or stripped != run.text
            or not stripped.isdigit()
            or len(stripped) > 4
            or run.rotation_angle != 0
            or run.baseline is None
        ):
            return False
        previous = self.previous_non_space_run(index)
        following = self.next_non_space_run(index)
        context_runs = [
            candidate
            for candidate in (previous, following)
            if candidate is not None
            and candidate.baseline is not None
            and candidate.rotation_angle == 0
            and candidate.stripped_text
            and not candidate.stripped_text.isdigit()
        ]
        if not context_runs:
            return False
        context_font_size = max(candidate.font_size for candidate in context_runs)
        if context_font_size <= 0.0 or run.font_size >= context_font_size * 0.8:
            return False
        run_baseline = baseline_midpoint(run.baseline, 0)
        baseline_raises = [
            run_baseline - baseline_midpoint(candidate.baseline, 0)
            for candidate in context_runs
            if candidate.baseline is not None
        ]
        if not baseline_raises:
            return False
        baseline_raise = max(baseline_raises)
        if baseline_raise < max(1.5, context_font_size * 0.18):
            return False
        attach_gap = max(run.space_width * 0.5, context_font_size * 0.25, 2.0)
        attached_prev = (
            previous is not None
            and bool(previous.stripped_text)
            and run.x0 - previous.x1 <= attach_gap
        )
        attached_next = (
            following is not None
            and bool(following.stripped_text)
            and following.x0 - run.x1 <= attach_gap
        )
        return attached_prev or attached_next

    def is_subscript_like_numeric_run(self, run: TextRun, index: int) -> bool:
        stripped = run.stripped_text
        if (
            not stripped
            or stripped != run.text
            or not stripped.isdigit()
            or len(stripped) > 3
            or run.rotation_angle != 0
            or run.baseline is None
        ):
            return False
        previous = self.previous_non_space_run(index)
        if previous is None or not chemical_subscript_prefix_text(previous.stripped_text):
            return False
        following = self.next_non_space_run(index)
        context_runs = [
            candidate
            for candidate in (previous, following)
            if candidate is not None
            and candidate.baseline is not None
            and candidate.rotation_angle == 0
            and candidate.stripped_text
            and not candidate.stripped_text.isdigit()
        ]
        if not context_runs:
            return False
        context_font_size = max(candidate.font_size for candidate in context_runs)
        if context_font_size <= 0.0 or run.font_size >= context_font_size * 0.82:
            return False
        run_baseline = baseline_midpoint(run.baseline, 0)
        baseline_drops = [
            baseline_midpoint(candidate.baseline, 0) - run_baseline
            for candidate in context_runs
            if candidate.baseline is not None
        ]
        if not baseline_drops:
            return False
        baseline_drop = max(baseline_drops)
        if baseline_drop < max(0.45, context_font_size * 0.05):
            return False
        attach_gap = max(run.space_width * 0.5, context_font_size * 0.2, 2.0)
        return run.x0 - previous.x1 <= attach_gap

    def is_formula_subscript_like_numeric_run(self, run: TextRun, index: int) -> bool:
        """Recognize numeric subscripts attached to mathematical variables."""
        stripped = run.stripped_text
        if (
            not stripped
            or stripped != run.text
            or not stripped.isdigit()
            or len(stripped) > 3
            or run.rotation_angle != 0
            or run.baseline is None
        ):
            return False
        previous = self.previous_non_space_run(index)
        if previous is None or not previous.stripped_text[-1:].isalpha():
            return False
        if previous.baseline is None:
            return False
        previous_height = previous.height_value
        if previous_height <= 0.0 or run.height_value >= previous_height * 0.9:
            return False
        baseline_drop = baseline_midpoint(previous.baseline, 0) - baseline_midpoint(run.baseline, 0)
        if baseline_drop < max(0.45, previous_height * 0.05):
            return False
        attach_gap = max(run.space_width * 0.5, previous_height * 0.2, 2.0)
        return run.x0 - previous.x1 <= attach_gap

    def is_unit_exponent_run(self, run: TextRun, index: int) -> bool:
        stripped = run.stripped_text
        if (
            not stripped
            or stripped != run.text
            or not stripped.isdigit()
            or len(stripped) > 2
            or run.rotation_angle != 0
        ):
            return False
        following = min(
            (
                candidate
                for candidate in self.runs
                if candidate is not run
                and candidate.has_text
                and candidate.x0 >= run.x1 - 0.01
                and candidate.stripped_text.lstrip().startswith((")", "]", "}"))
            ),
            key=lambda candidate: candidate.x0,
            default=None,
        )
        if following is None:
            return False

        left_candidates = sorted(
            (
                candidate
                for candidate in self.runs
                if candidate is not run
                and candidate.has_text
                and candidate.x0 < run.x0
                and candidate.x1 <= run.x1 + 0.01
            ),
            key=lambda candidate: candidate.x0,
        )
        previous = max(
            (
                candidate
                for candidate in left_candidates
                if candidate.stripped_text.rstrip().endswith(("in", "cm", "mm", "ft", "yd"))
            ),
            key=lambda candidate: candidate.x1,
            default=None,
        )
        if previous is None:
            for left, right in zip(left_candidates, left_candidates[1:], strict=False):
                if (
                    left.stripped_text.rstrip().endswith("i")
                    and right.stripped_text.lstrip().startswith("n")
                    and right.x0 - left.x1 <= max(run.space_width * 0.5, 2.0)
                ):
                    previous = right

        prefix_runs = [
            candidate
            for candidate in left_candidates
            if candidate.x1 <= run.x0 + max(1.0, run.space_width * 0.25)
        ]
        prefix = "".join(candidate.stripped_text for candidate in prefix_runs).rstrip()
        if any(prefix.endswith(unit) for unit in ("in", "cm", "mm", "ft", "yd")):
            attach_gap = max(run.space_width, run.font_size * 0.55, 4.0)
            return (
                bool(prefix_runs)
                and run.x0 - prefix_runs[-1].x1 <= attach_gap
                and following.x0 - run.x1 <= attach_gap
            )
        if previous is None or following is None or run.baseline is None:
            return False
        attach_gap = max(run.space_width, previous.font_size * 0.55, 4.0)
        return run.x0 - previous.x1 <= attach_gap and following.x0 - run.x1 <= attach_gap

    def previous_non_space_run(self, index: int) -> TextRun | None:
        for reverse_index in range(index - 1, -1, -1):
            run = self.runs[reverse_index]
            if run.has_text:
                return run
        return None

    def next_non_space_run(self, index: int) -> TextRun | None:
        for next_index in range(index + 1, len(self.runs)):
            run = self.runs[next_index]
            if run.has_text:
                return run
        return None

    @staticmethod
    def advance_bbox(run: TextRun) -> tuple[float, float, float, float]:
        return run.advance_bbox

    def is_duplicate_overlap(
        self,
        prev_bbox: tuple[float, float, float, float],
        prev_text: str,
        run: TextRun,
        text: str,
        *,
        run_bbox: tuple[float, float, float, float] | None = None,
    ) -> bool:
        x0, y0, x1, y1 = run_bbox if run_bbox is not None else self.advance_bbox(run)
        px0, py0, px1, py1 = prev_bbox
        ox = (x1 if x1 < px1 else px1) - (x0 if x0 > px0 else px0)
        oy = (y1 if y1 < py1 else py1) - (y0 if y0 > py0 else py0)
        if ox <= 0 or oy <= 0:
            return False
        box_area = (y1 - y0) * (x1 - x0)
        if box_area <= 0:
            return False
        overlap_ratio = (ox * oy) / box_area
        if text == prev_text:
            return overlap_ratio > 0.5
        return overlap_ratio > 0.8 and len(text) == len(prev_text) and len(text) <= 2

    def is_recent_duplicate_overlap(
        self,
        recent_runs: list[tuple[tuple[float, float, float, float], str]],
        run: TextRun,
        text: str,
    ) -> bool:
        if not recent_runs:
            return False
        x0, y0, x1, y1 = self.advance_bbox(run)
        box_area = (y1 - y0) * (x1 - x0)
        if box_area <= 0:
            return False
        text_length = len(text)
        for (px0, py0, px1, py1), prev_text in reversed(recent_runs):
            ox = (x1 if x1 < px1 else px1) - (x0 if x0 > px0 else px0)
            oy = (y1 if y1 < py1 else py1) - (y0 if y0 > py0 else py0)
            if ox <= 0 or oy <= 0:
                continue
            overlap_ratio = (ox * oy) / box_area
            if (text == prev_text and overlap_ratio > 0.5) or (
                overlap_ratio > 0.8 and text_length == len(prev_text) and text_length <= 2
            ):
                return True
        return False

    def should_drop_explicit_space(
        self,
        index: int,
        run: TextRun,
        prev_run: TextRun,
        prev_last_char: str,
    ) -> bool:
        next_text = (
            self.next_non_space_texts[index] if index < len(self.next_non_space_texts) else ""
        )
        next_x0 = self.next_non_space_x0s[index] if index < len(self.next_non_space_x0s) else 0.0
        leading_gap = run.x0 - prev_run.x1
        following_gap = next_x0 - run.x1 if next_text else 0.0
        if run.x1 - run.x0 >= max(1.0, run.space_width * 0.5):
            return False
        tight_gap = max(run.space_width * 0.55, run.height * 0.35, 2.0)
        if (
            next_text
            and prev_last_char in "$,(0123456789"
            and next_text[:1] in "),0123456789"
            and leading_gap <= tight_gap
            and following_gap <= tight_gap
        ):
            return True
        return (
            bool(next_text)
            and prev_last_char.isalpha()
            and next_text[:1].isalpha()
            and (self.is_table_like_line or self.is_tracked_glyph_line)
            and min(leading_gap, following_gap) <= max(2.0, run.height * 0.1)
        )

    def atom_separator(
        self,
        previous: LayoutLineTextAtom,
        atom: LayoutLineTextAtom,
    ) -> tuple[str, str]:
        prev_run = previous.run
        run = atom.run
        prev_text = previous.text
        text = atom.text
        if prev_text.endswith(" ") or text.startswith(" "):
            return "", "join"
        prev_last_char = prev_text[-1:]
        first_char = text[:1]
        if not prev_last_char or not first_char:
            return "", "join"
        if (
            prev_run is run
            and " " in run.text
            and previous.has_glyph_geometry
            and atom.has_glyph_geometry
        ):
            return "", "same_run_explicit_space_join"

        prev_x0, internal_prev_y0, prev_x1, internal_prev_y1 = previous.advance_bbox
        x0, y0, internal_x1, y1 = atom.advance_bbox
        height = y1 - y0
        x_gap = x0 - prev_x1
        spacing_gap = x_gap
        space_width = run.coords[TextRun.SPACE_WIDTH]
        estimated_char_width = self.estimated_char_width
        prev_stripped: str | None = None
        stripped: str | None = None
        if estimated_char_width is not None:
            stripped = text.strip()
            prev_stripped = prev_text.strip()
            tight_fragment_gap = max(1.8, min(space_width, height) * 0.25)
            if (
                should_use_estimated_word_spacing(prev_stripped, stripped)
                and x_gap > tight_fragment_gap
            ):
                spacing_gap = x0 - (prev_x0 + len(prev_stripped) * estimated_char_width)
        baseline_delta = self.atom_baseline_delta(previous, atom)
        if inline_marker_text(text):
            return "", "inline_marker_join"
        if self.is_formula_like_line and self.is_formula_numeric_atom(previous, atom):
            return " ", "formula_numeric_boundary_space"
        if self.is_formula_like_line and self.is_formula_fraction_denominator(previous, atom):
            return "/", "formula_fraction_separator"
        if script_digit_text(prev_text) and text[:1] in ")]},.;:":
            return "", "script_digit_suffix_join"
        if script_digit_text(text):
            return "", "script_digit_join"
        if self.is_formula_like_line and self.is_formula_script_atom(previous, atom):
            return " ", "formula_script_boundary_space"
        if baseline_delta is not None and baseline_delta > max(height * 0.42, 2.0):
            return " ", "baseline_space"

        if self.is_column_gap(spacing_gap, height, space_width):
            return " ", "column_space"

        if prev_stripped is None:
            prev_stripped = prev_text.strip()
            stripped = text.strip()
        assert stripped is not None
        threshold = self.word_gap_threshold(run, height)

        if (
            self.is_table_like_line
            and self.has_explicit_spaces
            and prev_last_char.isalpha()
            and first_char in "$0123456789("
            and x_gap >= -2.0
        ):
            return " ", "table_space"
        if (
            self.is_all_caps_line
            and " " in prev_stripped
            and prev_last_char.isupper()
            and first_char.isupper()
            and len(stripped) >= 4
            and x_gap >= -max(0.6, height * 0.08)
        ):
            return " ", "all_caps_space"
        if not (prev_run.visible and run.visible) and should_insert_tight_word_space(
            prev_text=prev_stripped,
            text=stripped,
            x_gap=spacing_gap,
            height=height,
            space_width=space_width,
        ):
            return " ", "hidden_tight_word_space"
        if should_insert_hidden_ocr_overlap_space(
            prev_text=prev_stripped,
            text=stripped,
            x_gap=spacing_gap,
            height=height,
            space_width=space_width,
            prev_visible=prev_run.visible,
            visible=run.visible,
        ):
            return " ", "hidden_overlap_space"
        if self.should_join_word_fragments(
            prev_stripped,
            stripped,
            spacing_gap=spacing_gap,
            height=height,
            space_width=space_width,
            prev_visible=prev_run.visible,
            visible=run.visible,
            allow_short_prefix=(len(prev_run.stripped_text) <= 2 and len(run.stripped_text) <= 2),
        ):
            return "", "word_fragment_join"
        if (
            self.has_explicit_spaces
            and not self.is_table_like_line
            and prev_last_char.isalnum()
            and first_char.isalnum()
            and len(prev_stripped) > 1
            and len(stripped) > 1
            and spacing_gap > max(0.55, min(space_width, height) * 0.1)
            and (prev_last_char.isupper() or first_char.isupper() or prev_last_char.isdigit())
        ):
            return " ", "explicit_context_space"
        if (
            not (previous.has_glyph_geometry and atom.has_glyph_geometry)
            and " " in prev_stripped
            and prev_last_char.islower()
            and first_char.islower()
            and spacing_gap >= max(0.25, min(space_width, height) * 0.08)
            and spacing_gap <= max(0.5, height * 0.04)
            and should_insert_phrase_continuation_space(prev_stripped, stripped)
            and len(stripped.split(" ", 1)[0]) >= 3
        ):
            return " ", "phrase_continuation_space"
        if (
            not self.is_table_like_line
            and not self.is_tracked_glyph_line
            and prev_last_char.islower()
            and first_char.islower()
            and len(prev_stripped) > 1
            and len(stripped) > 1
            and spacing_gap > max(0.45, min(space_width, height) * 0.12)
        ):
            return " ", "lowercase_gap_space"
        if (
            spacing_gap > threshold
            and (
                not self.has_explicit_spaces
                or self.is_tracked_glyph_line
                or (previous.has_glyph_geometry and atom.has_glyph_geometry)
                or (
                    self.is_table_like_line
                    and spacing_gap > max(space_width * 2.2, height * 1.4, 24.0)
                )
            )
            and not (
                self.explicit_spaces_control_glyph_gaps
                and previous.has_glyph_geometry
                and atom.has_glyph_geometry
                and not self.is_table_like_line
            )
        ):
            return " ", "word_gap_space"
        if spacing_gap >= -max(threshold, 2.5) and (
            (prev_last_char.islower() and first_char.isupper())
            or (
                prev_last_char.isdigit()
                and first_char.isdigit()
                and not digit_fragments_are_tightly_joined(
                    prev_stripped,
                    stripped,
                    x_gap=spacing_gap,
                    height=height,
                    space_width=space_width,
                )
                and ("," in prev_text or "," in text or len(prev_stripped) >= 3)
                and len(stripped) >= 3
            )
        ):
            if prev_run is run and x_gap <= max(0.5, min(space_width, height) * 0.2):
                return "", "same_run_case_digit_join"
            if compact_unit_suffix_should_join(
                prev_stripped,
                stripped,
                x_gap=spacing_gap,
                height=height,
                space_width=space_width,
            ):
                return "", "compact_unit_suffix_join"
            return " ", "case_digit_boundary_space"
        return "", "join"

    def is_formula_script_atom(
        self,
        previous: LayoutLineTextAtom,
        atom: LayoutLineTextAtom,
    ) -> bool:
        text = atom.text.strip()
        if len(text) != 1 or not text.isalpha():
            return False
        previous_height = previous.advance_bbox[3] - previous.advance_bbox[1]
        height = atom.advance_bbox[3] - atom.advance_bbox[1]
        if previous_height <= 0.0 or height >= previous_height * 0.84:
            return False
        baseline_delta = self.atom_baseline_delta(previous, atom)
        if baseline_delta is None or baseline_delta < max(0.7, previous_height * 0.1):
            return False
        x_gap = atom.advance_bbox[0] - previous.advance_bbox[2]
        return x_gap <= max(2.0, previous_height * 0.25)

    def is_formula_numeric_atom(
        self,
        previous: LayoutLineTextAtom,
        atom: LayoutLineTextAtom,
    ) -> bool:
        text = atom.text.strip()
        if len(text) != 1 or text not in "0123456789" or not previous.text.strip()[-1:].isalpha():
            return False
        previous_height = previous.advance_bbox[3] - previous.advance_bbox[1]
        height = atom.advance_bbox[3] - atom.advance_bbox[1]
        if previous_height <= 0.0 or height >= previous_height * 0.9:
            return False
        baseline_delta = self.atom_baseline_delta(previous, atom)
        if baseline_delta is None or abs(baseline_delta) < max(0.45, previous_height * 0.04):
            return False
        x_gap = atom.advance_bbox[0] - previous.advance_bbox[2]
        return x_gap <= max(2.0, previous_height * 0.25)

    def is_formula_fraction_denominator(
        self,
        previous: LayoutLineTextAtom,
        atom: LayoutLineTextAtom,
    ) -> bool:
        """Recognize a same-size lower glyph stacked under a formula glyph."""
        text = atom.text.strip()
        if not text:
            return False
        previous_text = previous.text.strip()
        if not previous_text:
            return False
        if text[:1] == "∂" and previous_text.endswith((")", "]", "}")):
            return True
        if text[:1] not in "√GT":
            return False
        if text[:1] == "T" and previous_text in {"t", "s"}:
            pass
        elif not previous_text[-1:].isdigit():
            return False
        previous_height = previous.advance_bbox[3] - previous.advance_bbox[1]
        height = atom.advance_bbox[3] - atom.advance_bbox[1]
        if previous_height <= 0.0 or height < previous_height * 0.82:
            return False
        if previous.baseline is None or atom.baseline is None:
            return False
        previous_baseline = baseline_midpoint(previous.baseline, previous.run.rotation_angle)
        atom_baseline = baseline_midpoint(atom.baseline, atom.run.rotation_angle)
        lower_by = previous_baseline - atom_baseline
        if lower_by < max(2.0, previous_height * 0.35):
            return False
        x_gap = atom.advance_bbox[0] - previous.advance_bbox[2]
        return x_gap <= max(2.0, previous_height * 0.25)

    def is_column_gap(
        self,
        spacing_gap: float,
        height: float,
        space_width: float,
    ) -> bool:
        if spacing_gap <= 0.0:
            return False
        threshold = max(
            self.column_gap_threshold,
            space_width * 5.0,
            height * 1.55,
            12.0,
        )
        return spacing_gap >= threshold

    def should_join_word_fragments(
        self,
        prev_text: str,
        text: str,
        *,
        spacing_gap: float,
        height: float,
        space_width: float,
        prev_visible: bool,
        visible: bool,
        allow_short_prefix: bool,
    ) -> bool:
        if self.is_tracked_glyph_line:
            return False
        if self.is_table_like_line:
            # Tables frequently split a word into adjacent text-showing
            # operators while also containing numeric cells.  Preserve the
            # table spacing rules for real cell gaps, but join only very tight
            # fragments (for example ``Vo`` + ``lume``).
            if spacing_gap > max(1.8, min(space_width, height) * 0.25):
                return False
            return should_join_plausible_split_word(
                prev_text,
                text,
                x_gap=spacing_gap,
                height=height,
                space_width=space_width,
                prev_visible=prev_visible,
                visible=visible,
                allow_short_prefix=allow_short_prefix,
            )
        return should_join_plausible_split_word(
            prev_text,
            text,
            x_gap=spacing_gap,
            height=height,
            space_width=space_width,
            prev_visible=prev_visible,
            visible=visible,
            allow_short_prefix=allow_short_prefix,
        )

    def word_gap_threshold(self, run: TextRun, height: float) -> float:
        threshold = max(run.space_width * 0.15, height * 0.08, 0.75)
        if run.baseline is not None:
            threshold = max(threshold, run.font_size * 0.08)
        if self.estimated_char_width is not None:
            threshold = min(threshold, max(self.estimated_char_width * 0.32, 0.75))
        if self.is_all_caps_line:
            threshold = min(threshold, 0.40)
        if self.is_tracked_glyph_line:
            return (
                self.tracked_word_gap
                if self.tracked_word_gap is not None
                else max(threshold, run.space_width * 2.2, height * 1.4)
            )
        return threshold

    @staticmethod
    def atom_baseline_delta(
        left: LayoutLineTextAtom,
        right: LayoutLineTextAtom,
    ) -> float | None:
        left_baseline = left.baseline
        right_baseline = right.baseline
        if left_baseline is None or right_baseline is None:
            return None
        if left.run.rotation_angle in (90, 270) or right.run.rotation_angle in (
            90,
            270,
        ):
            left_mid = (left_baseline[0] + left_baseline[2]) * 0.5
            right_mid = (right_baseline[0] + right_baseline[2]) * 0.5
        else:
            left_mid = (left_baseline[1] + left_baseline[3]) * 0.5
            right_mid = (right_baseline[1] + right_baseline[3]) * 0.5
        return abs(right_mid - left_mid)


def runs_are_left_to_right(runs: list[TextRun]) -> bool:
    if len(runs) < 2:
        return True

    x0_idx = TextRun.X0
    x1_idx = TextRun.X1

    previous = runs[0]
    previous_coords = previous.coords
    prev_x0 = previous_coords[x0_idx]
    prev_x1 = previous_coords[x1_idx]
    prev_height = previous.height_value
    prev_order = previous.order

    for idx in range(1, len(runs)):
        run = runs[idx]
        coords = run.coords
        x0 = coords[x0_idx]
        if x0 < prev_x0:
            return False
        if x0 == prev_x0 and run.order < prev_order:
            return False
        x1 = coords[x1_idx]
        height = run.height_value
        overlap = (prev_x1 if prev_x1 < x1 else x1) - (prev_x0 if prev_x0 > x0 else x0)
        if overlap > (prev_height if prev_height < height else height) * 0.25:
            return False
        prev_x0 = x0
        prev_x1 = x1
        prev_height = height
        prev_order = run.order
    return True


def runs_are_right_to_left(runs: list[TextRun]) -> bool:
    if len(runs) < 2:
        return False
    ordered = runs
    for run in runs:
        if not run.has_text:
            ordered = [text_run for text_run in runs if text_run.has_text]
            if len(ordered) < 2:
                return False
            break

    stream_sorted = sorted(ordered, key=lambda r: (r.order, r.stream_order))
    decreases = 0
    increases = 0
    text_run_x0 = TextRun.X0
    prev_x0 = stream_sorted[0].coords[text_run_x0]
    for idx in range(1, len(stream_sorted)):
        x0 = stream_sorted[idx].coords[text_run_x0]
        if x0 < prev_x0:
            decreases += 1
        elif x0 > prev_x0:
            increases += 1
        prev_x0 = x0
    return decreases > increases


def has_interleaved_horizontal_overlap(runs: list[TextRun]) -> bool:
    previous: TextRun | None = None
    prev_x0 = 0.0
    prev_x1 = 0.0
    prev_space_width = 0.0
    for idx in range(len(runs)):
        run = runs[idx]
        if not run.has_text:
            continue
        coords = run.coords
        x0 = coords[TextRun.X0]
        x1 = coords[TextRun.X1]
        space_width = run.space_width
        if previous is not None:
            overlap = (prev_x1 if prev_x1 < x1 else x1) - (prev_x0 if prev_x0 > x0 else x0)
            min_width = min(prev_x1 - prev_x0, x1 - x0)
            threshold = max(2.5, min_width * 0.45, max(prev_space_width, space_width) * 0.8)
            if overlap > threshold:
                return True
        previous = run
        prev_x0 = x0
        prev_x1 = x1
        prev_space_width = space_width
    return False


def is_tracked_glyph_run_line(non_space_runs: list[TextRun], *, has_explicit_spaces: bool) -> bool:
    if len(non_space_runs) < 6:
        return False

    single_glyph_runs = sum(1 for r in non_space_runs if len(r.stripped_text) == 1)
    if single_glyph_runs < len(non_space_runs) * 0.72:
        return False

    text_chars = sum(len(r.stripped_text) for r in non_space_runs)
    if text_chars > len(non_space_runs) * 1.5:
        return False

    positive_gaps = []
    previous = non_space_runs[0]
    for run in non_space_runs[1:]:
        gap = run.x0 - previous.x1
        if gap > 0:
            positive_gaps.append(gap)
        previous = run

    if len(positive_gaps) < len(non_space_runs) * 0.45:
        return False

    positive_gaps.sort()
    typical_gap = positive_gaps[len(positive_gaps) // 2]
    typical_height = max(r.height for r in non_space_runs)
    return typical_gap <= max(typical_height * 1.2, 24.0)


def explicit_spaces_should_control_glyph_gaps(
    non_space_runs: list[TextRun],
    *,
    explicit_space_count: int,
) -> bool:
    if explicit_space_count < 2 or len(non_space_runs) < 8:
        return False
    single_glyph_runs = sum(1 for run in non_space_runs if len(run.stripped_text) == 1)
    if single_glyph_runs < len(non_space_runs) * 0.72:
        return False
    text_chars = sum(len(run.stripped_text) for run in non_space_runs)
    return not text_chars > len(non_space_runs) * 1.5


def tracked_glyph_word_gap_threshold(runs: list[TextRun]) -> float | None:
    gaps: list[float] = []
    previous: TextRun | None = None
    for run in runs:
        if not run.has_text:
            continue
        if previous is not None:
            gap = run.x0 - previous.x1
            if gap > 0:
                gaps.append(gap)
        previous = run
    if len(gaps) < 5:
        return None
    gaps.sort()
    median_gap = gaps[len(gaps) // 2]
    upper_gap = gaps[int(len(gaps) * 0.8)]
    if upper_gap <= median_gap * 1.8:
        return None
    typical_height = max((r.height for r in runs if r.has_text), default=0.0)
    return max(median_gap * 2.2, typical_height * 0.22, 1.5)


def column_gap_threshold_for_runs(runs: list[TextRun]) -> float:
    non_space_runs = [run for run in runs if run.has_text]
    heights = [run.height for run in non_space_runs if run.height > 0.0]
    spaces = [run.space_width for run in non_space_runs if run.space_width > 0.0]
    typical_height = median_low(heights) if heights else 0.0
    typical_space = median_low(spaces) if spaces else 0.0

    gaps: list[float] = []
    previous: TextRun | None = None
    for run in non_space_runs:
        if previous is not None:
            gap = run.x0 - previous.x1
            if gap > 0.0:
                gaps.append(gap)
        previous = run
    typical_gap = median_low(gaps) if gaps else 0.0
    return max(
        typical_gap * 4.0,
        typical_space * 5.0,
        typical_height * 1.55,
        12.0,
    )


def estimated_char_width_for_suspect_line(sorted_runs: list[TextRun]) -> float | None:
    non_space_runs = [
        run
        for run in sorted_runs
        if run.visible and run.has_text and run.stripped_text and run.stripped_text.isalpha()
    ]
    if len(non_space_runs) < 5:
        return None
    suspect_runs = 0
    for run in non_space_runs:
        text_len = len(run.stripped_text)
        width = run.x1 - run.x0
        if width <= 0.0:
            suspect_runs += 1
            continue
        if abs((width / max(1, text_len)) - run.space_width) <= max(1.0, run.space_width * 0.05):
            suspect_runs += 1
    if suspect_runs < max(3, len(non_space_runs) // 3):
        return None

    ratios: list[float] = []
    previous: TextRun | None = None
    for run in non_space_runs:
        if previous is not None:
            prev_len = len(previous.stripped_text)
            if prev_len > 0:
                delta = run.x0 - previous.x0
                ratio = delta / prev_len
                space_width = max(1.0, min(previous.space_width, run.space_width))
                if space_width * 0.18 <= ratio <= space_width * 0.95:
                    ratios.append(ratio)
        previous = run
    if len(ratios) < 3:
        return None
    ratios.sort()
    median_ratio = ratios[len(ratios) // 2]
    typical_space = median_low([run.space_width for run in non_space_runs if run.space_width > 0.0])
    if typical_space <= 0.0:
        return median_ratio
    return min(median_ratio, typical_space * 0.48)


def should_use_estimated_word_spacing(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    if previous == "T" and current in {"he", "hes", "hese", "his"}:
        return True
    return not (not previous[-1].isalpha() or not current[0].isalpha())


@lru_cache(maxsize=4096)
def ranked_alpha_word(text: str) -> int | None:
    normalized = text.casefold()
    if not normalized.isalpha():
        return None
    return word_rank(normalized)


def repair_table_split_word_boundaries(text: str) -> str:
    """Join dictionary-backed fragments split by table glyph spacing."""
    tokens = text.split(" ")
    if len(tokens) < 2:
        return text
    index = 0
    while index + 1 < len(tokens):
        left = tokens[index]
        right = tokens[index + 1]
        if table_split_word_join_is_plausible(left, right):
            tokens[index : index + 2] = [left + right]
            if index > 0:
                index -= 1
            continue
        index += 1
    repaired = " ".join(tokens)
    return re.sub(r"\s+([.,;:)\]])", r"\1", repaired)


def table_split_word_join_is_plausible(left: str, right: str) -> bool:
    if not left.isalpha() or not right.isalpha():
        return False
    joined = left + right
    joined_rank = ranked_alpha_word(joined)
    if joined_rank is None or len(joined) < 3:
        return False
    left_rank = ranked_alpha_word(left)
    right_rank = ranked_alpha_word(right)
    if left_rank is None or right_rank is None:
        return True
    if joined_rank < min(left_rank, right_rank):
        return True
    return joined_rank <= 10_000 and max(left_rank, right_rank) >= 20_000


def trailing_alpha_token(text: str) -> str:
    index = len(text)
    while index > 0 and text[index - 1].isalpha():
        index -= 1
    return text[index:]


def leading_alpha_token(text: str) -> str:
    index = 0
    limit = len(text)
    while index < limit and text[index].isalpha():
        index += 1
    return text[:index]


def is_high_frequency_boundary_word(text: str) -> bool:
    rank = ranked_alpha_word(text.strip())
    return rank is not None and rank <= 250


def should_insert_phrase_continuation_space(previous: str, current: str) -> bool:
    tail = trailing_alpha_token(previous.strip())
    head = leading_alpha_token(current.strip())
    if len(head) < 3 or not tail:
        return False
    return len(tail) >= 3 or is_high_frequency_boundary_word(tail)


def should_join_plausible_split_word(
    previous: str,
    current: str,
    *,
    x_gap: float,
    height: float,
    space_width: float,
    prev_visible: bool,
    visible: bool,
    allow_short_prefix: bool = False,
) -> bool:
    if not (prev_visible and visible):
        return False
    prev = previous.strip()
    text = current.strip()
    if not prev or not text:
        return False
    tail = trailing_alpha_token(prev)
    head = leading_alpha_token(text)
    if len(tail) < (1 if allow_short_prefix else 3) or not head:
        return False
    if not (tail[-1].islower() and head[0].islower()):
        return False
    if x_gap < -max(0.5, min(space_width, height) * 0.2):
        return False
    if x_gap > max(space_width * 1.45, height * 0.45, 4.5):
        return False
    if allow_short_prefix and len(tail) < 3:
        # A producer may split one word into several short text-showing
        # operators (``V`` + ``o`` + ``l`` + ``u`` + ``m`` + ``e``). There is
        # no useful dictionary candidate until the final fragment arrives, so
        # keep only tight, lowercase joins in this opt-in path.
        return x_gap <= max(1.8, min(space_width, height) * 0.25)
    joined = f"{tail}{head}"
    joined_rank = ranked_alpha_word(joined)
    if joined_rank is None or joined_rank > 150_000:
        return False
    tail_rank = ranked_alpha_word(tail)
    head_rank = ranked_alpha_word(head)
    if tail_rank is None:
        return True
    if head_rank is None and joined_rank <= 75_000 and len(tail) <= 5:
        return True
    if head_rank is not None and joined_rank < min(tail_rank, head_rank):
        return True
    if len(head) <= 3 and joined_rank <= max(tail_rank * 8, 150_000):
        return True
    return bool(head_rank is None and joined_rank < tail_rank)


def digit_fragments_are_tightly_joined(
    previous: str,
    current: str,
    *,
    x_gap: float,
    height: float,
    space_width: float,
) -> bool:
    prev = previous.strip()
    text = current.strip()
    if not prev or not text or not prev[-1].isdigit() or not text[0].isdigit():
        return False
    return x_gap <= max(0.25, min(space_width, height) * 0.1)


def should_insert_tight_word_space(
    *,
    prev_text: str,
    text: str,
    x_gap: float,
    height: float,
    space_width: float,
) -> bool:
    if not prev_text or not text:
        return False
    prev = prev_text.strip()
    current = text.strip()
    if len(prev) <= 1 or len(current) <= 1:
        return False
    prev_last = prev[-1]
    current_first = current[0]
    if not prev_last.isalpha() or not current_first.isalpha():
        return False

    max_overlap = max(0.25, min(space_width, height) * 0.05)
    if x_gap < -max_overlap:
        return False

    if is_high_frequency_boundary_word(prev) or is_high_frequency_boundary_word(current):
        return True
    return bool(prev.isupper() and current_first.islower() and len(prev) <= 8)


def should_insert_hidden_ocr_overlap_space(
    *,
    prev_text: str,
    text: str,
    x_gap: float,
    height: float,
    space_width: float,
    prev_visible: bool,
    visible: bool,
) -> bool:
    if prev_visible or visible:
        return False
    prev = prev_text.strip()
    current = text.strip()
    if len(prev) <= 1 or len(current) <= 1:
        return False
    prev_last = prev[-1]
    current_first = current[0]
    if not prev_last.isalnum() or not current_first.isalnum():
        return False
    max_overlap = max(1.25, min(space_width, height) * 0.35)
    if x_gap < -max_overlap:
        return False
    if prev_last.isdigit() and current_first.isdigit():
        return False
    if prev.isupper() and current.isupper():
        return True
    if prev_last.isdigit() and current_first.isalpha():
        return True
    if prev_last.isalpha() and current_first.isdigit():
        return True
    return prev_last.islower() != current_first.islower()


def split_glued_numeric_label_boundaries(text: str) -> str:
    if not text:
        return text
    output: list[str] = []
    for index, ch in enumerate(text):
        if index > 0 and should_split_glued_numeric_label(text, index):
            output.append(" ")
        output.append(ch)
    return "".join(output)


def should_split_glued_numeric_label(text: str, index: int) -> bool:
    ch = text[index]
    prev = text[index - 1]
    if ch.isalpha() and prev.isdigit():
        left = text[max(0, index - 8) : index]
        return any(c in left for c in "./,:") and ch.isupper()
    if ch.isdigit() and prev.isalpha():
        left = text[max(0, index - 4) : index].casefold()
        return (
            left.endswith((" m", " m.", " g", " g.", " l", " l."))
            and len(text) > index + 1
            and text[index + 1].isspace()
        )
    return bool(ch.isdigit() and prev == ":")


def strip_private_use_chars(text: str) -> str:
    cleaned = "".join(ch for ch in text if not is_private_use_or_control(ch) and ch not in "»«•·●")
    if "...." in cleaned:
        while "...." in cleaned:
            cleaned = cleaned.replace("....", "..")
    if "----" in cleaned:
        while "----" in cleaned:
            cleaned = cleaned.replace("----", "--")
    return cleaned


def collapse_repeated_spaces(text: str) -> str:
    if "  " not in text:
        return text
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def baseline_midpoint(
    baseline: tuple[float, float, float, float],
    rotation_angle: int,
) -> float:
    if rotation_angle in (90, 270):
        return (baseline[0] + baseline[2]) * 0.5
    return (baseline[1] + baseline[3]) * 0.5


def formula_like_runs(runs: list[TextRun]) -> bool:
    text = "".join(run.text for run in runs if run.has_text)
    return bool(FORMULA_MARKERS.intersection(text))


def reorder_stacked_formula_numerators(runs: list[TextRun]) -> list[TextRun]:
    """Place a vertically stacked numeric numerator before its denominator.

    PDF math producers commonly paint a fraction's numerator and denominator
    as independent glyphs at the same x position.  The ordinary horizontal
    sort consequently emits the denominator first.  Keep the correction
    limited to a small digit/formula overlap so ordinary subscripts and table
    values retain their existing order.
    """
    reordered = list(runs)
    for index in range(1, len(reordered)):
        numerator = reordered[index]
        numerator_text = numerator.stripped_text
        if not (
            (numerator_text.isdigit() and len(numerator_text) <= 2) or numerator_text in {"t", "s"}
        ):
            continue
        denominator_index = next(
            (
                candidate_index
                for candidate_index in range(index - 1, -1, -1)
                if stacked_formula_denominator(
                    reordered[candidate_index],
                    numerator,
                    following=reordered[index + 1] if index + 1 < len(reordered) else None,
                )
            ),
            None,
        )
        if denominator_index is None:
            continue
        denominator = reordered[denominator_index]
        reordered[denominator_index:index] = [numerator, *reordered[denominator_index:index]]
        reordered[index] = denominator
    return reordered


def stacked_formula_denominator(
    denominator: TextRun,
    numerator: TextRun,
    *,
    following: TextRun | None = None,
) -> bool:
    if not denominator.has_text or not numerator.has_text:
        return False
    if denominator.stripped_text.isdigit():
        return False
    if denominator.stripped_text[:1] not in "√GT":
        return False
    numerator_text = numerator.stripped_text
    if denominator.stripped_text[:1] == "T" and numerator_text in {"t", "s"}:
        if following is None or following.stripped_text[:1] not in ",;:)]}]:":
            return False
    elif not numerator_text.isdigit():
        return False
    if denominator.rotation_angle != 0 or numerator.rotation_angle != 0:
        return False
    denominator_height = denominator.height_value
    numerator_height = numerator.height_value
    if denominator_height <= 0.0 or numerator_height <= 0.0:
        return False
    if numerator_height < denominator_height * 0.7 or numerator_height > denominator_height * 1.3:
        return False
    overlap = min(denominator.x1, numerator.x1) - max(denominator.x0, numerator.x0)
    if overlap <= 0.0:
        return False
    vertical_gap = numerator.y0 - denominator.y0
    return vertical_gap >= max(2.0, denominator_height * 0.18)


def script_digit_text(text: str) -> bool:
    if len(text) == 1:
        return text in SCRIPT_DIGITS
    stripped = text.strip()
    return bool(stripped) and all(ch in SCRIPT_DIGITS for ch in stripped)


def inline_marker_text(text: str) -> bool:
    if len(text) == 1:
        return text in INLINE_MARKERS
    return text.strip() in INLINE_MARKERS


def compact_unit_suffix_should_join(
    previous: str,
    current: str,
    *,
    x_gap: float,
    height: float,
    space_width: float,
) -> bool:
    prev = previous.strip()
    text = current.strip()
    if text != "V" or not prev or not any(ch.isdigit() for ch in prev):
        return False
    if prev[-1:] not in {"k", "K", "m", "M"}:
        return False
    return x_gap <= max(0.5, min(space_width, height) * 0.15)


def chemical_subscript_prefix_text(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1:].isalpha() and stripped[-1:].isupper()


def is_private_use_or_control(ch: str) -> bool:
    codepoint = ord(ch)
    if ch in "\t\n\r":
        return False
    if codepoint == 0xFFFD or codepoint == 0x00AD:
        return True
    if 0xE000 <= codepoint <= 0xF8FF:
        return True
    category = unicodedata.category(ch)
    return category in {"Cc", "Cf", "Cs", "Co", "Cn"}


def is_structural_list_marker_run(run: TextRun) -> bool:
    return run.stripped_text == "\u25cf"


def is_decorative_leader(text: str) -> bool:
    if not text:
        return False
    if len(text) < 3:
        return False
    first_text_char = text[0]
    leader_chars = LEADER_START_CHARS
    if first_text_char not in leader_chars and not first_text_char.isspace():
        return False
    if first_text_char in leader_chars:
        last_non_space = len(text) - 1
        while last_non_space >= 0 and text[last_non_space].isspace():
            last_non_space -= 1
        if last_non_space < 2:
            return False
        for ch in text[: last_non_space + 1]:
            if ch not in leader_chars and not ch.isspace():
                return False
        return True
    stripped = text.strip()
    if len(stripped) < 3:
        return False
    first = stripped[0]
    if first not in leader_chars:
        return False
    return all(not (ch not in leader_chars and not ch.isspace()) for ch in stripped)


def is_tiny_page_footer(text: str) -> bool:
    return bool(FOOTER_RE.match(text))


def trailing_tiny_page_label_run_indexes(sorted_runs: list[TextRun]) -> set[int]:
    text_indexes = [index for index, run in enumerate(sorted_runs) if run.stripped_text]
    if len(text_indexes) < 3:
        return set()
    digit_index = text_indexes[-1]
    page_index = text_indexes[-2]
    previous_index = text_indexes[-3]
    digit_run = sorted_runs[digit_index]
    page_run = sorted_runs[page_index]
    previous_run = sorted_runs[previous_index]
    if page_run.stripped_text.casefold() != "page":
        return set()
    if not digit_run.stripped_text.isdigit():
        return set()
    page_width = page_run.x1 - page_run.x0
    digit_width = digit_run.x1 - digit_run.x0
    pair_gap = page_run.x0 - previous_run.x1
    if page_width <= 0.0 or digit_width <= 0.0:
        return set()
    significant_font = max(
        (
            run.font_size
            for run in sorted_runs
            if run.stripped_text
            and run.stripped_text.casefold() != "page"
            and not run.stripped_text.isdigit()
        ),
        default=0.0,
    )
    compact_pair = (
        page_width <= 16.0
        and digit_width <= 14.0
        and pair_gap >= max(32.0, previous_run.space_width * 8.0)
    )
    if not compact_pair and (
        page_run.font_size > 6.5
        or digit_run.font_size > 6.5
        or page_width > 20.0
        or digit_width > 16.0
    ):
        return set()
    if (
        significant_font > 0.0
        and not compact_pair
        and max(page_run.font_size, digit_run.font_size) >= significant_font * 0.7
    ):
        return set()
    if digit_run.x0 < page_run.x0:
        return set()
    return {page_index, digit_index}
