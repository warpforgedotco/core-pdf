# SPDX-License-Identifier: AGPL-3.0-only
"""Assemble glyph runs into layout lines, words, and their reconstructed text."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from core_pdf.impl._impl.layout import text_rules as rules
from core_pdf.impl._impl.model.runs import (
    EMPTY_LAYOUT_LINE_TEXT,
    LayoutLineText,
    LayoutLineTextSegment,
    TextRun,
)
from core_pdf.impl._impl.model.text import WORD_GAP_SIZE_FACTOR, word_gap_threshold

# A digit is a superscript of the preceding text when it is clearly shorter and its
# baseline sits clearly above; the run-level and atom-level passes share the rule.
SUPERSCRIPT_HEIGHT_RATIO = 0.9
SUPERSCRIPT_BASELINE_MIN = 0.45
SUPERSCRIPT_BASELINE_RATIO = 0.05


def is_superscript_metrics(previous_height: float, height: float, baseline_raise: float) -> bool:
    if previous_height <= 0.0 or height >= previous_height * SUPERSCRIPT_HEIGHT_RATIO:
        return False
    return baseline_raise >= max(
        SUPERSCRIPT_BASELINE_MIN, previous_height * SUPERSCRIPT_BASELINE_RATIO
    )


SUPERSCRIPT_DIGIT_TRANSLATION = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
SUBSCRIPT_DIGIT_TRANSLATION = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


# Builder-only atoms are short-lived and never escape into the immutable layout result.
@dataclass(slots=True)
class LayoutLineTextAtom:
    text: str
    run: TextRun
    advance_bbox: tuple[float, float, float, float]
    baseline: tuple[float, float, float, float] | None
    has_glyph_geometry: bool


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
        return LayoutLineText(text, (line_text_segment(run, text, ""),))

    angle = runs[0].rotation_angle
    if angle == 0 and len(runs) <= 3:
        sorted_runs = (
            runs
            if rules.runs_are_left_to_right(runs)
            else sorted(runs, key=lambda r: (r.x0, r.order))
        )
        return GlyphLineBuilder(sorted_runs).build()

    if angle == 0 and rules.runs_are_left_to_right(runs):
        sorted_runs = runs
    elif rules.runs_are_right_to_left(runs):
        sorted_runs = sorted(runs, key=lambda r: (r.order, r.stream_order))
    else:
        if angle == 90:
            sorted_runs = sorted(runs, key=lambda r: (r.y0, r.order))
        elif angle == 270:
            sorted_runs = sorted(runs, key=lambda r: (-r.y1, r.order))
        else:
            sorted_runs = sorted(runs, key=lambda r: (r.x0, r.order))
            if rules.has_interleaved_horizontal_overlap(sorted_runs):
                sorted_runs = sorted(runs, key=lambda r: (r.order, r.stream_order))

    is_formula_like_line = rules.formula_like_runs(sorted_runs)
    if angle == 0 and is_formula_like_line:
        sorted_runs = rules.reorder_stacked_formula_numerators(sorted_runs)
        # Reordering rewrites the run sequence, so the classification is re-derived from
        # it.  Every other line keeps the value computed above rather than rebuilding the
        # joined line text a second time.
        is_formula_like_line = rules.formula_like_runs(sorted_runs)

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
    is_tracked_glyph_line = rules.is_tracked_glyph_run_line(non_space_runs)

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
) -> LayoutLineTextSegment:
    return LayoutLineTextSegment(
        text=text,
        separator_before=separator_before,
        advance_bbox=run.advance_bbox,
        rotation_angle=run.rotation_angle,
    )


def line_text_segment_from_atom(
    atom: LayoutLineTextAtom,
    separator_before: str,
) -> LayoutLineTextSegment:
    run = atom.run
    return LayoutLineTextSegment(
        text=atom.text,
        separator_before=separator_before,
        advance_bbox=atom.advance_bbox,
        rotation_angle=run.rotation_angle,
    )


def render_single_run_text(run: TextRun) -> str:
    text = run.text
    if not text:
        return ""
    if rules.is_structural_list_marker_run(run):
        return ""
    if not text.isprintable() and any(rules.is_private_use_or_control(ch) for ch in text):
        text = rules.strip_private_use_chars(text)
        if not text:
            return ""
    if rules.is_tiny_page_footer(text) and run.font_size <= 5.0:
        return ""
    return rules.collapse_repeated_spaces(text)


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
                segments.append(line_text_segment(run, " ", " "))
            previous_run = run
            continue
        if not run.has_text:
            continue
        if rules.is_structural_list_marker_run(run):
            continue
        if not text.isprintable() and any(rules.is_private_use_or_control(ch) for ch in text):
            text = rules.strip_private_use_chars(text)
        text = text.strip()
        if text:
            separator = ""
            if (
                parts
                and parts[-1] != " "
                and geometric_cell_spacing
                and previous_run is not None
                and rotated_table_run_gap(previous_run, run)
                > max(1.0, min(previous_run.font_size, run.font_size) * 0.40)
            ):
                parts.append(" ")
                separator = " "
            parts.append(text)
            segments.append(line_text_segment(run, text, separator))
            previous_run = run
    combined = rules.split_glued_numeric_label_boundaries("".join(parts))
    if not combined:
        return EMPTY_LAYOUT_LINE_TEXT
    return LayoutLineText(combined, tuple(segments))


def rotated_table_run_gap(previous: TextRun, current: TextRun) -> float:
    if current.rotation_angle == 270:
        return previous.y0 - current.y1
    return current.y0 - previous.y1


def internal_is_short_digit_run(
    run: TextRun,
    *,
    max_length: int,
    require_baseline: bool = True,
) -> bool:
    """Whether ``run`` is a short, unrotated, digits-only run with no surrounding space."""
    stripped = run.stripped_text
    return bool(
        stripped
        and stripped == run.text
        and stripped.isdigit()
        and len(stripped) <= max_length
        and run.rotation_angle == 0
        and (run.baseline is not None or not require_baseline)
    )


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
        self.page_label_indexes = rules.trailing_tiny_page_label_run_indexes(runs)
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
            rules.tracked_glyph_word_gap_threshold(runs) if is_tracked_glyph_line else None
        )
        non_space_runs = [run for run in runs if run.has_text]
        self.explicit_spaces_control_glyph_gaps = rules.explicit_spaces_should_control_glyph_gaps(
            non_space_runs,
            explicit_space_count=sum(1 for run in runs if run.text_is_space),
        )
        self.next_non_space_texts: list[str] = []
        self.next_non_space_x0s: list[float] = []
        self.estimated_char_width = (
            None
            if is_table_like_line or is_tracked_glyph_line or self.has_explicit_spaces
            else rules.estimated_char_width_for_suspect_line(runs)
        )
        self.column_gap_threshold = rules.column_gap_threshold_for_runs(runs)
        if self.has_explicit_spaces:
            self.internal_prepare_explicit_space_context()

    def build(self) -> LayoutLineText:
        parts: list[str] = []
        segments: list[LayoutLineTextSegment] = []
        append_part = parts.append
        append_segment = segments.append

        prev_run: TextRun | None = None
        prev_run_text = ""
        prev_last_char = ""
        prev_atom: LayoutLineTextAtom | None = None
        recent_emitted_runs: list[tuple[tuple[float, float, float, float], str]] = []

        for index, run in enumerate(self.runs):
            text = self.normalized_text(run, index)
            if not text:
                continue
            stripped = text.strip()
            if not stripped and not text.isspace():
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
                if prev_atom is not None:
                    separator_before = self.atom_separator(prev_atom, atom)
                    if separator_before:
                        append_part(separator_before)
                append_part(atom_text)
                append_segment(
                    line_text_segment_from_atom(
                        atom,
                        separator_before,
                    )
                )
                prev_atom = atom
                emitted_run = True

            if not emitted_run:
                continue
            prev_run = run
            prev_run_text = text
            prev_last_char = text[-1:]
            recent_emitted_runs.append((run.advance_bbox, text))
            if len(recent_emitted_runs) > 256:
                del recent_emitted_runs[:64]

        combined = "".join(parts)
        if self.suppress_tiny_page_footer and rules.is_tiny_page_footer(combined):
            return EMPTY_LAYOUT_LINE_TEXT
        text = rules.collapse_repeated_spaces(combined)
        text = rules.repair_table_split_word_boundaries(text)
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
            # A text-showing operation that already carries whitespace has
            # authoritative word boundaries. Keep it as one atom so ordinary
            # authored lines do not get exploded into hundreds of glyph objects
            # merely to join them back together unchanged.
            if any(character.isspace() for character in text):
                return (
                    LayoutLineTextAtom(
                        text=text,
                        run=run,
                        advance_bbox=run.advance_bbox,
                        baseline=run.baseline,
                        has_glyph_geometry=False,
                    ),
                )
            atoms = [
                LayoutLineTextAtom(
                    text=cluster.text,
                    run=run,
                    advance_bbox=cluster.advance_bbox,
                    baseline=cluster.baseline,
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
                advance_bbox=run.advance_bbox,
                baseline=run.baseline,
                has_glyph_geometry=False,
            ),
        )

    def internal_prepare_explicit_space_context(self) -> None:
        runs = self.runs
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

    def normalized_text(self, run: TextRun, index: int) -> str:
        text = run.text
        if not text:
            return ""
        if index in self.page_label_indexes:
            if run.stripped_text.casefold() == "page":
                return ""
            text = run.stripped_text
        if rules.is_structural_list_marker_run(run):
            return ""
        if not text.isprintable() and any(rules.is_private_use_or_control(ch) for ch in text):
            text = rules.strip_private_use_chars(text)
            if not text:
                return ""
        if rules.is_tiny_page_footer(text) and run.font_size <= 5.0:
            return ""
        first_char = text[:1]
        if (
            first_char in rules.LEADER_START_CHARS or first_char.isspace()
        ) and rules.is_decorative_leader(text):
            return ""
        if self.is_trademark_marker_run(run, index):
            return "™"
        if self.is_superscript_like_numeric_run(run, index) or self.is_unit_exponent_run(run):
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
        return self.internal_is_shifted_script_run(
            run,
            self.previous_non_space_run(index),
            self.next_non_space_run(index),
            context_text_ok=lambda text: text != "TM",
            font_size_ratio=0.82,
            baseline_drops=False,
            shift_minimum=1.5,
            shift_factor=0.18,
            attach_factor=0.25,
            attach_previous_only=False,
        )

    def is_superscript_like_numeric_run(self, run: TextRun, index: int) -> bool:
        if not internal_is_short_digit_run(run, max_length=4):
            return False
        return self.internal_is_shifted_script_run(
            run,
            self.previous_non_space_run(index),
            self.next_non_space_run(index),
            context_text_ok=lambda text: not text.isdigit(),
            font_size_ratio=0.8,
            baseline_drops=False,
            shift_minimum=1.5,
            shift_factor=0.18,
            attach_factor=0.25,
            attach_previous_only=False,
        )

    def is_subscript_like_numeric_run(self, run: TextRun, index: int) -> bool:
        if not internal_is_short_digit_run(run, max_length=3):
            return False
        previous = self.previous_non_space_run(index)
        if previous is None or not rules.chemical_subscript_prefix_text(previous.stripped_text):
            return False
        return self.internal_is_shifted_script_run(
            run,
            previous,
            self.next_non_space_run(index),
            context_text_ok=lambda text: not text.isdigit(),
            font_size_ratio=0.82,
            baseline_drops=True,
            shift_minimum=0.45,
            shift_factor=0.05,
            attach_factor=0.2,
            attach_previous_only=True,
        )

    def internal_is_shifted_script_run(
        self,
        run: TextRun,
        previous: TextRun | None,
        following: TextRun | None,
        *,
        context_text_ok: Callable[[str], bool],
        font_size_ratio: float,
        baseline_drops: bool,
        shift_minimum: float,
        shift_factor: float,
        attach_factor: float,
        attach_previous_only: bool,
    ) -> bool:
        """Shared gate for runs shifted off the surrounding baseline.

        Classifies a run as script-like when its neighbours accept it as
        context, its font is small enough relative to theirs, its baseline is
        raised (or dropped) far enough, and it sits tightly attached to a
        neighbour.
        """
        if run.baseline is None:
            return False
        context_runs = [
            candidate
            for candidate in (previous, following)
            if candidate is not None
            and candidate.baseline is not None
            and candidate.rotation_angle == 0
            and candidate.stripped_text
            and context_text_ok(candidate.stripped_text)
        ]
        if not context_runs:
            return False
        context_font_size = max(candidate.font_size for candidate in context_runs)
        if context_font_size <= 0.0 or run.font_size >= context_font_size * font_size_ratio:
            return False
        run_baseline = rules.baseline_midpoint(run.baseline, 0)
        baseline_shift = max(
            (
                rules.baseline_midpoint(candidate.baseline, 0) - run_baseline
                if baseline_drops
                else run_baseline - rules.baseline_midpoint(candidate.baseline, 0)
            )
            for candidate in context_runs
            if candidate.baseline is not None
        )
        if baseline_shift < max(shift_minimum, context_font_size * shift_factor):
            return False
        attach_gap = max(run.space_width * 0.5, context_font_size * attach_factor, 2.0)
        if attach_previous_only:
            return previous is not None and run.x0 - previous.x1 <= attach_gap
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

    def is_formula_subscript_like_numeric_run(self, run: TextRun, index: int) -> bool:
        """Recognize numeric subscripts attached to mathematical variables."""
        run_baseline = run.baseline
        if not internal_is_short_digit_run(run, max_length=3) or run_baseline is None:
            return False
        previous = self.previous_non_space_run(index)
        if previous is None or not previous.stripped_text[-1:].isalpha():
            return False
        previous_baseline = previous.baseline
        if previous_baseline is None:
            return False
        previous_height = previous.height
        baseline_drop = rules.baseline_midpoint(previous_baseline, 0) - rules.baseline_midpoint(
            run_baseline, 0
        )
        if not is_superscript_metrics(previous_height, run.height, baseline_drop):
            return False
        attach_gap = max(run.space_width * 0.5, previous_height * 0.2, 2.0)
        return run.x0 - previous.x1 <= attach_gap

    def is_unit_exponent_run(self, run: TextRun) -> bool:
        if not internal_is_short_digit_run(run, max_length=2, require_baseline=False):
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

    def is_recent_duplicate_overlap(
        self,
        recent_runs: list[tuple[tuple[float, float, float, float], str]],
        run: TextRun,
        text: str,
    ) -> bool:
        if not recent_runs:
            return False
        x0, y0, x1, y1 = run.advance_bbox
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
    ) -> str:
        prev_run = previous.run
        run = atom.run
        prev_text = previous.text
        text = atom.text
        if prev_text.endswith(" ") or text.startswith(" "):
            return ""
        prev_last_char = prev_text[-1:]
        first_char = text[:1]
        if not prev_last_char or not first_char:
            return ""
        if (
            prev_run is run
            and " " in run.text
            and previous.has_glyph_geometry
            and atom.has_glyph_geometry
        ):
            return ""

        prev_x0, internal_prev_y0, prev_x1, internal_prev_y1 = previous.advance_bbox
        x0, y0, internal_x1, y1 = atom.advance_bbox
        height = y1 - y0
        x_gap = x0 - prev_x1
        spacing_gap = x_gap
        space_width = run.space_width
        estimated_char_width = self.estimated_char_width
        prev_stripped: str | None = None
        stripped: str | None = None
        if estimated_char_width is not None:
            stripped = text.strip()
            prev_stripped = prev_text.strip()
            tight_fragment_gap = max(1.8, min(space_width, height) * 0.25)
            if (
                rules.should_use_estimated_word_spacing(prev_stripped, stripped)
                and x_gap > tight_fragment_gap
            ):
                spacing_gap = x0 - (prev_x0 + len(prev_stripped) * estimated_char_width)
        baseline_delta = self.atom_baseline_delta(previous, atom)
        if rules.inline_marker_text(text):
            return ""
        if self.is_formula_like_line and self.is_formula_numeric_atom(previous, atom):
            return " "
        if self.is_formula_like_line and self.is_formula_fraction_denominator(previous, atom):
            return "/"
        if rules.script_digit_text(prev_text) and text[:1] in ")]},.;:":
            return ""
        if rules.script_digit_text(text):
            return ""
        if self.is_formula_like_line and self.is_formula_script_atom(previous, atom):
            return " "
        if baseline_delta is not None and baseline_delta > max(height * 0.42, 2.0):
            return " "

        if self.is_column_gap(spacing_gap, height, space_width):
            return " "

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
            return " "
        if (
            self.is_all_caps_line
            and " " in prev_stripped
            and prev_last_char.isupper()
            and first_char.isupper()
            and len(stripped) >= 4
            and x_gap >= -max(0.6, height * 0.08)
        ):
            return " "
        if not (prev_run.visible and run.visible) and rules.should_insert_tight_word_space(
            prev_text=prev_stripped,
            text=stripped,
            x_gap=spacing_gap,
            height=height,
            space_width=space_width,
        ):
            return " "
        if rules.should_insert_hidden_text_overlap_space(
            prev_text=prev_stripped,
            text=stripped,
            x_gap=spacing_gap,
            height=height,
            space_width=space_width,
            prev_visible=prev_run.visible,
            visible=run.visible,
        ):
            return " "
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
            return ""
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
            return " "
        if (
            not (previous.has_glyph_geometry and atom.has_glyph_geometry)
            and " " in prev_stripped
            and prev_last_char.islower()
            and first_char.islower()
            and spacing_gap >= max(0.25, min(space_width, height) * 0.08)
            and spacing_gap <= max(0.5, height * 0.04)
            and rules.should_insert_phrase_continuation_space(prev_stripped, stripped)
            and len(stripped.split(" ", 1)[0]) >= 3
        ):
            return " "
        if (
            not self.is_table_like_line
            and not self.is_tracked_glyph_line
            and prev_last_char.islower()
            and first_char.islower()
            and len(prev_stripped) > 1
            and len(stripped) > 1
            and spacing_gap > max(0.45, min(space_width, height) * 0.12)
        ):
            return " "
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
            return " "
        if spacing_gap >= -max(threshold, 2.5) and (
            (prev_last_char.islower() and first_char.isupper())
            or (
                prev_last_char.isdigit()
                and first_char.isdigit()
                and not rules.digit_fragments_are_tightly_joined(
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
                return ""
            if rules.compact_unit_suffix_should_join(
                prev_stripped,
                stripped,
                x_gap=spacing_gap,
                height=height,
                space_width=space_width,
            ):
                return ""
            return " "
        return ""

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
        baseline_delta = self.atom_baseline_delta(previous, atom)
        if baseline_delta is None or not is_superscript_metrics(
            previous_height, height, baseline_delta
        ):
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
        previous_baseline = rules.baseline_midpoint(previous.baseline, previous.run.rotation_angle)
        atom_baseline = rules.baseline_midpoint(atom.baseline, atom.run.rotation_angle)
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
        # Tables frequently split a word into adjacent text-showing
        # operators while also containing numeric cells.  Preserve the
        # table spacing rules for real cell gaps, but join only very tight
        # fragments (for example ``Vo`` + ``lume``).
        if self.is_table_like_line and spacing_gap > max(1.8, min(space_width, height) * 0.25):
            return False
        return rules.should_join_plausible_split_word(
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
        threshold = word_gap_threshold(run.space_width, height)
        if run.baseline is not None:
            threshold = max(threshold, run.font_size * WORD_GAP_SIZE_FACTOR)
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
