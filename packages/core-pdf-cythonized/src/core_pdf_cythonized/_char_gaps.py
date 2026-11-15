# SPDX-License-Identifier: AGPL-3.0-only

import cython
from cython.cimports.core_pdf_cythonized._pymath import py_max, py_min
from cython.cimports.libc.math import fabs

# core_pdf.impl.base.text's word_gap_threshold: WORD_GAP_SPACE_FACTOR, WORD_GAP_SIZE_FACTOR
# and WORD_GAP_MIN.
WORD_GAP_SPACE_FACTOR = cython.declare(cython.double, 0.15)
WORD_GAP_SIZE_FACTOR = cython.declare(cython.double, 0.08)
WORD_GAP_MIN = cython.declare(cython.double, 0.75)

# An int a double holds exactly, and whose sums and differences with another such int a
# double also holds exactly, as Python's int arithmetic keeps them.
EXACT_INT = cython.declare(object, 2**52)


@cython.cfunc
@cython.inline
def exact_number(value: object, out: cython.p_double) -> cython.bint:
    # Only a float or such an int: anything else declines, for Python's arithmetic.
    if type(value) is float:
        out[0] = cython.cast(cython.double, value)
        return True
    if type(value) is int and -EXACT_INT <= value <= EXACT_INT:
        out[0] = cython.cast(cython.double, value)
        return True
    return False


@cython.cfunc
@cython.inline
def exact_item(sequence: object, index: cython.Py_ssize_t, out: cython.p_double) -> cython.bint:
    # A four-number box or baseline is a tuple: anything else declines.
    if type(sequence) is not tuple or len(cython.cast(tuple, sequence)) != 4:
        return False
    return exact_number(cython.cast(tuple, sequence)[index], out)


@cython.cfunc
@cython.inline
def is_quarter_turn(angle: object) -> cython.bint:
    return angle == 90 or angle == 270


@cython.cclass
class CharGapRules:
    """GlyphLineBuilder.atom_separator for two atoms of one character each, neither
    whitespace, on a line that is not formula-like, in C.

    Built from the line's own values: estimated_char_width, column_gap_threshold and the
    line's kinds. A character is its own stripped text, so atom_separator's rules needing a
    fragment longer than one character (should_insert_tight_word_space,
    should_insert_hidden_text_overlap_space, the explicit-space and lowercase-pair rules,
    the long digit groups and compact_unit_suffix_should_join) or a space inside one (the
    all-caps and phrase-continuation rules) never apply, and should_join_word_fragments
    can only join a one-letter tail, when both runs are at most two characters long. The
    rest are evaluated in atom_separator's order, with Python's max, min and comparisons.
    separator() returns None, for atom_separator itself, when a number is neither a float
    nor an int a double holds exactly, or a box is not a tuple of four.
    """

    usable: cython.bint
    has_estimated_char_width: cython.bint
    estimated_char_width: cython.double
    column_gap_threshold: cython.double
    is_table_like_line: cython.bint
    has_explicit_spaces: cython.bint
    is_all_caps_line: cython.bint
    is_tracked_glyph_line: cython.bint
    has_tracked_word_gap: cython.bint
    tracked_word_gap: cython.double
    explicit_spaces_control_glyph_gaps: cython.bint

    def __init__(
        self,
        estimated_char_width: object,
        column_gap_threshold: object,
        is_table_like_line: cython.bint,
        has_explicit_spaces: cython.bint,
        is_all_caps_line: cython.bint,
        is_tracked_glyph_line: cython.bint,
        tracked_word_gap: object,
        explicit_spaces_control_glyph_gaps: cython.bint,
    ):
        self.usable = exact_number(column_gap_threshold, cython.address(self.column_gap_threshold))
        self.has_estimated_char_width = estimated_char_width is not None
        if self.has_estimated_char_width and not exact_number(
            estimated_char_width, cython.address(self.estimated_char_width)
        ):
            self.usable = False
        self.has_tracked_word_gap = tracked_word_gap is not None
        if self.has_tracked_word_gap and not exact_number(
            tracked_word_gap, cython.address(self.tracked_word_gap)
        ):
            self.usable = False
        self.is_table_like_line = is_table_like_line
        self.has_explicit_spaces = has_explicit_spaces
        self.is_all_caps_line = is_all_caps_line
        self.is_tracked_glyph_line = is_tracked_glyph_line
        self.explicit_spaces_control_glyph_gaps = explicit_spaces_control_glyph_gaps

    def separator(
        self,
        prev_run: object,
        run: object,
        prev_char: cython.Py_UCS4,
        char: cython.Py_UCS4,
        prev_box: object,
        box: object,
        prev_baseline: object,
        baseline: object,
        both_glyphs: cython.bint,
    ) -> object:
        prev_x0 = cython.declare(cython.double)
        prev_x1 = cython.declare(cython.double)
        x0 = cython.declare(cython.double)
        y0 = cython.declare(cython.double)
        y1 = cython.declare(cython.double)
        b0 = cython.declare(cython.double)
        b1 = cython.declare(cython.double)
        p0 = cython.declare(cython.double)
        p1 = cython.declare(cython.double)
        space_width = cython.declare(cython.double)
        value = cython.declare(cython.double)
        prev_y0 = cython.declare(cython.double)
        prev_y1 = cython.declare(cython.double)
        threshold = cython.declare(cython.double)
        tight = cython.declare(cython.double)
        if not self.usable:
            return None

        # inline_marker_text, then script_digit_text after a script digit and before it.
        if char in "™℠®©" or char in "⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉":
            return ""
        if prev_char in "⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉" and char in ")]},.;:":
            return ""
        same_run: cython.bint = prev_run is run
        if same_run and both_glyphs and " " in run.text:
            return ""

        if not (
            exact_item(prev_box, 0, cython.address(prev_x0))
            and exact_item(prev_box, 2, cython.address(prev_x1))
            and exact_item(box, 0, cython.address(x0))
            and exact_item(box, 1, cython.address(y0))
            and exact_item(box, 3, cython.address(y1))
            and exact_number(run.space_width, cython.address(space_width))
        ):
            return None
        height: cython.double = y1 - y0
        x_gap: cython.double = x0 - prev_x1
        spacing_gap: cython.double = x_gap
        if (
            self.has_estimated_char_width
            and prev_char.isalpha()
            and char.isalpha()
            and x_gap > py_max(1.8, py_min(space_width, height) * 0.25)
        ):
            spacing_gap = x0 - (prev_x0 + self.estimated_char_width)

        if prev_baseline is not None and baseline is not None:
            if is_quarter_turn(prev_run.rotation_angle) or is_quarter_turn(run.rotation_angle):
                if not (
                    exact_item(baseline, 0, cython.address(b0))
                    and exact_item(baseline, 2, cython.address(b1))
                    and exact_item(prev_baseline, 0, cython.address(p0))
                    and exact_item(prev_baseline, 2, cython.address(p1))
                ):
                    return None
            elif not (
                exact_item(baseline, 1, cython.address(b0))
                and exact_item(baseline, 3, cython.address(b1))
                and exact_item(prev_baseline, 1, cython.address(p0))
                and exact_item(prev_baseline, 3, cython.address(p1))
            ):
                return None
            if fabs((b0 + b1) * 0.5 - (p0 + p1) * 0.5) > py_max(height * 0.42, 2.0):
                return " "

        if (
            not same_run
            and prev_run.rotation_angle == run.rotation_angle
            and run.rotation_angle == 0
            and not (prev_run.is_vertical or run.is_vertical)
            and not both_glyphs
            and prev_char.isalnum()
            and char.isalnum()
        ):
            # -max(prev_run.font_size, run.font_size, prev_run.height, height)
            if not exact_number(prev_run.font_size, cython.address(value)):
                return None
            if not exact_number(run.font_size, cython.address(b0)):
                return None
            if not (
                exact_number(prev_run.y0, cython.address(prev_y0))
                and exact_number(prev_run.y1, cython.address(prev_y1))
            ):
                return None
            value = py_max(py_max(py_max(value, b0), prev_y1 - prev_y0), height)
            if x_gap < -value:
                return " "

        if spacing_gap > 0.0 and spacing_gap >= py_max(
            py_max(py_max(self.column_gap_threshold, space_width * 5.0), height * 1.55), 12.0
        ):
            return " "

        if (
            self.is_table_like_line
            and self.has_explicit_spaces
            and prev_char.isalpha()
            and char in "$0123456789("
            and x_gap >= -2.0
        ):
            return " "

        # GlyphLineBuilder.word_gap_threshold.
        threshold = py_max(
            py_max(space_width * WORD_GAP_SPACE_FACTOR, height * WORD_GAP_SIZE_FACTOR), WORD_GAP_MIN
        )
        if run.baseline is not None:
            if not exact_number(run.font_size, cython.address(value)):
                return None
            threshold = py_max(threshold, value * WORD_GAP_SIZE_FACTOR)
        if self.has_estimated_char_width:
            threshold = py_min(threshold, py_max(self.estimated_char_width * 0.32, 0.75))
        if self.is_all_caps_line:
            threshold = py_min(threshold, 0.40)
        if self.is_tracked_glyph_line:
            if self.has_tracked_word_gap:
                threshold = self.tracked_word_gap
            else:
                threshold = py_max(py_max(threshold, space_width * 2.2), height * 1.4)

        # should_join_word_fragments, for a one-letter tail and head.
        tight = py_max(1.8, py_min(space_width, height) * 0.25)
        if (
            not self.is_tracked_glyph_line
            and prev_run.visible
            and run.visible
            and len(prev_run.stripped_text) <= 2
            and len(run.stripped_text) <= 2
            and not (self.is_table_like_line and spacing_gap > tight)
            and prev_char.isalpha()
            and char.isalpha()
            and prev_char.islower()
            and char.islower()
            and not spacing_gap < -py_max(0.5, py_min(space_width, height) * 0.2)
            and not spacing_gap > py_max(py_max(space_width * 1.45, height * 0.45), 4.5)
            and spacing_gap <= tight
        ):
            return ""

        if (
            spacing_gap > threshold
            and (
                not self.has_explicit_spaces
                or self.is_tracked_glyph_line
                or both_glyphs
                or (
                    self.is_table_like_line
                    and spacing_gap > py_max(py_max(space_width * 2.2, height * 1.4), 24.0)
                )
            )
            and not (
                self.explicit_spaces_control_glyph_gaps
                and both_glyphs
                and not self.is_table_like_line
            )
        ):
            return " "
        if spacing_gap >= -py_max(threshold, 2.5) and prev_char.islower() and char.isupper():
            if same_run and x_gap <= py_max(0.5, py_min(space_width, height) * 0.2):
                return ""
            return " "
        return ""
