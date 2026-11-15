# SPDX-License-Identifier: AGPL-3.0-only

"""Check the compiled CharGapRules against GlyphLineBuilder.atom_separator.

Line reconstruction separates two atoms of one character each, neither whitespace, on a
line that is not formula-like, with core_pdf_cythonized's CharGapRules.separator rather than
atom_separator, whose rules it compiles for such a pair. This compares the two on random
pairs, drawn around every threshold atom_separator tests (gaps, heights, space widths, font
sizes, baselines, letters of each case, digits, brackets, inline markers and script digits,
NaN and infinities) under every kind of line, on edge cases, and on every pair the first
pages of the upstream test PDFs reconstruct. A pair whose numbers are not all floats or
ints a double holds exactly, or whose box is not a tuple, must be declined (None), and is
then checked to reach atom_separator.

uv run --all-packages python scripts/check_char_gaps.py [random cases]
"""

from __future__ import annotations

import glob
import itertools
import logging
import math
import random
import sys
import warnings
from typing import Any

from core_pdf import PdfDocument
from core_pdf.impl.base.runs import TextRun
from core_pdf.impl.layout import reconstruction
from core_pdf.impl.layout.reconstruction import GlyphLineBuilder, LayoutLineTextAtom
from core_pdf_cythonized import CharGapRules

CORPUS = (
    "tests/upstream/pdfplumber/tests/pdfs/*.pdf",
    "tests/upstream/unstructured/example-docs/pdf/*.pdf",
    "tests/upstream/llama_index/docs/examples/data/10q/*.pdf",
)
CHARS = (
    "aezAEZ019$(),.;:]}™©®℠⁰¹⁹₀₉"  # cases, digits, brackets, inline markers, script digits
    "éßªǅΣσЖж中あ½٣_-'\"/­​"  # other letters, numbers, marks, controls
)
SPECIAL = (math.nan, math.inf, -math.inf, 0.0, -0.0)
# How often a number is one of SPECIAL, by case: most cases have none, some have many.
SPECIAL_RATES = (0.0, 0.0, 0.08, 0.3)
FIELDS = (
    "estimated_char_width",
    "column_gap_threshold",
    "is_table_like_line",
    "has_explicit_spaces",
    "is_all_caps_line",
    "is_tracked_glyph_line",
    "tracked_word_gap",
    "explicit_spaces_control_glyph_gaps",
)


def builder(line: dict[str, Any]) -> GlyphLineBuilder:
    built = object.__new__(GlyphLineBuilder)
    for name, value in line.items():
        setattr(built, name, value)
    built.is_formula_like_line = False
    return built


def reference(line: dict[str, Any], args: tuple[Any, ...]) -> str:
    # atom_separator reads the atoms' has_glyph_geometry only together.
    prev_run, run, prev_char, char, prev_box, box, prev_baseline, baseline, both = args
    return builder(line).atom_separator(
        LayoutLineTextAtom(prev_char, prev_run, prev_box, prev_baseline, both),
        LayoutLineTextAtom(char, run, box, baseline, both),
    )


def compiled(line: dict[str, Any], args: tuple[Any, ...]) -> str | None:
    return CharGapRules(*(line[name] for name in FIELDS)).separator(*args)


def number(rng: random.Random, scale: float, specials: float) -> float:
    if rng.random() < specials:
        return rng.choice(SPECIAL)
    pick = rng.random()
    if pick < 0.1:
        return float(rng.randint(-3, 30))
    if pick < 0.2:
        return round(rng.uniform(-scale, scale), 2)
    return rng.uniform(-scale, scale) if rng.random() < 0.2 else rng.uniform(0.0, scale)


def gap(rng: random.Random, specials: float) -> float:
    if rng.random() < specials:
        return rng.choice(SPECIAL)
    pick = rng.random()
    if pick < 0.3:
        return rng.uniform(-1.0, 1.0)
    if pick < 0.5:
        return rng.choice((-2.5, -2.0, -0.5, 0.25, 0.4, 0.45, 0.5, 0.75, 1.8, 2.0, 4.5, 12.0))
    if pick < 0.8:
        return rng.uniform(-5.0, 5.0)
    return rng.uniform(-40.0, 40.0)


def random_line(rng: random.Random, specials: float) -> dict[str, Any]:
    return {
        "estimated_char_width": None if rng.random() < 0.5 else abs(number(rng, 8.0, specials)),
        "column_gap_threshold": number(rng, 30.0, specials),
        "is_table_like_line": rng.random() < 0.3,
        "has_explicit_spaces": rng.random() < 0.3,
        "is_all_caps_line": rng.random() < 0.2,
        "is_tracked_glyph_line": rng.random() < 0.2,
        "tracked_word_gap": None if rng.random() < 0.5 else number(rng, 10.0, specials),
        "explicit_spaces_control_glyph_gaps": rng.random() < 0.3,
    }


def random_run(rng: random.Random, chars: str, specials: float) -> TextRun:
    angle = rng.choice((0, 0, 0, 90, 180, 270))
    y0 = number(rng, 50.0, specials)
    height = abs(number(rng, 20.0, specials))
    text = rng.choice((chars, chars[:1], chars + "x", chars + " x", "abcd", " " + chars))
    return TextRun(
        text,
        0.0,
        y0,
        10.0,
        y0 + height,
        0.0,
        0.0,
        abs(number(rng, 20.0, specials)),
        abs(number(rng, 8.0, specials)),
        0,
        0,
        0,
        is_vertical=rng.random() < 0.1,
        rotation_angle=angle,
        visible=rng.random() < 0.85,
        baseline=None if rng.random() < 0.2 else (0.0, y0, 10.0, y0),
    )


def random_baseline(
    rng: random.Random, near: tuple[float, float, float, float] | None, specials: float
) -> tuple[float, float, float, float] | None:
    if rng.random() < 0.15:
        return None
    base = number(rng, 50.0, specials) if near is None else near[1]
    shift = rng.choice((0.0, 0.0, rng.uniform(-3.0, 3.0), rng.uniform(-15.0, 15.0)))
    return (base + shift, base + shift, base + shift + 5.0, base + shift)


def random_case(rng: random.Random, specials: float) -> tuple[dict[str, Any], tuple[Any, ...]]:
    line = random_line(rng, specials)
    prev_char = rng.choice(CHARS)
    char = rng.choice(CHARS)
    prev_run = random_run(rng, prev_char, specials)
    run = prev_run if rng.random() < 0.4 else random_run(rng, char, specials)
    prev_x0 = number(rng, 300.0, specials)
    width = abs(number(rng, 10.0, specials))
    prev_y0 = number(rng, 50.0, specials)
    prev_height = abs(number(rng, 20.0, specials))
    prev_box = (prev_x0, prev_y0, prev_x0 + width, prev_y0 + prev_height)
    x0 = prev_box[2] + gap(rng, specials)
    y0 = prev_y0 + rng.choice((0.0, 0.0, rng.uniform(-3.0, 3.0)))
    height = prev_height if rng.random() < 0.6 else abs(number(rng, 20.0, specials))
    box = (x0, y0, x0 + abs(number(rng, 10.0, specials)), y0 + height)
    prev_baseline = random_baseline(rng, None, specials)
    baseline = random_baseline(rng, prev_baseline, specials)
    both = rng.random() < 0.6
    return line, (prev_run, run, prev_char, char, prev_box, box, prev_baseline, baseline, both)


def edge_cases() -> list[tuple[dict[str, Any], tuple[Any, ...]]]:
    """Every character pair from CHARS at a few fixed gaps, on a plain line and on a table
    line with explicit spaces, within one run and across two short runs."""
    cases = []
    plain = dict.fromkeys(FIELDS, False) | {
        "estimated_char_width": None,
        "column_gap_threshold": 20.0,
        "tracked_word_gap": None,
    }
    table = plain | {"is_table_like_line": True, "has_explicit_spaces": True}
    for prev_char, char in itertools.product(CHARS, repeat=2):
        prev_run = TextRun(prev_char, 0, 0, 5, 10, 0, 0, 10.0, 2.5, 0, 0, 0, baseline=(0, 0, 5, 0))
        run = TextRun(char, 5, 0, 10, 10, 0, 0, 10.0, 2.5, 0, 0, 0, baseline=(5, 0, 10, 0))
        long_run = TextRun("abcd", 0, 0, 20, 10, 0, 0, 10.0, 2.5, 0, 0, 0)
        for line, offset, (left, right), both in itertools.product(
            (plain, table),
            (-3.0, -0.3, 0.0, 0.3, 1.0, 1.9, 3.0, 25.0),
            ((prev_run, run), (long_run, long_run)),
            (False, True),
        ):
            cases.append(
                (
                    line,
                    (
                        left,
                        right,
                        prev_char,
                        char,
                        (0.0, 0.0, 5.0, 10.0),
                        (5.0 + offset, 0.0, 10.0 + offset, 10.0),
                        (0.0, 0.0, 5.0, 0.0),
                        (5.0, 0.0, 10.0, 0.0),
                        both,
                    ),
                )
            )
    return cases


def declined_cases() -> list[tuple[dict[str, Any], tuple[Any, ...]]]:
    """Numbers a double does not hold as Python does: a huge int box, a box that is a list,
    a huge int space width or font size, and a huge int estimated width."""
    line = dict.fromkeys(FIELDS, False) | {
        "estimated_char_width": None,
        "column_gap_threshold": 20.0,
        "tracked_word_gap": None,
    }
    run = TextRun("ab", 0, 0, 5, 10, 0, 0, 10.0, 2.5, 0, 0, 0, baseline=(0, 0, 5, 0))
    huge = 2**60 + 1
    box = (0.0, 0.0, 5.0, 10.0)
    args = (run, run, "a", "b", box, (5.5, 0.0, 10.0, 10.0), None, None, True)
    cases = [
        (line, (run, run, "a", "b", box, (huge, 0, huge + 5, 10), None, None, True)),
        (line, (run, run, "a", "b", box, [5.5, 0.0, 10.0, 10.0], None, None, True)),
        (line | {"estimated_char_width": huge}, args),
    ]
    wide = TextRun("ab", 0, 0, 5, 10, 0, 0, 10.0, huge, 0, 0, 0)
    cases.append((line, (wide, wide, "a", "b", box, (5.5, 0.0, 10.0, 10.0), None, None, True)))
    return cases


class CheckedRules(CharGapRules):
    """CharGapRules as build() constructs it, comparing every separator() with
    atom_separator on the line build() would pass it."""

    pairs = 0
    failures: list[str] = []

    def __init__(self, *values: Any) -> None:
        super().__init__(*values)
        self.line = dict(zip(FIELDS, values, strict=True))

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
    ) -> str | None:
        args = (prev_run, run, prev_char, char, prev_box, box, prev_baseline, baseline, both_glyphs)
        got = super().separator(*args)
        CheckedRules.pairs += 1
        if got is not None:
            want = reference(self.line, args)
            if got != want and len(CheckedRules.failures) < 20:
                CheckedRules.failures.append(f"corpus pair {args[2:]!r}: {got!r} != {want!r}")
        return got


def check_corpus() -> int:
    paths = sorted({path for pattern in CORPUS for path in glob.glob(pattern)})
    if not paths:
        print("no upstream test PDFs: run git submodule update --init --recursive --depth 1")
        return 1
    original = reconstruction.CharGapRules
    # build() looks CharGapRules up in its module on every line.
    setattr(reconstruction, "CharGapRules", CheckedRules)  # noqa: B010
    try:
        for path in paths:
            try:
                with PdfDocument.open(path) as document:
                    document.extract(pages=range(1, min(3, len(document.pages)) + 1))
            except Exception:  # noqa: BLE001, S112
                continue
    finally:
        setattr(reconstruction, "CharGapRules", original)  # noqa: B010
    for failure in CheckedRules.failures:
        print(failure)
    print(f"{CheckedRules.pairs} pairs from {len(paths)} upstream PDFs")
    return len(CheckedRules.failures)


def main(argv: list[str]) -> int:
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    count = int(argv[0]) if argv else 400_000
    rng = random.Random(20261114)
    failures = []
    random_cases = (random_case(rng, specials=SPECIAL_RATES[index % 4]) for index in range(count))
    checked = 0
    for line, args in itertools.chain(edge_cases(), random_cases):
        checked += 1
        got = compiled(line, args)
        want = reference(line, args)
        if got != want:
            failures.append(f"{args[2:4]!r} {line!r} {args!r}: {got!r} != {want!r}")
    # Integer coordinates a double holds exactly give atom_separator's answer too.
    for line, args in edge_cases()[:5000]:
        prev_run, run, prev_char, char, prev_box, box, prev_baseline, baseline, both = args
        ints = (
            prev_run,
            run,
            prev_char,
            char,
            tuple(round(value) for value in prev_box),
            tuple(round(value) for value in box),
            prev_baseline,
            baseline,
            both,
        )
        checked += 1
        if (got := compiled(line, ints)) != (want := reference(line, ints)):
            failures.append(f"int box {ints[2:]!r}: {got!r} != {want!r}")
    for line, args in declined_cases():
        checked += 1
        if (got := compiled(line, args)) is not None:
            failures.append(f"not declined {args[2:]!r}: {got!r}")
        reference(line, args)
    for failure in failures[:20]:
        print(failure)
    print(f"{checked} pairs, {len(failures)} differ from atom_separator")
    corpus_failures = check_corpus()
    return 1 if failures or corpus_failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
