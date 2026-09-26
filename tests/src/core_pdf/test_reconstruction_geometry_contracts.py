import pytest

from core_pdf.impl.glyphs import GlyphCluster
from core_pdf.impl.layout_reconstruction import (
    GlyphLineBuilder,
    LayoutLineTextAtom,
    is_superscript_metrics,
    reconstruct_layout_line_text,
)
from core_pdf.impl.runs import TextRun


def make_run(text: str, x: float = 0, height: float = 10, baseline: float = 10) -> TextRun:
    return TextRun(
        text,
        x,
        0,
        x + 5,
        height,
        x,
        baseline,
        10,
        2,
        0,
        0,
        0,
        baseline=(x, baseline, x + 5, baseline),
    )


@pytest.mark.parametrize("text", ["a", "fi", " ", "\t", "\n", "\u2003", "a b", "\ue000", "a\x00b"])
@pytest.mark.parametrize("clustered", [False, True])
def test_nonempty_text_always_produces_nonempty_atoms(text, clustered):
    run = make_run(text)
    if clustered:
        run = run.replace(
            glyph_clusters=tuple(
                GlyphCluster(i, char, (), run.advance_bbox, run.advance_bbox, run.baseline, None)
                for i, char in enumerate(text)
            )
        )
    atoms = GlyphLineBuilder([run]).text_atoms(run, text)
    assert atoms
    assert all(atom.text for atom in atoms)
    assert "".join(atom.text for atom in atoms) == text
    assert all(atom.run is run for atom in atoms)


@pytest.mark.parametrize(
    ("text", "expected"), [("\ue000", ""), ("\x00", ""), ("a\x00b", "ab"), ("a  b", "a b")]
)
def test_single_and_builder_paths_agree_on_text_cleanup(text, expected):
    run = make_run(text)
    assert reconstruct_layout_line_text([run]).text == expected
    assert GlyphLineBuilder([run]).build().text == expected


@pytest.mark.parametrize(
    ("previous_height", "height", "raise_by", "expected"),
    [
        (0, 0, 2, False),
        (10, 9, 2, False),
        (10, 8, 0.49, False),
        (10, 8, 0.5, True),
        (4, 3, 0.44, False),
        (4, 3, 0.45, True),
    ],
)
def test_superscript_geometry_requires_both_size_and_baseline_change(
    previous_height, height, raise_by, expected
):
    assert is_superscript_metrics(previous_height, height, raise_by) is expected


@pytest.mark.parametrize(
    ("previous_text", "text", "height", "baseline", "x", "expected"),
    [
        ("2", "√", 10, 5, 5, True),
        ("2", "G", 10, 5, 5, True),
        ("t", "T", 10, 5, 5, True),
        ("s", "T", 10, 5, 5, True),
        (")", "∂", 10, 5, 5, True),
        ("", "√", 10, 5, 5, False),
        ("2", "", 10, 5, 5, False),
        ("2", "x", 10, 5, 5, False),
        ("x", "√", 10, 5, 5, False),
        ("2", "√", 8, 5, 5, False),
        ("2", "√", 10, 8, 5, False),
        ("2", "√", 10, 5, 20, False),
    ],
)
def test_stacked_fraction_denominator_requires_compatible_geometry(
    previous_text, text, height, baseline, x, expected
):
    previous = make_run(previous_text)
    current = make_run(text, x, height, baseline)

    def atom(run):
        return LayoutLineTextAtom(run.text, run, run.advance_bbox, run.baseline, True)

    builder = GlyphLineBuilder([previous, current], is_formula_like_line=True)
    assert builder.is_formula_fraction_denominator(atom(previous), atom(current)) is expected


@pytest.mark.parametrize("kind", ["letter", "digit"])
@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("raised", True),
        ("lowered", True),
        ("unshifted", False),
        ("small-shift", False),
        ("full-height", False),
        ("zero-context-height", False),
        ("distant", False),
        ("attachment-boundary", True),
        ("missing-baseline", False),
        ("missing-context-baseline", False),
        ("empty", False),
        ("multiple-characters", False),
        ("punctuation", False),
    ],
)
def test_formula_script_atoms_require_shape_shift_and_attachment(kind, case, expected):
    previous = make_run("x")
    current = make_run("i" if kind == "letter" else "2", x=5, height=6, baseline=12)
    if case == "lowered":
        current = make_run(current.text, x=5, height=6, baseline=8)
    elif case == "unshifted":
        current = make_run(current.text, x=5, height=6, baseline=10)
    elif case == "small-shift":
        current = make_run(current.text, x=5, height=6, baseline=10.1)
    elif case == "full-height":
        current = make_run(current.text, x=5, height=10, baseline=12)
    elif case == "zero-context-height":
        previous = make_run("x", height=0)
    elif case == "distant":
        current = make_run(current.text, x=8, height=6, baseline=12)
    elif case == "attachment-boundary":
        current = make_run(current.text, x=7.5, height=6, baseline=12)
    elif case == "missing-baseline":
        current = current.replace(baseline=None)
    elif case == "missing-context-baseline":
        previous = previous.replace(baseline=None)
    elif case in {"empty", "multiple-characters", "punctuation"}:
        current = current.replace(
            text={"empty": "", "multiple-characters": "ii", "punctuation": "+"}[case]
        )

    def atom(run):
        return LayoutLineTextAtom(run.text, run, run.advance_bbox, run.baseline, True)

    builder = GlyphLineBuilder([previous, current], is_formula_like_line=True)
    classify = (
        builder.is_formula_script_atom if kind == "letter" else builder.is_formula_numeric_atom
    )
    assert classify(atom(previous), atom(current)) is expected


@pytest.mark.parametrize(
    ("prefix", "text", "height", "baseline", "x", "expected"),
    [
        ("x", "2", 6, 9, 5, True),
        ("x", "123", 6, 9, 5, True),
        ("x", "1234", 6, 9, 5, False),
        ("x", "i", 6, 9, 5, False),
        ("+", "2", 6, 9, 5, False),
        ("x", "2", 6, 10, 5, False),
        ("x", "2", 6, 11, 5, False),
        ("x", "2", 10, 9, 5, False),
        ("x", "2", 6, 9, 7, True),
        ("x", "2", 6, 9, 7.1, False),
    ],
)
def test_formula_numeric_subscript_requires_a_lower_attached_small_run(
    prefix, text, height, baseline, x, expected
):
    previous = make_run(prefix)
    current = make_run(text, x=x, height=height, baseline=baseline)
    builder = GlyphLineBuilder([previous, current], is_formula_like_line=True)
    assert builder.is_formula_subscript_like_numeric_run(current, 1) is expected


@pytest.mark.parametrize("missing", ["previous", "current", "context"])
def test_formula_numeric_subscript_requires_context_and_baselines(missing):
    previous = make_run("x")
    current = make_run("2", x=5, height=6, baseline=9)
    if missing == "previous":
        previous = previous.replace(baseline=None)
    elif missing == "current":
        current = current.replace(baseline=None)
    runs = [current] if missing == "context" else [previous, current]
    builder = GlyphLineBuilder(runs, is_formula_like_line=True)
    assert not builder.is_formula_subscript_like_numeric_run(current, len(runs) - 1)


@pytest.mark.parametrize(("formula", "expected"), [(False, "x2"), (True, "x₂")])
def test_formula_context_controls_numeric_subscript_normalization(formula, expected):
    previous = make_run("x")
    current = make_run("2", x=5, height=6, baseline=9)
    builder = GlyphLineBuilder([previous, current], is_formula_like_line=formula)
    assert builder.build().text == expected
    assert previous.text == "x"
    assert current.text == "2"


def recent(*entries: tuple[tuple[float, float, float, float], str]) -> list:
    tracked = []
    reach = float("-inf")
    for box, text in entries:
        reach = max(reach, box[2])
        tracked.append((box, text, reach))
    return tracked


def test_duplicate_overlap_is_found_behind_an_entry_further_left() -> None:
    # The early stop may only skip entries that cannot reach the run: here the
    # newest entry ends left of it, but an older one still overlaps.
    run = make_run("a", x=12)
    entries = recent(((10.0, 0.0, 20.0, 10.0), "a"), ((0.0, 0.0, 5.0, 10.0), "b"))
    assert GlyphLineBuilder([run]).is_recent_duplicate_overlap(entries, run, "a")


def test_duplicate_overlap_stops_once_nothing_older_can_reach() -> None:
    run = make_run("a", x=30)
    entries = recent(((0.0, 0.0, 5.0, 10.0), "a"), ((10.0, 0.0, 20.0, 10.0), "a"))
    assert not GlyphLineBuilder([run]).is_recent_duplicate_overlap(entries, run, "a")
