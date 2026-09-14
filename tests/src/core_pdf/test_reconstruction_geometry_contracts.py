"""Text atom and script geometry contracts for line reconstruction."""

import pytest

from core_pdf.impl._impl.layout.reconstruction import (
    GlyphLineBuilder,
    LayoutLineTextAtom,
    is_superscript_metrics,
    reconstruct_layout_line_text,
)
from core_pdf.impl._impl.model.glyphs import GlyphCluster
from core_pdf.impl._impl.model.runs import TextRun


def internal_run(text: str, x: float = 0, height: float = 10, baseline: float = 10) -> TextRun:
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
    run = internal_run(text)
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
    run = internal_run(text)
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
    previous = internal_run(previous_text)
    current = internal_run(text, x, height, baseline)

    def atom(run):
        return LayoutLineTextAtom(run.text, run, run.advance_bbox, run.baseline, True)

    builder = GlyphLineBuilder([previous, current], is_formula_like_line=True)
    assert builder.is_formula_fraction_denominator(atom(previous), atom(current)) is expected
