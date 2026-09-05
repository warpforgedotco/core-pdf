import pytest

from core_pdf.impl._impl.layout.reconstruction import (
    GlyphLineBuilder,
    reconstruct_layout_line_text,
)
from core_pdf.impl._impl.layout.text_rules import (
    should_join_plausible_split_word,
)
from core_pdf.impl._impl.model.glyphs import (
    GlyphObservation,
    glyph_cluster_from_observations,
)
from tests.helpers.extract_fakes import text_run


def test_complete_whitespace_run_stays_one_text_atom() -> None:
    clusters = []
    for index, character in enumerate("A B"):
        bbox = (float(index), 0.0, float(index + 1), 1.0)
        observation = GlyphObservation(character, bbox, bbox, index)
        cluster = glyph_cluster_from_observations(index, character, (observation,))
        assert cluster is not None
        clusters.append(cluster)
    run = text_run(
        "A B",
        0.0,
        0.0,
        3.0,
        1.0,
        tx=0.0,
        ty=0.0,
        font_size=12.0,
        space_width=1.0,
        glyph_clusters=tuple(clusters),
    )
    builder = GlyphLineBuilder([run])

    atoms = builder.text_atoms(run, run.text)

    assert len(atoms) == 1
    assert atoms[0].text == "A B"
    assert builder.build().text == "A B"


def test_table_word_fragments_can_join_when_the_gap_is_tight() -> None:
    assert should_join_plausible_split_word(
        "Vo",
        "lume",
        x_gap=1.5,
        height=10.2,
        space_width=5.0,
        prev_visible=True,
        visible=True,
        allow_short_prefix=True,
    )


def test_short_word_fragments_still_require_explicit_opt_in() -> None:
    assert not should_join_plausible_split_word(
        "Vo",
        "lume",
        x_gap=1.5,
        height=10.2,
        space_width=5.0,
        prev_visible=True,
        visible=True,
    )


@pytest.mark.parametrize(
    "text",
    [
        "SERVICE MODUL E Tempera ture Ox idizer We ight",
        "Primary Fuel Secondary Oxidizer",
        "Coef . Var . 105 .1",
        "now here",
        "a part",
    ],
)
def test_reconstruction_keeps_explicit_word_and_punctuation_boundaries(text: str) -> None:
    # Poppler 26.07.0 -raw preserves each exact string authored in one Tj.
    # Dictionary rank is not evidence that its explicit spaces were accidental.
    assert GlyphLineBuilder([text_run(text)]).build().text == text


@pytest.mark.parametrize("text", ["---", "...", "●"])
def test_prepared_punctuation_is_not_discarded_as_decoration(text: str) -> None:
    # These are already decoded observations: layout must preserve their glyph
    # text regardless of how a particular PDF font maps its character codes.
    run = text_run(text)

    assert reconstruct_layout_line_text([run]).text == text
    assert GlyphLineBuilder([run]).build().text == text
