# SPDX-License-Identifier: AGPL-3.0-only
import pytest

from core_pdf.impl._impl.layout.reconstruction import reconstruct_layout_line_text
from core_pdf.impl._impl.model.glyphs import (
    GlyphObservation,
    glyph_cluster_from_observations,
)
from tests.helpers.extract_fakes import text_run


def test_overlapping_phrases_keep_a_boundary_when_the_next_run_restarts() -> None:
    # A real clipped-layer PDF is covered by test_extract_clipped_text.py;
    # Poppler 26.07.0 `pdftotext -raw` retains the boundary between these phrases.
    runs = [
        text_run("nowhere nowhere", 0.0, 0.0, 70.0, 10.0),
        text_run("now here now here", 0.0, 0.0, 80.0, 10.0, order=1),
    ]

    reconstructed = reconstruct_layout_line_text(runs)

    assert reconstructed.text == "nowhere nowhere now here now here"
    assert reconstructed.segments[1].separator_before == " "
    assert [run.text for run in runs] == ["nowhere nowhere", "now here now here"]


@pytest.mark.parametrize("gap", [-2.0, -0.5, 0.0])
@pytest.mark.parametrize(("prefix", "suffix"), [("inter", "face"), ("name", "_suffix")])
def test_font_changes_and_small_overlaps_preserve_words(
    prefix: str, suffix: str, gap: float
) -> None:
    runs = [
        text_run(prefix, 0.0, 0.0, 25.0, 10.0, font_name="First"),
        text_run(suffix, 25.0 + gap, 0.0, 50.0 + gap, 10.0, font_name="Second", order=1),
    ]

    assert reconstruct_layout_line_text(runs).text == prefix + suffix


def test_ligature_cluster_stays_joined_at_a_font_boundary() -> None:
    # The expanded "ffi" is one source glyph; its text must remain one atom.
    box = (5.0, 0.0, 17.0, 10.0)
    observation = GlyphObservation("ffi", box, box, 1)
    cluster = glyph_cluster_from_observations(1, "ffi", (observation,))
    assert cluster is not None
    runs = [
        text_run("o", 0.0, 0.0, 5.0, 10.0, font_name="First"),
        text_run("ffi", *box, order=1, font_name="Second", glyph_clusters=(cluster,)),
        text_run("ce", 17.0, 0.0, 27.0, 10.0, order=2, font_name="First"),
    ]

    reconstructed = reconstruct_layout_line_text(runs)

    assert reconstructed.text == "office"
    assert [segment.text for segment in reconstructed.segments] == ["o", "ffi", "ce"]
