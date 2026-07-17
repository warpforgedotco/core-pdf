import pytest

from core_pdf.impl.engine.layout import glyphs
from core_pdf.impl.engine.layout.geometry import RectBox
from core_pdf.impl.engine.layout.glyphs import (
    GlyphObservation,
    glyph_cluster_from_observations,
    glyph_unicode_confidence,
)


def test_unicode_confidence_is_independent_of_paint_visibility() -> None:
    assert glyph_unicode_confidence("A", "to_unicode", visible=False) == 1.0


def test_hidden_unsupported_glyph_still_has_low_unicode_confidence() -> None:
    assert glyph_unicode_confidence("\ue000", "to_unicode", visible=False) == 0.20


def test_unicode_confidence_reuses_semantic_result_across_paint_visibility() -> None:
    glyphs._cached_glyph_unicode_confidence.cache_clear()

    assert glyph_unicode_confidence("A", "to_unicode", visible=True) == 1.0
    assert glyph_unicode_confidence("A", "to_unicode", visible=False) == 1.0

    cache_info = glyphs._cached_glyph_unicode_confidence.cache_info()
    assert cache_info.hits == 1
    assert cache_info.misses == 1


@pytest.mark.parametrize("confidence", [None, 0.84])
def test_single_glyph_cluster_reuses_observation_geometry_and_confidence(
    confidence: float | None,
) -> None:
    observation = GlyphObservation(
        text="A",
        ink_rect=RectBox(1.0, 2.0, 3.0, 4.0),
        advance_rect=RectBox(0.5, 1.5, 3.5, 4.5),
        seqno=7,
        font_name="Example",
        baseline=(0.5, 1.5, 3.5, 1.5),
        writing_mode="horizontal",
        rotation_angle=0,
        confidence=confidence,
    )

    cluster = glyph_cluster_from_observations(
        12,
        "A",
        (observation,),
        kind="single_glyph",
        provenance=(("source", "test"),),
    )

    assert cluster is not None
    assert cluster.glyphs == (observation,)
    assert cluster.advance_bbox == observation.advance_bbox
    assert cluster.ink_bbox == observation.ink_bbox
    assert cluster.baseline == observation.baseline
    assert cluster.font_name == observation.font_name
    assert cluster.seqno == observation.seqno
    assert cluster.confidence == confidence
    assert cluster.provenance == (("source", "test"),)
