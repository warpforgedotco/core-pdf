from dataclasses import replace

import pytest

from core_pdf.impl.engine.layout.geometry import RectBox
from core_pdf.impl.engine.layout.glyphs import (
    GlyphObservation,
    GlyphUnicodeSemantics,
    glyph_cluster_from_observations,
    glyph_unicode_confidence,
    glyph_unicode_semantics,
)
from core_pdf.impl.engine.layout.models import TextRun


def test_authoritative_unicode_has_full_confidence() -> None:
    assert glyph_unicode_confidence("A", "to_unicode") == 1.0


def test_unsupported_glyph_has_low_unicode_confidence() -> None:
    assert glyph_unicode_confidence("\ue000", "to_unicode") == 0.20


def test_identity_cmap_value_remains_an_unknown_identifier() -> None:
    assert glyph_unicode_semantics("A", "identity") is GlyphUnicodeSemantics.UNKNOWN_IDENTIFIER
    assert glyph_unicode_semantics("A", "to_unicode") is GlyphUnicodeSemantics.AUTHORITATIVE


def test_unicode_confidence_reuses_cached_result() -> None:
    glyph_unicode_confidence.cache_clear()

    assert glyph_unicode_confidence("A", "to_unicode") == 1.0
    assert glyph_unicode_confidence("A", "to_unicode") == 1.0

    cache_info = glyph_unicode_confidence.cache_info()
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


def test_text_run_replacement_drops_clusters_that_describe_old_text() -> None:
    observation = GlyphObservation(
        text="A",
        ink_rect=RectBox(1.0, 2.0, 3.0, 4.0),
        advance_rect=RectBox(1.0, 2.0, 3.0, 4.0),
        seqno=7,
    )
    cluster = glyph_cluster_from_observations(
        0,
        "A",
        (observation,),
        kind="single_glyph",
    )
    assert cluster is not None
    run = TextRun(
        "A",
        1.0,
        2.0,
        3.0,
        4.0,
        0.0,
        0.0,
        12.0,
        4.0,
        0,
        0,
        0,
        glyph_clusters=(cluster,),
    )

    replacement = run.replace(text="B")
    repaired_cluster = replace(cluster, text="B")
    repaired = run.replace(text="B", glyph_clusters=(repaired_cluster,))

    assert replacement.text == "B"
    assert replacement.glyph_clusters == ()
    assert repaired.glyph_clusters == (repaired_cluster,)
