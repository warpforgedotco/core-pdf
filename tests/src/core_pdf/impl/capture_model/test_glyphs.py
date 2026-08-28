import pytest

from core_pdf.impl.capture_model.glyphs import (
    GlyphCluster,
    GlyphObservation,
    GlyphUnicodeSemantics,
    glyph_cluster_from_observations,
    glyph_unicode_confidence,
    glyph_unicode_semantics,
)
from core_pdf.impl.capture_model.runs import TextRun


def test_unicode_confidence_scores_authoritative_and_unsupported_glyphs() -> None:
    assert glyph_unicode_confidence("A", "to_unicode") == 1.0
    assert glyph_unicode_confidence("\ue000", "to_unicode") == 0.20


@pytest.mark.parametrize(
    ("text", "source", "expected"),
    [
        ("A", "to_unicode", GlyphUnicodeSemantics.AUTHORITATIVE),
        ("A", "encoding", GlyphUnicodeSemantics.HEURISTIC),
        ("A", "identity", GlyphUnicodeSemantics.UNKNOWN_IDENTIFIER),
        ("\ue000", "to_unicode", GlyphUnicodeSemantics.UNSUPPORTED),
    ],
)
def test_unicode_semantics_distinguish_mapping_evidence(
    text: str,
    source: str,
    expected: GlyphUnicodeSemantics,
) -> None:
    assert glyph_unicode_semantics(text, source) is expected


def test_vector_outline_counts_as_glyph_paint() -> None:
    observation = GlyphObservation(
        "A",
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
        1,
        font_decoder=object(),
        glyph_transform=(0.01, 0.0, 0.0, 0.01, 0.0, 0.0),
    )

    assert observation.has_paint is True

    observation.paint_glyph = False

    assert observation.has_paint is False


@pytest.mark.parametrize("confidence", [None, 0.84])
def test_single_glyph_cluster_reuses_observation_geometry_and_confidence(
    confidence: float | None,
) -> None:
    observation = GlyphObservation(
        text="A",
        ink_bbox=(1.0, 2.0, 3.0, 4.0),
        advance_bbox=(0.5, 1.5, 3.5, 4.5),
        seqno=7,
        font_name="Example",
        baseline=(0.5, 1.5, 3.5, 1.5),
        rotation_angle=0,
        confidence=confidence,
    )

    cluster = glyph_cluster_from_observations(
        12,
        "A",
        (observation,),
    )

    assert cluster is not None
    assert cluster.glyphs == (observation,)
    assert cluster.advance_bbox == observation.advance_bbox
    assert cluster.ink_bbox == observation.ink_bbox
    assert cluster.baseline == observation.baseline
    assert cluster.confidence == confidence


def test_multi_glyph_cluster_unions_geometry_and_uses_weakest_confidence() -> None:
    first = GlyphObservation(
        "A",
        (1.0, 2.0, 3.0, 4.0),
        (0.5, 1.5, 3.5, 4.5),
        1,
        baseline=(0.5, 1.5, 3.5, 1.5),
        confidence=0.9,
    )
    second = GlyphObservation(
        "B",
        (3.0, 2.0, 5.0, 4.0),
        (2.5, 1.5, 5.5, 4.5),
        2,
        baseline=(2.5, 1.5, 5.5, 1.5),
        confidence=0.7,
    )

    cluster = glyph_cluster_from_observations(4, "AB", (first, second))

    assert cluster is not None
    assert cluster.advance_bbox == (0.5, 1.5, 5.5, 4.5)
    assert cluster.ink_bbox == (1.0, 2.0, 5.0, 4.0)
    assert cluster.baseline == first.baseline
    assert cluster.confidence == 0.7


def test_text_run_replacement_drops_clusters_that_describe_old_text() -> None:
    observation = GlyphObservation(
        text="A",
        ink_bbox=(1.0, 2.0, 3.0, 4.0),
        advance_bbox=(1.0, 2.0, 3.0, 4.0),
        seqno=7,
    )
    cluster = glyph_cluster_from_observations(
        0,
        "A",
        (observation,),
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
    repaired_cluster = GlyphCluster(
        cluster.cluster_id,
        "B",
        cluster.glyphs,
        cluster.advance_bbox,
        cluster.ink_bbox,
        cluster.baseline,
        cluster.confidence,
    )
    repaired = run.replace(text="B", glyph_clusters=(repaired_cluster,))

    assert replacement.text == "B"
    assert replacement.glyph_clusters == ()
    assert repaired.glyph_clusters == (repaired_cluster,)
