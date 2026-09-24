from copy import replace

import pytest

from core_pdf.impl.extract.capture import (
    hidden_text_is_trusted,
    layout_bbox_for_run,
    promote_hidden_run,
)
from core_pdf.impl.extract.contracts import GlyphEvidence, TextQualityStats
from core_pdf.impl.glyphs import GlyphCluster
from core_pdf.impl.runs import TextRun


def run() -> TextRun:
    return TextRun(
        "word",
        10,
        -100,
        50,
        100,
        10,
        20,
        10,
        2,
        0,
        0,
        0,
        ink_bbox=(12, 18, 48, 25),
        baseline=(10, 20, 50, 20),
    )


@pytest.mark.parametrize("font_size", [10, -10])
@pytest.mark.parametrize("baseline", [None, (10, 20, 50, 20), (10, 18, 50, 22)])
def test_large_font_bounds_use_occurrence_ink_and_baseline(font_size, baseline):
    source = run().replace(font_size=font_size, baseline=baseline)
    assert layout_bbox_for_run(source) == (10, 18, 50, 25 if baseline is None else 28)
    assert (source.y0, source.y1) == (-100, 100)
    assert source.ink_bbox == (12, 18, 48, 25)


@pytest.mark.parametrize(
    "reason", ["vertical", "quarter-turn", "zero-font", "normal-height", "empty-ink", "large-ink"]
)
def test_geometry_stays_intact_without_evidence_of_inflated_font_bounds(reason):
    source = run()
    changes = {
        "vertical": {"is_vertical": True},
        "quarter-turn": {"rotation_angle": 90},
        "zero-font": {"font_size": 0},
        "normal-height": {"y0": 0, "y1": 25},
        "empty-ink": {"ink_bbox": (12, 20, 48, 20)},
        "large-ink": {"ink_bbox": (12, -40, 48, 40)},
    }
    source = source.replace(**changes[reason])
    assert layout_bbox_for_run(source) == (source.x0, source.y0, source.x1, source.y1)


def test_nonblank_cluster_ink_overrides_font_wide_run_ink():
    source = run().replace(ink_bbox=(10, -100, 50, 100))
    visible = GlyphCluster(
        0, "word", (), source.advance_bbox, (12, 15, 48, 30), source.baseline, None
    )
    blank = GlyphCluster(
        1, " ", (), source.advance_bbox, (10, -100, 50, 100), source.baseline, None
    )
    source = source.replace(glyph_clusters=(visible, blank))
    assert layout_bbox_for_run(source) == (10, 15, 50, 30)


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("authoritative", True),
        ("short", False),
        ("painted-boundary", False),
        ("below-painted-boundary", True),
        ("suspicious-boundary", True),
        ("too-suspicious", False),
        ("actual-text", True),
        ("insufficient-actual-text", False),
        ("no-glyphs", False),
        ("low-confidence", False),
        ("unsupported", False),
        ("heuristic", True),
        ("insufficient-mapping", False),
        ("unknown", False),
        ("poor-words", False),
        ("noisy", False),
    ],
)
def test_hidden_text_requires_sufficient_clean_semantic_evidence(case, expected):
    native, painted, suspicious = 100, 0, 0
    quality = TextQualityStats(wordlike_ratio=0.65)
    glyphs = GlyphEvidence(
        glyph_count=100, authoritative_glyphs=90, heuristic_glyphs=9, unknown_glyphs=1
    )
    if case == "short":
        native = 99
    elif case == "painted-boundary":
        painted = 20
    elif case == "below-painted-boundary":
        painted = 19
    elif case == "suspicious-boundary":
        suspicious = 1
    elif case == "too-suspicious":
        suspicious = 2
    elif case in {"actual-text", "insufficient-actual-text"}:
        glyphs = GlyphEvidence(actual_text_characters=80 if case == "actual-text" else 79)
    elif case == "no-glyphs":
        glyphs = GlyphEvidence()
    elif case == "low-confidence":
        glyphs = replace(glyphs, low_confidence_glyphs=2)
    elif case == "unsupported":
        glyphs = replace(glyphs, unsupported_glyphs=2)
    elif case in {"heuristic", "insufficient-mapping", "unknown", "poor-words", "noisy"}:
        glyphs = GlyphEvidence(glyph_count=100, heuristic_glyphs=99, unknown_glyphs=1)
        if case == "insufficient-mapping":
            glyphs = replace(glyphs, heuristic_glyphs=98)
        elif case == "unknown":
            glyphs = replace(glyphs, unknown_glyphs=2)
        elif case == "poor-words":
            quality = replace(quality, wordlike_ratio=0.64)
        elif case == "noisy":
            quality = replace(quality, symbol_ratio=1)
    assert (
        hidden_text_is_trusted(
            native_characters=native,
            painted_characters=painted,
            suspicious_characters=suspicious,
            quality=quality,
            glyphs=glyphs,
        )
        is expected
    )


def test_hidden_text_promotion_preserves_capture_visibility_and_geometry():
    source = run().replace(visible=False, inside_active_clip=False)
    promoted = promote_hidden_run(source)
    assert promoted.visible
    assert not promoted.inside_active_clip
    assert not source.visible
    assert promoted.advance_bbox == source.advance_bbox
    assert promoted.glyph_clusters is source.glyph_clusters
    assert dict(promoted.provenance)["extraction_visibility"] == "trusted-hidden-layer"
    assert "extraction_visibility" not in dict(source.provenance)
