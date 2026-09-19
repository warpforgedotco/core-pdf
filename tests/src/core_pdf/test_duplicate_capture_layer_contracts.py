"""Duplicate layers require local content and glyph evidence, not just overlap."""

from dataclasses import replace

import pytest

from core_pdf.impl._impl.extract.capture import (
    internal_clip_bbox,
    internal_discard_duplicate_clipped_layers,
    internal_discard_duplicate_nested_layers,
    internal_glyphs_covered_by_extended_run,
)
from core_pdf.impl._impl.model.glyphs import GlyphCluster
from core_pdf.impl._impl.model.runs import TextRun


def run(text: str, depth: int = 0, x: float = 0) -> TextRun:
    clusters = tuple(
        GlyphCluster(i, char, (), (x + i, 0, x + i + 1, 10), (x + i, 0, x + i + 1, 10), None, None)
        for i, char in enumerate(text)
    )
    return TextRun(text, x, 0, x + len(text), 10, x, 2, 10, 2, 0, 0, depth, glyph_clusters=clusters)


@pytest.mark.parametrize("count", [23, 24, 25])
@pytest.mark.parametrize("kind", ["form", "clip"])
def test_duplicate_removal_requires_enough_matching_words_and_retains_unmatched_content(
    count, kind
):
    text = " ".join(f"word{i}" for i in range(count))
    primary, overlay = run(text), run(text, depth=1)
    unique = run("unique content", depth=1, x=1000)
    if kind == "clip":
        primary.provenance = (("clip_bbox", (0, 0, 2000, 100)),)
        overlay.provenance = unique.provenance = (("clip_bbox", (0, 0, 1500, 50)),)
        filter_runs = internal_discard_duplicate_clipped_layers
    else:
        filter_runs = internal_discard_duplicate_nested_layers
    sources = (primary, overlay, unique)
    expected = (primary, unique) if count >= 24 else sources
    assert filter_runs(sources) == expected
    assert overlay.text == text
    assert unique.text == "unique content"


@pytest.mark.parametrize("difference", ["case", "punctuation", "word-boundary", "location"])
def test_overlapping_text_is_not_removed_when_content_or_location_differs(difference):
    text = " ".join(["now here"] * 12)
    primary, overlay = run(text), run(text, depth=1)
    if difference == "case":
        overlay = run(text.upper(), depth=1)
    elif difference == "punctuation":
        overlay = run(text.replace("here", "here!"), depth=1)
    elif difference == "word-boundary":
        overlay = run(text.replace("now here", "no wh ere"), depth=1)
    else:
        overlay = run(text, depth=1, x=1000)
    sources = (primary, overlay)
    assert internal_discard_duplicate_nested_layers(sources) == sources


@pytest.mark.parametrize(
    "case",
    ["identical-glyphs", "missing-glyphs", "changed-text", "changed-position", "repeated-glyph"],
)
def test_extended_overlay_replaces_shorter_copy_only_with_exact_glyph_multiset(case):
    text = " ".join(f"word{i}" for i in range(24))
    primary = run(text)
    extended = run(text + " appendix", depth=1)
    expected = case == "identical-glyphs"
    if case == "missing-glyphs":
        primary = primary.replace(glyph_clusters=())
    elif case == "changed-text":
        primary.text = text + "!"
    elif case == "changed-position":
        first = replace(extended.glyph_clusters[0], advance_bbox=(0, 1, 1, 11))
        extended = extended.replace(glyph_clusters=(first, *extended.glyph_clusters[1:]))
    elif case == "repeated-glyph":
        primary = primary.replace(
            text=primary.text[0] + primary.text,
            glyph_clusters=(primary.glyph_clusters[0], *primary.glyph_clusters),
        )
    assert internal_glyphs_covered_by_extended_run(primary, extended) is expected
    sources = (primary, extended)
    assert internal_discard_duplicate_nested_layers(sources) == (
        (extended,) if expected else sources
    )
    assert extended.text.endswith(" appendix")


def test_inconsistent_extended_glyph_text_cannot_prove_coverage():
    primary = run("abc")
    extended = run("abcdef").replace(glyph_clusters=run("abc").glyph_clusters)
    assert not internal_glyphs_covered_by_extended_run(primary, extended)


@pytest.mark.parametrize(
    "value",
    [
        (0, 0, 10, 10),
        ["0", "1", "10", "11"],
        (0, 0, 0, 10),
        (0, 0, 10, -1),
        (0, "bad", 10, 10),
        (0, None, 10, 10),
        (1, 2),
        None,
    ],
)
def test_clip_geometry_is_validated_before_layer_grouping(value):
    source = run("text").replace(provenance=(("clip_bbox", value),))
    expected = (
        (0.0, 0.0, 10.0, 10.0)
        if value == (0, 0, 10, 10)
        else (0.0, 1.0, 10.0, 11.0)
        if value == ["0", "1", "10", "11"]
        else None
    )
    assert internal_clip_bbox(source) == expected
