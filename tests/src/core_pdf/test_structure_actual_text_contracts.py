from types import SimpleNamespace

import pytest

from core_pdf.impl.extract.capture import (
    apply_structure_actual_text,
    run_mcid,
    run_uses_actual_text,
    structure_actual_text_owner,
)
from core_pdf.impl.model.glyphs import GlyphCluster
from core_pdf.impl.model.runs import TextRun


def run(text: str, mcid: object = 0, x: float = 0) -> TextRun:
    box = (x, 1, x + 5, 11)
    baseline = (x, 3, x + 5, 3)
    return TextRun(
        text,
        *box,
        x,
        3,
        10,
        2,
        0,
        0,
        0,
        baseline=baseline,
        provenance=(("mcid", mcid),),
        glyph_clusters=(GlyphCluster(0, text, (), box, box, baseline, None),),
    )


@pytest.mark.parametrize("replacement", ["replacement", ""])
def test_shared_parent_replacement_merges_geometry_once_without_mutating_sources(replacement):
    parent_props = {}
    parents = [
        SimpleNamespace(props=parent_props, actual_text=replacement, parent=None) for _ in range(2)
    ]
    structure = [
        SimpleNamespace(props={}, actual_text="child", parent=parent) for parent in parents
    ]
    first = run("one", 0).replace(confidence=0.9, visible=False, inside_active_clip=False)
    second = run("two", 1, 20).replace(confidence=0.6)
    unrelated = run("outside", None, 10)
    result = apply_structure_actual_text(None, (first, unrelated, second), structure)
    assert len(result) == 2
    merged = result[0]
    assert result[1] is unrelated
    assert merged.text == replacement
    assert (merged.x0, merged.y0, merged.x1, merged.y1) == (0, 1, 25, 11)
    assert merged.advance_bbox == merged.ink_bbox == (0, 1, 25, 11)
    assert merged.baseline == (0, 3, 25, 3)
    assert merged.confidence == 0.6
    assert merged.visible
    assert merged.inside_active_clip
    assert merged.glyph_clusters == first.glyph_clusters + second.glyph_clusters
    assert run_uses_actual_text(merged)
    assert first.text == "one"
    assert second.text == "two"
    assert first.advance_bbox == (0, 1, 5, 11)
    assert first.confidence == 0.9
    assert not first.visible
    assert len(first.glyph_clusters) == 1


def test_equal_but_distinct_owner_dictionaries_do_not_merge():
    structure = [SimpleNamespace(props={}, actual_text="same", parent=None) for _ in range(2)]
    result = apply_structure_actual_text(None, (run("a", 0), run("b", 1, 10)), structure)
    assert [item.text for item in result] == ["same", "same"]
    assert result[0].advance_bbox == (0, 1, 5, 11)
    assert result[1].advance_bbox == (10, 1, 15, 11)


@pytest.mark.parametrize("with_props", [False, True])
def test_parent_cycle_stops_at_repeated_identity_and_keeps_outer_replacement(with_props):
    child = SimpleNamespace(actual_text="child", parent=None)
    parent = SimpleNamespace(actual_text="parent", parent=child)
    child.parent = parent
    if with_props:
        child.props, parent.props = {}, {}
    marker = id(parent.props) if with_props else id(parent)
    assert structure_actual_text_owner(child) == (marker, "parent")


@pytest.mark.parametrize("value", [None, 0, False, b"replacement"])
def test_non_string_actual_text_does_not_replace(value):
    source = run("original")
    result = apply_structure_actual_text(None, (source,), [SimpleNamespace(actual_text=value)])
    assert result == (source,)
    assert result[0] is source


@pytest.mark.parametrize("error_type", [IndexError, TypeError, ValueError])
@pytest.mark.parametrize("boundary", ["page", "lookup", "parent"])
def test_malformed_structure_boundaries_preserve_available_text(error_type, boundary):
    class Broken:
        @property
        def structure(self):
            raise error_type("broken structure")

        def __len__(self):
            return 1

        def __getitem__(self, index):
            raise error_type("broken lookup")

        actual_text = "retained replacement"

        @property
        def parent(self):
            raise error_type("broken parent")

    source = run("original")
    if boundary == "page":
        result = apply_structure_actual_text(Broken(), (source,))
    elif boundary == "lookup":
        result = apply_structure_actual_text(None, (source,), Broken())
    else:
        result = apply_structure_actual_text(None, (source,), [Broken()])
    assert result[0].text == ("retained replacement" if boundary == "parent" else "original")


@pytest.mark.parametrize("mcid", [-1, 3, True, "0", None])
def test_invalid_or_out_of_range_mcid_does_not_select_a_structure_element(mcid):
    source = run("original", mcid)
    result = apply_structure_actual_text(None, (source,), [SimpleNamespace(actual_text="bad")])
    assert result == (source,)


def test_missing_mcid_avoids_lazy_structure_loading():
    class Page:
        @property
        def structure(self):
            raise AssertionError("unneeded structure access")

    sources = (run("original", None),)
    assert apply_structure_actual_text(Page(), sources) is sources


def test_absent_structure_preserves_original_run_tuple():
    sources = (run("original"),)
    assert apply_structure_actual_text(SimpleNamespace(structure=None), sources) is sources


def test_last_integer_mcid_wins_without_treating_boolean_as_an_integer():
    source = run("original").replace(provenance=(("mcid", 2), ("mcid", 7), ("mcid", True)))
    assert run_mcid(source) == 7
