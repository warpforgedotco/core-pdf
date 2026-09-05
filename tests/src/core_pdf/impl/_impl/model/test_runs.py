# SPDX-License-Identifier: AGPL-3.0-only
"""Run replacement preserves fields and invalidates evidence after edits.

A former handwritten copier lost ``font_name`` on rotated pages. Keep full-field
round trips alongside the dependent-evidence checks for the dataclass copier.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from core_pdf.impl._impl.model.glyphs import GlyphCluster
from core_pdf.impl._impl.model.runs import TextRun

CLUSTER = GlyphCluster(
    7,
    "sample",
    (),
    (1.0, 2.0, 3.0, 4.0),
    (1.5, 2.5, 3.5, 4.5),
    (1.0, 2.0, 3.0, 2.0),
    0.75,
)

OTHER_CLUSTER = GlyphCluster(
    8,
    "other",
    (),
    (10.0, 20.0, 30.0, 40.0),
    (15.0, 25.0, 35.0, 45.0),
    (10.0, 20.0, 30.0, 20.0),
    0.25,
)

# A distinctive, non-default value for every constructor parameter, so a field
# `replace` drops reads as a changed value rather than a coincidental match.
FIELDS: dict[str, Any] = {
    "text": "sample",
    "x0": 1.0,
    "y0": 2.0,
    "x1": 3.0,
    "y1": 4.0,
    "tx": 5.0,
    "ty": 6.0,
    "font_size": 7.0,
    "space_width": 8.0,
    "order": 9,
    "stream_order": 10,
    "xobject_depth": 11,
    "font_name": "Helvetica",
    "is_vertical": True,
    "rotation_angle": 90,
    "visible": False,
    "inside_active_clip": False,
    "line_break_before": True,
    "seqno": 12,
    "fill_color": (0.1, 0.2, 0.3),
    "advance_bbox": (1.0, 2.0, 3.0, 4.0),
    "ink_bbox": (1.5, 2.5, 3.5, 4.5),
    "baseline": (1.0, 2.0, 3.0, 2.0),
    "provenance": (("source", "capture"),),
    "confidence": 0.75,
    "glyph_clusters": (CLUSTER,),
}

REPLACEMENTS: dict[str, Any] = {
    "text": "other",
    "x0": 10.0,
    "y0": 20.0,
    "x1": 30.0,
    "y1": 40.0,
    "tx": 50.0,
    "ty": 60.0,
    "font_size": 70.0,
    "space_width": 80.0,
    "order": 90,
    "stream_order": 100,
    "xobject_depth": 110,
    "font_name": "Times",
    "is_vertical": False,
    "rotation_angle": 270,
    "visible": True,
    "inside_active_clip": True,
    "line_break_before": False,
    "seqno": 120,
    "fill_color": (0.4, 0.5, 0.6),
    "advance_bbox": (10.0, 20.0, 30.0, 40.0),
    "ink_bbox": (15.0, 25.0, 35.0, 45.0),
    "baseline": (10.0, 20.0, 30.0, 20.0),
    "provenance": (("source", "external"),),
    "confidence": 0.25,
    "glyph_clusters": (OTHER_CLUSTER,),
}


def field_value(run: TextRun, name: str) -> Any:
    return getattr(run, name)


def test_field_tables_cover_every_constructor_parameter() -> None:
    parameters = {name for name in inspect.signature(TextRun.__init__).parameters if name != "self"}
    assert parameters == set(FIELDS)
    assert parameters == set(REPLACEMENTS)


def test_replace_without_arguments_preserves_every_field() -> None:
    run = TextRun(**FIELDS)

    copy = run.replace()

    for name, expected in FIELDS.items():
        assert field_value(copy, name) == expected, f"replace() dropped {name}"


def test_replace_applies_every_field_it_is_given() -> None:
    run = TextRun(**FIELDS)

    updated = run.replace(**REPLACEMENTS)

    for name, expected in REPLACEMENTS.items():
        assert field_value(updated, name) == expected, f"replace() ignored {name}"


def test_replace_preserves_font_name_when_geometry_changes() -> None:
    """The rotated-page path: geometry is replaced, the font name must survive."""
    run = TextRun(**FIELDS)

    rotated = run.replace(x0=10.0, y0=20.0, x1=30.0, y1=40.0, rotation_angle=180)

    assert rotated.font_name == "Helvetica"
    assert rotated.rotation_angle == 180


def test_runs_remain_mutable_identity_records() -> None:
    run = TextRun(**FIELDS)
    copy = run.replace()

    assert copy is not run
    assert copy != run
    assert len({run, copy}) == 2
    copy.text = "changed"
    assert run.text == "sample"
    assert not hasattr(run, "__dict__")


def test_replace_rejects_unknown_fields() -> None:
    with pytest.raises(TypeError, match="font_szie"):
        TextRun(**FIELDS).replace(font_szie=999)


def test_replace_text_discards_source_clusters_but_preserves_geometry() -> None:
    run = TextRun(**FIELDS)

    updated = run.replace(text="other")

    assert updated.glyph_clusters == ()
    assert updated.baseline == run.baseline
    assert updated.ink_bbox == run.ink_bbox


@pytest.mark.parametrize("changes", [{"x0": 0.0}, {"tx": 0.0}, {"rotation_angle": 180}])
def test_repositioning_a_run_discards_untransformed_evidence(changes: dict[str, Any]) -> None:
    run = TextRun(**FIELDS)

    updated = run.replace(**changes)

    assert updated.baseline is None
    assert updated.glyph_clusters == ()
    assert run.baseline == CLUSTER.baseline
    assert run.glyph_clusters == (CLUSTER,)


def test_refining_ink_bounds_keeps_baseline_and_invalidates_stale_clusters() -> None:
    run = TextRun(**FIELDS)

    updated = run.replace(ink_bbox=(1.6, 2.6, 3.4, 4.4))

    assert updated.baseline == run.baseline
    assert updated.glyph_clusters == ()
