# SPDX-License-Identifier: AGPL-3.0-only
"""``TextRun.replace`` field-preservation contract.

``replace`` rebuilds a run from a hand-written list of ``kwargs.get(...)``
lines, one per constructor parameter. A parameter missing from that list is
silently dropped -- ``font_name`` was, so every run on a rotated page
(``spec/s_07_document/page_boxes.py``) came back with ``font_name=None``.
The signature check below fails when a new field is added to ``__init__``
without being added here, and the round-trips then prove ``replace`` carries
it both when defaulted and when passed explicitly.
"""

from __future__ import annotations

import inspect
from typing import Any

from core_pdf.impl.model.glyphs import GlyphCluster
from core_pdf.impl.model.runs import TextRun

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
    "provenance": (("source", "ocr"),),
    "confidence": 0.25,
    "glyph_clusters": (OTHER_CLUSTER,),
}

# The eight packed coordinates live in `coords`, not in same-named attributes.
COORD_INDEX = {
    "x0": TextRun.X0,
    "y0": TextRun.Y0,
    "x1": TextRun.X1,
    "y1": TextRun.Y1,
    "tx": TextRun.TX,
    "ty": TextRun.TY,
    "font_size": TextRun.FONT_SIZE,
    "space_width": TextRun.SPACE_WIDTH,
}


def field_value(run: TextRun, name: str) -> Any:
    index = COORD_INDEX.get(name)
    if index is not None:
        return run.coords[index]
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
