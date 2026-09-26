"""A knockout group of plain fills is set up over the pixels its members can paint."""

import math
from typing import Any

import numpy
import pytest

from core_pdf import PdfDocument
from core_pdf.impl import render_display as display
from core_pdf.impl.render_display import plain_fill_members_box
from core_pdf.impl.render_model import DisplayItem, DisplayListItem, PathPaintItem, PathPaintKind
from core_pdf.impl.render_target import RasterTarget
from tests.src.core_pdf.pdf_bytes import one_page_pdf


def fill(bbox: tuple[float, float, float, float] | None, **changes: Any) -> PathPaintItem:
    values: dict[str, Any] = {
        "paint_kind": PathPaintKind.FILL,
        "seqno": 1,
        "bbox": bbox,
        "path": None,
        "fill": (0.0, 0.0, 0.0),
        "fill_opacity": 1.0,
        "stroke_color": None,
        "stroke_opacity": None,
        "line_width": 1.0,
        "line_cap": 0,
        "line_join": 0,
        "dash_pattern": None,
        "fill_rule": "nonzero",
        "blend_mode": None,
        "soft_mask_alpha": None,
        "edge_array": numpy.zeros((0, 4)),
    }
    values.update(changes)
    return PathPaintItem(**values)


def test_plain_fills_give_the_union_of_their_boxes() -> None:
    items: list[DisplayItem] = [fill((1.0, 2.0, 3.0, 4.0)), fill((0.5, 3.0, 2.0, 6.0))]
    assert plain_fill_members_box(items, 0) == (True, (0.5, 2.0, 3.0, 6.0))


@pytest.mark.parametrize(
    "member",
    [
        fill((0.0, 0.0, 1.0, 1.0), paint_kind=PathPaintKind.STROKE),
        fill((0.0, 0.0, 1.0, 1.0), edge_array=None),
        fill((0.0, 0.0, 1.0, 1.0), blend_mode="Multiply"),
        fill((0.0, 0.0, 1.0, 1.0), fill_pattern=object()),
        fill((0.0, 0.0, math.nan, 1.0)),
        fill(None),
        DisplayListItem(kind="clip", seqno=1, data={}),
    ],
)
def test_any_other_member_leaves_the_group_unbounded(member: DisplayItem) -> None:
    assert plain_fill_members_box([fill((1.0, 2.0, 3.0, 4.0)), member], 0) == (False, None)


def test_text_items_paint_nothing_and_are_passed_over() -> None:
    text = DisplayListItem(kind="text", seqno=1, data={})
    assert plain_fill_members_box([text, fill((1.0, 2.0, 3.0, 4.0)), text], 0) == (
        True,
        (1.0, 2.0, 3.0, 4.0),
    )


def test_a_group_of_no_fills_paints_nothing() -> None:
    assert plain_fill_members_box([], 0) == (True, None)
    text = DisplayListItem(kind="text", seqno=1, data={})
    assert plain_fill_members_box([text], 0) == (True, None)


CONTENT = (
    b"0.2 0.6 0.9 rg 10 60 150 60 re f "
    b"BT /F1 28 Tf -3 Tc 0.9 0.1 0.1 rg 20 80 Td (WAVAWO) Tj "
    b"0 0 0 rg 0 -6 Td (WAVAWO) Tj ET "
    b"BT /F1 20 Tf 0 0.5 0 rg 30 70 Td [(AV) 400 (AV)] TJ ET"
)


def rendered() -> bytes:
    with PdfDocument(one_page_pdf(CONTENT)) as document:
        return document.pages[0].render().rasterize(scale=2.0).array().tobytes()


def test_a_region_set_up_group_renders_as_a_page_sized_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    regions: list[object] = []
    original = RasterTarget.knockout_group_region

    def recorded(self: RasterTarget, item: DisplayItem) -> Any:
        region = original(self, item)
        regions.append(region)
        return region

    monkeypatch.setattr(RasterTarget, "knockout_group_region", recorded)
    bounded = rendered()
    assert regions
    assert all(region is not None for region in regions)
    monkeypatch.setattr(display, "plain_fill_members_box", lambda *_: (False, None))
    assert rendered() == bounded


def test_a_glyph_fill_skips_its_elementary_group_and_paints_the_same(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    taken: list[bool] = []
    original = RasterTarget.knockout_glyph_fill

    def recorded(self: RasterTarget, item: DisplayItem) -> bool:
        result = original(self, item)
        taken.append(result)
        return result

    monkeypatch.setattr(RasterTarget, "knockout_glyph_fill", recorded)
    direct = rendered()
    assert any(taken)
    monkeypatch.setattr(RasterTarget, "knockout_glyph_fill", lambda *_: False)
    assert rendered() == direct
