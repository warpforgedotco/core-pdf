"""A knockout group of plain fills is set up over the pixels its members can paint."""

import math
from typing import Any

import numpy
import pytest

from core_pdf import PdfDocument
from core_pdf.impl.render import display
from core_pdf.impl.render.display import plain_fill_members_box
from core_pdf.impl.render.model import DisplayItem, DisplayListItem, PathPaintItem, PathPaintKind
from core_pdf.impl.render.target import RasterTarget


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
    assert plain_fill_members_box(items, 0) == (0.5, 2.0, 3.0, 6.0)


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
    assert plain_fill_members_box([fill((1.0, 2.0, 3.0, 4.0)), member], 0) is None


def test_text_items_paint_nothing_and_are_passed_over() -> None:
    text = DisplayListItem(kind="text", seqno=1, data={})
    assert plain_fill_members_box([text, fill((1.0, 2.0, 3.0, 4.0)), text], 0) == (
        1.0,
        2.0,
        3.0,
        4.0,
    )


def test_no_fills_leave_the_group_unbounded() -> None:
    assert plain_fill_members_box([], 0) is None
    assert plain_fill_members_box([DisplayListItem(kind="text", seqno=1, data={})], 0) is None


CONTENT = (
    b"0.2 0.6 0.9 rg 10 60 150 60 re f "
    b"BT /F1 28 Tf -3 Tc 0.9 0.1 0.1 rg 20 80 Td (WAVAWO) Tj "
    b"0 0 0 rg 0 -6 Td (WAVAWO) Tj ET "
    b"BT /F1 20 Tf 0 0.5 0 rg 30 70 Td [(AV) 400 (AV)] TJ ET"
)


def one_page_pdf(content: bytes) -> bytes:
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        4: b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        5: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    data = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for number, body in objects.items():
        offsets[number] = len(data)
        data += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(data)
    data += b"xref\n0 6\n0000000000 65535 f \n"
    data += b"".join(b"%010d 00000 n \n" % offsets[number] for number in range(1, 6))
    data += b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % xref
    return bytes(data)


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
    monkeypatch.setattr(display, "plain_fill_members_box", lambda *_: None)
    assert rendered() == bounded
