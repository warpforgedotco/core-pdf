import pytest

from core_pdf.impl.capture_records import CapturedPath
from core_pdf.impl.render_blend import color_rgba
from core_pdf.impl.render_display import DisplayList
from core_pdf.impl.render_model import (
    ClipItem,
    ControlItem,
    DisplayListItem,
    GlyphBitmapItem,
    GroupBeginItem,
    ScopeBeginItem,
    ShadingItem,
    display_item,
)
from core_pdf.impl.render_page import RenderedPage
from tests.src.core_pdf.raster_support import make_target

PATH = CapturedPath()
PATH.rect(1.0, 2.0, 3.0, 4.0)

PAYLOADS = [
    ("clip", {"path": PATH, "fill_rule": "evenodd"}),
    ("clip", {"bbox": (1, 2, 3, 4), "fill": None, "path": PATH, "items": ()}),
    ("group-begin", {"fill_opacity": 1.0, "blend_mode": None, "group_knockout": True}),
    ("group-begin", {"group_isolated": False, "soft_mask_alpha": 0.5, "extra": [1]}),
    ("group-end", {}),
    ("group-end", {"bbox": None, "path": None}),
    ("scope-begin", {}),
    ("scope-begin", {"path": PATH}),
    ("scope-end", {"items": ()}),
    ("state-push", {}),
    ("state-pop", {"bbox": (0, 0, 1, 1)}),
    ("shading", {"dictionary": {"ShadingType": 2}, "rect": (0, 0, 5, 6), "fill_opacity": 0.5}),
    ("shading", {"bbox": (0, 0, 1, 1), "color_rendering": None}),
    (
        "glyph",
        {
            "text": "a",
            "bbox": (0, 0, 1, 1),
            "fill_color": (0.0,),
            "blend_mode": "Multiply",
            "graphics_soft_mask": "not a mask",
            "bitmap": [1, 2],
        },
    ),
    ("text", {"text": "a", "bbox": (0, 0, 1, 1)}),
    ("annotation", {"subtype": "Link", "rect": (0, 0, 1, 1)}),
    ("widget", {"name": "field", "rect": (0, 0, 1, 1)}),
]


@pytest.mark.parametrize(("kind", "payload"), PAYLOADS)
def test_appended_items_serialize_their_payload_in_order(kind, payload):
    display = DisplayList(10, 10)
    display.append(kind, 3, **payload)
    page = RenderedPage(1, 10, 10, 0, display)
    (entry,) = page.to_dict()["display_list"]
    expected = dict(payload)
    if "graphics_soft_mask" in expected:
        expected["graphics_soft_mask"] = None
    assert entry["kind"] == kind
    assert entry["seqno"] == 3
    assert list(entry["data"].items()) == list(expected.items())


@pytest.mark.parametrize(
    ("kind", "item_type"),
    [
        ("clip", ClipItem),
        ("scope-begin", ScopeBeginItem),
        ("group-begin", GroupBeginItem),
        ("glyph", GlyphBitmapItem),
        ("shading", ShadingItem),
        ("state-push", ControlItem),
        ("state-pop", ControlItem),
        ("group-end", ControlItem),
        ("scope-end", ControlItem),
        ("text", DisplayListItem),
        ("annotation", DisplayListItem),
        ("widget", DisplayListItem),
    ],
)
def test_each_kind_builds_its_typed_item(kind, item_type):
    item = display_item(kind, 1, {})
    assert type(item) is item_type
    assert item.kind == kind


def test_a_glyph_bitmap_item_paints_like_the_direct_call():
    data = {
        "bbox": (1.0, 1.0, 3.0, 3.0),
        "fill_color": (1.0, 0.0, 0.0),
        "fill_opacity": 0.5,
        "bitmap": [0b01, 0b10],
        "bitmap_width": 2,
        "bitmap_height": 2,
    }
    painted = make_target(4, 4)
    painted.paint_item(display_item("glyph", 0, data))
    direct = make_target(4, 4)
    rgba = color_rgba((1.0, 0.0, 0.0), 0.5)
    direct.draw_glyph_bitmap(data["bbox"], data["bitmap"], rgba, None, 2, 2)
    assert any(painted.pixels)
    assert painted.pixels == direct.pixels
    hidden = make_target(4, 4)
    hidden.paint_item(display_item("glyph", 0, {**data, "visible": False}))
    assert not any(hidden.pixels)
