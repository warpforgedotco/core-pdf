import pytest

from core_pdf.impl.capture_records import CapturedPath
from core_pdf.impl.render_display import DisplayList
from core_pdf.impl.render_page import RenderedPage

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
