from copy import replace

import numpy as np
import pytest

from core_pdf.impl.fonts.decoder import internal_outline_arrays
from core_pdf.impl.model.glyphs import GlyphObservation
from core_pdf.impl.render.commands import (
    internal_append_glyph_paint,
    internal_glyph_outline_path,
)
from core_pdf.impl.render.display import DisplayList


class ScalarOutline:
    def __init__(self, contours):
        self.contours = contours
        self.calls = []

    def glyph_outline(self, code, gid, text):
        self.calls.append((code, gid, text))
        return self.contours


class ArrayOutline(ScalarOutline):
    def glyph_outline_arrays(self, code, gid, text):
        self.calls.append((code, gid, text))
        return internal_outline_arrays(self.contours)


def internal_glyph(decoder):
    return GlyphObservation(
        "A",
        (0, 0, 3, 3),
        (0, 0, 3, 3),
        7,
        char_code=65,
        font_decoder=decoder,
        glyph_transform=(1, 0, 0, 1, 0, 0),
        fill=(1, 0, 0),
    )


@pytest.mark.parametrize(
    "contours",
    [
        (),
        ((),),
        (((1, 1),),),
        (((1, 1), (1, 1)),),
        (((0, 0), (3, 0), (1, 3)),),
        (((0, 0), (3, 0), (1, 3), (0, 0)),),
        (((99, 99), (99, 99)), ((0, 0), (3, 0), (1, 3))),
    ],
)
@pytest.mark.parametrize(
    "matrix",
    [(1, 0, 0, 1, 0, 0), (0, 2, -3, 0, 5, 7), (1, 0.25, -0.5, 2, 0.1, 0.2), (0, 0, 0, 0, 5, 6)],
)
def test_scalar_and_array_outlines_have_identical_transformed_paths(contours, matrix):
    results = [
        internal_glyph_outline_path(replace(internal_glyph(kind(contours)), glyph_transform=matrix))
        for kind in (ScalarOutline, ArrayOutline)
    ]
    scalar, array = results
    if scalar is None:
        assert array is None
        return
    assert array is not None
    scalar_path, scalar_box, _ = scalar
    array_path, array_box, edges = array
    assert [sub.points for sub in scalar_path.subpaths] == [
        sub.points for sub in array_path.subpaths
    ]
    assert scalar_box == array_box
    np.testing.assert_array_equal(edges, np.asarray(scalar_path.fill_edges()))
    assert all(sub.closed for sub in array_path.subpaths)


@pytest.mark.parametrize("mode", range(8))
@pytest.mark.parametrize("include_paint", [False, True])
@pytest.mark.parametrize("visible", [False, True])
def test_text_modes_separate_paint_from_accumulated_clipping(mode, include_paint, visible):
    decoder = ArrayOutline((((0, 0), (3, 0), (1, 3)),))
    glyph = replace(
        internal_glyph(decoder), text_render_mode=mode, visible=visible, clip_glyph=mode >= 4
    )
    display = DisplayList(10, 10)
    clipping = []
    assert internal_append_glyph_paint(display, glyph, clipping, include_paint=include_paint)
    assert bool(clipping) is (mode >= 4)
    paints = visible and include_paint and mode not in (3, 7)
    assert len(display.items) == int(paints)
    if paints:
        assert (
            display.items[0].kind
            == {0: "fill", 1: "stroke", 2: "fillstroke", 4: "fill", 5: "stroke", 6: "fillstroke"}[
                mode
            ]
        )
        assert display.items[0].seqno == 7


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"bitmap_code": 8, "cid": 9, "char_code": 10}, 8),
        ({"cid": 9, "char_code": 10}, 9),
        ({"char_code": 10}, 10),
    ],
)
def test_outline_code_precedence_retains_gid_and_text(fields, expected):
    decoder = ScalarOutline((((0, 0), (1, 1)),))
    glyph = replace(internal_glyph(decoder), gid=12, **fields)
    assert internal_glyph_outline_path(glyph) is not None
    assert decoder.calls == [(expected, 12, "A")]


@pytest.mark.parametrize(
    "fields",
    [
        {"paint_glyph": False},
        {"glyph_transform": None},
        {"font_decoder": None},
        {"char_code": None},
        {"font_decoder": object()},
    ],
)
def test_missing_outline_inputs_return_no_outline(fields):
    decoder = ScalarOutline((((0, 0), (1, 1)),))
    glyph = replace(internal_glyph(decoder), **fields)
    assert internal_glyph_outline_path(glyph) is None
    assert decoder.calls == []


def test_missing_outline_requests_bitmap_fallback_without_partial_paint():
    display = DisplayList(10, 10)
    clipping = []
    assert not internal_append_glyph_paint(display, internal_glyph(ScalarOutline(())), clipping)
    assert display.items == []
    assert clipping == []
