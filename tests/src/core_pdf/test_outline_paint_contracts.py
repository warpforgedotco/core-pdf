from copy import replace

import numpy
import pytest

from core_pdf.impl.capture_records import CapturedPath, CapturedSubpath
from core_pdf.impl.fonts_glyph_geometry import outline_arrays
from core_pdf.impl.glyphs import GlyphObservation
from core_pdf.impl.render_commands import (
    append_glyph_paint,
    glyph_outline_path,
)
from core_pdf.impl.render_display import DisplayList


class ArrayOutline:
    def __init__(self, contours):
        self.contours = contours
        self.calls = []

    def glyph_outline_arrays(self, code, gid, text):
        self.calls.append((code, gid, text))
        return outline_arrays(self.contours)


class ContourOutline:
    def __init__(self, contours):
        self.contours = contours
        self.calls = []

    def glyph_outline(self, code, gid, text):
        self.calls.append((code, gid, text))
        return self.contours


def reference_outline(contours, transform):
    subpaths = []
    for contour in contours:
        if len(contour) < 2:
            continue
        subpath = CapturedSubpath(list(contour), closed=True).transformed(transform)
        points = subpath.points
        if len(points) >= 2 and points[0] == points[-1]:
            points.pop()
        if len(points) >= 2:
            subpaths.append(subpath)
    if not subpaths:
        return None
    path = CapturedPath(subpaths)
    return path, path.bbox()


def make_glyph(decoder):
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
def test_array_outlines_match_transformed_contour_paths(contours, matrix):
    scalar = reference_outline(contours, matrix)
    array = glyph_outline_path(replace(make_glyph(ArrayOutline(contours)), glyph_transform=matrix))
    if scalar is None:
        assert array is None
        return
    assert array is not None
    scalar_path, scalar_box = scalar
    array_path, array_box, edges = array
    assert [sub.points for sub in scalar_path.subpaths] == [
        sub.points for sub in array_path.subpaths
    ]
    assert scalar_box == array_box
    numpy.testing.assert_array_equal(edges, numpy.asarray(scalar_path.fill_edges()))
    assert all(sub.closed for sub in array_path.subpaths)


@pytest.mark.parametrize("mode", range(8))
@pytest.mark.parametrize("include_paint", [False, True])
@pytest.mark.parametrize("visible", [False, True])
def test_text_modes_separate_paint_from_accumulated_clipping(mode, include_paint, visible):
    decoder = ArrayOutline((((0, 0), (3, 0), (1, 3)),))
    glyph = replace(
        make_glyph(decoder), text_render_mode=mode, visible=visible, clip_glyph=mode >= 4
    )
    display = DisplayList(10, 10)
    clipping = []
    assert append_glyph_paint(display, glyph, clipping, include_paint=include_paint)
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
    decoder = ArrayOutline((((0, 0), (3, 0), (1, 3)),))
    glyph = replace(make_glyph(decoder), gid=12, **fields)
    assert glyph_outline_path(glyph) is not None
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
    decoder = ArrayOutline((((0, 0), (3, 0), (1, 3)),))
    glyph = replace(make_glyph(decoder), **fields)
    assert glyph_outline_path(glyph) is None
    assert decoder.calls == []


def test_missing_outline_requests_bitmap_fallback_without_partial_paint():
    display = DisplayList(10, 10)
    clipping = []
    assert not append_glyph_paint(display, make_glyph(ArrayOutline(())), clipping)
    assert display.items == []
    assert clipping == []


def test_decoders_without_outline_arrays_paint_no_outline():
    decoder = ContourOutline((((0, 0), (3, 0), (1, 3)),))
    assert glyph_outline_path(make_glyph(decoder)) is None
    assert decoder.calls == []
