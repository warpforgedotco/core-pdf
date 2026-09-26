# SPDX-License-Identifier: AGPL-3.0-only

"""A group composites over what it painted, not over the page.

Outside its paint window a group is untouched -- transparent if it was
isolated, equal to the backdrop it was seeded with if it was not -- and every
compositing formula leaves the destination alone where the source contributes
nothing. So the window has to cover every write into the group's buffer, and
these pin the paths that were found not to."""

from typing import Any

import numpy
import pytest

from core_pdf.impl.render.target import RasterTarget
from tests.src.core_pdf.raster_support import make_target


def grouped_target(width: int = 40, height: int = 40, *, isolated: bool = True) -> RasterTarget:
    target = make_target(width=width, height=height)
    target.push_group(bytearray(width * height * 4), None, None, isolated=isolated)
    return target


def window(target: RasterTarget) -> list[int]:
    assert target.paint_window is not None
    return list(target.paint_window)


def test_the_page_tracks_no_window() -> None:
    # Most documents have no groups at all, and tracking a window they would
    # never read is pure cost.
    assert make_target().paint_window is None


def test_an_isolated_group_composites_over_what_it_painted() -> None:
    target = grouped_target()
    target.fill_rect((4.0, 30.0, 9.0, 36.0), (255, 0, 0, 255))
    group = target.buffer_stack[-1]
    assert target.group_window(group) == (slice(4, 10), slice(4, 9))


def test_a_group_that_painted_nothing_composites_nothing() -> None:
    target = grouped_target()
    group = target.buffer_stack[-1]
    assert group.paint_window == []
    assert target.group_window(group) is None


def test_the_window_covers_every_fill() -> None:
    target = grouped_target()
    target.fill_rect((4.0, 30.0, 9.0, 36.0), (255, 0, 0, 255))
    target.fill_rect((20.0, 5.0, 25.0, 11.0), (0, 255, 0, 255))
    assert window(target) == [4, 35, 4, 25]


def test_a_per_pixel_span_extends_the_window() -> None:
    """blend_px is the one primitive that writes without going through a plane
    recorder. Short spans -- the corners of a rounded rectangle -- take it, and
    for 9 documents in the corpus those corners were the only thing outside the
    window the group composited over."""
    target = grouped_target()
    assert target.group_source_alpha is None
    assert target.group_source_shape is None
    target.blend_px((7 * target.width + 11) * 4, (255, 0, 0, 255), None)
    assert window(target) == [7, 8, 11, 12]


def test_a_fully_transparent_pixel_paints_nothing_and_records_nothing() -> None:
    target = grouped_target()
    target.blend_px((7 * target.width + 11) * 4, (255, 0, 0, 0), None)
    assert window(target) == [7, 8, 11, 12]
    assert not any(target.pixels)


@pytest.mark.parametrize("isolated", [True, False])
def test_the_window_survives_a_nested_group(isolated: bool) -> None:
    target = grouped_target(isolated=isolated)
    target.push_group(bytearray(40 * 40 * 4), None, None, isolated=True)
    target.fill_rect((4.0, 30.0, 9.0, 36.0), (255, 0, 0, 255))
    assert window(target) == [4, 10, 4, 9]
    target.composite_group(target.pop_group())
    # Compositing the child in is itself a paint into the parent.
    assert window(target) == [4, 10, 4, 9]


def test_leaving_the_last_group_stops_the_tracking() -> None:
    target = grouped_target()
    target.fill_rect((4.0, 30.0, 9.0, 36.0), (255, 0, 0, 255))
    target.composite_group(target.pop_group())
    assert target.paint_window is None


def test_a_stroke_records_its_own_region_rather_than_the_scratch_it_uses() -> None:
    """paint_stroke_once paints into a throwaway full-page coverage buffer, so
    the window must not follow it there."""
    from core_pdf.impl.capture.records import CapturedPath
    from core_pdf.impl.render.target import paint_stroke_once

    target = grouped_target()
    path = CapturedPath()
    path.move_to(5.0, 30.0)
    path.line_to(9.0, 30.0)
    paint_stroke_once(target, path, 1.0, (255, 0, 0, 255), None, None, 0, 0)
    y0, y1, x0, x1 = window(target)
    assert y1 - y0 < 6
    assert x1 - x0 < 10


def test_every_written_pixel_lies_inside_the_window() -> None:
    """The invariant itself, over the primitives that reach a group."""
    target = grouped_target(width=60, height=60)
    target.fill_rect((4.0, 50.0, 9.0, 56.0), (255, 0, 0, 255))
    target.blend_px((40 * target.width + 51) * 4, (0, 255, 0, 200), None)
    target.fill_line(2.0, 20.0, 30.0, 20.0, 1.0, (0, 0, 255, 255))
    view: Any = numpy.frombuffer(bytes(target.pixels), dtype=numpy.uint8).reshape(60, 60, 4)
    rows = numpy.flatnonzero((view[..., 3] != 0).any(axis=1))
    columns = numpy.flatnonzero((view[..., 3] != 0).any(axis=0))
    y0, y1, x0, x1 = window(target)
    assert y0 <= int(rows[0])
    assert y1 >= int(rows[-1]) + 1
    assert x0 <= int(columns[0])
    assert x1 >= int(columns[-1]) + 1


def test_a_diagonal_line_with_no_plane_recording_extends_the_window() -> None:
    """A slanted segment over more than 64 pixels of box is rasterized by numpy
    and recorded through the planes; with neither plane recording, nothing
    extended the window. On pdfminer.six cmp_itext_logo that left a stroke's
    rows outside it."""
    target = grouped_target(width=60, height=60)
    assert target.group_source_alpha is None
    assert target.group_source_shape is None
    target.fill_line(5.0, 10.0, 40.0, 30.0, 2.5, (0, 0, 255, 255))
    view: Any = numpy.frombuffer(bytes(target.pixels), dtype=numpy.uint8).reshape(60, 60, 4)
    rows = numpy.flatnonzero((view[..., 3] != 0).any(axis=1))
    columns = numpy.flatnonzero((view[..., 3] != 0).any(axis=0))
    assert rows.size
    y0, y1, x0, x1 = window(target)
    assert y0 <= int(rows[0])
    assert y1 >= int(rows[-1]) + 1
    assert x0 <= int(columns[0])
    assert x1 >= int(columns[-1]) + 1


def test_a_stroke_leaves_its_reused_scratch_zeroed() -> None:
    """paint_stroke_once keeps its coverage buffer between strokes and clears
    only the window it painted, so a later stroke starts from zeros and paints
    what it would into a fresh buffer."""
    from core_pdf.impl.capture.records import CapturedPath
    from core_pdf.impl.render.target import paint_stroke_once

    def stroke(target: RasterTarget, x0: float, y0: float, x1: float, y1: float) -> None:
        path = CapturedPath()
        path.move_to(x0, y0)
        path.line_to(x1, y1)
        paint_stroke_once(target, path, 2.0, (255, 0, 0, 128), None, None, 1, 0)

    reused = grouped_target(width=60, height=60)
    stroke(reused, 5.0, 10.0, 40.0, 30.0)
    assert reused.stroke_scratch is not None
    assert not any(reused.stroke_scratch)
    stroke(reused, 10.0, 50.0, 50.0, 45.0)
    assert not any(reused.stroke_scratch)

    fresh = grouped_target(width=60, height=60)
    stroke(fresh, 5.0, 10.0, 40.0, 30.0)
    fresh.stroke_scratch = None
    stroke(fresh, 10.0, 50.0, 50.0, 45.0)
    assert bytes(reused.pixels) == bytes(fresh.pixels)
    assert window(reused) == window(fresh)
