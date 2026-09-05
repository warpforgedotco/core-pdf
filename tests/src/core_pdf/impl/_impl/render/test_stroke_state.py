# SPDX-License-Identifier: AGPL-3.0-only
"""Stroke-state regressions checked with qpdf 12.3.2 and Poppler 26.07.0."""

import numpy
import pytest

from tests.helpers.pdf_bytes import one_page_pdf, open_pdf


@pytest.mark.parametrize("scale", [1, 3, 6])
def test_explicit_zero_width_remains_one_device_pixel(scale: int) -> None:
    # pdftoppm at 72/216/432 DPI keeps this hairline one device pixel wide.
    data = one_page_pdf(b"0 w 1 4 m 7 4 l S", media_box=(0, 0, 8, 8))
    with open_pdf(data) as document:
        pixels = document.pages[0].render().rasterize(scale=scale).array()
    # Our coverage sampler may split that pixel over adjacent rows.
    assert abs(int(pixels[:, 4 * scale, 3].sum()) - 255) <= 1


@pytest.mark.parametrize("path", [b"4 1 m 4 7 l", b"1 1 m 7 7 l"])
@pytest.mark.parametrize("scale", [1, 3, 6])
def test_vertical_and_diagonal_hairlines_keep_device_width(path: bytes, scale: int) -> None:
    # Poppler at 72/216/432 DPI confirms width stays one device pixel for
    # both directions. Integrated coverage permits orientation/AA differences.
    data = one_page_pdf(b"0 w " + path + b" S", media_box=(0, 0, 8, 8))
    with open_pdf(data) as document:
        pixels = document.pages[0].render().rasterize(scale=scale).array()
    coverage = float(pixels[:, :, 3].sum()) / 255.0
    assert 0.9 <= coverage / (6 * scale) <= 1.5


@pytest.mark.parametrize(
    ("path", "crop"),
    [
        (b"1 3.99 m 7 3.99 l", (0, 4, 8, 8)),
        (b"3.99 1 m 3.99 7 l", (4, 0, 8, 8)),
        (b"1 1 m 7 7 l", (4, 4, 8, 8)),
    ],
)
@pytest.mark.parametrize("scale", [1, 3, 6])
def test_crop_preserves_hairline_coverage_crossing_the_crop_boundary(
    path: bytes, crop: tuple[int, int, int, int], scale: int
) -> None:
    data = one_page_pdf(b"0 w " + path + b" S", media_box=(0, 0, 8, 8))
    with open_pdf(data) as document:
        page = document.pages[0].render()
        full = page.rasterize(scale=scale).array()
        cropped = page.rasterize(scale=scale, crop=crop).array()
    assert cropped[:, :, 3].any()
    numpy.testing.assert_array_equal(cropped, full[: (8 - crop[1]) * scale, crop[0] * scale :])


@pytest.mark.parametrize("pattern", [b"[3] 0", b"[3 3] 0", b"[3 3] 2"])
def test_collinear_vertices_do_not_restart_the_dash_phase(pattern: bytes) -> None:
    rasters = []
    for path in (b"1 4 m 19 4 l S", b"1 4 m 5 4 l 19 4 l S"):
        data = one_page_pdf(b"2 w " + pattern + b" d " + path, media_box=(0, 0, 20, 8))
        with open_pdf(data) as document:
            rasters.append(document.pages[0].render().rasterize().array())
    numpy.testing.assert_array_equal(*rasters)
    # These exact centerline patterns were verified with pdftoppm -r 72.
    expected = ".#...###...###...##." if pattern.endswith(b"2") else ".###...###...###...."
    assert "".join("#" if alpha >= 128 else "." for alpha in rasters[0][4, :, 3]) == expected


@pytest.mark.parametrize("cap", [0, 1, 2])
def test_a_dash_crossing_a_vertex_has_one_join_and_only_endpoint_caps(cap: int) -> None:
    # Poppler's 288-DPI rasters are byte-identical for these two descriptions.
    rasters = []
    for path in (
        b"[6 2] 0 d 1 4 m 5 4 l 5 11 l S",
        b"1 4 m 5 4 l 5 6 l S 5 8 m 5 11 l S",
    ):
        data = one_page_pdf(f"2 w {cap} J 1 j ".encode() + path, media_box=(0, 0, 12, 12))
        with open_pdf(data) as document:
            rasters.append(document.pages[0].render().rasterize(scale=4).array())
    numpy.testing.assert_array_equal(*rasters)


def test_closed_path_dash_phase_continues_over_the_closing_edge() -> None:
    data = one_page_pdf(b"2 w [5 3] 2 d 3 3 m 9 3 l 9 9 l 3 9 l h S", media_box=(0, 0, 12, 12))
    with open_pdf(data) as document:
        pixels = document.pages[0].render().rasterize(scale=4).array()
    # Poppler confirms the closing edge is painted below y=5 and above y=8,
    # with a gap in between. Samples stay inside strokes, away from edge AA.
    assert pixels[47 - 4 * 4, 3 * 4, 3] == 255
    assert pixels[47 - 6 * 4, 3 * 4, 3] == 0
    assert pixels[47 - 8 * 4, 3 * 4, 3] == 255


def test_dash_phase_restarts_for_each_new_subpath() -> None:
    data = one_page_pdf(b"2 w [3 3] 0 d 1 4 m 11 4 l 1 8 m 11 8 l S", media_box=(0, 0, 12, 12))
    with open_pdf(data) as document:
        pixels = document.pages[0].render().rasterize().array()
    numpy.testing.assert_array_equal(pixels[4, :, 3], pixels[8, :, 3])


def test_zero_length_dashes_with_round_caps_paint_dots() -> None:
    data = one_page_pdf(b"2 w 1 J [0 3] 0 d 1 4 m 10 4 l S", media_box=(0, 0, 12, 8))
    with open_pdf(data) as document:
        pixels = document.pages[0].render().rasterize().array()
    # Poppler confirms the dot centers and the intervening gaps.
    assert numpy.all(pixels[4, [1, 4, 7], 3] >= 150)
    assert list(pixels[4, [2, 5, 8], 3]) == [0, 0, 0]


def test_subpixel_round_dashes_use_fractional_coverage() -> None:
    # Poppler paints this width-.5 round dot at 72 DPI with 40/255 pixels of
    # integrated coverage. Analytic circular coverage is pi/16 ~= .196 pixels;
    # either treatment must preserve a faint dot rather than a solid pixel.
    data = one_page_pdf(b".5 w 1 J [0 3] 0 d 4 4 m 4.1 4 l S", media_box=(0, 0, 8, 8))
    with open_pdf(data) as document:
        pixels = document.pages[0].render().rasterize().array()
    assert 35 <= int(pixels[:, :, 3].sum()) <= 55
    assert int(pixels[:, :, 3].max()) < 25
