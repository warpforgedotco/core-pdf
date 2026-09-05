# SPDX-License-Identifier: AGPL-3.0-only
"""Opacity ownership checked with qpdf 12.3.2 and Poppler 26.07.0."""

import numpy
import pytest

from tests.helpers.pdf_bytes import one_page_pdf, open_pdf, stream_obj


def internal_group_pdf(content: bytes, *, masked: bool = False, outer_alpha: float = 0.5) -> bytes:
    return one_page_pdf(
        b"/GS gs /Fm Do",
        media_box=(0, 0, 8, 8),
        resources=b"<< /ExtGState << /GS << /ca "
        + str(outer_alpha).encode()
        + b" >> >> /XObject << /Fm 6 0 R >> >>",
        extra_objects=[
            stream_obj(
                content,
                b"/Type /XObject /Subtype /Form /BBox [0 0 8 8] "
                b"/Group << /S /Transparency /I true >> "
                b"/Resources << /ExtGState << /Opaque << /ca 1 /CA 1 >> "
                b"/Half << /ca .5 /CA .5 >> /Multiply << /BM /Multiply >> "
                b"/Screen << /BM /Screen >> >> /XObject << /Im 7 0 R >> "
                b"/Shading << /Red << /ShadingType 2 /ColorSpace /DeviceRGB "
                b"/Coords [0 0 8 0] /Function << /FunctionType 2 /Domain [0 1] "
                b"/C0 [1 0 0] /C1 [1 0 0] /N 1 >> /Extend [true true] >> >> >>",
            ),
            stream_obj(
                bytes((255, 0, 0)),
                b"/Type /XObject /Subtype /Image /Width 1 /Height 1 "
                b"/ColorSpace /DeviceRGB /BitsPerComponent 8"
                + (b" /SMask 8 0 R" if masked else b""),
            ),
            stream_obj(
                b"\x80",
                b"/Type /XObject /Subtype /Image /Width 1 /Height 1 "
                b"/ColorSpace /DeviceGray /BitsPerComponent 8",
            ),
        ],
    )


PAINT_OPERATIONS = (
    b"1 0 0 rg 0 0 8 8 re f",
    b"1 0 0 rg 0 0 m 8 0 l 4 8 l h f",
    b"1 0 0 RG 4 w 0 4 m 8 4 l S",
    b"8 0 0 8 0 0 cm /Im Do",
    b"8 0 0 8 0 0 cm BI /W 1 /H 1 /BPC 8 /CS /RGB ID \xff\x00\x00 EI",
    b"1 0 0 rg 8 0 0 8 0 0 cm BI /W 1 /H 1 /IM true ID \x00 EI",
    b"/Red sh",
)


@pytest.mark.parametrize("paint", PAINT_OPERATIONS)
@pytest.mark.parametrize("inner_state", [b"", b"/Opaque gs ", b"/Half gs "])
def test_isolated_group_applies_its_opacity_once_to_every_paint_path(
    paint: bytes, inner_state: bytes
) -> None:
    # Poppler's center sample is (255,127,127), or (255,191,191) when
    # the group explicitly paints at its own half opacity as well.
    data = internal_group_pdf(inner_state + paint)
    with open_pdf(data) as document:
        pixels = document.pages[0].render().rasterize(background=(255, 255, 255, 255)).array()
    shade = 191 if inner_state == b"/Half gs " else 127
    assert tuple(pixels[4, 4]) == (255, shade, shade, 255)


@pytest.mark.parametrize(("inner_state", "shade"), [(b"", 191), (b"/Half gs ", 223)])
def test_full_soft_mask_is_not_multiplied_by_its_captured_average(
    inner_state: bytes, shade: int
) -> None:
    data = internal_group_pdf(inner_state + b"8 0 0 8 0 0 cm /Im Do", masked=True)
    with open_pdf(data) as document:
        pixels = document.pages[0].render().rasterize(background=(255, 255, 255, 255)).array()
    assert tuple(pixels[4, 4]) == (255, shade, shade, 255)


@pytest.mark.parametrize("mode", [b"Multiply", b"Screen"])
def test_blending_against_a_transparent_isolated_backdrop_preserves_source_color(
    mode: bytes,
) -> None:
    data = internal_group_pdf(b"/" + mode + b" gs 1 0 0 rg 0 0 8 8 re f", outer_alpha=1)
    with open_pdf(data) as document:
        pixels = document.pages[0].render().rasterize().array()
    numpy.testing.assert_array_equal(pixels, numpy.broadcast_to((255, 0, 0, 255), pixels.shape))


@pytest.mark.parametrize(
    ("mode", "expected"), [(b"Multiply", (127, 0, 0, 255)), (b"Screen", (255, 0, 128, 255))]
)
def test_blending_weights_a_partially_transparent_group_backdrop(
    mode: bytes, expected: tuple[int, int, int, int]
) -> None:
    # These interior RGBA colors agree with Poppler's RGB reference raster.
    data = internal_group_pdf(
        b"/Half gs 0 0 1 rg 0 0 8 8 re f /Opaque gs /" + mode + b" gs 1 0 0 rg 0 0 8 8 re f",
        outer_alpha=1,
    )
    with open_pdf(data) as document:
        pixels = document.pages[0].render().rasterize().array()
    assert tuple(pixels[4, 4]) == expected


def test_group_opacity_is_applied_after_overlapping_contents_are_painted() -> None:
    data = internal_group_pdf(b"1 0 0 rg 0 0 6 8 re f 2 0 6 8 re f")
    with open_pdf(data) as document:
        pixels = document.pages[0].render().rasterize().array()
    assert numpy.all(pixels[:, :, 3] == 128)


def test_nested_groups_apply_each_group_opacity_at_its_own_boundary() -> None:
    data = one_page_pdf(
        b"/Half gs /Outer Do",
        media_box=(0, 0, 8, 8),
        resources=b"<< /ExtGState << /Half << /ca .5 >> >> /XObject << /Outer 6 0 R >> >>",
        extra_objects=[
            stream_obj(
                b"/Half gs /Inner Do",
                b"/Type /XObject /Subtype /Form /BBox [0 0 8 8] "
                b"/Group << /S /Transparency /I true >> "
                b"/Resources << /ExtGState << /Half << /ca .5 >> >> "
                b"/XObject << /Inner 7 0 R >> >>",
            ),
            stream_obj(
                b"1 0 0 rg 0 0 8 8 re f",
                b"/Type /XObject /Subtype /Form /BBox [0 0 8 8] "
                b"/Group << /S /Transparency /I true >> /Resources << >>",
            ),
        ],
    )
    with open_pdf(data) as document:
        pixels = document.pages[0].render().rasterize(background=(255, 255, 255, 255)).array()
    assert tuple(pixels[4, 4]) == (255, 191, 191, 255)
