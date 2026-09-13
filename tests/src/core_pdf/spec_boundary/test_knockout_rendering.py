# SPDX-License-Identifier: AGPL-3.0-only
"""RGB Form knockout composition, with independent overlap and backdrop controls."""

from typing import Any

import numpy
import pytest

from core_pdf import PdfDocument


def internal_stream(content: bytes, dictionary: bytes = b"") -> bytes:
    return (
        b"<< "
        + dictionary
        + f" /Length {len(content)} >>\nstream\n".encode()
        + content
        + b"\nendstream"
    )


def internal_pdf(
    content: bytes,
    *,
    isolated: bool = False,
    knockout: bool = True,
    opacity: float = 1.0,
    alpha_is_shape: bool = False,
    resources: bytes = b"",
    extras: tuple[bytes, ...] = (),
    background: bool = True,
) -> bytes:
    group = (
        f"/Group << /S /Transparency /CS /DeviceRGB /I {str(isolated).lower()} "
        f"/K {str(knockout).lower()} >>"
    ).encode()
    state = (
        f"/ExtGState << /Half << /ca 0.5 /CA 0.5 >> /Zero << /ca 0 /CA 0 >> "
        f"/Shape << /ca 0.5 /CA 0.5 /AIS true >> /Opaque << /ca 1 /CA 1 >> "
        f"/Multiply << /BM /Multiply /ca 0.5 >> "
        f"/Outer << /ca {opacity} /AIS {str(alpha_is_shape).lower()} >> >>"
    ).encode()
    page_content = (
        b"0.2 0.6 0.8 rg 0 0 24 20 re f " if background else b""
    ) + b"q /Outer gs /Fm Do Q"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 24 20] "
        b"/Resources << /XObject << /Fm 5 0 R >> " + state + b" >> /Contents 4 0 R >>",
        internal_stream(page_content),
        internal_stream(
            content,
            b"/Type /XObject /Subtype /Form /BBox [0 0 24 20] "
            + group
            + b" /Resources << "
            + state
            + b" "
            + resources
            + b" >>",
        ),
        *extras,
    ]
    data = b"%PDF-1.7\n"
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(data)
    data += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    data += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets[1:])
    return (
        data
        + (f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").encode()
    )


def internal_raster(content: bytes, **kwargs: Any) -> numpy.ndarray:
    with PdfDocument(internal_pdf(content, **kwargs)) as document:
        return document.pages[0].render().rasterize().array().copy()


internal_RED = b"1 0 0 rg 2 2 12 16 re f "
internal_BLUE = b"0 0 1 rg 8 4 12 12 re f "


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("opacity", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("second_state", [b"/Half gs ", b"/Zero gs ", b"/Multiply gs "])
def test_top_object_replaces_prior_paint_at_its_own_opacity(
    isolated: bool, opacity: float, second_state: bytes
) -> None:
    # ISO 32000-2 11.4.6: a full source shape replaces earlier contributions,
    # while the element itself blends against the group's initial backdrop.
    kwargs = {"isolated": isolated, "opacity": opacity}
    actual = internal_raster(internal_RED + second_state + internal_BLUE, **kwargs)
    top_only = internal_raster(second_state + internal_BLUE, **kwargs)
    first_only = internal_raster(internal_RED, **kwargs)
    numpy.testing.assert_allclose(actual[8, 10], top_only[8, 10], atol=1, rtol=0)
    numpy.testing.assert_array_equal(actual[8, 4], first_only[8, 4])
    numpy.testing.assert_array_equal(actual[0, 23], [51, 153, 204, 255])


@pytest.mark.parametrize("isolated", [False, True])
def test_nonknockout_positive_control_composites_over_previous_elements(isolated: bool) -> None:
    actual = internal_raster(
        internal_RED + b"/Half gs " + internal_BLUE, isolated=isolated, knockout=False
    )
    numpy.testing.assert_allclose(actual[8, 10], [127, 0, 128, 255], atol=1, rtol=0)


@pytest.mark.parametrize("isolated", [False, True])
def test_alpha_is_shape_retains_previous_contribution(isolated: bool) -> None:
    # AIS true turns ca into fractional shape; default AIS false is opacity.
    actual = internal_raster(internal_RED + b"/Shape gs " + internal_BLUE, isolated=isolated)
    numpy.testing.assert_allclose(actual[8, 10], [127, 0, 128, 255], atol=1, rtol=0)
    opacity_only = internal_raster(internal_RED + b"/Half gs " + internal_BLUE, isolated=isolated)
    assert not numpy.array_equal(actual[8, 10], opacity_only[8, 10])


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("child_isolated", [False, True])
@pytest.mark.parametrize("child_knockout", [False, True])
@pytest.mark.parametrize("child_opacity", [0.0, 0.5, 1.0])
def test_nested_group_is_one_element_with_its_own_shape(
    isolated: bool, child_isolated: bool, child_knockout: bool, child_opacity: float
) -> None:
    child = internal_stream(
        b"/Paint gs 0 0 1 rg 8 4 12 12 re f",
        (
            f"/Subtype /Form /BBox [0 0 24 20] /Group << /S /Transparency "
            f"/I {str(child_isolated).lower()} /K {str(child_knockout).lower()} >> "
            f"/Resources << /ExtGState << /Paint << /BM /Multiply /ca {child_opacity} >> >> >>"
        ).encode(),
    )
    kwargs = {"isolated": isolated, "resources": b"/XObject << /Child 6 0 R >>", "extras": (child,)}
    actual = internal_raster(internal_RED + b"/Child Do", **kwargs)
    top_only = internal_raster(b"/Child Do", **kwargs)
    numpy.testing.assert_allclose(actual[8, 10], top_only[8, 10], atol=1, rtol=0)
    numpy.testing.assert_allclose(actual[8, 4], [255, 0, 0, 255], atol=1, rtol=0)


@pytest.mark.parametrize("alpha_is_shape", [False, True])
def test_outer_group_alpha_is_shape_controls_knockout_in_parent(alpha_is_shape: bool) -> None:
    child = internal_stream(
        internal_BLUE,
        b"/Subtype /Form /BBox [0 0 24 20] /Group << /S /Transparency /I true >> /Resources << >>",
    )
    state = b"/Shape gs " if alpha_is_shape else b"/Half gs "
    actual = internal_raster(
        internal_RED + state + b"/Child Do",
        resources=b"/XObject << /Child 6 0 R >>",
        extras=(child,),
    )
    expected = [127, 0, 128, 255] if alpha_is_shape else [25, 76, 230, 255]
    numpy.testing.assert_allclose(actual[8, 10], expected, atol=1, rtol=0)


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("paint", ["stroke", "shading", "text", "image", "stencil"])
def test_transparent_element_knocks_out_only_its_painted_shape(isolated: bool, paint: str) -> None:
    resources = b""
    extras: tuple[bytes, ...] = ()
    if paint == "stroke":
        top = b"/Zero gs 0 0 1 RG 4 w 4 6 m 18 14 l S"
    elif paint == "shading":
        top = b"/Zero gs /Sh sh"
        resources = (
            b"/Shading << /Sh << /ShadingType 2 /ColorSpace /DeviceRGB /Coords [8 0 18 0] "
            b"/Function << /FunctionType 2 /Domain [0 1] /C0 [0 0 1] /C1 [0 0 1] /N 1 >> >> >>"
        )
    elif paint == "text":
        top = b"/Zero gs BT /F1 16 Tf 5 4 Td (X) Tj ET"
        resources = b"/Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >>"
    else:
        top = b"/Zero gs q 12 0 0 12 6 4 cm /Im Do Q"
        dictionary = b"/Subtype /Image /Width 2 /Height 1 "
        if paint == "stencil":
            dictionary += b"/ImageMask true /BitsPerComponent 1"
            samples = b"\x40"
        else:
            dictionary += b"/ColorSpace /DeviceRGB /BitsPerComponent 8"
            samples = bytes((0, 0, 255, 0, 0, 255))
        extras = (internal_stream(samples, dictionary),)
        resources = b"/XObject << /Im 6 0 R >>"
    kwargs = {"isolated": isolated, "resources": resources, "extras": extras}
    actual = internal_raster(internal_RED + top, **kwargs)
    # An opaque black copy reveals exactly the same raster geometry. This also
    # tests glyph holes, stencil holes and antialiased stroke edges.
    coverage = (
        internal_raster(top.replace(b"/Zero gs", b"/Opaque gs"), background=False, **kwargs)[..., 3]
        / 255.0
    )
    first = internal_raster(internal_RED, **kwargs).astype(float)
    backdrop = numpy.broadcast_to([51, 153, 204, 255], first.shape)
    expected = first * (1 - coverage[..., None]) + backdrop * coverage[..., None]
    assert numpy.any(coverage > 0)
    assert numpy.any(coverage == 0)
    numpy.testing.assert_allclose(actual, expected, atol=2, rtol=0)


@pytest.mark.parametrize("isolated", [False, True])
def test_fractional_rectangle_edges_replace_by_coverage_once(isolated: bool) -> None:
    top = b"0 0 1 rg 8.25 4.25 10.5 11.5 re f"
    coverage = internal_raster(top, isolated=True, background=False)[..., 3] / 255.0
    first = internal_raster(internal_RED, isolated=isolated).astype(float)
    actual = internal_raster(internal_RED + b"/Half gs " + top, isolated=isolated)
    background = numpy.array([51, 153, 204, 255])
    source = numpy.array([0, 0, 255, 255])
    alpha = 128 / 255
    expected = first * (1 - coverage[..., None]) + coverage[..., None] * (
        source * alpha + background * (1 - alpha)
    )
    assert numpy.any((coverage > 0) & (coverage < 1))
    numpy.testing.assert_allclose(actual, expected, atol=2, rtol=0)


@pytest.mark.parametrize("alpha_is_shape", [False, True])
def test_image_soft_mask_is_opacity_unless_ais_selects_shape(alpha_is_shape: bool) -> None:
    image = internal_stream(
        bytes((0, 0, 255, 0, 0, 255)),
        b"/Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceRGB "
        b"/BitsPerComponent 8 /SMask 7 0 R",
    )
    mask = internal_stream(
        bytes((0, 128)),
        b"/Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8",
    )
    state = b"/Shape gs /Opaque gs " if alpha_is_shape else b""
    actual = internal_raster(
        b"1 0 0 rg 0 0 24 20 re f " + state + b"q 12 0 0 12 6 4 cm /Im Do Q",
        resources=b"/XObject << /Im 6 0 R >>",
        extras=(image, mask),
    )
    numpy.testing.assert_array_equal(
        actual[10, 8], [255, 0, 0, 255] if alpha_is_shape else [51, 153, 204, 255]
    )
    expected = [127, 0, 128, 255] if alpha_is_shape else [25, 76, 230, 255]
    numpy.testing.assert_allclose(actual[10, 16], expected, atol=1, rtol=0)


def test_color_key_mask_constrains_shape_even_at_zero_opacity() -> None:
    image = internal_stream(
        bytes((0, 0, 255, 255, 0, 0)),
        b"/Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceRGB "
        b"/BitsPerComponent 8 /Mask [0 0 0 0 255 255]",
    )
    actual = internal_raster(
        b"1 0 0 rg 0 0 24 20 re f /Zero gs q 12 0 0 12 6 4 cm /Im Do Q",
        resources=b"/XObject << /Im 6 0 R >>",
        extras=(image,),
    )
    numpy.testing.assert_array_equal(actual[10, 8], [255, 0, 0, 255])
    numpy.testing.assert_array_equal(actual[10, 16], [51, 153, 204, 255])


@pytest.mark.parametrize("isolated", [False, True])
def test_knockout_shape_respects_clip_and_restores_parent_paint(isolated: bool) -> None:
    actual = internal_raster(
        internal_RED + b"q 8 4 3 12 re W n /Zero gs " + internal_BLUE + b"Q 0 1 0 rg 18 2 4 4 re f",
        isolated=isolated,
    )
    numpy.testing.assert_array_equal(actual[8, 9], [51, 153, 204, 255])
    numpy.testing.assert_array_equal(actual[8, 12], [255, 0, 0, 255])
    numpy.testing.assert_array_equal(actual[16, 20], [0, 255, 0, 255])


@pytest.mark.parametrize("alpha_is_shape", [False, True])
@pytest.mark.parametrize("inner_opacity", [0.0, 0.5, 1.0])
def test_pattern_shape_and_outer_constant_apply_once(
    alpha_is_shape: bool, inner_opacity: float
) -> None:
    pattern = internal_stream(
        b"/Paint gs " + internal_BLUE,
        b"/PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 24 20] /XStep 24 /YStep 20 "
        + (f"/Resources << /ExtGState << /Paint << /ca {inner_opacity} >> >> >>").encode(),
    )
    state = b"/Shape gs " if alpha_is_shape else b"/Half gs "
    actual = internal_raster(
        internal_RED + state + b"/Pattern cs /Pt scn 0 0 24 20 re f",
        resources=b"/Pattern << /Pt 6 0 R >>",
        extras=(pattern,),
    )
    shape = 0.5 if alpha_is_shape else 1.0
    alpha = round(round(inner_opacity * 255) * 0.5) / 255.0
    expected = (
        numpy.array([255, 0, 0]) * (1 - shape)
        + numpy.array([0, 0, 255]) * alpha
        + numpy.array([51, 153, 204]) * (shape - alpha)
    )
    numpy.testing.assert_allclose(actual[8, 10, :3], expected, atol=1, rtol=0)
    numpy.testing.assert_array_equal(actual[8, 4], [255, 0, 0, 255])


@pytest.mark.parametrize("isolated", [False, True])
def test_combined_fill_stroke_is_one_element_without_doubled_border(isolated: bool) -> None:
    actual = internal_raster(
        internal_RED + b"/Half gs 0 0 1 rg 0 1 0 RG 4 w 6 6 12 8 re B",
        isolated=isolated,
    )
    # ISO 32000-2 11.7.4.4: stroke replaces the fill where the two overlap.
    numpy.testing.assert_allclose(actual[8, 7], [25, 204, 102, 255], atol=1, rtol=0)
    numpy.testing.assert_allclose(actual[8, 11], [25, 76, 230, 255], atol=1, rtol=0)


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("separate", [False, True])
def test_pattern_preserves_separate_overlapping_stroke_objects(
    isolated: bool, separate: bool
) -> None:
    # The pattern's implicit group is non-knockout even inside a /K Form.
    # Two half-opacity strokes accumulate; two subpaths of one stroke do not.
    crossing = b"3 3 m 21 17 l " + (b"S " if separate else b"") + b"3 17 m 21 3 l S"
    pattern = internal_stream(
        b"/Paint gs 0 0 1 RG 6 w " + crossing,
        b"/PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 24 20] /XStep 24 /YStep 20 "
        b"/Resources << /ExtGState << /Paint << /CA 0.5 >> >> >>",
    )
    actual = internal_raster(
        b"/Pattern cs /Pt scn 0 0 24 20 re f",
        isolated=isolated,
        resources=b"/Pattern << /Pt 6 0 R >>",
        extras=(pattern,),
    )
    alpha = 192.0 / 255.0 if separate else 128.0 / 255.0
    expected = numpy.rint(
        numpy.array([51, 153, 204]) * (1 - alpha) + numpy.array([0, 0, 255]) * alpha
    )
    numpy.testing.assert_allclose(actual[10, 12, :3], expected, atol=1, rtol=0)
