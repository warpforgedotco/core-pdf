# SPDX-License-Identifier: AGPL-3.0-only
"""Form bounds are scoped raster clips, including their transformed quadrilateral."""

import numpy
import pytest

from core_pdf import PdfDocument

internal_GROUPS = [
    b"",
    b"/Group << /S /Transparency /I true >>",
    b"/Group << /S /Transparency /I false >>",
]
internal_PAINT = {
    "fill": b"1 0 0 rg -100 -100 200 200 re f",
    "stroke": b"1 0 0 RG 16 w -100 16 m 100 16 l S",
    "shading": b"/Sh sh",
    "image": b"q 48 0 0 48 0 0 cm /Im Do Q",
    "text": b"1 0 0 rg BT /F 28 Tf 1 0 0 1 2 10 Tm (HH) Tj ET",
}


def internal_stream(content: bytes, dictionary: bytes = b"") -> bytes:
    return (
        b"<< "
        + dictionary
        + f" /Length {len(content)} >>\nstream\n".encode()
        + content
        + b"\nendstream"
    )


def internal_document(
    content: bytes,
    *,
    bbox: bytes = b"8 8 24 24",
    matrix: bytes = b"1 0 0 1 0 0",
    group: bytes = b"",
    prefix: bytes = b"",
    suffix: bytes = b"",
    child_content: bytes = b"",
    child_bbox: bytes = b"0 0 48 48",
    child_matrix: bytes = b"1 0 0 1 0 0",
    child_group: bytes = b"",
) -> bytes:
    resources = (
        b"/Font << /F << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> "
        b"/XObject << /Im 6 0 R /Child 7 0 R >> "
        b"/Shading << /Sh << /ShadingType 2 /ColorSpace /DeviceRGB "
        b"/Coords [0 0 48 0] /Extend [true true] /Function << /FunctionType 2 "
        b"/Domain [0 1] /C0 [1 0 0] /C1 [1 0 0] /N 1 >> >> >>"
    )

    def form(paint: bytes, bounds: bytes, transform: bytes, transparency: bytes) -> bytes:
        return internal_stream(
            paint,
            b"/Type /XObject /Subtype /Form /BBox ["
            + bounds
            + b"] /Matrix ["
            + transform
            + b"] "
            + transparency
            + b" /Resources << "
            + resources
            + b" >>",
        )

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 48 48] "
        b"/Resources << /XObject << /Fm 5 0 R >> >> /Contents 4 0 R >>",
        internal_stream(b"q 1 1 1 rg 0 0 48 48 re f Q " + prefix + b" /Fm Do " + suffix),
        form(content, bbox, matrix, group),
        internal_stream(
            bytes([255, 0, 0, 0, 0, 255]),
            b"/Type /XObject /Subtype /Image /Width 2 /Height 1 "
            b"/ColorSpace /DeviceRGB /BitsPerComponent 8",
        ),
        form(child_content, child_bbox, child_matrix, child_group),
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
        + f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )


def internal_raster(
    data: bytes, *, crop: tuple[float, float, float, float] | None = None
) -> numpy.ndarray:
    with PdfDocument(data) as document:
        return document.pages[0].render().rasterize(crop=crop).array().copy()


def internal_assert_pixel(pixels: numpy.ndarray, x: int, y: int, rgb: list[int]) -> None:
    numpy.testing.assert_array_equal(pixels[47 - y, x], [*rgb, 255])


@pytest.mark.parametrize("group", internal_GROUPS, ids=["ordinary", "isolated", "nonisolated"])
@pytest.mark.parametrize("paint", internal_PAINT)
def test_form_bbox_matches_an_explicit_clip_for_each_paint_kind(group: bytes, paint: str) -> None:
    # ISO 32000-2 8.10.1–2: Form execution intersects the current clip with BBox.
    content = internal_PAINT[paint]
    actual = internal_raster(internal_document(content, group=group))
    expected = internal_raster(
        internal_document(
            b"q 8 8 16 16 re W n " + content + b" Q",
            bbox=b"0 0 48 48",
            group=group,
        )
    )
    assert numpy.any(expected[:, :, 1] < 255), "The positive control must paint inside the clip."
    numpy.testing.assert_array_equal(actual, expected)
    internal_assert_pixel(actual, 30, 16, [255, 255, 255])


@pytest.mark.parametrize("group", internal_GROUPS, ids=["ordinary", "isolated", "nonisolated"])
@pytest.mark.parametrize("paint", ["fill", "shading", "image"])
@pytest.mark.parametrize(
    ("matrix", "inside", "outside"),
    [
        (b"1 0.5 0.5 1 8 8", (17, 17), (9, 24)),
        (b"1 1 -1 1 24 4", (24, 16), (13, 5)),
    ],
    ids=["sheared", "rotated"],
)
def test_form_bbox_clips_the_transformed_quad_not_its_axis_aligned_envelope(
    group: bytes,
    paint: str,
    matrix: bytes,
    inside: tuple[int, int],
    outside: tuple[int, int],
) -> None:
    content = internal_PAINT[paint]
    actual = internal_raster(
        internal_document(content, bbox=b"0 0 12 12", matrix=matrix, group=group)
    )
    internal_assert_pixel(actual, *inside, [255, 0, 0])
    # This sample is inside the transformed bounding envelope but outside the quad.
    internal_assert_pixel(actual, *outside, [255, 255, 255])
    expected = internal_raster(
        internal_document(
            b"q 0 0 12 12 re W n " + content + b" Q",
            bbox=b"-100 -100 100 100",
            matrix=matrix,
            group=group,
        )
    )
    numpy.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("group", internal_GROUPS, ids=["ordinary", "isolated", "nonisolated"])
def test_form_bbox_uses_invocation_ctm_as_well_as_its_own_matrix(group: bytes) -> None:
    pixels = internal_raster(
        internal_document(
            internal_PAINT["fill"],
            bbox=b"0 0 8 8",
            matrix=b"1 0 0 1 2 3",
            prefix=b"q 2 0 0 2 4 6 cm",
            suffix=b"Q",
            group=group,
        )
    )
    internal_assert_pixel(pixels, 10, 14, [255, 0, 0])
    internal_assert_pixel(pixels, 6, 14, [255, 255, 255])
    internal_assert_pixel(pixels, 10, 30, [255, 255, 255])


def test_finite_extreme_form_bbox_does_not_overflow_when_constructing_its_clip() -> None:
    # PDF numbers use decimal syntax; both endpoints are finite even though
    # subtracting the minimum from the maximum overflows a binary64 width.
    endpoint = b"1" + b"0" * 308
    bounds = b"-" + endpoint + b" -" + endpoint + b" " + endpoint + b" " + endpoint
    pixels = internal_raster(
        internal_document(
            b"1 0 0 rg 8 8 8 8 re f",
            bbox=bounds,
            suffix=b"0 0 1 rg 32 32 8 8 re f",
        )
    )
    internal_assert_pixel(pixels, 12, 12, [255, 0, 0])
    internal_assert_pixel(pixels, 36, 36, [0, 0, 255])
    internal_assert_pixel(pixels, 24, 24, [255, 255, 255])


def test_singular_form_matrix_collapses_its_clip_without_leaking_to_parent_paint() -> None:
    pixels = internal_raster(
        internal_document(
            internal_PAINT["shading"],
            matrix=b"1 1 1 1 0 0",
            suffix=b"0 0 1 rg 32 32 8 8 re f",
        )
    )
    expected = numpy.full((48, 48, 4), 255, dtype=numpy.uint8)
    expected[8:16, 32:40] = [0, 0, 255, 255]
    numpy.testing.assert_array_equal(pixels, expected)


@pytest.mark.parametrize("group", internal_GROUPS, ids=["ordinary", "isolated", "nonisolated"])
@pytest.mark.parametrize("bbox", [b"12 8 12 24", b"8 12 24 12", b"12 12 12 12"])
def test_zero_area_form_bbox_suppresses_paint_without_leaking_its_empty_clip(
    group: bytes, bbox: bytes
) -> None:
    pixels = internal_raster(
        internal_document(
            internal_PAINT["fill"],
            bbox=bbox,
            group=group,
            suffix=b"0 0 1 rg 36 36 8 8 re f",
        )
    )
    expected = numpy.full((48, 48, 4), 255, dtype=numpy.uint8)
    expected[4:12, 36:44] = [0, 0, 255, 255]
    numpy.testing.assert_array_equal(pixels, expected)


@pytest.mark.parametrize("group", internal_GROUPS, ids=["ordinary", "isolated", "nonisolated"])
def test_nested_forms_intersect_both_bounds_and_restore_the_parent_form_clip(group: bytes) -> None:
    pixels = internal_raster(
        internal_document(
            b"/Child Do 0 1 0 rg 8 20 32 8 re f",
            bbox=b"8 8 24 28",
            group=group,
            child_content=internal_PAINT["fill"],
            child_bbox=b"0 0 16 16",
            child_matrix=b"1 0 0 1 16 0",
            child_group=group,
        )
    )
    internal_assert_pixel(pixels, 20, 12, [255, 0, 0])
    internal_assert_pixel(pixels, 28, 12, [255, 255, 255])
    internal_assert_pixel(pixels, 20, 4, [255, 255, 255])
    internal_assert_pixel(pixels, 12, 24, [0, 255, 0])
    internal_assert_pixel(pixels, 28, 24, [255, 255, 255])


@pytest.mark.parametrize("malformed", [False, True], ids=["balanced", "unbalanced-q-Q"])
@pytest.mark.parametrize("group", internal_GROUPS, ids=["ordinary", "isolated", "nonisolated"])
def test_form_clip_restores_the_parent_clip_after_local_graphics_saves(
    group: bytes, malformed: bool
) -> None:
    content = b"q 8 8 8 8 re W n " + internal_PAINT["fill"]
    content = b"Q Q " + content + b" q" if malformed else content + b" Q"
    pixels = internal_raster(
        internal_document(
            content,
            group=group,
            prefix=b"q 0 0 24 48 re W n",
            suffix=b"0 1 0 rg 0 32 48 8 re f Q 0 0 1 rg 36 4 8 8 re f",
        )
    )
    internal_assert_pixel(pixels, 12, 12, [255, 0, 0])
    internal_assert_pixel(pixels, 4, 12, [255, 255, 255])
    internal_assert_pixel(pixels, 4, 36, [0, 255, 0])
    internal_assert_pixel(pixels, 28, 36, [255, 255, 255])
    internal_assert_pixel(pixels, 40, 8, [0, 0, 255])


@pytest.mark.parametrize("paint", ["fill", "shading", "image"])
def test_render_crop_keeps_form_clip_in_original_page_coordinates(paint: str) -> None:
    data = internal_document(internal_PAINT[paint])
    full = internal_raster(data)
    cropped = internal_raster(data, crop=(4, 4, 32, 32))
    numpy.testing.assert_array_equal(cropped, full[16:44, 4:32])
    numpy.testing.assert_array_equal(cropped[16, 24], [255, 255, 255, 255])
    numpy.testing.assert_array_equal(cropped[16, 12], [255, 0, 0, 255])


@pytest.mark.parametrize("group", internal_GROUPS, ids=["ordinary", "isolated", "nonisolated"])
def test_form_clip_does_not_stretch_the_remaining_image_samples(group: bytes) -> None:
    pixels = internal_raster(
        internal_document(internal_PAINT["image"], bbox=b"16 8 40 24", group=group)
    )
    # The original image placement puts its red/blue division at page x=24.
    # Restricting the destination to [16,40] must not move that division to x=28.
    internal_assert_pixel(pixels, 20, 16, [255, 0, 0])
    internal_assert_pixel(pixels, 26, 16, [0, 0, 255])
    internal_assert_pixel(pixels, 36, 16, [0, 0, 255])
    internal_assert_pixel(pixels, 12, 16, [255, 255, 255])
    internal_assert_pixel(pixels, 44, 16, [255, 255, 255])


def test_form_bbox_raster_clip_preserves_captured_text_and_drawing_geometry() -> None:
    content = (
        b"1 0 0 rg 0 0 48 48 re f "
        b"0 g BT /F 8 Tf 1 0 0 1 10 12 Tm (Hi) Tj "
        b"1 0 0 1 2 30 Tm (Outside) Tj ET"
    )
    with PdfDocument(internal_document(content)) as document:
        page = document.pages[0]
        # Existing capture filters text wholly outside the Form envelope.
        # Adding a raster clip preserves that policy and the drawing's geometry.
        assert "".join(char.text for char in page.chars) == "Hi"
        assert page.extract().text == "Hi"
        assert page.get_drawings()[-1].rect == (0, 0, 48, 48)
        pixels = page.render().rasterize().array()
        numpy.testing.assert_array_equal(pixels[:24], 255)
        internal_assert_pixel(pixels, 20, 20, [255, 0, 0])
