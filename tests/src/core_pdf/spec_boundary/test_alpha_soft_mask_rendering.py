# SPDX-License-Identifier: AGPL-3.0-only
"""Graphics-state Alpha masks retain spatial coverage and their establishment state."""

import imagecodecs
import numpy
import pytest

from core_pdf import PdfDocument

internal_WIDTH = 32
internal_HEIGHT = 24
internal_FULL_RED = b"1 0 0 rg 0 0 32 24 re f"
internal_BANDS = b"0 g 0 0 12 24 re f /Half gs 12 0 12 24 re f"
internal_INVERT = b"/TR << /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [0] /N 1 >>"
internal_PAINT = {
    "fill": internal_FULL_RED,
    "stroke": b"1 0 0 RG 12 w 0 12 m 32 12 l S",
    "shading": b"/Sh sh",
    "pattern": b"/Pattern cs /P scn 0 0 32 24 re f",
    "image": b"q 32 0 0 24 0 0 cm /Im Do Q",
    "inline-image": (b"q 32 0 0 24 0 0 cm BI /W 1 /H 1 /CS /RGB /BPC 8 ID \xff\x00\x00 EI Q"),
}


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
    mask_content: bytes = internal_BANDS,
    mask_bbox: bytes = b"0 0 32 24",
    mask_matrix: bytes = b"1 0 0 1 0 0",
    mask_group: bytes = b"/I true",
    mask_has_resources: bool = True,
    transfer: bytes = b"",
    background: bytes = b"1 1 1",
    form_content: bytes = b"",
    form_group: bytes = b"",
    form_uses_right_piece: bool = False,
    other_mask: bytes = b"0 0 32 12 re f",
    pattern_content: bytes = b"1 0 0 rg 0 0 8 8 re f",
    pattern_has_resources: bool = False,
    image_dictionary: bytes = b"",
    image_data: bytes = b"\xff\x00\x00\xff\x00\x00",
    native_mask_dictionary: bytes = b"/ColorSpace /DeviceGray /BitsPerComponent 8",
    native_mask_data: bytes = b"\x00\xff",
) -> bytes:
    states = (
        b"/ExtGState << /SM << /SMask << /S /Alpha /G 5 0 R "
        + transfer
        + b" >> >> /OtherMask << /SMask << /S /Alpha /G 11 0 R >> >> "
        b"/None << /SMask /None >> /Half << /ca 0.5 /CA 0.5 >> "
        b"/Opaque << /ca 1 /CA 1 >> /Shape << /AIS true >> "
        b"/Multiply << /BM /Multiply >> /TKOff << /TK false >> >>"
    )
    resources = (
        states + b" /Font << /F 7 0 R /H << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> "
        b"/XObject << /Fm 6 0 R /Im 9 0 R /Piece 13 0 R >> /Pattern << /P 12 0 R >> "
        b"/Shading << /Sh << /ShadingType 2 /ColorSpace /DeviceRGB "
        b"/Coords [0 0 32 0] /Extend [true true] /Function << /FunctionType 2 "
        b"/Domain [0 1] /C0 [1 0 0] /C1 [1 0 0] /N 1 >> >> >>"
    )

    def form(
        paint: bytes,
        bbox: bytes,
        matrix: bytes,
        group: bytes,
        *,
        has_resources: bool = True,
        right_piece: bool = False,
    ) -> bytes:
        form_resources = (
            resources.replace(b"/Piece 13 0 R", b"/Piece 14 0 R") if right_piece else resources
        )
        return internal_stream(
            paint,
            b"/Type /XObject /Subtype /Form /BBox ["
            + bbox
            + b"] /Matrix ["
            + matrix
            + b"] "
            + group
            + (b" /Resources << " + form_resources + b" >>" if has_resources else b""),
        )

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 32 24] /Resources << "
        + resources
        + b" >> /Contents 4 0 R >>",
        internal_stream(b"q " + background + b" rg 0 0 32 24 re f Q " + content),
        form(
            mask_content,
            mask_bbox,
            mask_matrix,
            b"/Group << /S /Transparency " + mask_group + b" >>",
            has_resources=mask_has_resources,
        ),
        form(
            form_content,
            b"0 0 32 24",
            b"1 0 0 1 0 0",
            form_group,
            right_piece=form_uses_right_piece,
        ),
        b"<< /Type /Font /Subtype /Type3 /FontBBox [0 0 1000 1000] "
        b"/FontMatrix [0.001 0 0 0.001 0 0] /FirstChar 65 /LastChar 65 /Widths [1000] "
        b"/Encoding << /Type /Encoding /Differences [65 /A] >> /CharProcs << /A 8 0 R >> "
        b"/Resources << " + states + b" >> >>",
        internal_stream(b"1000 0 d0 1 0 0 rg 0 0 1000 1000 re f"),
        internal_stream(
            image_data,
            b"/Type /XObject /Subtype /Image /Width 2 /Height 1 "
            b"/ColorSpace /DeviceRGB /BitsPerComponent 8 " + image_dictionary,
        ),
        internal_stream(
            native_mask_data,
            b"/Type /XObject /Subtype /Image /Width 2 /Height 1 " + native_mask_dictionary,
        ),
        form(
            other_mask,
            b"0 0 32 24",
            b"1 0 0 1 0 0",
            b"/Group << /S /Transparency /I true >>",
        ),
        internal_stream(
            pattern_content,
            b"/Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 "
            b"/BBox [0 0 8 8] /XStep 8 /YStep 8 /Resources << "
            + (resources if pattern_has_resources else b"")
            + b" >>",
        ),
        form(b"0 0 12 24 re f", b"0 0 32 24", b"1 0 0 1 0 0", b""),
        form(b"20 0 12 24 re f", b"0 0 32 24", b"1 0 0 1 0 0", b""),
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
    numpy.testing.assert_allclose(pixels[internal_HEIGHT - 1 - y, x], [*rgb, 255], atol=1, rtol=0)


@pytest.mark.parametrize("paint", internal_PAINT)
def test_alpha_mask_applies_spatial_coverage_to_each_paint_kind(paint: str) -> None:
    # ISO 32000-2 11.5.2 / 11.6.5.1: mask values come from group alpha,
    # not the group's colors or an average of its painted coverage.
    pixels = internal_raster(internal_pdf(b"/SM gs " + internal_PAINT[paint]))
    internal_assert_pixel(pixels, 6, 12, [255, 0, 0])
    internal_assert_pixel(pixels, 18, 12, [255, 127, 127])
    internal_assert_pixel(pixels, 28, 12, [255, 255, 255])


def test_spatial_mask_multiplies_the_painted_objects_constant_opacity_once() -> None:
    pixels = internal_raster(internal_pdf(b"/SM gs /Half gs " + internal_FULL_RED))
    internal_assert_pixel(pixels, 6, 12, [255, 127, 127])
    internal_assert_pixel(pixels, 18, 12, [255, 191, 191])
    internal_assert_pixel(pixels, 28, 12, [255, 255, 255])


@pytest.mark.parametrize("mask_color", [b"0 0 0", b"1 1 1", b"0 0 1"])
def test_alpha_mask_ignores_color_and_nonisolated_backdrop(mask_color: bytes) -> None:
    pixels = internal_raster(
        internal_pdf(
            b"/SM gs " + internal_FULL_RED,
            mask_content=mask_color + b" rg /Half gs 0 0 16 24 re f",
            mask_group=b"/I false",
            background=b"0.2 0.6 0.8",
        )
    )
    internal_assert_pixel(pixels, 8, 12, [153, 76, 102])
    internal_assert_pixel(pixels, 24, 12, [51, 153, 204])


@pytest.mark.parametrize("inverted", [False, True])
def test_empty_mask_uses_transfer_of_zero_inside_and_outside_its_bbox(inverted: bool) -> None:
    pixels = internal_raster(
        internal_pdf(
            b"/SM gs " + internal_FULL_RED,
            mask_content=b"",
            mask_bbox=b"8 8 16 16",
            transfer=internal_INVERT if inverted else b"",
        )
    )
    expected = [255, 0, 0, 255] if inverted else [255, 255, 255, 255]
    numpy.testing.assert_array_equal(pixels, numpy.broadcast_to(expected, pixels.shape))


def test_inverting_transfer_keeps_opaque_mask_values_outside_the_group_bbox() -> None:
    pixels = internal_raster(
        internal_pdf(
            b"/SM gs " + internal_FULL_RED,
            mask_content=b"0 0 32 24 re f",
            mask_bbox=b"8 8 16 16",
            transfer=internal_INVERT,
        )
    )
    internal_assert_pixel(pixels, 12, 12, [255, 255, 255])
    internal_assert_pixel(pixels, 4, 12, [255, 0, 0])
    internal_assert_pixel(pixels, 24, 12, [255, 0, 0])


def test_mask_ctm_is_frozen_when_gs_establishes_it() -> None:
    # The mask covers x=8..16 despite the subsequent translation of the object.
    pixels = internal_raster(
        internal_pdf(
            b"q 1 0 0 1 8 0 cm /SM gs 1 0 0 1 8 0 cm "
            b"1 0 0 rg -16 0 32 24 re f Q 0 0 1 rg 24 4 4 4 re f",
            mask_content=b"0 0 8 24 re f",
            mask_bbox=b"0 0 8 24",
        )
    )
    internal_assert_pixel(pixels, 12, 12, [255, 0, 0])
    internal_assert_pixel(pixels, 4, 12, [255, 255, 255])
    internal_assert_pixel(pixels, 20, 12, [255, 255, 255])
    internal_assert_pixel(pixels, 26, 6, [0, 0, 255])


def test_reestablishing_the_same_mask_resource_retains_each_invocations_ctm() -> None:
    pixels = internal_raster(
        internal_pdf(
            b"/SM gs " + internal_FULL_RED + b" q 1 0 0 1 16 0 cm /SM gs "
            b"0 0 1 rg -16 0 32 24 re f Q",
            mask_content=b"0 0 8 24 re f",
            mask_bbox=b"0 0 8 24",
        )
    )
    internal_assert_pixel(pixels, 4, 12, [255, 0, 0])
    internal_assert_pixel(pixels, 12, 12, [255, 255, 255])
    internal_assert_pixel(pixels, 20, 12, [0, 0, 255])
    internal_assert_pixel(pixels, 28, 12, [255, 255, 255])


@pytest.mark.parametrize(("installed_width", "paint_width"), [(1, 8), (8, 1)])
def test_mask_inherits_line_width_at_painting_after_its_ctm_was_established(
    installed_width: int, paint_width: int
) -> None:
    # The explicit establishment-time exception is the CTM (11.6.5.1).
    # Other Form parameters inherit invocation state (8.10.1 / 11.6.6).
    prefix = f"{installed_width} w /SM gs {paint_width} w ".encode()
    actual = internal_raster(
        internal_pdf(prefix + internal_FULL_RED, mask_content=b"0 12 m 32 12 l S")
    )
    reference = internal_raster(
        internal_pdf(
            prefix + internal_FULL_RED,
            mask_content=f"{paint_width} w 0 12 m 32 12 l S".encode(),
        )
    )
    assert actual[11, 16, 1] < 255, "Both widths must paint the center of the mask's stroke."
    internal_assert_pixel(actual, 16, 15, [255, 0, 0] if paint_width == 8 else [255, 255, 255])
    internal_assert_pixel(actual, 16, 18, [255, 255, 255])
    numpy.testing.assert_array_equal(actual, reference)


def test_reusing_an_installed_mask_observes_changed_line_width() -> None:
    pixels = internal_raster(
        internal_pdf(
            b"8 w /SM gs 1 0 0 rg 0 0 16 24 re f 1 w 0 0 1 rg 16 0 16 24 re f",
            mask_content=b"0 12 m 32 12 l S",
        )
    )
    internal_assert_pixel(pixels, 8, 15, [255, 0, 0])
    internal_assert_pixel(pixels, 24, 15, [255, 255, 255])
    assert pixels[11, 24, 0] < 255, "The second invocation must retain its thin blue stroke."


def test_mask_without_resources_keeps_installation_scope_when_used_inside_another_form() -> None:
    data = internal_pdf(
        b"/SM gs /Fm Do",
        mask_content=b"/Piece Do",
        mask_has_resources=False,
        form_content=internal_FULL_RED,
        form_uses_right_piece=True,
    )
    pixels = internal_raster(data)
    internal_assert_pixel(pixels, 6, 12, [255, 0, 0])
    internal_assert_pixel(pixels, 26, 12, [255, 255, 255])
    # Prove that the consuming Form really has a different /Piece binding.
    reference = internal_raster(
        internal_pdf(b"/Fm Do", form_content=b"1 0 0 rg /Piece Do", form_uses_right_piece=True)
    )
    internal_assert_pixel(reference, 6, 12, [255, 255, 255])
    internal_assert_pixel(reference, 26, 12, [255, 0, 0])


def test_mask_form_matrix_and_rotated_bbox_remain_in_page_coordinates_when_cropped() -> None:
    data = internal_pdf(
        b"/SM gs " + internal_FULL_RED,
        mask_content=b"-100 -100 200 200 re f",
        mask_bbox=b"0 0 8 8",
        mask_matrix=b"1 1 -1 1 16 4",
    )
    pixels = internal_raster(data)
    internal_assert_pixel(pixels, 16, 12, [255, 0, 0])
    # In the axis-aligned envelope, but outside the transformed quadrilateral.
    internal_assert_pixel(pixels, 9, 5, [255, 255, 255])
    cropped = internal_raster(data, crop=(4, 4, 28, 20))
    numpy.testing.assert_array_equal(cropped, pixels[4:20, 4:28])


def test_none_clears_the_mask_and_q_q_restores_the_previous_mask() -> None:
    pixels = internal_raster(
        internal_pdf(
            b"/SM gs q /None gs 1 0 0 rg 0 16 32 8 re f Q "
            b"0 0 1 rg 0 8 32 8 re f /None gs 0 1 0 rg 0 0 32 8 re f"
        )
    )
    internal_assert_pixel(pixels, 28, 20, [255, 0, 0])
    internal_assert_pixel(pixels, 6, 12, [0, 0, 255])
    internal_assert_pixel(pixels, 18, 12, [127, 127, 255])
    internal_assert_pixel(pixels, 28, 12, [255, 255, 255])
    internal_assert_pixel(pixels, 28, 4, [0, 255, 0])


@pytest.mark.parametrize("group", [b"", b"/I true", b"/I false"])
def test_form_inherits_mask_or_applies_it_once_to_its_completed_transparency_group(
    group: bytes,
) -> None:
    form_group = b"/Group << /S /Transparency " + group + b" >>" if group else b""
    pixels = internal_raster(
        internal_pdf(
            b"/SM gs /Fm Do",
            mask_content=b"/Half gs 0 0 32 24 re f",
            form_content=internal_FULL_RED + b" 0 0 1 rg 0 0 32 24 re f",
            form_group=form_group,
        )
    )
    # A transparency group's children start without the inherited soft mask.
    expected = [127, 127, 255] if group else [127, 63, 191]
    internal_assert_pixel(pixels, 16, 12, expected)


def test_mask_group_resets_invoking_opacity_blend_mode_and_soft_mask() -> None:
    pixels = internal_raster(
        internal_pdf(
            b"/OtherMask gs /Half gs /Multiply gs /SM gs " + internal_FULL_RED,
            mask_content=b"0 0 32 24 re f",
            background=b"0.2 0.6 0.8",
        )
    )
    # Only the painted object's ca and BM apply; the prior lower-half mask was replaced.
    for y in [6, 18]:
        internal_assert_pixel(pixels, 16, y, [51, 76, 102])


def test_mask_group_can_establish_a_nested_alpha_mask() -> None:
    pixels = internal_raster(
        internal_pdf(
            b"/SM gs " + internal_FULL_RED,
            mask_content=b"/OtherMask gs /Half gs 0 0 32 24 re f",
        )
    )
    internal_assert_pixel(pixels, 16, 6, [255, 127, 127])
    internal_assert_pixel(pixels, 16, 18, [255, 255, 255])


@pytest.mark.parametrize("nested_mask", [False, True])
def test_masks_defined_inside_pattern_cells_translate_with_every_tile(nested_mask: bool) -> None:
    mask_content = (b"/OtherMask gs " if nested_mask else b"") + b"0 0 4 8 re f"
    pixels = internal_raster(
        internal_pdf(
            b"/Pattern cs /P scn 0 0 32 24 re f",
            mask_bbox=b"0 0 8 8",
            mask_content=mask_content,
            other_mask=b"0 0 2 8 re f",
            pattern_content=b"/SM gs 1 0 0 rg 0 0 8 8 re f",
            pattern_has_resources=True,
        )
    )
    expected = numpy.full((internal_HEIGHT, internal_WIDTH, 4), 255, dtype=numpy.uint8)
    painted_columns = numpy.arange(internal_WIDTH) % 8 < (2 if nested_mask else 4)
    expected[:, painted_columns] = [255, 0, 0, 255]
    numpy.testing.assert_array_equal(pixels, expected)


@pytest.mark.parametrize("cycle", ["direct", "two-groups", "through-pattern"])
def test_recursive_mask_factor_is_skipped_without_losing_other_mask_artwork(cycle: str) -> None:
    # Reader recovery drops the recursive factor; the enclosing mask's valid
    # paint remains usable, and its nested state must not escape to the page.
    mask_content = {
        "direct": b"/SM gs 0 0 12 24 re f",
        "two-groups": b"/OtherMask gs 0 0 12 24 re f",
        "through-pattern": b"/Pattern cs /P scn 0 0 12 24 re f",
    }[cycle]
    content = b"/SM gs " + internal_FULL_RED + b" /None gs 0 0 1 rg 24 16 8 8 re f"
    actual = internal_raster(
        internal_pdf(
            content,
            mask_content=mask_content,
            other_mask=b"/SM gs 0 0 12 24 re f",
            pattern_content=b"/SM gs 0 0 8 8 re f",
            pattern_has_resources=True,
        )
    )
    reference = internal_raster(internal_pdf(content, mask_content=b"0 0 12 24 re f"))
    internal_assert_pixel(actual, 6, 12, [255, 0, 0])
    internal_assert_pixel(actual, 26, 12, [255, 255, 255])
    internal_assert_pixel(actual, 28, 20, [0, 0, 255])
    numpy.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("has_previous_mask", [False, True])
def test_invalid_mask_group_dictionary_keeps_the_previous_graphics_state(
    has_previous_mask: bool,
) -> None:
    prefix = b"/OtherMask gs " if has_previous_mask else b""
    pixels = internal_raster(
        internal_pdf(
            prefix + b"/SM gs " + internal_FULL_RED + b" /None gs 0 0 1 rg 24 16 8 8 re f",
            mask_bbox=b"0 0 32",
        )
    )
    internal_assert_pixel(pixels, 8, 6, [255, 0, 0])
    internal_assert_pixel(pixels, 8, 18, [255, 255, 255] if has_previous_mask else [255, 0, 0])
    internal_assert_pixel(pixels, 28, 20, [0, 0, 255])


def test_malformed_mask_content_unwinds_its_clipping_and_graphics_saves() -> None:
    pixels = internal_raster(
        internal_pdf(
            b"/SM gs " + internal_FULL_RED + b" /None gs 0 0 1 rg 24 16 8 8 re f",
            mask_content=b"Q Q q 0 0 12 24 re W n /Half gs /Multiply gs 0 0 32 24 re f q",
        )
    )
    internal_assert_pixel(pixels, 6, 12, [255, 127, 127])
    internal_assert_pixel(pixels, 26, 12, [255, 255, 255])
    internal_assert_pixel(pixels, 28, 20, [0, 0, 255])


def test_mask_program_text_and_drawings_do_not_become_page_extraction() -> None:
    data = internal_pdf(
        b"/SM gs 0 0 1 rg BT /H 8 Tf 1 0 0 1 4 4 Tm (Page) Tj ET",
        mask_content=b"0 0 32 24 re f BT /H 8 Tf 1 0 0 1 4 16 Tm (MaskOnly) Tj ET",
    )
    with PdfDocument(data) as document:
        page = document.pages[0]
        assert "".join(char.text for char in page.chars) == "Page"
        assert page.extract().text == "Page"
        # The only page drawing is the explicit background, never the mask's rectangle.
        assert len(page.get_drawings()) == 1
        assert numpy.any(page.render().rasterize().array()[..., 0] < 255)


def test_type3_text_program_uses_the_spatial_mask_once_and_retains_advances() -> None:
    content = b"BT /F 8 Tf 1 0 0 1 4 4 Tm [(A) 0 (AA)] TJ ET"
    actual = internal_raster(internal_pdf(b"/SM gs " + content))
    reference = internal_raster(
        internal_pdf(b"/SM gs 1 0 0 rg 4 4 8 8 re f 12 4 8 8 re f 20 4 8 8 re f")
    )
    numpy.testing.assert_allclose(actual, reference, atol=1, rtol=0)
    internal_assert_pixel(actual, 18, 8, [255, 127, 127])
    internal_assert_pixel(actual, 26, 8, [255, 255, 255])
    with PdfDocument(internal_pdf(b"/SM gs " + content)) as document:
        assert "".join(char.text for char in document.pages[0].chars) == "AAA"


def test_soft_mask_changes_inside_a_text_object_apply_and_persist_after_et() -> None:
    pixels = internal_raster(
        internal_pdf(
            b"BT /F 8 Tf 1 0 0 1 4 4 Tm /SM gs (AA) Tj /None gs (A) Tj ET 0 0 1 rg 24 16 8 8 re f"
        )
    )
    internal_assert_pixel(pixels, 6, 8, [255, 0, 0])
    internal_assert_pixel(pixels, 18, 8, [255, 127, 127])
    internal_assert_pixel(pixels, 26, 8, [255, 0, 0])
    internal_assert_pixel(pixels, 28, 20, [0, 0, 255])


def test_outline_text_receives_fractional_mask_opacity() -> None:
    text = b"1 0 0 rg BT /H 20 Tf 1 0 0 1 8 4 Tm (H) Tj ET"
    reference = internal_raster(internal_pdf(text))
    actual = internal_raster(
        internal_pdf(b"/SM gs " + text, mask_content=b"/Half gs 0 0 32 24 re f")
    )
    solid = numpy.all(reference[..., :3] == [255, 0, 0], axis=2)
    assert numpy.any(solid), "The outline control must include fully covered pixels."
    numpy.testing.assert_allclose(
        actual[solid], numpy.broadcast_to([255, 127, 127, 255], actual[solid].shape), atol=1, rtol=0
    )


def test_soft_mask_does_not_replace_or_modify_text_clipping() -> None:
    text_clip = b"BT /H 20 Tf 7 Tr 1 0 0 1 8 4 Tm (H) Tj ET "
    content = b"q /SM gs " + text_clip + b"/None gs " + internal_FULL_RED + b" Q"
    actual = internal_raster(internal_pdf(content, mask_content=b""))
    reference = internal_raster(internal_pdf(b"q " + text_clip + internal_FULL_RED + b" Q"))
    assert numpy.any(numpy.all(reference[..., :3] == [255, 0, 0], axis=2))
    numpy.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("native", ["soft", "explicit", "color-key"])
def test_image_native_mask_overrides_graphics_state_mask_but_retains_ca(native: str) -> None:
    # ISO 32000-2 11.6.4.3: an image's SMask or Mask replaces the current GS mask.
    dictionary = {
        "soft": b"/SMask 10 0 R",
        "explicit": b"/Mask 10 0 R",
        "color-key": b"/Mask [255 255 0 0 0 0]",
    }[native]
    mask_dictionary = (
        b"/ImageMask true /BitsPerComponent 1"
        if native == "explicit"
        else b"/ColorSpace /DeviceGray /BitsPerComponent 8"
    )

    def image_document(content: bytes) -> bytes:
        return internal_pdf(
            content,
            image_dictionary=dictionary,
            image_data=b"\xff\x00\x00\x00\x00\xff",
            native_mask_dictionary=mask_dictionary,
            native_mask_data=b"\x00" if native == "explicit" else b"\x00\xff",
            mask_content=b"",
        )

    actual = internal_raster(image_document(b"/SM gs /Half gs " + internal_PAINT["image"]))
    reference = internal_raster(image_document(b"/Half gs " + internal_PAINT["image"]))
    internal_assert_pixel(reference, 28, 12, [127, 127, 255])
    numpy.testing.assert_array_equal(actual, reference)


def test_jpx_embedded_opacity_overrides_graphics_state_mask() -> None:
    samples = numpy.array([[[255, 0, 0, 128], [0, 0, 255, 255]]], dtype=numpy.uint8)
    data = internal_pdf(
        b"/SM gs " + internal_PAINT["image"],
        mask_content=b"",
        image_dictionary=b"/Filter /JPXDecode /SMaskInData 1",
        image_data=bytes(imagecodecs.jpeg2k_encode(samples, reversible=True)),
    )
    pixels = internal_raster(data)
    internal_assert_pixel(pixels, 4, 12, [255, 127, 127])
    internal_assert_pixel(pixels, 28, 12, [0, 0, 255])


@pytest.mark.parametrize("alpha_is_shape", [False, True])
def test_soft_mask_shape_and_opacity_have_distinct_knockout_behavior(alpha_is_shape: bool) -> None:
    content = (
        internal_FULL_RED
        + b" /SM gs "
        + (b"/Shape gs " if alpha_is_shape else b"")
        + b"0 0 1 rg 0 0 32 24 re f"
    )
    pixels = internal_raster(
        internal_pdf(
            b"/Fm Do",
            form_content=content,
            form_group=b"/Group << /S /Transparency /I true /K true >>",
        )
    )
    internal_assert_pixel(pixels, 6, 12, [0, 0, 255])
    internal_assert_pixel(pixels, 18, 12, [127, 0, 128] if alpha_is_shape else [127, 127, 255])
    internal_assert_pixel(pixels, 28, 12, [255, 0, 0] if alpha_is_shape else [255, 255, 255])
