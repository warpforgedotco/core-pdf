# SPDX-License-Identifier: AGPL-3.0-only
"""Text objects preserve their implicit knockout grouping and inherited paint state."""

import numpy
import pytest

from core_pdf import PdfDocument

internal_BACKDROP = numpy.array([51, 153, 204], dtype=float)
internal_RED = numpy.array([255, 0, 0], dtype=float)
internal_BLUE = numpy.array([0, 0, 255], dtype=float)
internal_ALPHA = 128 / 255
internal_RED_GLYPH = b"0 0 d0 1 0 0 rg 0 0 1000 1000 re f"
internal_BLUE_GLYPH = b"0 0 d0 0 0 1 rg 0 0 1000 1000 re f"
internal_TEXT_BEGIN = b"BT /F 12 Tf 1 0 0 1 4 4 Tm "
internal_SHOWS = [b"(AB) Tj", b"[(A) 0 (B)] TJ", b"(A) Tj (B) Tj"]


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
    first_glyph: bytes = internal_RED_GLYPH,
    second_glyph: bytes = internal_BLUE_GLYPH,
    form_group: bytes | None = None,
    page_prefix: bytes = b"",
) -> bytes:
    states = (
        b"/ExtGState << /Half << /ca 0.5 /CA 0.5 >> /Zero << /ca 0 /CA 0 >> "
        b"/Opaque << /ca 1 /CA 1 >> /Shape << /ca 0.5 /CA 0.5 /AIS true >> "
        b"/Multiply << /ca 0.5 /CA 0.5 /BM /Multiply >> "
        b"/TKOn << /TK true >> /TKOff << /TK false >> "
        b"/Changed << /ca 0.25 /CA 0.25 /BM /Multiply /TK false >> >>"
    )
    resources = (
        b"/Font << /F 5 0 R /H << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> "
        b"/XObject << /Fm 8 0 R >> " + states
    )
    page_content = (
        b"0.2 0.6 0.8 rg 0 0 24 20 re f "
        + page_prefix
        + (content if form_group is None else b"/Fm Do")
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 24 20] /Resources << "
        + resources
        + b" >> /Contents 4 0 R >>",
        internal_stream(page_content),
        b"<< /Type /Font /Subtype /Type3 /FontBBox [0 0 1000 1000] "
        b"/FontMatrix [0.001 0 0 0.001 0 0] /FirstChar 65 /LastChar 66 /Widths [0 0] "
        b"/Encoding << /Type /Encoding /Differences [65 /A /B] >> "
        b"/CharProcs << /A 6 0 R /B 7 0 R >> /Resources << " + states + b" >> >>",
        internal_stream(first_glyph),
        internal_stream(second_glyph),
        internal_stream(
            content,
            b"/Subtype /Form /BBox [0 0 24 20] "
            + (form_group or b"")
            + b" /Resources << "
            + resources
            + b" >>",
        ),
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


def internal_raster(data: bytes) -> numpy.ndarray:
    with PdfDocument(data) as document:
        return document.pages[0].render().rasterize().array().copy()


def internal_over(backdrop: numpy.ndarray, source: numpy.ndarray, alpha: float) -> numpy.ndarray:
    return numpy.rint(backdrop * (1 - alpha) + source * alpha)


@pytest.mark.parametrize(
    "knockout", [b"", b"/TKOn gs ", b"/TKOff gs "], ids=["default", "on", "off"]
)
@pytest.mark.parametrize("show", internal_SHOWS, ids=["Tj", "TJ", "multiple-Tj"])
def test_overlapping_glyphs_share_one_text_object_across_show_operators(
    knockout: bytes, show: bytes
) -> None:
    # ISO 32000-2 9.3.8: TK defaults true; the whole BT..ET is one
    # non-isolated knockout group, regardless of how many show operators it uses.
    pixels = internal_raster(
        internal_pdf(knockout + b"/Half gs " + internal_TEXT_BEGIN + show + b" ET")
    )
    initial = (
        internal_over(internal_BACKDROP, internal_RED, internal_ALPHA)
        if knockout == b"/TKOff gs "
        else internal_BACKDROP
    )
    expected = internal_over(initial, internal_BLUE, internal_ALPHA)
    numpy.testing.assert_allclose(pixels[6, 14, :3], expected, atol=1, rtol=0)
    numpy.testing.assert_array_equal(pixels[0, 23], [51, 153, 204, 255])


@pytest.mark.parametrize(
    "knockout", [b"", b"/TKOn gs ", b"/TKOff gs "], ids=["default", "on", "off"]
)
def test_separate_text_objects_accumulate_over_the_previous_text_object(knockout: bytes) -> None:
    content = (
        knockout
        + b"/Half gs "
        + internal_TEXT_BEGIN
        + b"(A) Tj ET "
        + internal_TEXT_BEGIN
        + b"(B) Tj ET"
    )
    pixels = internal_raster(internal_pdf(content))
    expected = internal_over(
        internal_over(internal_BACKDROP, internal_RED, internal_ALPHA),
        internal_BLUE,
        internal_ALPHA,
    )
    numpy.testing.assert_allclose(pixels[6, 14, :3], expected, atol=1, rtol=0)


@pytest.mark.parametrize("knockout", [True, False])
def test_transparent_glyph_replaces_prior_glyph_only_when_text_knockout_is_enabled(
    knockout: bool,
) -> None:
    content = (b"/TKOn gs " if knockout else b"/TKOff gs ") + b"/Half gs "
    pixels = internal_raster(
        internal_pdf(
            content + internal_TEXT_BEGIN + b"(AB) Tj ET",
            second_glyph=b"0 0 d0 /Zero gs 0 0 1 rg 0 0 1000 1000 re f",
        )
    )
    expected = (
        internal_BACKDROP
        if knockout
        else internal_over(internal_BACKDROP, internal_RED, internal_ALPHA)
    )
    numpy.testing.assert_allclose(pixels[6, 14, :3], expected, atol=1, rtol=0)


@pytest.mark.parametrize("knockout", [True, False])
def test_alpha_is_shape_retains_prior_glyph_contribution(knockout: bool) -> None:
    content = (b"/TKOn gs " if knockout else b"/TKOff gs ") + b"/Shape gs "
    pixels = internal_raster(internal_pdf(content + internal_TEXT_BEGIN + b"(AB) Tj ET"))
    expected = internal_over(
        internal_over(internal_BACKDROP, internal_RED, internal_ALPHA),
        internal_BLUE,
        internal_ALPHA,
    )
    numpy.testing.assert_allclose(pixels[6, 14, :3], expected, atol=1, rtol=0)


@pytest.mark.parametrize("knockout", [True, False])
def test_text_group_preserves_inherited_multiply_and_opacity_on_its_glyphs(knockout: bool) -> None:
    content = (b"/TKOn gs " if knockout else b"/TKOff gs ") + b"/Multiply gs "
    pixels = internal_raster(internal_pdf(content + internal_TEXT_BEGIN + b"(AB) Tj ET"))
    initial = internal_BACKDROP
    if not knockout:
        initial = internal_over(initial, initial * internal_RED / 255, internal_ALPHA)
    expected = internal_over(initial, initial * internal_BLUE / 255, internal_ALPHA)
    numpy.testing.assert_allclose(pixels[6, 14, :3], expected, atol=1, rtol=0)


@pytest.mark.parametrize("initial_knockout", [True, False])
def test_tk_changes_inside_bt_are_ignored_but_other_graphics_changes_persist(
    initial_knockout: bool,
) -> None:
    initial = b"/TKOn gs " if initial_knockout else b"/TKOff gs "
    requested = b"/TKOff gs " if initial_knockout else b"/TKOn gs "
    content = (
        initial
        + b"/Half gs "
        + internal_TEXT_BEGIN
        + b"(A) Tj "
        + requested
        + b"/Changed gs (B) Tj ET 0 0 1 rg 18 4 4 12 re f "
        + b"/Half gs BT /F 4 Tf 1 0 0 1 4 0 Tm (AB) Tj ET"
    )
    pixels = internal_raster(internal_pdf(content))
    backdrop = internal_BACKDROP
    if not initial_knockout:
        backdrop = internal_over(backdrop, internal_RED, internal_ALPHA)
    quarter = 64 / 255
    expected = internal_over(backdrop, backdrop * internal_BLUE / 255, quarter)
    numpy.testing.assert_allclose(pixels[6, 14, :3], expected, atol=1, rtol=0)
    # ET must not restore ca/BM changed inside BT: this path keeps both.
    following = internal_over(internal_BACKDROP, internal_BACKDROP * internal_BLUE / 255, quarter)
    numpy.testing.assert_allclose(pixels[10, 20, :3], following, atol=1, rtol=0)
    # The ignored TK setting cannot take effect on the next text object either.
    later_backdrop = internal_BACKDROP
    if not initial_knockout:
        later_backdrop = internal_over(
            later_backdrop, later_backdrop * internal_RED / 255, internal_ALPHA
        )
    later = internal_over(later_backdrop, later_backdrop * internal_BLUE / 255, internal_ALPHA)
    numpy.testing.assert_allclose(pixels[18, 7, :3], later, atol=1, rtol=0)


def test_saved_graphics_state_restores_tk_and_forms_inherit_it() -> None:
    form = b"/Half gs " + internal_TEXT_BEGIN + b"(AB) Tj ET"
    pixels = internal_raster(
        internal_pdf(
            form,
            form_group=b"",
            page_prefix=b"/TKOff gs q /TKOn gs Q ",
        )
    )
    expected = internal_over(
        internal_over(internal_BACKDROP, internal_RED, internal_ALPHA),
        internal_BLUE,
        internal_ALPHA,
    )
    numpy.testing.assert_allclose(pixels[6, 14, :3], expected, atol=1, rtol=0)


@pytest.mark.parametrize("isolated", [True, False])
@pytest.mark.parametrize("knockout", [True, False])
def test_text_inside_a_knockout_form_respects_the_enclosing_group(
    isolated: bool, knockout: bool
) -> None:
    group = b"/Group << /S /Transparency /K true /I " + (b"true" if isolated else b"false") + b" >>"
    content = (b"/TKOn gs " if knockout else b"/TKOff gs ") + b"/Half gs "
    text = content + internal_TEXT_BEGIN + b"(AB) Tj ET"
    actual = internal_raster(internal_pdf(b"0 1 0 rg 0 0 24 20 re f " + text, form_group=group))
    top_only = internal_raster(
        internal_pdf(content + internal_TEXT_BEGIN + b"(B) Tj ET", form_group=group)
    )
    numpy.testing.assert_allclose(actual[6, 14], top_only[6, 14], atol=1, rtol=0)
    numpy.testing.assert_array_equal(actual[0, 23], [0, 255, 0, 255])


def test_multiple_paints_in_one_type3_glyph_form_one_knockout_element() -> None:
    glyph = b"0 0 d0 1 0 0 rg 0 0 1000 1000 re f 0 0 1 rg 0 0 1000 1000 re f"
    pixels = internal_raster(
        internal_pdf(b"/Half gs " + internal_TEXT_BEGIN + b"(A) Tj ET", first_glyph=glyph)
    )
    expected = internal_over(
        internal_over(internal_BACKDROP, internal_RED, internal_ALPHA),
        internal_BLUE,
        internal_ALPHA,
    )
    numpy.testing.assert_allclose(pixels[6, 14, :3], expected, atol=1, rtol=0)


@pytest.mark.parametrize("opacity", [b"/Half gs ", b"/Opaque gs "], ids=["half", "opaque"])
def test_type3_glyph_paints_its_program_without_a_second_substitute_outline(opacity: bytes) -> None:
    data = internal_pdf(opacity + internal_TEXT_BEGIN + b"(A) Tj ET")
    actual = internal_raster(data)
    reference = internal_raster(internal_pdf(opacity + b"1 0 0 rg 4 4 12 12 re f"))
    # Compare every pixel, including the interior where an inappropriate
    # substitute A outline used to paint a second time over the CharProc.
    numpy.testing.assert_allclose(actual, reference, atol=1, rtol=0)
    with PdfDocument(data) as document:
        assert "".join(run.text for run in document.pages[0].chars) == "A"


@pytest.mark.parametrize("knockout", [True, False])
@pytest.mark.parametrize("render_mode", [0, 1, 2], ids=["fill", "stroke", "fillstroke"])
def test_outline_glyph_overlap_keeps_each_glyph_as_one_element(
    knockout: bool, render_mode: int
) -> None:
    prefix = (b"/TKOn gs " if knockout else b"/TKOff gs ") + b"/Half gs "
    begin = f"3 w BT /H 16 Tf {render_mode} Tr 1 0 0 1 4 4 Tm ".encode()
    content = (
        prefix
        + b"1 0 0 rg 1 0 0 RG "
        + begin
        + b"(H) Tj 0 0 1 rg 0 0 1 RG 1 0 0 1 4 4 Tm (H) Tj ET"
    )
    pixels = internal_raster(internal_pdf(content))
    coverage = internal_raster(internal_pdf(b"/TKOff gs 0 0 1 rg 0 0 1 RG " + begin + b"(H) Tj ET"))
    fully_covered = numpy.all(coverage[..., :3] == [0, 0, 255], axis=2)
    assert numpy.any(fully_covered), "The outline control must contain solid glyph coverage."
    initial = (
        internal_BACKDROP
        if knockout
        else internal_over(internal_BACKDROP, internal_RED, internal_ALPHA)
    )
    expected = internal_over(initial, internal_BLUE, internal_ALPHA)
    numpy.testing.assert_allclose(
        pixels[fully_covered, :3],
        numpy.broadcast_to(expected, pixels[fully_covered, :3].shape),
        atol=1,
        rtol=0,
    )


@pytest.mark.parametrize("render_mode", [4, 7])
def test_text_clip_survives_et_and_parent_clip_is_restored(render_mode: int) -> None:
    text = f"BT /H 16 Tf {render_mode} Tr 1 0 0 1 4 4 Tm (X) Tj ET ".encode()
    content = (
        b"q 4 4 8 12 re W n /Half gs "
        + text
        + b"/Opaque gs 0 0 1 rg 0 0 24 20 re f Q 0 1 0 rg 18 4 4 12 re f"
    )
    actual = internal_raster(internal_pdf(b"/TKOn gs " + content))
    reference = internal_raster(internal_pdf(b"/TKOff gs " + content))
    assert numpy.any(numpy.all(reference[:, :12, :3] == [0, 0, 255], axis=2))
    numpy.testing.assert_array_equal(actual, reference)
    numpy.testing.assert_array_equal(actual[10, 20], [0, 255, 0, 255])
