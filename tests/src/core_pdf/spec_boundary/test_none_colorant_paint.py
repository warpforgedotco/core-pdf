# SPDX-License-Identifier: AGPL-3.0-only
"""None colourants discard paint, not geometry (ISO 32000-2, 8.6.6.4-.5)."""

from typing import Any

import numpy
import pytest

from core_pdf import PdfDocument
from core_pdf.impl._impl.capture import recording
from core_pdf.impl._impl.graphics import images, shading
from core_pdf.impl._impl.graphics.color_spec import internal_color_space_paints
from core_pdf.impl._impl.graphics.images import decode_pdf_image, prepare_image
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.image_spec import ImageSource, SoftMask

SPACES = (
    pytest.param(b"[/Separation /None /DeviceRGB 6 0 R]", 1, id="separation"),
    pytest.param(b"[/DeviceN [/None /None] /DeviceRGB 6 0 R]", 2, id="devicen"),
    pytest.param(
        b"[/Indexed [/Separation /None /DeviceRGB 6 0 R] 0 <ff>]", 1, id="indexed-separation"
    ),
    pytest.param(
        b"[/Indexed [/DeviceN [/None /None] /DeviceRGB 6 0 R] 0 <ffff>]",
        2,
        id="indexed-devicen",
    ),
)
PAINTS = (
    "fill",
    "stroke",
    "fillstroke",
    "image8",
    "image16",
    "inline-image",
    "stencil",
    "inline-stencil",
    "shading",
    "shading-pattern",
    "colored-pattern",
    "uncolored-pattern",
)
BACKGROUND = b"0.2 0.4 0.6 rg 0 0 24 24 re f "


def internal_stream(raw: bytes, entries: bytes = b"") -> bytes:
    return b"<< " + entries + f" /Length {len(raw)} >>\nstream\n".encode() + raw + b"\nendstream"


def internal_document(
    content: bytes,
    space: bytes,
    components: int,
    *,
    extra: bytes = b"<< >>",
    tint: bytes | None = None,
) -> bytes:
    # A constant magenta alternate makes both painting white and invoking the
    # global tint visibly wrong on the opaque blue-gray backdrop.
    if tint is None:
        tint = internal_stream(
            b"\xff\x00\xff",
            b"/FunctionType 0 /Domain ["
            + b"0 1 " * components
            + b"] /Size ["
            + b"1 " * components
            + b"] /Encode ["
            + b"0 0 " * components
            + b"] /BitsPerSample 8 /Range [0 1 0 1 0 1]",
        )
    bodies = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 24 24] /Resources << "
        b"/ColorSpace << /C 5 0 R /U [/Pattern 5 0 R] >> "
        b"/Font << /F 7 0 R >> /XObject << /Im 8 0 R >> "
        b"/Shading << /S 8 0 R >> /Pattern << /P 8 0 R >> "
        b"/ExtGState << /Alpha << /ca 0.5 /CA 0.5 >> >> "
        b">> /Contents 4 0 R >>",
        internal_stream(BACKGROUND + content),
        space,
        tint,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        extra,
    )
    result = b"%PDF-2.0\n"
    offsets = [0]
    for number, body in enumerate(bodies, 1):
        offsets.append(len(result))
        result += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(result)
    result += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    result += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets[1:])
    return (
        result
        + f"trailer << /Root 1 0 R /Size {len(offsets)} >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )


def internal_pixels(pdf: bytes) -> numpy.ndarray[Any, Any]:
    with PdfDocument(pdf) as document:
        return document.pages[0].render().rasterize().array().copy()


def internal_operands(space: bytes, components: int) -> bytes:
    return b"0" if space.startswith(b"[/Indexed") else b" ".join([b"0.7"] * components)


def internal_paint(space: bytes, components: int, paint: str) -> tuple[bytes, bytes]:
    values = internal_operands(space, components)
    selected = b"/C cs " + values + b" scn /C CS " + values + b" SCN "
    extra = b"<< >>"
    if paint in {"fill", "stroke", "fillstroke"}:
        op = {"fill": b"f", "stroke": b"S", "fillstroke": b"B"}[paint]
        operation = selected + b"2 w 4 4 16 16 re " + op
    elif paint in {"image8", "image16", "inline-image"}:
        depth = 16 if paint == "image16" else 8
        count = 1 if space.startswith(b"[/Indexed") else components
        raw = bytes([0 if space.startswith(b"[/Indexed") else 127]) * count * (depth // 8)
        if paint == "inline-image":
            operation = b"16 0 0 16 4 4 cm BI /W 1 /H 1 /BPC 8 /CS /C ID " + raw + b" EI"
        else:
            extra = internal_stream(
                raw,
                b"/Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace 5 0 R "
                + f"/BitsPerComponent {depth}".encode(),
            )
            operation = b"16 0 0 16 4 4 cm /Im Do"
    elif paint in {"stencil", "inline-stencil"}:
        extra = internal_stream(
            b"\x00",
            b"/Type /XObject /Subtype /Image /Width 1 /Height 1 /ImageMask true "
            b"/BitsPerComponent 1",
        )
        operation = (
            selected
            + b"16 0 0 16 4 4 cm "
            + (
                b"BI /W 1 /H 1 /BPC 1 /IM true ID \x00 EI"
                if paint == "inline-stencil"
                else b"/Im Do"
            )
        )
    elif paint in {"shading", "shading-pattern"}:
        extra = (
            b"<< /ShadingType 2 /ColorSpace 5 0 R /Coords [0 0 24 0] /Extend [true true] "
            b"/Function << /FunctionType 2 /Domain [0 1] /C0 ["
            + values
            + b"] /C1 ["
            + values
            + b"] /N 1 >> >>"
        )
        operation = b"/S sh"
        if paint == "shading-pattern":
            extra = b"<< /PatternType 2 /Shading " + extra + b" >>"
            operation = b"/Pattern cs /P scn 0 0 24 24 re f"
    else:
        colored = paint == "colored-pattern"
        extra = internal_stream(
            (selected if colored else b"") + b"0 0 4 4 re f",
            b"/Type /Pattern /PatternType 1 "
            + (b"/PaintType 1 " if colored else b"/PaintType 2 ")
            + b"/TilingType 1 /BBox [0 0 4 4] /XStep 4 /YStep 4 "
            b"/Resources << /ColorSpace << /C 5 0 R >> >>",
        )
        operation = (
            b"/Pattern cs /P scn " if colored else b"/U cs " + values + b" /P scn "
        ) + b"4 4 16 16 re f"
    return b"q /Alpha gs " + operation + b" Q", extra


@pytest.mark.parametrize(("space", "components"), SPACES)
@pytest.mark.parametrize("paint", [paint for paint in PAINTS if "shading" not in paint])
def test_none_paint_leaves_every_backdrop_pixel_unchanged(
    space: bytes,
    components: int,
    paint: str,
) -> None:
    content, extra = internal_paint(space, components, paint)
    actual = internal_pixels(internal_document(content, space, components, extra=extra))
    expected = internal_pixels(internal_document(b"", space, components))
    numpy.testing.assert_array_equal(actual, expected)
    real_space = space.replace(b"/None", b"/SpotA", 1).replace(b"/None", b"/SpotB", 1)
    control = internal_pixels(internal_document(content, real_space, components, extra=extra))
    assert numpy.any(control != expected)


@pytest.mark.parametrize(("space", "components"), SPACES[:2])
@pytest.mark.parametrize("paint", ["shading", "shading-pattern"])
def test_none_shading_leaves_backdrop_unchanged(space: bytes, components: int, paint: str) -> None:
    # Axial/radial shadings cannot use Indexed spaces (8.7.4.2).
    content, extra = internal_paint(space, components, paint)
    expected = internal_pixels(internal_document(b"", space, components))
    actual = internal_pixels(internal_document(content, space, components, extra=extra))
    numpy.testing.assert_array_equal(actual, expected)
    real_space = space.replace(b"/None", b"/SpotA", 1).replace(b"/None", b"/SpotB", 1)
    control = internal_pixels(internal_document(content, real_space, components, extra=extra))
    assert numpy.any(control != expected)


@pytest.mark.parametrize(("space", "components"), SPACES)
@pytest.mark.parametrize("none_fill", [False, True], ids=["none-stroke", "none-fill"])
def test_fill_and_stroke_suppress_only_the_none_contribution(
    space: bytes,
    components: int,
    none_fill: bool,
) -> None:
    values = internal_operands(space, components)
    selected = (
        b"/C cs " + values + b" scn 0 1 0 RG "
        if none_fill
        else b"/C CS " + values + b" SCN 0 1 0 rg "
    )
    actual = internal_pixels(internal_document(selected + b"2 w 4 4 16 16 re B", space, components))
    expected = internal_pixels(
        internal_document(
            b"0 1 0 rg 0 1 0 RG 2 w 4 4 16 16 re " + (b"S" if none_fill else b"f"),
            space,
            components,
        )
    )
    numpy.testing.assert_array_equal(actual, expected)
    assert numpy.any(numpy.all(actual[:, :, :3] == (0, 255, 0), axis=2))


@pytest.mark.parametrize(("space", "components"), SPACES)
def test_none_path_keeps_clip_and_q_Q_restores_color(space: bytes, components: int) -> None:
    values = internal_operands(space, components)
    prefix = b"0 1 0 rg q /C cs " + values + b" scn /C CS " + values + b" SCN "
    suffix = b"0 0 1 rg 0 0 24 24 re f Q 0 0 3 3 re f"
    actual = internal_pixels(
        internal_document(prefix + b"4 4 16 16 re W B " + suffix, space, components)
    )
    expected = internal_pixels(
        internal_document(b"0 1 0 rg q 4 4 16 16 re W n " + suffix, space, components)
    )
    numpy.testing.assert_array_equal(actual, expected)
    with PdfDocument(
        internal_document(prefix + b"4 4 16 16 re B Q", space, components)
    ) as document:
        drawing = document.pages[0].get_drawings()[-1]
        assert drawing.kind == "fillstroke"
        assert drawing.rect == (4, 4, 20, 20)


@pytest.mark.parametrize(("space", "components"), SPACES)
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_none_text_is_extracted_and_advances_without_paint(
    space: bytes,
    components: int,
    mode: int,
) -> None:
    values = internal_operands(space, components)
    text = f"BT /F 10 Tf {mode} Tr 1 0 0 1 1 5 Tm (H) Tj ".encode()
    prefix = b"/C cs " + values + b" scn /C CS " + values + b" SCN "
    suffix = b"0 1 0 rg 0 1 0 RG (I) Tj ET"
    pdf = internal_document(prefix + text + suffix, space, components)
    expected = internal_document(
        text.replace(f"{mode} Tr".encode(), b"3 Tr") + f"{mode} Tr ".encode() + suffix,
        space,
        components,
    )
    numpy.testing.assert_array_equal(internal_pixels(pdf), internal_pixels(expected))
    with PdfDocument(pdf) as document:
        page = document.pages[0]
        assert "".join(run.text for run in page.chars) == "HI"
        glyphs = page.get_page_program().glyphs
        assert glyphs[0].text_render_mode == 3
        assert glyphs[1].advance_bbox[0] == pytest.approx(glyphs[0].advance_bbox[2])
        assert "HI" in page.extract().text


@pytest.mark.parametrize(("space", "components"), SPACES)
@pytest.mark.parametrize("mode", [2, 6])
@pytest.mark.parametrize("none_fill", [False, True], ids=["none-stroke", "none-fill"])
def test_none_text_fillstroke_keeps_the_real_contribution(
    space: bytes,
    components: int,
    mode: int,
    none_fill: bool,
) -> None:
    values = internal_operands(space, components)
    selected = (
        b"/C cs " + values + b" scn 0 1 0 RG "
        if none_fill
        else b"/C CS " + values + b" SCN 0 1 0 rg "
    )
    text = f"1 w BT /F 18 Tf {mode} Tr 1 0 0 1 3 3 Tm (H) Tj ET".encode()
    actual = internal_pixels(internal_document(selected + text, space, components))
    remaining_mode = (1 if none_fill else 0) + (4 if mode >= 4 else 0)
    expected = internal_pixels(
        internal_document(
            b"0 1 0 rg 0 1 0 RG "
            + text.replace(f"{mode} Tr".encode(), f"{remaining_mode} Tr".encode()),
            space,
            components,
        )
    )
    numpy.testing.assert_array_equal(actual, expected)
    assert numpy.any(actual[:, :, 1] > 102)


@pytest.mark.parametrize(("space", "components"), SPACES)
@pytest.mark.parametrize("mode", [4, 5, 6])
def test_none_text_still_clips_after_ET_and_Q_restores_clip(
    space: bytes,
    components: int,
    mode: int,
) -> None:
    values = internal_operands(space, components)
    prefix = b"q /C cs " + values + b" scn /C CS " + values + b" SCN "
    text = f"BT /F 18 Tf {mode} Tr 1 0 0 1 3 3 Tm (H) Tj ET ".encode()
    suffix = b"0 1 0 rg 0 0 24 24 re f Q 1 0 0 rg 0 0 2 2 re f"
    actual = internal_pixels(internal_document(prefix + text + suffix, space, components))
    expected = internal_pixels(
        internal_document(
            b"q " + text.replace(f"{mode} Tr".encode(), b"7 Tr") + suffix, space, components
        )
    )
    numpy.testing.assert_array_equal(actual, expected)
    assert numpy.any(numpy.all(actual[:, :, :3] == (0, 255, 0), axis=2))
    assert numpy.any(numpy.all(actual[:, :, :3] == (51, 102, 153), axis=2))


@pytest.mark.parametrize(("space", "components"), SPACES)
@pytest.mark.parametrize("paint", ["image8", "inline-image", "stencil", "inline-stencil"])
def test_none_images_retain_extraction_records_without_decoding(
    monkeypatch: pytest.MonkeyPatch,
    space: bytes,
    components: int,
    paint: str,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("discarded image was decoded")

    content, extra = internal_paint(space, components, paint)
    with PdfDocument(internal_document(content, space, components, extra=extra)) as document:
        # Explicit image extraction may decode a stencil's reusable mask.
        # Rendering it in a None fill must not request that decode.
        assert len(document.pages[0].extract_images()) == 1
        monkeypatch.setattr(images, "decode_stream_data", fail)
        monkeypatch.setattr(images, "internal_decode_image_samples", fail)
        document.pages[0].render().rasterize()


def test_none_preparation_skips_tint_functions_samples_and_soft_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("discarded paint performed conversion or decode")

    space = ["Separation", "None", "DeviceRGB", {}]
    dictionary = {"ColorSpace": space, "Width": 1, "Height": 1, "BitsPerComponent": 16}
    monkeypatch.setattr(images, "internal_decode_image_samples", fail)
    monkeypatch.setattr(images, "decode_stream_data", fail)
    monkeypatch.setattr(shading, "internal_compile_pdf_function", fail)
    assert decode_pdf_image(b"", dictionary) is None
    assert (
        prepare_image(ImageSource(b"", dictionary, soft_mask=SoftMask(b"", {"Matte": [0]}))) is None
    )
    assert shading.prepare_shading({"ColorSpace": space, "ShadingType": 2}) is None
    original = recording.color_operands_to_srgb

    def check_tint(spec: Any, *args: Any, **kwargs: Any) -> Any:
        if spec.kind in {"Separation", "DeviceN"}:
            fail()
        return original(spec, *args, **kwargs)

    monkeypatch.setattr(recording, "color_operands_to_srgb", check_tint)
    internal_pixels(
        internal_document(b"/C cs 1 scn 0 0 24 24 re f", b"[/Separation /None /DeviceRGB 6 0 R]", 1)
    )


@pytest.mark.parametrize("depth", [8, 16])
def test_color_image_uses_its_own_space_even_when_fill_is_none(depth: int) -> None:
    raw = b"\x00" * (depth // 8) + b"\xff" * (depth // 8) + b"\x00" * (depth // 8)
    image = internal_stream(
        raw,
        b"/Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceRGB "
        + f"/BitsPerComponent {depth}".encode(),
    )
    actual = internal_pixels(
        internal_document(
            b"/C cs 1 scn 16 0 0 16 4 4 cm /Im Do",
            b"[/Separation /None /DeviceRGB 6 0 R]",
            1,
            extra=image,
        )
    )
    assert tuple(actual[12, 12, :3]) == (0, 255, 0)


@pytest.mark.parametrize("paint", ["fill", "image8", "image16"])
def test_mixed_none_reaches_global_tint_with_unchanged_values(paint: str) -> None:
    # First (None) component controls red; the real spot component controls
    # green. Dropping or zeroing None changes the rendered colour.
    tint = internal_stream(
        bytes((0, 0, 0, 255, 0, 0, 0, 255, 0, 255, 255, 0)),
        b"/FunctionType 0 /Domain [0 1 0 1] /Size [2 2] /BitsPerSample 8 /Range [0 1 0 1 0 1]",
    )
    space = b"[/DeviceN [/None /Spot] /DeviceRGB 6 0 R]"
    content, extra = internal_paint(space, 2, paint)
    # Use the full-range endpoint to make expected byte colours exact at both depths.
    content = content.replace(b"0.7 0.7", b"1 1").replace(b"/Alpha gs", b"")
    extra = extra.replace(b"\x7f", b"\xff")
    actual = internal_pixels(internal_document(content, space, 2, extra=extra, tint=tint))
    assert tuple(actual[12, 12, :3]) == (255, 255, 0)


def internal_unreadable_stream() -> PdfStream:
    def fail(*args: Any, **kwargs: Any) -> bytes:
        pytest.fail("no-paint probe decoded an unused stream")

    return PdfStream({"N": 3}, b"encoded", decoder=fail)


@pytest.mark.parametrize("wrapper", ["direct", "indexed", "pattern"])
@pytest.mark.parametrize("kind", ["icc", "separation", "devicen", "mixed-none"])
def test_paint_probe_does_not_read_real_colorant_profiles_palettes_or_tints(
    wrapper: str,
    kind: str,
) -> None:
    stream = internal_unreadable_stream()
    icc = ["ICCBased", stream]
    space: object = {
        "icc": icc,
        "separation": ["Separation", "Spot", icc, stream],
        "devicen": ["DeviceN", ["Spot"], icc, stream],
        "mixed-none": ["DeviceN", ["None", "Spot"], icc, stream],
    }[kind]
    if wrapper == "indexed":
        space = ["Indexed", space, 0, stream]
    elif wrapper == "pattern":
        space = ["Pattern", space]
    assert internal_color_space_paints(space)


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("prepare", [False, True], ids=["decode", "prepare"])
def test_zero_width_image_skips_its_unreadable_profile(indexed: bool, prepare: bool) -> None:
    stream = internal_unreadable_stream()
    space = ["ICCBased", stream]
    if indexed:
        space = ["Indexed", space, 0, stream]
    dictionary = {"Width": 0, "Height": 1, "BitsPerComponent": 8, "ColorSpace": space}
    assert (
        prepare_image(ImageSource(b"", dictionary))
        if prepare
        else decode_pdf_image(b"", dictionary)
    ) is None


@pytest.mark.parametrize("device_n", [False, True], ids=["separation", "devicen"])
@pytest.mark.parametrize("indexed", [False, True])
def test_all_none_image_skips_unused_profile_palette_tint_and_mask(
    device_n: bool,
    indexed: bool,
) -> None:
    stream = internal_unreadable_stream()
    space = (
        ["DeviceN", ["None", "None"], ["ICCBased", stream], stream]
        if device_n
        else [
            "Separation",
            "None",
            ["ICCBased", stream],
            stream,
        ]
    )
    if indexed:
        space = ["Indexed", space, 0, stream]
    dictionary = {"Width": 1, "Height": 1, "BitsPerComponent": 16, "ColorSpace": space}
    assert decode_pdf_image(b"", dictionary) is None
    source = ImageSource(b"", dictionary, soft_mask=SoftMask(b"", {"Matte": [0]}))
    assert prepare_image(source) is None


@pytest.mark.parametrize("device_n", [False, True], ids=["separation", "devicen"])
def test_all_none_shading_skips_unused_profile_and_functions(device_n: bool) -> None:
    stream = internal_unreadable_stream()
    space = (
        ["DeviceN", ["None", "None"], ["ICCBased", stream], stream]
        if device_n
        else [
            "Separation",
            "None",
            ["ICCBased", stream],
            stream,
        ]
    )
    assert (
        shading.prepare_shading(
            {
                "ShadingType": 2,
                "Coords": [0, 0, 24, 0],
                "ColorSpace": space,
                "Function": stream,
            }
        )
        is None
    )


def test_invalid_shading_skips_its_unreadable_profile() -> None:
    assert (
        shading.prepare_shading(
            {
                "ShadingType": 2,
                "Coords": [],
                "ColorSpace": ["ICCBased", internal_unreadable_stream()],
            }
        )
        is None
    )


@pytest.mark.parametrize(
    "space",
    [
        None,
        [],
        "UnresolvedName",
        ["Indexed"],
        ["Pattern", "DeviceRGB", 1],
        ["Separation", "None"],
        ["Separation", 1, "DeviceRGB", {}],
        ["DeviceN", "None", "DeviceRGB", {}],
        ["DeviceN", [], "DeviceRGB", {}],
        ["DeviceN", ["None", 1], "DeviceRGB", {}],
    ],
)
def test_paint_probe_defers_unknown_and_malformed_descriptions(space: object) -> None:
    assert internal_color_space_paints(space)


@pytest.mark.parametrize("kind", ["Indexed", "Pattern"])
def test_paint_probe_defers_cyclic_descriptions(kind: str) -> None:
    cyclic: list[object] = [kind, None, 0, b""] if kind == "Indexed" else [kind, None]
    cyclic[1] = cyclic
    assert internal_color_space_paints(cyclic)
