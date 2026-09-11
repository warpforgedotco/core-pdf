# SPDX-License-Identifier: AGPL-3.0-only
"""Effective document versions reach page, pattern, image and group blending."""

import numpy
import pytest

from core_pdf import PdfDocument
from core_pdf.impl._impl.render.blend import (
    internal_blend_solid_array_numpy,
    internal_composite_blended_group_numpy,
)
from core_pdf.impl._impl.render.clipping import internal_ClipState
from core_pdf.impl._impl.render.target import internal_RasterTarget
from core_pdf.impl._impl.runtime.array_views import uint8_image_view
from core_pdf_spec.standards import PdfVersion, SemanticContext


def internal_stream(content: bytes, dictionary: bytes = b"") -> bytes:
    return (
        b"<< "
        + dictionary
        + f" /Length {len(content)} >>\nstream\n".encode()
        + content
        + b"\nendstream"
    )


def internal_document(
    mode: str,
    version: str,
    paint: str = "rectangle",
    *,
    catalog: bytes = b"",
    opacity: float = 1.0,
    width: int = 4,
) -> bytes:
    backdrop = 0 if mode == "ColorDodge" else 1
    source = 1 - backdrop
    path = f"{source} g 0 0 {width} 4 re f".encode()
    resource = b""
    extras = []
    if paint == "polygon":
        path = f"{source} g 0 0 m {width} 0 l {width} 4 l 0 4 l h f".encode()
    elif paint in {"image", "affine-image"}:
        resource = b"/XObject << /Im 5 0 R >>"
        path = b"4 0 0 4 0 0 cm /Im Do" if paint == "image" else b"4 1 1 4 -1 -1 cm /Im Do"
        extras.append(
            internal_stream(
                bytes([source * 255]),
                b"/Type /XObject /Subtype /Image "
                b"/Width 1 /Height 1 /BitsPerComponent 8 /ColorSpace /DeviceGray",
            )
        )
    elif paint in {"group", "nested-group"}:
        resource = b"/XObject << /Fm 5 0 R >>"
        group = b"/Type /XObject /Subtype /Form /BBox [0 0 4 4] "
        group += b"/Group << /S /Transparency /I true /CS /DeviceRGB >> "
        if paint == "nested-group":
            extras.append(
                internal_stream(
                    b"/Inner Do", group + b"/Resources << /XObject << /Inner 6 0 R >> >>"
                )
            )
        extras.append(internal_stream(path, group + b"/Resources << >>"))
        path = b"/Fm Do"
    elif paint == "shading":
        resource = b"/Shading << /Sh 5 0 R >>"
        path = b"/Sh sh"
        extras.append(
            f"<< /ShadingType 2 /ColorSpace /DeviceGray /Coords [0 0 4 0] "
            f"/Extend [true true] /Function << /FunctionType 2 /Domain [0 1] "
            f"/C0 [{source}] /C1 [{source}] /N 1 >> >>".encode()
        )
    elif paint == "pattern":
        resource = b"/Pattern << /Pt 5 0 R >>"
        extras.append(
            internal_stream(
                f"{source} g 0 0 1 1 re f".encode(),
                b"/Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 "
                b"/BBox [0 0 1 1] /XStep 1 /YStep 1 /Resources << >>",
            )
        )
        path = b"/Pattern cs /Pt scn 0 0 4 4 re f"
    content = f"{backdrop} g 0 0 {width} 4 re f q /GS gs ".encode() + path + b" Q"
    bodies = [
        b"<< /Type /Catalog /Pages 2 0 R " + catalog + b" >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} 4] /Resources << ".encode()
        + resource
        + f" /ExtGState << /GS << /BM /{mode} /ca {opacity} >> >> >> ".encode()
        + b"/Contents 4 0 R >>",
        internal_stream(content),
        *extras,
    ]
    output = f"%PDF-{version}\n".encode()
    offsets = [0]
    for index, body in enumerate(bodies, 1):
        offsets.append(len(output))
        output += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(output)
    output += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    output += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets[1:])
    return (
        output
        + f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )


@pytest.mark.parametrize("mode", ["ColorDodge", "ColorBurn"])
@pytest.mark.parametrize(
    "paint",
    [
        "rectangle",
        "polygon",
        "image",
        "affine-image",
        "group",
        "nested-group",
        "shading",
        "pattern",
    ],
)
@pytest.mark.parametrize(
    ("version", "catalog", "revised"),
    [
        ("1.7", b"", False),
        ("2.0", b"", True),
        ("1.4", b"/Version /2.0", True),
        ("1.7", b"/Extensions << /ADBE << /BaseVersion /1.7 /ExtensionLevel 5 >> >>", True),
    ],
)
def test_document_blend_pixels_follow_effective_version_across_paint_paths(
    mode: str, paint: str, version: str, catalog: bytes, revised: bool
) -> None:
    with PdfDocument(internal_document(mode, version, paint, catalog=catalog)) as document:
        rendered = document.pages[0].render()
        assert rendered.semantic_context == document.resolver.semantic_context
        pixels = rendered.rasterize().array()
    channel = (0 if revised else 255) if mode == "ColorDodge" else (255 if revised else 0)
    assert pixels[2, 1].tolist() == [channel, channel, channel, 255]


@pytest.mark.parametrize("paint", ["rectangle", "polygon", "group", "nested-group"])
@pytest.mark.parametrize("mode", ["ColorDodge", "ColorBurn"])
@pytest.mark.parametrize("version", ["1.7", "2.0"])
def test_document_blends_partial_alpha_and_group_opacity_once(
    paint: str, mode: str, version: str
) -> None:
    with PdfDocument(internal_document(mode, version, paint, opacity=0.5, width=64)) as document:
        pixels = document.pages[0].render().rasterize().array()
    channel = (
        (0 if version == "2.0" else 128)
        if mode == "ColorDodge"
        else (255 if version == "2.0" else 127)
    )
    assert pixels[2, 1].tolist() == [channel, channel, channel, 255]


def internal_target(pixels: bytearray, context: SemanticContext | None) -> internal_RasterTarget:
    width = len(pixels) // 4
    return internal_RasterTarget(
        pixels,
        None,
        clip=internal_ClipState(crop_x0=0, crop_y1=1, scale=1, width=width, height=1),
        width=width,
        height=1,
        scale=1,
        crop_x0=0,
        crop_y0=0,
        crop_y1=1,
        page_view=uint8_image_view(pixels, (1, width, 4)),
        semantic_context=context,
    )


@pytest.mark.parametrize("mode", ["ColorDodge", "ColorBurn", "Multiply", "Screen", "Normal"])
@pytest.mark.parametrize(
    "context",
    [
        None,
        SemanticContext(PdfVersion(1, 7)),
        SemanticContext(PdfVersion(2, 0)),
        SemanticContext(None),
    ],
)
def test_scalar_vector_and_group_compositing_agree_with_alpha_and_strided_views(
    mode: str, context: SemanticContext | None
) -> None:
    destinations = numpy.random.default_rng(2026).integers(0, 256, (4, 16, 4), dtype=numpy.uint8)
    destinations[0, :4] = [[0, 0, 0, 0], [0, 0, 0, 255], [255, 255, 255, 255], [255, 0, 0, 128]]
    for rgba in [(255, 255, 255, 255), (0, 0, 0, 255), (0, 255, 90, 128), (33, 77, 111, 0)]:
        expected = bytearray(destinations[:, ::2].tobytes())
        target = internal_target(expected, context)
        for offset in range(0, len(expected), 4):
            target.blend_px(offset, rgba, mode.lower())
        actual = destinations.copy()
        internal_blend_solid_array_numpy(actual[:, ::2], rgba, mode, semantic_context=context)
        assert actual[:, ::2].tobytes() == expected
        numpy.testing.assert_array_equal(actual[:, 1::2], destinations[:, 1::2])
        grouped = destinations.copy()
        source = numpy.empty_like(grouped[:, ::2])
        source[...] = rgba
        internal_composite_blended_group_numpy(
            grouped[:, ::2], source, None, None, mode, semantic_context=context
        )
        assert grouped[:, ::2].tobytes() == expected


@pytest.mark.parametrize("mode", ["ColorDodge", "ColorBurn"])
def test_reader_unknown_version_uses_explicit_current_equations(mode: str) -> None:
    with PdfDocument(internal_document(mode, "9.9")) as document:
        rendered = document.pages[0].render()
        assert rendered.semantic_context is not None
        assert rendered.semantic_context.version == PdfVersion(9, 9)
        pixels = rendered.rasterize().array()
    assert pixels[2, 1].tolist() == (
        [0, 0, 0, 255] if mode == "ColorDodge" else [255, 255, 255, 255]
    )


@pytest.mark.parametrize("mode", ["ColorDodge", "ColorBurn"])
@pytest.mark.parametrize("version", [PdfVersion(1, 7), PdfVersion(2, 0)])
def test_transparent_group_boundary_preserves_hidden_backdrop_colour(
    mode: str, version: PdfVersion
) -> None:
    context = SemanticContext(version)
    destination = numpy.asarray([[[19, 31, 47, 0], [23, 37, 53, 128]]], dtype=numpy.uint8)
    original = destination.copy()
    source = numpy.asarray([[[255, 255, 255, 0], [255, 255, 255, 255]]], dtype=numpy.uint8)
    internal_composite_blended_group_numpy(
        destination, source, 0.0, None, mode, semantic_context=context
    )
    numpy.testing.assert_array_equal(destination, original)
    internal_composite_blended_group_numpy(
        destination, source, 0.5, None, mode, semantic_context=context
    )
    numpy.testing.assert_array_equal(destination[0, 0], original[0, 0])
    expected = bytearray(original.tobytes())
    target = internal_target(expected, context)
    target.blend_px(4, (255, 255, 255, 128), mode.lower())
    assert destination.tobytes() == expected
