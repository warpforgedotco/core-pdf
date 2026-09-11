# SPDX-License-Identifier: AGPL-3.0-only
"""Backdrop-aware Form and pattern composition in the selected RGB renderer."""

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


def internal_document(
    *,
    isolation: bytes = b"/I false",
    opacity: float = 0.5,
    inner_opacity: float = 1.0,
    outer_blend: str = "Normal",
    pattern: bool = False,
    nested: bool = False,
    repeats: int = 1,
    empty: bool = False,
) -> bytes:
    state = (
        f"/ExtGState << /Inner << /BM /Multiply /ca {inner_opacity} >> "
        f"/Outer << /BM /{outer_blend} /ca {opacity} >> >>"
    ).encode()
    paint = b"/Inner gs 0.8 0.4 0.2 rg 0 0 8 8 re f " * repeats if not empty else b""
    group = b"/Type /XObject /Subtype /Form /BBox [0 0 8 8] "
    group += b"/Group << /S /Transparency /CS /DeviceRGB " + isolation + b" >> "
    page_resource = b"/XObject << /Fm 5 0 R >> "
    invoke = b"/Fm Do"
    extras = []
    if pattern:
        page_resource = b"/Pattern << /Pt 5 0 R >> "
        invoke = b"/Pattern cs /Pt scn 0 0 8 8 re f"
        cell = b"/PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 8 8] /XStep 8 /YStep 8 "
        extras.append(internal_stream(paint, cell + b"/Resources << " + state + b" >>"))
    elif nested:
        extras.append(
            internal_stream(
                b"/Child Do",
                group + b"/Resources << /XObject << /Child 6 0 R >> >>",
            )
        )
        extras.append(internal_stream(paint, group + b"/Resources << " + state + b" >>"))
    else:
        extras.append(internal_stream(paint, group + b"/Resources << " + state + b" >>"))
    content = b"0.2 0.6 0.8 rg 0 0 12 8 re f q /Outer gs " + invoke + b" Q"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 12 8] /Resources << "
        + page_resource
        + state
        + b" >> /Contents 4 0 R >>",
        internal_stream(content),
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


def internal_raster(data: bytes) -> numpy.ndarray:
    with PdfDocument(data) as document:
        return document.pages[0].render().rasterize().array().copy()


@pytest.mark.parametrize("isolation", [b"", b"/I false", b"/I true"])
@pytest.mark.parametrize("opacity", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("inner_opacity", [0.5, 1.0])
def test_form_isolation_selects_backdrop_and_applies_outer_opacity_once(
    isolation: bytes, opacity: float, inner_opacity: float
) -> None:
    # ISO 32000-2 11.4.4: Multiply sees the page in a non-isolated group,
    # and a transparent backdrop in an isolated group. Outer alpha applies once.
    pixels = internal_raster(
        internal_document(isolation=isolation, opacity=opacity, inner_opacity=inner_opacity)
    )
    backdrop = numpy.array([51, 153, 204], dtype=float)
    source = numpy.array([204, 102, 51], dtype=float)
    color = source if isolation == b"/I true" else backdrop * source / 255.0
    alpha = round(round(inner_opacity * 255) * opacity) / 255.0
    expected = numpy.rint(backdrop * (1 - alpha) + color * alpha)
    numpy.testing.assert_allclose(pixels[4, 4, :3], expected, atol=1, rtol=0)
    numpy.testing.assert_array_equal(pixels[4, 10], [51, 153, 204, 255])


@pytest.mark.parametrize("pattern", [False, True], ids=["form", "pattern"])
@pytest.mark.parametrize("repeats", [1, 2])
@pytest.mark.parametrize("outer_blend", ["Normal", "Screen"])
def test_nonisolated_rgb_appearance_is_composited_with_outer_blend_once(
    pattern: bool, repeats: int, outer_blend: str
) -> None:
    pixels = internal_raster(
        internal_document(pattern=pattern, repeats=repeats, outer_blend=outer_blend)
    )
    backdrop = numpy.array([51, 153, 204], dtype=float)
    result = backdrop.copy()
    for _ in range(repeats):
        result = numpy.rint(result * numpy.array([0.8, 0.4, 0.2]))
    if outer_blend == "Screen":
        result = 255 - (255 - backdrop) * (255 - result) / 255
    expected = numpy.rint(backdrop * (127 / 255) + result * (128 / 255))
    numpy.testing.assert_allclose(pixels[4, 4, :3], expected, atol=1, rtol=0)
    numpy.testing.assert_array_equal(pixels[4, 10], [51, 153, 204, 255])


@pytest.mark.parametrize("inner_opacity", [0.5, 1.0])
def test_nested_nonisolated_form_preserves_parent_backdrop_and_group_alpha(
    inner_opacity: float,
) -> None:
    single = internal_raster(internal_document(inner_opacity=inner_opacity))
    nested = internal_raster(internal_document(inner_opacity=inner_opacity, nested=True))
    numpy.testing.assert_allclose(nested, single, atol=1, rtol=0)


@pytest.mark.parametrize("pattern", [False, True])
def test_empty_nonisolated_group_does_not_repaint_the_backdrop(pattern: bool) -> None:
    pixels = internal_raster(internal_document(empty=True, pattern=pattern, outer_blend="Screen"))
    numpy.testing.assert_array_equal(pixels, numpy.broadcast_to([51, 153, 204, 255], pixels.shape))
