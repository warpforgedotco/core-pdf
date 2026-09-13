# SPDX-License-Identifier: AGPL-3.0-only
"""Calculator functions reach reader paint paths without weakening decode failures."""

import zlib
from typing import Any

import numpy
import pytest

from core_pdf import PdfDocument
from core_pdf.impl._impl.graphics.functions import internal_compile_pdf_function
from core_pdf_spec.exceptions import PdfDecryptionError, PdfParseError, PdfUnsupportedError
from core_pdf_spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError
from core_pdf_spec.s_07_syntax.stream import PdfStream


def internal_calculator(decoder: Any) -> PdfStream:
    dictionary = {
        "FunctionType": "4",
        "Domain": ["0", "1"],
        "Range": ["0", "1"],
        "Filter": "FlateDecode",
    }
    return PdfStream(dictionary, zlib.compress(b"{ dup mul }"), spec=dictionary, decoder=decoder)


def internal_wrap(stream: PdfStream, wrapper: str) -> object:
    if wrapper == "array":
        return [stream]
    if wrapper == "stitching":
        return {
            "FunctionType": 3,
            "Domain": [0, 1],
            "Functions": [stream],
            "Bounds": [],
            "Encode": [0, 1],
        }
    return stream


@pytest.mark.parametrize("wrapper", ["direct", "array", "stitching"])
def test_compressed_calculator_decodes_once_and_preserves_the_original_stream(wrapper: str) -> None:
    calls: list[None] = []

    def decode(
        data: bytes | memoryview, dictionary: object, *, parent_dictionary: Any = None
    ) -> bytes:
        calls.append(None)
        assert dictionary is stream.spec
        assert parent_dictionary["FunctionType"] == 4
        assert parent_dictionary["Domain"] == (0.0, 1.0)
        assert parent_dictionary["Range"] == (0.0, 1.0)
        return zlib.decompress(data)

    stream = internal_calculator(decode)
    original_data = stream.raw_data
    evaluate = internal_compile_pdf_function(internal_wrap(stream, wrapper))
    assert evaluate(0.25) == (0.0625,)
    assert evaluate(0.75) == (0.5625,)
    assert len(calls) == 1
    assert stream.raw_data is original_data
    assert stream.decoder is decode
    assert stream.spec is stream.dictionary
    assert stream.dictionary["FunctionType"] == "4"
    assert stream.dictionary["Domain"] == ["0", "1"]


@pytest.mark.parametrize("wrapper", ["direct", "stitching"])
@pytest.mark.parametrize("error_type", [PdfDecryptionError, RuntimeError])
def test_calculator_authentication_and_unexpected_decoder_errors_propagate(
    wrapper: str, error_type: type[Exception]
) -> None:
    failure = error_type("decoder failure must propagate")

    def decode(*args: object, **kwargs: object) -> bytes:
        raise failure

    with pytest.raises(error_type) as raised:
        internal_compile_pdf_function(internal_wrap(internal_calculator(decode), wrapper))
    assert raised.value is failure


@pytest.mark.parametrize(
    "error_type",
    [FilterParseError, FilterUnsupportedError, PdfParseError, PdfUnsupportedError, ValueError],
)
def test_known_calculator_decode_errors_become_numeric_fallback_errors(
    error_type: type[Exception],
) -> None:
    failure = error_type("controlled stream failure")

    def decode(*args: object, **kwargs: object) -> bytes:
        raise failure

    with pytest.raises(ValueError, match="invalid calculator PDF function") as raised:
        internal_compile_pdf_function(internal_calculator(decode))
    assert raised.value.__cause__ is failure


def test_calculator_evaluator_has_a_fresh_stack_after_a_numeric_failure() -> None:
    evaluate = internal_compile_pdf_function(
        PdfStream(
            {"FunctionType": 4, "Domain": [0, 1], "Range": [0, 1]},
            b"{ dup 0 eq { pop 99 0 div } { dup mul } ifelse }",
        )
    )
    assert evaluate(0.5) == (0.25,)
    with pytest.raises(ValueError):
        evaluate(0.0)
    assert evaluate(0.75) == (0.5625,)


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
    program: bytes,
    *,
    domain: bytes = b"0 1",
    output_range: bytes = b"0 1 0 1 0 1",
    color_space: bytes = b"[/Separation /Ink /DeviceRGB 5 0 R]",
    image_data: bytes = b"\x40",
    image_bits: int = 8,
    function_reference: bytes = b"5 0 R",
    function_object: bytes | None = None,
) -> bytes:
    resources = (
        b"/ColorSpace << /C " + color_space + b" >> /XObject << /Im 7 0 R >> "
        b"/Shading << /Sh << /ShadingType 2 /ColorSpace /DeviceRGB "
        b"/Coords [0 0 32 0] /Extend [true true] /Function "
        + function_reference
        + b" >> >> /ExtGState << /Half << /ca 0.5 >> /None << /SMask /None >> "
        b"/SM << /SMask << /S /Alpha /G 6 0 R /TR " + function_reference + b" >> >> >>"
    )
    if function_object is None:
        function_object = internal_stream(
            zlib.compress(program),
            b"/FunctionType 4 /Filter /FlateDecode /Domain ["
            + domain
            + b"] /Range ["
            + output_range
            + b"]",
        )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 32 24] /Resources << "
        + resources
        + b" >> /Contents 4 0 R >>",
        internal_stream(b"q 1 1 1 rg 0 0 32 24 re f Q " + content),
        function_object,
        internal_stream(
            b"0 0 12 24 re f /Half gs 12 0 12 24 re f",
            b"/Type /XObject /Subtype /Form /BBox [0 0 32 24] "
            b"/Group << /S /Transparency /I true >> /Resources << " + resources + b" >>",
        ),
        internal_stream(
            image_data,
            b"/Type /XObject /Subtype /Image /Width 1 /Height 1 "
            + f"/BitsPerComponent {image_bits} /ColorSpace ".encode()
            + color_space,
        ),
        b"<< /FunctionType 3 /Domain [0 1] /Functions [5 0 R] /Bounds [] /Encode [0 1] >>",
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


def internal_assert_rgb(pixels: numpy.ndarray, x: int, rgb: list[int]) -> None:
    numpy.testing.assert_allclose(pixels[12, x], [*rgb, 255], atol=1, rtol=0)


@pytest.mark.parametrize("function_reference", [b"5 0 R", b"8 0 R"], ids=["direct", "stitching"])
def test_calculator_alpha_transfer_maps_group_alpha_including_zero_outside_paint(
    function_reference: bytes,
) -> None:
    # ISO 32000-2 11.6.5.1: TR applies to computed alpha, including TR(0)
    # outside mask artwork; a Type 4 stream is a function under 7.10.5.
    pixels = internal_raster(
        internal_pdf(
            b"/SM gs 1 0 0 rg 0 0 32 24 re f",
            b"{ 1 exch sub }",
            output_range=b"0 1",
            function_reference=function_reference,
        )
    )
    internal_assert_rgb(pixels, 6, [255, 255, 255])
    internal_assert_rgb(pixels, 18, [255, 128, 128])
    internal_assert_rgb(pixels, 28, [255, 0, 0])


@pytest.mark.parametrize("space", ["separation", "devicen"])
@pytest.mark.parametrize("paint", ["path", "image", "image16"])
def test_calculator_tint_transform_reaches_path_and_image_colors(space: str, paint: str) -> None:
    devicen = space == "devicen"
    color_space = (
        b"[/DeviceN [/First /Second] /DeviceRGB 5 0 R]"
        if devicen
        else b"[/Separation /Ink /DeviceRGB 5 0 R]"
    )
    content = (
        b"q 32 0 0 24 0 0 cm /Im Do Q"
        if paint in {"image", "image16"}
        else b"/C cs " + (b"0.25 0.75" if devicen else b"0.25") + b" scn 0 0 32 24 re f"
    )
    image_data = b"\x40\xc0" if devicen else b"\x40"
    if paint == "image16":
        image_data = b"\x40\x00\xc0\x00" if devicen else b"\x40\x00"
    pixels = internal_raster(
        internal_pdf(
            content,
            b"{ 0 }" if devicen else b"{ dup 1 exch sub 0 }",
            domain=b"0 1 0 1" if devicen else b"0 1",
            color_space=color_space,
            image_data=image_data,
            image_bits=16 if paint == "image16" else 8,
        )
    )
    internal_assert_rgb(pixels, 16, [64, 192 if devicen and paint == "image" else 191, 0])


@pytest.mark.parametrize("function_reference", [b"5 0 R", b"8 0 R"], ids=["direct", "stitching"])
def test_calculator_axial_shading_matches_an_exponential_positive_control(
    function_reference: bytes,
) -> None:
    actual = internal_raster(
        internal_pdf(b"/Sh sh", b"{ dup dup }", function_reference=function_reference)
    )
    reference = internal_raster(
        internal_pdf(
            b"/Sh sh",
            b"",
            function_reference=function_reference,
            function_object=b"<< /FunctionType 2 /Domain [0 1] /N 1 /C0 [0 0 0] /C1 [1 1 1] >>",
        )
    )
    assert reference[12, 4, 0] < reference[12, 28, 0]
    numpy.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("program", [b"{ quit }", b"{ 0 div 0 0 }"])
def test_malformed_calculator_tint_keeps_reader_subtractive_fallback(program: bytes) -> None:
    pixels = internal_raster(internal_pdf(b"/C cs 0.25 scn 0 0 32 24 re f", program))
    internal_assert_rgb(pixels, 16, [191, 191, 191])


def test_invalid_calculator_shading_is_skipped_without_losing_following_paint() -> None:
    pixels = internal_raster(
        internal_pdf(b"/Sh sh 0 0 1 rg 24 0 8 24 re f", b"{ forbidden_operator }")
    )
    internal_assert_rgb(pixels, 8, [255, 255, 255])
    internal_assert_rgb(pixels, 28, [0, 0, 255])


def test_failing_calculator_mask_transfer_keeps_reader_unmasked_fallback() -> None:
    pixels = internal_raster(
        internal_pdf(
            b"/SM gs 1 0 0 rg 0 0 32 24 re f /None gs 0 0 1 rg 24 0 8 24 re f",
            b"{ 0 div }",
            output_range=b"0 1",
        )
    )
    internal_assert_rgb(pixels, 6, [255, 0, 0])
    internal_assert_rgb(pixels, 18, [255, 0, 0])
    internal_assert_rgb(pixels, 28, [0, 0, 255])
