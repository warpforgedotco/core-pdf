# SPDX-License-Identifier: AGPL-3.0-only
"""Sixteen-bit image words retain precision through Decode, colour conversion and masks."""

import zlib

import imagecodecs
import numpy
import pytest

from core_pdf import PdfDocument
from core_pdf.impl._impl.graphics.color_math import d50_xyz_to_srgb
from core_pdf.impl._impl.graphics.device_profiles import default_cmyk_transform
from core_pdf.impl._impl.graphics.icc_profiles import IccTransform, parse_icc_transform
from core_pdf.impl._impl.graphics.images import decode_pdf_image, prepare_image
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.color_math import lab_components_to_xyz
from core_pdf_spec.s_08_graphics.image_spec import ImageSource, SoftMask
from core_pdf_spec.standards import PdfVersion, SemanticContext


def internal_words(values: list[int]) -> bytes:
    return numpy.asarray(values, dtype=">u2").tobytes()


def internal_dictionary(width: int, **entries: object) -> dict:
    return {
        "Width": width,
        "Height": 1,
        "BitsPerComponent": 16,
        "ColorSpace": "DeviceGray",
        **entries,
    }


@pytest.mark.parametrize("filtered", [False, True])
def test_gray_words_decode_most_significant_byte_first(filtered: bool) -> None:
    words = internal_words([0, 255, 256, 32768, 65535])
    raw = zlib.compress(words) if filtered else words
    dictionary = internal_dictionary(5, **({"Filter": "FlateDecode"} if filtered else {}))
    image = prepare_image(ImageSource(raw, dictionary))
    assert image is not None
    assert image.raster.array.tolist() == [[[0], [1], [1], [128], [255]]]


def test_decode_uses_low_bits_before_output_quantization_and_clamps() -> None:
    image = prepare_image(
        ImageSource(
            internal_words([32767, 32768, 32769]),
            internal_dictionary(3, Decode=[-32767, 32768]),
        )
    )
    assert image is not None
    assert image.raster.array.tolist() == [[[0], [255], [255]]]


def test_colour_key_mask_compares_original_full_words_before_reversed_decode() -> None:
    image = prepare_image(
        ImageSource(
            internal_words([32768, 32769]),
            internal_dictionary(2, Decode=[1, 0], Mask=[32768, 32768]),
        )
    )
    assert image is not None
    assert image.raster.has_alpha
    assert image.raster.array.tolist() == [[[127, 0], [127, 255]]]


def test_rgb_words_keep_interleaved_component_order() -> None:
    image = prepare_image(
        ImageSource(
            internal_words([65535, 0, 32768, 0, 65535, 257]),
            internal_dictionary(2, ColorSpace="DeviceRGB"),
        )
    )
    assert image is not None
    assert image.raster.array.tolist() == [[[255, 0, 128], [0, 255, 1]]]


def test_tiff_predictor_words_reconstruct_before_decode() -> None:
    # Differencing is modulo 65536, per component, not per byte.
    encoded = zlib.compress(internal_words([32768, 1, 32766]))
    image = prepare_image(
        ImageSource(
            encoded,
            internal_dictionary(
                3,
                Filter="FlateDecode",
                DecodeParms={"Predictor": 2, "Colors": 1, "Columns": 3, "BitsPerComponent": 16},
                Mask=[32769, 32769],
            ),
        )
    )
    assert image is not None
    assert image.raster.array.tolist() == [[[128, 255], [128, 0], [255, 255]]]


def test_embedded_icc_receives_uint16_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = bytes(imagecodecs.cms_profile("srgb"))
    actual: list[numpy.ndarray] = []
    original = IccTransform.apply_uint16

    def record(self: IccTransform, samples: numpy.ndarray) -> numpy.ndarray:
        actual.append(samples.copy())
        return original(self, samples)

    monkeypatch.setattr(IccTransform, "apply_uint16", record)
    values = [1000, 33000, 65000, 1001, 33001, 65001]
    image = prepare_image(
        ImageSource(
            internal_words(values),
            internal_dictionary(2, ColorSpace=["ICCBased", PdfStream({"N": 3}, profile)]),
        )
    )
    assert image is not None
    assert len(actual) == 1
    assert actual[0].dtype == numpy.uint16
    assert actual[0].reshape(-1).tolist() == values
    expected = original(
        parse_icc_transform(profile), numpy.asarray(values, dtype=numpy.uint16).reshape(2, 3)
    )
    numpy.testing.assert_array_equal(image.raster.array.reshape(2, 3), expected)


def test_separation_function_receives_unquantized_tints() -> None:
    tint = {"FunctionType": 2, "Domain": [0, 1], "N": 1, "C0": [-32767], "C1": [32768]}
    image = prepare_image(
        ImageSource(
            internal_words([32767, 32768]),
            internal_dictionary(2, ColorSpace=["Separation", "Ink", "DeviceGray", tint]),
        )
    )
    assert image is not None
    assert image.raster.array.tolist() == [[[0], [255]]]


def test_cmyk_device_profile_receives_full_words() -> None:
    values = numpy.asarray([[1000, 30000, 65000, 10000]], dtype=numpy.uint16)
    transform = default_cmyk_transform()
    assert transform is not None
    image = prepare_image(
        ImageSource(values.astype(">u2").tobytes(), internal_dictionary(1, ColorSpace="DeviceCMYK"))
    )
    assert image is not None
    numpy.testing.assert_array_equal(
        image.raster.array.reshape(1, 3), transform.apply_uint16(values)
    )


def test_lab_decode_maps_to_actual_components_before_colour_conversion() -> None:
    white = (0.9642, 1.0, 0.8249)
    image = prepare_image(
        ImageSource(
            internal_words([65535, 32768, 0]),
            internal_dictionary(
                1,
                ColorSpace=["Lab", {"WhitePoint": white, "Range": [-20, 20, -40, 40]}],
                Decode=[0, 75, -10, 10, -20, 20],
            ),
        )
    )
    assert image is not None
    components = numpy.asarray([[75, -10 + 20 * 32768 / 65535, -20]], dtype=numpy.float32)
    expected = numpy.rint(
        numpy.clip(d50_xyz_to_srgb(lab_components_to_xyz(components, white)), 0, 1) * 255
    ).astype(numpy.uint8)
    numpy.testing.assert_array_equal(image.raster.array.reshape(1, 3), expected)


def test_sixteen_bit_soft_mask_keeps_native_resolution_and_decode() -> None:
    image = prepare_image(
        ImageSource(
            internal_words([65535, 65535, 65535, 65535]),
            internal_dictionary(4),
            soft_mask=SoftMask(
                zlib.compress(internal_words([0, 65535])),
                internal_dictionary(2, Filter="FlateDecode", Decode=[1, 0]),
            ),
        )
    )
    assert image is not None
    assert image.soft_mask is not None
    assert image.soft_mask.array.tolist() == [[[255], [0]]]
    assert image.raster.array.tolist() == [[[255, 255], [255, 255], [255, 0], [255, 0]]]


def test_matte_unblending_uses_low_bits_of_soft_mask_before_quantization() -> None:
    image = prepare_image(
        ImageSource(
            internal_words([65534, 32768]),
            internal_dictionary(2),
            soft_mask=SoftMask(internal_words([1, 32767]), internal_dictionary(2, Matte=[1])),
        )
    )
    assert image is not None
    assert image.raster.array.tolist() == [[[0, 0], [0, 127]]]


def test_indexed_matte_is_expressed_in_base_color_space() -> None:
    image = prepare_image(
        ImageSource(
            internal_words([0, 1]),
            internal_dictionary(
                2,
                ColorSpace=["Indexed", "DeviceRGB", 1, bytes([128, 0, 0, 0, 128, 0])],
            ),
            soft_mask=SoftMask(
                bytes([128, 128]), internal_dictionary(2, BitsPerComponent=8, Matte=[0, 0, 0])
            ),
        )
    )
    assert image is not None
    assert image.raster.array.tolist() == [[[255, 0, 0, 128], [0, 255, 0, 128]]]


def test_matte_requires_matching_image_and_soft_mask_dimensions() -> None:
    image = prepare_image(
        ImageSource(
            internal_words([32768, 32768]),
            internal_dictionary(2),
            soft_mask=SoftMask(internal_words([32768]), internal_dictionary(1, Matte=[0])),
        )
    )
    assert image is None


@pytest.mark.parametrize("raw", [b"", b"\xff", b"\xff\xff\x00"])
def test_truncated_words_are_not_reinterpreted_as_eight_bit_images(raw: bytes) -> None:
    assert decode_pdf_image(raw, internal_dictionary(2)) is None


@pytest.mark.parametrize("outer_filter", [False, True])
def test_lossless_jpx_preserves_unsigned_sixteen_bit_samples(outer_filter: bool) -> None:
    original = numpy.asarray([[0, 255, 256, 32768, 65535]], dtype=numpy.uint16)
    compressed = bytes(imagecodecs.jpeg2k_encode(original, reversible=True))
    dictionary = internal_dictionary(5, Filter="JPXDecode", Decode=[1, 0])
    if outer_filter:
        compressed = zlib.compress(compressed)
        dictionary["Filter"] = ["FlateDecode", "JPXDecode"]
    dictionary.pop("BitsPerComponent")
    image = prepare_image(ImageSource(compressed, dictionary))
    assert image is not None
    # JPX ignores Decode and determines its own sample precision.
    assert image.raster.array.tolist() == [[[0], [1], [1], [128], [255]]]


def internal_document(
    image: bytes,
    dictionary: bytes = b"",
    mask: bytes | None = None,
    *,
    version: bytes = b"1.5",
    catalog_version: bytes | None = None,
    content_override: bytes | None = None,
    color_space: bytes = b"DeviceGray",
) -> bytes:
    content = content_override or b"q 2 0 0 1 0 0 cm /Im Do Q"
    bodies = [
        b"<< /Type /Catalog /Pages 2 0 R "
        + (b"/Version /" + catalog_version if catalog_version else b"")
        + b" >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 2 1] "
        b"/Resources << /XObject << /Im 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /XObject /Subtype /Image /Width 2 /Height 1 "
        b"/ColorSpace /"
        + color_space
        + b" /BitsPerComponent 16 "
        + dictionary
        + b" /Length "
        + str(len(image)).encode()
        + b" >>\nstream\n"
        + image
        + b"\nendstream",
    ]
    if mask is not None:
        bodies.append(mask)
    output = b"%PDF-" + version + b"\n"
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


def test_document_render_uses_source_precision_for_color_key_alpha() -> None:
    data = internal_document(internal_words([32768, 32769]), b"/Mask [32768 32768]")
    with PdfDocument(data) as document:
        pixels = document.pages[0].render().rasterize(background=(0, 0, 0, 255)).array()
    assert pixels.tolist() == [[[0, 0, 0, 255], [128, 128, 128, 255]]]


def test_document_inline_image_retains_full_words_and_decode() -> None:
    content = (
        b"q 2 0 0 1 0 0 cm BI /W 2 /H 1 /CS /G /BPC 16 /D [-32767 32768] ID "
        + internal_words([32767, 32768])
        + b" EI Q"
    )
    data = internal_document(internal_words([0, 0]), content_override=content)
    with PdfDocument(data) as document:
        pixels = document.pages[0].render().rasterize(background=(255, 0, 0, 255)).array()
    assert pixels.tolist() == [[[0, 0, 0, 255], [255, 255, 255, 255]]]


def test_document_render_uses_sixteen_bit_soft_mask_and_ignores_colour_key() -> None:
    mask = (
        b"<< /Type /XObject /Subtype /Image /Width 2 /Height 1 "
        b"/ColorSpace /DeviceGray /BitsPerComponent 16 /Length 4 >>\nstream\n"
        + internal_words([0, 65535])
        + b"\nendstream"
    )
    data = internal_document(
        internal_words([65535, 65535]), b"/SMask 6 0 R /Mask [65535 65535]", mask
    )
    with PdfDocument(data) as document:
        pixels = document.pages[0].render().rasterize(background=(0, 0, 0, 255)).array()
    assert pixels.tolist() == [[[0, 0, 0, 255], [255, 255, 255, 255]]]


def test_document_render_unblends_matte_before_compositing() -> None:
    mask = (
        b"<< /Type /XObject /Subtype /Image /Width 2 /Height 1 "
        b"/ColorSpace /DeviceGray /BitsPerComponent 16 /Matte [0] /Length 4 >>\nstream\n"
        + internal_words([32768, 65535])
        + b"\nendstream"
    )
    data = internal_document(internal_words([32768, 65535]), b"/SMask 6 0 R", mask)
    with PdfDocument(data) as document:
        pixels = document.pages[0].render().rasterize(background=(0, 0, 0, 255)).array()
    assert pixels.tolist() == [[[128, 128, 128, 255], [255, 255, 255, 255]]]


@pytest.mark.parametrize(
    ("version", "catalog_version", "expected"),
    [(b"1.7", None, [0, 255]), (b"2.0", None, [255, 0]), (b"1.7", b"2.0", [255, 0])],
)
@pytest.mark.parametrize("mask", [False, True])
def test_document_jpx_decode_uses_effective_version_for_image_and_soft_mask(
    version: bytes, catalog_version: bytes | None, expected: list[int], mask: bool
) -> None:
    compressed = bytes(
        imagecodecs.jpeg2k_encode(numpy.asarray([[0, 65535]], dtype=numpy.uint16), reversible=True)
    )
    dictionary = b"/Filter /JPXDecode /Decode [1 0]"
    mask_object = None
    image = compressed
    if mask:
        mask_object = (
            b"<< /Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceGray "
            + dictionary
            + b" /Length "
            + str(len(compressed)).encode()
            + b" >>\nstream\n"
            + compressed
            + b"\nendstream"
        )
        dictionary = b"/SMask 6 0 R"
        image = internal_words([65535, 65535])
    data = internal_document(
        image, dictionary, mask_object, version=version, catalog_version=catalog_version
    )
    with PdfDocument(data) as document:
        pixels = document.pages[0].render().rasterize(background=(0, 0, 0, 255)).array()
    assert pixels.tolist() == [[[value, value, value, 255] for value in expected]]


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("has_color_space", [False, True])
def test_pdf20_jpx_decode_applies_only_with_color_space(bits: int, has_color_space: bool) -> None:
    values = numpy.asarray(
        [[0, (1 << bits) - 1]], dtype=numpy.uint16 if bits == 16 else numpy.uint8
    )
    compressed = bytes(imagecodecs.jpeg2k_encode(values, reversible=True))
    dictionary = internal_dictionary(2, Filter="JPXDecode", Decode=[1, 0])
    if not has_color_space:
        dictionary.pop("ColorSpace")
    image = prepare_image(
        ImageSource(compressed, dictionary, semantic_context=SemanticContext(PdfVersion(2, 0)))
    )
    assert image is not None
    expected = [255, 0] if has_color_space else [0, 255]
    assert image.raster.array.tolist() == [[[value] for value in expected]]


@pytest.mark.parametrize("version", [None, PdfVersion(1, 8), PdfVersion(2, 1)])
def test_unknown_reader_version_preserves_legacy_jpx_decode(version: PdfVersion | None) -> None:
    compressed = bytes(
        imagecodecs.jpeg2k_encode(numpy.asarray([[0, 65535]], dtype=numpy.uint16), reversible=True)
    )
    image = prepare_image(
        ImageSource(
            compressed,
            internal_dictionary(2, Filter="JPXDecode", Decode=[1, 0]),
            semantic_context=SemanticContext(version),
        )
    )
    assert image is not None
    assert image.raster.array.tolist() == [[[0], [255]]]


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("selector", [0, 1, 2])
def test_jpx_embedded_alpha_selects_ignore_straight_or_premultiplied(
    bits: int, selector: int
) -> None:
    maximum = (1 << bits) - 1
    half = 1 << (bits - 1)
    samples = numpy.asarray(
        [[[half, 0, 0, half], [0, maximum, 0, maximum]]],
        dtype=numpy.uint16 if bits == 16 else numpy.uint8,
    )
    compressed = bytes(imagecodecs.jpeg2k_encode(samples, reversible=True))
    image = prepare_image(
        ImageSource(
            compressed,
            internal_dictionary(
                2, ColorSpace="DeviceRGB", Filter="JPXDecode", SMaskInData=selector
            ),
        )
    )
    assert image is not None
    if selector == 0:
        assert image.raster.array.tolist() == [[[128, 0, 0], [0, 255, 0]]]
    else:
        red = 255 if selector == 2 else 128
        assert image.raster.array.tolist() == [[[red, 0, 0, 128], [0, 255, 0, 255]]]


def test_premultiplied_jpx_unblending_uses_low_bits_of_embedded_alpha() -> None:
    samples = numpy.asarray([[[1, 1], [65535, 65535]]], dtype=numpy.uint16)
    compressed = bytes(imagecodecs.jpeg2k_encode(samples, reversible=True))
    image = prepare_image(
        ImageSource(compressed, internal_dictionary(2, Filter="JPXDecode", SMaskInData=2))
    )
    assert image is not None
    assert image.raster.array.tolist() == [[[255, 0], [255, 255]]]


@pytest.mark.parametrize("selector", [0, 1, 2])
def test_jpx_embedded_soft_mask_takes_precedence_over_color_key(selector: int) -> None:
    samples = numpy.asarray([[[32768, 32768], [65535, 65535]]], dtype=numpy.uint16)
    compressed = bytes(imagecodecs.jpeg2k_encode(samples, reversible=True))
    image = prepare_image(
        ImageSource(
            compressed,
            internal_dictionary(2, Filter="JPXDecode", SMaskInData=selector, Mask=[0, 65535]),
        )
    )
    assert image is not None
    assert image.raster.array[0, :, -1].tolist() == ([0, 0] if selector == 0 else [128, 255])


@pytest.mark.parametrize(("selector", "red"), [(0, 128), (1, 64), (2, 128)])
def test_document_render_uses_jpx_embedded_alpha(selector: int, red: int) -> None:
    samples = numpy.asarray([[[32768, 0, 0, 32768], [0, 65535, 0, 65535]]], dtype=numpy.uint16)
    compressed = bytes(imagecodecs.jpeg2k_encode(samples, reversible=True))
    data = internal_document(
        compressed,
        b"/Filter /JPXDecode /SMaskInData " + str(selector).encode(),
        color_space=b"DeviceRGB",
    )
    with PdfDocument(data) as document:
        pixels = document.pages[0].render().rasterize(background=(0, 0, 0, 255)).array()
    assert pixels.tolist() == [[[red, 0, 0, 255], [0, 255, 0, 255]]]


def test_non_jpx_image_ignores_smask_in_data_selector() -> None:
    image = prepare_image(
        ImageSource(internal_words([32768, 65535]), internal_dictionary(2, SMaskInData=2))
    )
    assert image is not None
    assert image.raster.array.tolist() == [[[128], [255]]]
