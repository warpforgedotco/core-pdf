"""JBIG2 arithmetic generic image exports against the reference decoder."""

import struct
from pathlib import Path

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf
from core_pdf.impl.spec.s_07_filters.jbig2.codec import parse_jbig2_file

from .test_pymupdf_geometry import internal_geometry_expected
from .test_pymupdf_text import real_pymupdf

pytestmark = pytest.mark.compat_differential


def internal_segment(
    number: int, kind: int, payload: bytes, *, references: tuple[int, ...] = (), page: int = 1
) -> bytes:
    count = len(references)
    if count <= 4:
        retention = bytes([(count << 5) | ((1 << (count + 1)) - 1)])
    else:
        retention = struct.pack(">I", 0xE0000000 | count) + ((1 << (count + 1)) - 1).to_bytes(
            (count + 8) // 8, "little"
        )
    width = 1 if number <= 256 else 2 if number <= 65536 else 4
    return (
        struct.pack(">IB", number, kind)
        + retention
        + b"".join(reference.to_bytes(width, "big") for reference in references)
        + struct.pack(">BI", page, len(payload))
        + payload
    )


def internal_generic_jbig2(width: int, height: int, prediction: bool) -> bytes:
    info = struct.pack(">IIIIBH", width, height, 72, 72, 0, 0)
    # Any sufficiently long arithmetic code string denotes a bitmap. Keep FF
    # out of the body so only the explicit final marker terminates the coder.
    code = bytes((index * 53 + 31) % 250 for index in range(4096)) + b"\xff\xac"
    region = (
        struct.pack(">IIII BB", width, height, 0, 0, 0, 8 if prediction else 0)
        + bytes([3, 255, 253, 255, 2, 254, 254, 254])
        + code
    )
    return (
        internal_segment(1, 48, info)
        + internal_segment(2, 39, region)
        + internal_segment(3, 49, b"")
    )


def internal_jbig2_pdf(width: int, height: int, data: bytes, *, globals_data: bytes = b"") -> bytes:
    with real_pymupdf.open() as document:
        page = document.new_page(width=100, height=100)
        image = document.get_new_xref()
        document.update_object(
            image,
            f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
            "/ColorSpace /DeviceGray /BitsPerComponent 1 >>",
        )
        document.update_stream(image, data, compress=False)
        document.xref_set_key(image, "Filter", "/JBIG2Decode")
        if globals_data:
            globals_xref = document.get_new_xref()
            document.update_object(globals_xref, "<< >>")
            document.update_stream(globals_xref, globals_data, compress=False)
            document.xref_set_key(image, "DecodeParms", f"<< /JBIG2Globals {globals_xref} 0 R >>")
        document.xref_set_key(page.xref, "Resources", f"<< /XObject << /Im {image} 0 R >> >>")
        content = document.get_new_xref()
        document.update_object(content, "<< >>")
        document.update_stream(content, b"q 80 0 0 80 10 10 cm /Im Do Q")
        page.set_contents(content)
        return document.tobytes()


@pytest.mark.parametrize("width", [1, 7, 8, 9, 17, 63])
@pytest.mark.parametrize("height", [1, 2, 19])
@pytest.mark.parametrize("prediction", [False, True])
def test_jbig2_template_zero_typical_row_prediction(
    width: int, height: int, prediction: bool
) -> None:
    source = internal_jbig2_pdf(width, height, internal_generic_jbig2(width, height, prediction))
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        assert actual[0].get_text("dict") == internal_geometry_expected(
            expected[0].get_text("dict")
        )


def test_jbig2_generic_prediction_region_from_fixture() -> None:
    path = Path("tests/fixtures/PyMuPDF/tests/resources/test_4790.pdf")
    with real_pymupdf.open(path) as fixture:
        xref = fixture[0].get_images()[0][0]
        segments = parse_jbig2_file(fixture.xref_stream_raw(xref))
    # Isolate the generic region from the fixture's separately encoded text
    # region. Its arithmetic bytes remain unchanged on both decoder inputs.
    info = next(segment.data for segment in segments if segment.segment_type == 48)
    region = next(segment.data for segment in segments if segment.segment_type == 39)
    width, height = struct.unpack_from(">II", info)
    data = internal_segment(1, 48, info) + internal_segment(2, 39, region)
    source = internal_jbig2_pdf(width, height, data)
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        assert actual[0].get_text("dict") == internal_geometry_expected(
            expected[0].get_text("dict")
        )


@pytest.fixture(scope="module")
def arithmetic_text_segments() -> tuple[bytes, bytes, bytes]:
    with real_pymupdf.open("tests/fixtures/PyMuPDF/tests/resources/test_4790.pdf") as fixture:
        segments = parse_jbig2_file(fixture.xref_stream_raw(fixture[0].get_images()[0][0]))
    by_kind = {segment.segment_type: segment.data for segment in segments}
    return by_kind[48], by_kind[0], by_kind[7]


@pytest.mark.parametrize("number", [17, 256, 257, 65536, 65537])
@pytest.mark.parametrize("extended", [False, True])
@pytest.mark.parametrize("global_dictionary", [False, True])
def test_jbig2_arithmetic_text_dictionary_references(
    arithmetic_text_segments: tuple[bytes, bytes, bytes],
    number: int,
    extended: bool,
    global_dictionary: bool,
) -> None:
    info, dictionary, text = arithmetic_text_segments
    references = tuple(range(number - 7, number)) if extended else (number - 1,)
    # Empty dictionaries add legal references without changing the symbol ID
    # table. Seven references select the extended count form and occupy one
    # full retention byte on both decoders (the reference rounds partial bytes
    # down, contrary to T.88 7.2.4).
    empty = dictionary[:10] + struct.pack(">II", 0, 0) + b"\xff\xac"
    dictionaries = b"".join(
        internal_segment(
            reference,
            0,
            dictionary if reference == number - 1 else empty,
            page=0 if global_dictionary else 1,
        )
        for reference in references
    )
    data = internal_segment(0, 48, info)
    if not global_dictionary:
        data += dictionaries
    data += internal_segment(number, 7, text, references=references)
    width, height = struct.unpack_from(">II", info)
    source = internal_jbig2_pdf(
        width, height, data, globals_data=dictionaries if global_dictionary else b""
    )
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        assert actual[0].get_text("dict") == internal_geometry_expected(
            expected[0].get_text("dict")
        )


@pytest.mark.parametrize("corner", [0, 1, 2, 3])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("operator", [0, 2])
def test_jbig2_refined_symbol_placement(
    arithmetic_text_segments: tuple[bytes, bytes, bytes],
    corner: int,
    transposed: bool,
    operator: int,
) -> None:
    info, dictionary, text = arithmetic_text_segments
    # Reuse the encoded dimensions, deltas, and all 36 instance refinements.
    # Changing the reference corner / axes affects placement without changing
    # their arithmetic code. A smaller page clips the outer region at byte-
    # unaligned coordinates, separately from symbol clipping within the region.
    info = struct.pack(">II", 513, 641) + info[8:]
    text = (
        struct.pack(
            ">IIiiBH",
            907,
            1259,
            13,
            27,
            0,
            2 | (corner << 4) | (int(transposed) << 6) | (operator << 7),
        )
        + text[19:]
    )
    data = (
        internal_segment(0, 48, info)
        + internal_segment(1, 0, dictionary)
        + internal_segment(2, 7, text, references=(1,))
    )
    source = internal_jbig2_pdf(513, 641, data)
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        assert actual[0].get_text("dict") == internal_geometry_expected(
            expected[0].get_text("dict")
        )


def test_jbig2_complete_symbol_text_and_generic_fixture() -> None:
    path = Path("tests/fixtures/PyMuPDF/tests/resources/test_4790.pdf")
    with real_pymupdf.open(path) as expected, compat_pymupdf.open(path) as actual:
        for actual_page, expected_page in zip(actual, expected, strict=True):
            assert actual_page.get_text("dict") == internal_geometry_expected(
                expected_page.get_text("dict")
            )
