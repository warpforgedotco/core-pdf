# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import zlib
from collections.abc import Callable
from typing import Any, cast

import pytest
from pdfminer.ascii85 import ascii85decode as pdfminer_ascii85decode
from pdfminer.ascii85 import asciihexdecode as pdfminer_asciihexdecode
from pdfminer.cmapdb import CMap as PdfMinerCMap
from pdfminer.cmapdb import CMapDB as PdfMinerCMapDB
from pdfminer.layout import LTChar as PdfMinerLTChar
from pdfminer.layout import LTComponent as PdfMinerLTComponent
from pdfminer.layout import LTImage as PdfMinerLTImage
from pdfminer.layout import LTTextLineHorizontal as PdfMinerLTTextLineHorizontal
from pdfminer.pdftypes import PDFStream as PdfMinerPDFStream
from pdfminer.psparser import KWD as PDFMINER_KWD
from pdfminer.psparser import LIT as PDFMINER_LIT
from pdfminer.psparser import PSLiteral as PdfMinerPSLiteral
from pdfminer.psparser import literal_name as pdfminer_literal_name
from pdfminer.utils import decode_text as pdfminer_decode_text

from core_pdf.integrations.pdfminer.ascii85 import ascii85decode, asciihexdecode
from core_pdf.integrations.pdfminer.cmapdb import CMap, CMapDB
from core_pdf.integrations.pdfminer.layout import (
    LTChar,
    LTComponent,
    LTImage,
    LTTextLineHorizontal,
)
from core_pdf.integrations.pdfminer.pdftypes import PDFStream
from core_pdf.integrations.pdfminer.psparser import KWD, LIT, PSLiteral, literal_name
from core_pdf.integrations.pdfminer.utils import decode_text


@pytest.mark.parametrize(
    ("encoded", "decoder", "pdfminer_decoder"),
    [
        (b"<~87cURD_*#TDfTZ)+T~>", ascii85decode, pdfminer_ascii85decode),
        (b"<~z~>", ascii85decode, pdfminer_ascii85decode),
        (b"61 62 2>", asciihexdecode, pdfminer_asciihexdecode),
        (b"61\n62\t63>", asciihexdecode, pdfminer_asciihexdecode),
    ],
)
def test_ascii_decoders_match_pdfminer(
    encoded: bytes,
    decoder: Callable[[bytes], bytes],
    pdfminer_decoder: Callable[[bytes], bytes],
) -> None:
    assert decoder(encoded) == pdfminer_decoder(encoded)


@pytest.mark.parametrize(
    "value",
    [
        b"plain ASCII",
        b"\xfe\xff\x00H\x00e\x00l\x00l\x00o",
        bytes((0x18, 0x1F, 0x80, 0x81, 0x8D, 0x95, 0xA0, 0xFF)),
    ],
)
def test_decode_text_matches_pdf_doc_encoding(value: bytes) -> None:
    assert decode_text(value) == pdfminer_decode_text(value)


def test_literal_and_keyword_contracts_match_pdfminer() -> None:
    core_literal = LIT("Font")
    pdfminer_literal = PDFMINER_LIT("Font")
    core_keyword = KWD(b"Tf")
    pdfminer_keyword = PDFMINER_KWD(b"Tf")

    assert (repr(core_literal), str(core_literal), literal_name(core_literal)) == (
        repr(pdfminer_literal),
        str(pdfminer_literal),
        pdfminer_literal_name(pdfminer_literal),
    )
    assert (repr(core_keyword), str(core_keyword)) == (
        repr(pdfminer_keyword),
        str(pdfminer_keyword),
    )
    assert core_literal is LIT("Font")
    assert pdfminer_literal is PDFMINER_LIT("Font")
    assert PSLiteral("Font") != PSLiteral("Font")
    assert PdfMinerPSLiteral("Font") != PdfMinerPSLiteral("Font")
    assert literal_name(LIT(b"Font")) == pdfminer_literal_name(PDFMINER_LIT(b"Font"))
    assert literal_name(42) == pdfminer_literal_name(42)


@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("Identity-H", b"\x00A\x12\x34\xff"),
        ("Identity-V", b"\x00A\x12\x34\xff"),
        ("OneByteIdentityH", b"A\x80\xff"),
        ("OneByteIdentityV", b"A\x80\xff"),
    ],
)
def test_identity_cmaps_match_pdfminer(name: str, data: bytes) -> None:
    result = CMapDB.get_cmap(name)
    expected = PdfMinerCMapDB.get_cmap(name)

    assert result.is_vertical() == expected.is_vertical()
    assert tuple(result.decode(data)) == tuple(expected.decode(data))


def test_nested_cmap_decoding_matches_pdfminer() -> None:
    result = CMap(CMapName="Custom")
    expected = PdfMinerCMap(CMapName="Custom")
    mapping: dict[int, object] = {0x01: {0x02: 100}, 0x03: 200}
    result.code2cid = mapping
    expected.code2cid = mapping

    assert repr(result) == repr(expected)
    assert list(result.decode(b"\x01\x02\x03\xff\x03")) == list(
        expected.decode(b"\x01\x02\x03\xff\x03")
    )


def _filter_snapshot(stream: Any) -> list[tuple[str, Any]]:
    return [(literal_name(name), params) for name, params in stream.get_filters()]


@pytest.mark.parametrize(
    "attrs",
    [
        {},
        {"Filter": LIT("FlateDecode")},
        {
            "Filter": [LIT("ASCIIHexDecode"), LIT("FlateDecode")],
            "DecodeParms": [{"Columns": 4}, None],
        },
    ],
)
def test_pdfstream_filter_parsing_matches_pdfminer(attrs: dict[str, Any]) -> None:
    pdfminer_attrs = {
        key: [PDFMINER_LIT(literal_name(item)) for item in value]
        if key == "Filter" and isinstance(value, list)
        else PDFMINER_LIT(literal_name(value))
        if key == "Filter"
        else value
        for key, value in attrs.items()
    }
    result = PDFStream(attrs, b"")
    expected = PdfMinerPDFStream(pdfminer_attrs, b"")

    expected_snapshot = [
        (pdfminer_literal_name(name), params) for name, params in expected.get_filters()
    ]
    assert _filter_snapshot(result) == expected_snapshot


@pytest.mark.parametrize(
    ("filter_name", "encoded"),
    [
        ("FlateDecode", zlib.compress(b"decoded payload")),
        ("ASCII85Decode", b"<~FCfN8+EV:.+Cf>-FD5Z2@;]Tu~>"),
        ("ASCIIHexDecode", b"6465636f646564207061796c6f6164>"),
    ],
)
def test_pdfstream_decoding_and_state_match_pdfminer(filter_name: str, encoded: bytes) -> None:
    result = PDFStream({"Filter": LIT(filter_name)}, encoded)
    expected = PdfMinerPDFStream({"Filter": PDFMINER_LIT(filter_name)}, encoded)

    assert result.get_data() == expected.get_data()
    assert (result.rawdata, result.data) == (expected.rawdata, expected.data)
    assert result.get_data() == expected.get_data()


def test_pdfstream_decipher_callback_matches_pdfminer() -> None:
    calls: list[tuple[int, int, bytes, dict[str, Any]]] = []

    def decipher(objid: int, genno: int, data: bytes, attrs: dict[str, Any]) -> bytes:
        calls.append((objid, genno, data, attrs))
        return data[::-1]

    result = PDFStream({}, b"payload", decipher)
    expected = PdfMinerPDFStream({}, b"payload", cast(Any, decipher))
    result.objid = expected.objid = 7
    result.genno = expected.genno = 2

    assert result.get_data() == expected.get_data()
    assert calls == [(7, 2, b"payload", {}), (7, 2, b"payload", {})]


def _component_snapshot(component: Any) -> tuple[Any, ...]:
    return (
        component.bbox,
        component.x0,
        component.y0,
        component.x1,
        component.y1,
        component.width,
        component.height,
        component.is_empty(),
    )


@pytest.mark.parametrize(
    ("left_bbox", "right_bbox"),
    [
        ((0, 0, 10, 10), (5, 5, 15, 15)),
        ((0, 0, 10, 10), (20, 2, 25, 8)),
        ((0, 0, 0, 10), (0, 20, 5, 25)),
    ],
)
def test_layout_component_geometry_matches_pdfminer(
    left_bbox: tuple[float, float, float, float],
    right_bbox: tuple[float, float, float, float],
) -> None:
    left = LTComponent(left_bbox)
    right = LTComponent(right_bbox)
    expected_left = PdfMinerLTComponent(left_bbox)
    expected_right = PdfMinerLTComponent(right_bbox)

    assert _component_snapshot(left) == _component_snapshot(expected_left)
    assert (
        left.is_hoverlap(right),
        left.hdistance(right),
        left.hoverlap(right),
        left.is_voverlap(right),
        left.vdistance(right),
        left.voverlap(right),
    ) == (
        expected_left.is_hoverlap(expected_right),
        expected_left.hdistance(expected_right),
        expected_left.hoverlap(expected_right),
        expected_left.is_voverlap(expected_right),
        expected_left.vdistance(expected_right),
        expected_left.voverlap(expected_right),
    )


def test_text_line_spacing_and_analysis_match_pdfminer() -> None:
    result = LTTextLineHorizontal(word_margin=0.1)
    expected = PdfMinerLTTextLineHorizontal(word_margin=0.1)

    class Font:
        fontname = "Helvetica"

        def is_vertical(self) -> bool:
            return False

        def get_descent(self) -> float:
            return 0

    for text, x in (("A", 0), ("B", 20)):
        arguments = {
            "matrix": (1, 0, 0, 1, x, 0),
            "font": Font(),
            "fontsize": 10,
            "scaling": 1,
            "rise": 0,
            "text": text,
            "textwidth": 0.5,
            "textdisp": 0,
            "ncs": object(),
            "graphicstate": object(),
        }
        result.add(LTChar(**cast(Any, arguments)))
        expected.add(PdfMinerLTChar(**cast(Any, arguments)))

    result.analyze(cast(Any, object()))
    expected.analyze(cast(Any, object()))

    assert result.bbox == expected.bbox
    assert result.get_text() == expected.get_text()
    assert result.get_text() == "A B\n"


def test_image_metadata_matches_pdfminer() -> None:
    attrs = {
        "Width": 640,
        "Height": 480,
        "ImageMask": False,
        "BitsPerComponent": 8,
        "ColorSpace": LIT("DeviceRGB"),
    }
    pdfminer_attrs = {**attrs, "ColorSpace": PDFMINER_LIT("DeviceRGB")}
    result = LTImage("image", PDFStream(attrs, b"data"), (1, 2, 11, 12))
    expected = PdfMinerLTImage("image", PdfMinerPDFStream(pdfminer_attrs, b"data"), (1, 2, 11, 12))

    assert _component_snapshot(result) == _component_snapshot(expected)
    assert (result.name, result.srcsize, result.imagemask, result.bits) == (
        expected.name,
        expected.srcsize,
        expected.imagemask,
        expected.bits,
    )
    assert [literal_name(value) for value in result.colorspace] == [
        pdfminer_literal_name(value) for value in expected.colorspace
    ]
