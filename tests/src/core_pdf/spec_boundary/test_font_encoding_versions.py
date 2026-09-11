# SPDX-License-Identifier: AGPL-3.0-only
"""Document context reaches both simple-font glyph selection and decoded text."""

from io import BytesIO
from pathlib import Path
from typing import Any, cast

import pytest

from core_pdf import PdfDocument
from core_pdf.impl._impl.fonts.decoder import FontDecoder
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax_primitives.text_string import decode_pdf_text_string
from core_pdf_spec.standards import PdfVersion, SemanticContext


def internal_pdf(version: str, *, catalog: bytes = b"", differences: bool = False) -> bytes:
    encoding = (
        b"<< /BaseEncoding /WinAnsiEncoding /Differences [128 /Euro 142 /Zcaron 158 /zcaron] >>"
        if differences
        else b"/WinAnsiEncoding"
    )
    content = b"BT /F1 12 Tf 20 60 Td <808e9e> Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R " + catalog + b" >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding " + encoding + b" >>",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
        b"<< /Title (price \xa0) >>",
    ]
    data = f"%PDF-{version}\n".encode()
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(data)
    data += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    data += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets[1:])
    data += f"trailer\n<< /Size {len(offsets)} /Root 1 0 R /Info 6 0 R >>\n".encode()
    return data + f"startxref\n{xref}\n%%EOF\n".encode()


@pytest.mark.parametrize(
    ("version", "catalog", "expected"),
    [
        ("1.0", b"", "•••"),
        ("1.1", b"", "•••"),
        ("1.2", b"", "•••"),
        ("1.3", b"", "€Žž"),
        ("1.7", b"", "€Žž"),
        ("2.0", b"", "€Žž"),
        ("9.0", b"", "€Žž"),
        ("1.2", b"/Version /1.4", "€Žž"),
    ],
)
def test_document_font_encoding_uses_effective_version(
    version: str, catalog: bytes, expected: str
) -> None:
    with PdfDocument(internal_pdf(version, catalog=catalog)) as document:
        page = document.pages[0]
        assert "".join(run.text for run in page.text_diagnostics().runs) == expected
        program = page.get_page_program()
        decoder = program.glyphs[0].font_decoder
        assert isinstance(decoder, FontDecoder)
        assert decoder.semantic_context == document.resolver.semantic_context
        assert tuple(decoder.simple_encoding_glyph_names[code] for code in (0x80, 0x8E, 0x9E)) == (
            ("bullet", "bullet", "bullet") if expected == "•••" else ("Euro", "Zcaron", "zcaron")
        )
        advances = []
        for glyph in program.glyphs:
            assert glyph.baseline is not None
            advances.append(glyph.baseline[2] - glyph.baseline[0])
        assert advances == pytest.approx(
            [4.2, 4.2, 4.2] if expected == "•••" else [0.0, 7.332, 6.0]
        )


@pytest.mark.parametrize("version", ["1.0", "1.1", "1.2"])
def test_legacy_document_explicit_glyph_differences_remain_authoritative(version: str) -> None:
    with PdfDocument(internal_pdf(version, differences=True)) as document:
        assert "".join(run.text for run in document.pages[0].text_diagnostics().runs) == "€Žž"


@pytest.mark.parametrize("version", ["1.0", "1.1", "1.2", "1.3", "2.0", "9.0"])
def test_reader_recovers_underdeclared_pdfdocencoding_euro(version: str) -> None:
    with PdfDocument(internal_pdf(version)) as document:
        info = document.resolver.resolve(document.trailer_dict.get("Info"))
        assert isinstance(info, dict)
        assert document.resolver.resolve_str(cast(PdfDict, info)["Title"]) == "price €"
        if version in {"1.0", "1.1", "1.2"}:
            with pytest.raises(ValueError, match="undefined PDFDocEncoding"):
                decode_pdf_text_string(b"price \xa0", context=document.standards.context)


@pytest.mark.parametrize(
    "context", [None, SemanticContext(None), SemanticContext(PdfVersion(9, 0))]
)
def test_reader_font_decoder_keeps_modern_recovery_without_known_version(
    context: SemanticContext | None,
) -> None:
    decoder = FontDecoder(
        {"Subtype": "Type1", "BaseFont": "Helvetica", "Encoding": "WinAnsiEncoding"},
        semantic_context=context,
    )
    assert "".join(decoder.encoding_decode_table[code] for code in (0x80, 0x8E, 0x9E)) == "€Žž"
    assert tuple(decoder.simple_encoding_glyph_names[code] for code in (0x80, 0x8E, 0x9E)) == (
        "Euro",
        "Zcaron",
        "zcaron",
    )


@pytest.mark.parametrize("version", ["1.2", "1.3"])
def test_facades_preserve_reference_modern_winansi_policy(version: str, tmp_path: Path) -> None:
    # Verified against pypdf, LlamaIndex PDFReader, pdfminer.six and pdfplumber:
    # each reference uses modern WinAnsi assignments under both PDF headers.
    # The native historical glyph selection above does not change that contract.
    from core_pdf.api.compat import pdfminer, pdfplumber, pypdf
    from core_pdf.api.compat.llamaindex import load_data

    data = internal_pdf(version)
    with pypdf.PdfReader(BytesIO(data)) as document:
        assert document.pages[0].extract_text() == "€Žž"
    assert pdfminer.extract_text(BytesIO(data)) == "€Žž\n\n\f"
    # These references also recompute standard-14 advances. Retaining modern
    # text while inheriting native historical bullet widths would still differ.
    expected_geometry = [
        (20.0, 57.516, 20.0, 69.516, 0.0),
        (20.0, 57.516, 27.332, 69.516, 7.332),
        (27.332, 57.516, 33.332, 69.516, 6.0),
    ]
    miner_chars = [
        char
        for page in pdfminer.extract_pages(BytesIO(data))
        for box in page
        if isinstance(box, pdfminer.LTTextBox)
        for line in box
        for char in line
        if isinstance(char, pdfminer.LTChar)
    ]
    assert len(miner_chars) == 3
    for char, expected in zip(miner_chars, expected_geometry, strict=True):
        assert (*char.bbox, char.adv) == pytest.approx(expected)
    with pdfplumber.open(BytesIO(data)) as document:
        assert document.pages[0].extract_text() == "€Žž"
        plumber_chars = document.pages[0].chars
        assert len(plumber_chars) == 3
        for char, expected in zip(plumber_chars, expected_geometry, strict=True):
            assert tuple(char[key] for key in ("x0", "y0", "x1", "y1", "adv")) == pytest.approx(
                expected
            )
    path = tmp_path / "historical-winansi.pdf"
    path.write_bytes(data)
    assert [document.text for document in load_data(path)] == ["€Žž"]


def internal_xray_pdf(
    version: str, *, encoding: str = "winansi", stream: str = "page", widths: bool = True
) -> bytes:
    content = b"BT /F1 12 Tf 20 60 Td <4180> Tj [<8e> 50 <9e>] TJ <42> Tj ET"
    content += b" 0 g 18 54 70 20 re f"
    cmap = (
        b"begincmap 1 begincodespacerange <00> <ff> endcodespacerange "
        b"3 beginbfchar <80> <0058> <8e> <0059> <9e> <005a> endbfchar endcmap"
    )
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding "
    font += (
        b"<< /BaseEncoding /WinAnsiEncoding /Differences [128 /Q 142 /R 158 /S] >>"
        if encoding == "differences"
        else b"/WinAnsiEncoding"
    )
    if encoding == "tounicode":
        font += b" /ToUnicode 8 0 R"
    if widths:
        font += b" /FirstChar 65 /LastChar 158 /Widths [" + b"600 " * 94 + b"]"
    font += b" >>"
    resources = b"<< /Font << /F1 4 0 R >> /XObject << /Fm1 6 0 R >> >>"
    page_content = content if stream == "page" else b"/Fm1 Do" if stream == "form" else b""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Resources "
        + resources
        + b" /Contents 5 0 R "
        + (b"/Annots [7 0 R]" if stream == "appearance" else b"")
        + b" >>",
        font,
        f"<< /Length {len(page_content)} >>\nstream\n".encode() + page_content + b"\nendstream",
        b"<< /Type /XObject /Subtype /Form /BBox [0 0 200 100] /Resources "
        + resources
        + f" /Length {len(content)} >>\nstream\n".encode()
        + content
        + b"\nendstream",
        b"<< /Type /Annot /Subtype /Stamp /Rect [0 0 200 100] /F 4 /AP << /N 6 0 R >> >>",
        f"<< /Length {len(cmap)} >>\nstream\n".encode() + cmap + b"\nendstream",
    ]
    data = f"%PDF-{version}\n".encode()
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(data)
    data += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    data += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets[1:])
    data += f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n".encode()
    return data + f"startxref\n{xref}\n%%EOF\n".encode()


@pytest.fixture
def internal_real_xray(monkeypatch: pytest.MonkeyPatch) -> Any:
    import sys

    real_pymupdf = pytest.importorskip("pymupdf")
    reference = Path(__file__).resolve().parents[3] / "fixtures" / "x-ray"
    monkeypatch.syspath_prepend(str(reference))
    monkeypatch.setitem(sys.modules, "fitz", real_pymupdf)
    for name in tuple(sys.modules):
        if name == "xray" or name.startswith("xray."):
            monkeypatch.delitem(sys.modules, name)
    return pytest.importorskip("xray")


@pytest.mark.parametrize("version", ["1.2", "1.3"])
@pytest.mark.parametrize("stream", ["page", "form", "appearance"])
@pytest.mark.parametrize(
    ("encoding", "expected"),
    [("winansi", "A€ŽžB"), ("differences", "AQRSB"), ("tounicode", "AXYZB")],
)
def test_xray_font_policy_matches_upstream_with_explicit_widths(
    version: str, stream: str, encoding: str, expected: str, internal_real_xray: Any
) -> None:
    import pymupdf

    from core_pdf.api.compat.xray import inspect, internal_XrayDocument

    data = internal_xray_pdf(version, encoding=encoding, stream=stream)
    expected_result = {1: [{"bbox": (18.0, 26.0, 88.0, 46.0), "text": expected}]}
    assert internal_real_xray.inspect(data) == expected_result
    assert inspect(data) == expected_result
    with internal_XrayDocument(data) as document, pymupdf.open(stream=data) as reference:
        assert document.resolver.semantic_context == SemanticContext(PdfVersion.parse(version))
        assert document.internal_font_semantic_context is None
        assert document.pages[0].render().semantic_context == document.resolver.semantic_context
        glyphs = document.pages[0].get_page_program().glyphs
        chars = [char for span in reference[0].get_texttrace() for char in span["chars"]]
        assert len(glyphs) == len(chars) == 5
        for glyph, char in zip(glyphs, chars, strict=True):
            assert glyph.baseline is not None
            # MuPDF's trace box uses its substitute glyph's ink metrics, while
            # the origins include the stated /Widths and the intervening TJ.
            assert glyph.baseline[0] == pytest.approx(char[2][0], abs=1e-5)
            assert glyph.baseline[2] - glyph.baseline[0] == pytest.approx(7.2)
            assert isinstance(glyph.font_decoder, FontDecoder)
            assert glyph.font_decoder.semantic_context is None
    with PdfDocument(data) as native:
        glyphs = native.pages[0].get_page_program().glyphs
        native_expected = "A•••B" if version == "1.2" and encoding == "winansi" else expected
        assert "".join(glyph.text for glyph in glyphs) == native_expected
        assert native.internal_font_semantic_context == native.resolver.semantic_context


@pytest.mark.parametrize("stream", ["page", "form", "appearance"])
def test_xray_legacy_implicit_advances_preserve_existing_modern_policy(stream: str) -> None:
    from core_pdf.api.compat.xray import inspect, internal_XrayDocument

    old = internal_xray_pdf("1.2", stream=stream, widths=False)
    modern = internal_xray_pdf("1.3", stream=stream, widths=False)
    assert inspect(old) == inspect(modern)
    with (
        internal_XrayDocument(old) as old_document,
        internal_XrayDocument(modern) as modern_document,
    ):
        old_glyphs = old_document.pages[0].get_page_program().glyphs
        modern_glyphs = modern_document.pages[0].get_page_program().glyphs
        assert "".join(glyph.text for glyph in old_glyphs) == "A€ŽžB"
        for old_glyph, modern_glyph in zip(old_glyphs, modern_glyphs, strict=True):
            assert old_glyph.baseline == modern_glyph.baseline
            assert old_glyph.ink_bbox == modern_glyph.ink_bbox
        # The existing Standard-14 metrics omit Euro; upstream MuPDF supplies
        # 556 units for Helvetica. This font-context fix preserves that separate
        # metric policy instead of changing native and other facade geometry.
        assert old_glyphs[1].baseline is not None
        assert old_glyphs[1].baseline[0] == old_glyphs[1].baseline[2]
    with PdfDocument(old) as native:
        glyphs = native.pages[0].get_page_program().glyphs
        assert "".join(glyph.text for glyph in glyphs) == "A•••B"
        assert glyphs[1].baseline is not None
        assert glyphs[1].baseline[2] - glyphs[1].baseline[0] == pytest.approx(4.2)
