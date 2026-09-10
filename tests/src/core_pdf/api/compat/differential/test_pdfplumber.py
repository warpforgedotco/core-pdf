from pathlib import Path
from typing import Any

import pytest

from .support import FIXTURES_ROOT, call_pair, differential_pdfs, pdf_id, words

real_pdfplumber = pytest.importorskip("pdfplumber")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("pdf_path", differential_pdfs(), ids=pdf_id)
def test_matches_real_library_on_fixture_corpus(pdf_path: Path) -> None:
    from core_pdf.api.compat import pdfplumber as compat_pdfplumber

    def snapshot(open_pdf: Any) -> tuple[tuple[str, list[dict[str, Any]], float, float], ...]:
        with open_pdf(pdf_path) as pdf:
            return tuple(
                (page.extract_text(), page.extract_words(), page.width, page.height)
                for page in pdf.pages
            )

    pair = call_pair(
        lambda: snapshot(real_pdfplumber.open), lambda: snapshot(compat_pdfplumber.open)
    )
    if pair is not None:
        expected_pages, actual_pages = pair
        assert len(actual_pages) == len(expected_pages)
        for actual, expected in zip(actual_pages, expected_pages, strict=True):
            assert actual[0] == expected[0]
            assert actual[1] == words(expected[1])
            assert actual[2:] == expected[2:]


def internal_text_pdf(
    content: bytes,
    *,
    free_generation: int = 65535,
    rotate_update: bool = False,
    empty_mapping: bool = False,
    broken_previous: bool = False,
) -> bytes:
    page = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 400] "
        b"/Resources << /Font << /F1 5 0 R /FV 6 0 R >> >> /Contents 4 0 R >>"
    )
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        page,
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type0 /BaseFont /HeiseiKakuGo-W5 "
        b"/Encoding /UniJIS-UCS2-V /DescendantFonts [7 0 R] >>",
        b"<< /Type /Font /Subtype /CIDFontType0 /BaseFont /HeiseiKakuGo-W5 "
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (Japan1) /Supplement 2 >> "
        b"/DW 1000 /FontDescriptor << /Type /FontDescriptor /FontName /HeiseiKakuGo-W5 "
        b"/Flags 4 /FontBBox [-1000 -1000 2000 2000] /Ascent 800 /Descent -200 >> >>",
    )
    if empty_mapping:
        cmap = (
            b"begincmap 1 begincodespacerange <00> <ff> endcodespacerange "
            b"2 beginbfchar <41> <> <42> <0042> endbfchar endcmap"
        )
        objects = (
            *objects[:4],
            b"<< /Type /Font /Subtype /Type1 /BaseFont /TestFont /FirstChar 65 /LastChar 66 "
            b"/Widths [600 700] /FontDescriptor << /Type /FontDescriptor /FontName /TestFont "
            b"/FontBBox [0 -200 1000 800] /Descent -200 >> /ToUnicode 8 0 R >>",
            *objects[5:],
            b"<< /Length " + str(len(cmap)).encode() + b" >>\nstream\n" + cmap + b"\nendstream",
        )
    data = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, value in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{number} 0 obj\n".encode() + value + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n0000000000 {free_generation:05d} f \n".encode())
    for offset in offsets:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    if rotate_update:
        page_offset = len(data)
        data.extend(b"3 0 obj\n" + page[:-2] + b" /Rotate 90 >>\nendobj\n")
        latest_xref = len(data)
        data.extend(f"xref\n3 1\n{page_offset:010d} 00000 n \n".encode())
        data.extend(
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Prev {xref} >>\n"
            f"startxref\n{latest_xref}\n%%EOF\n".encode()
        )
    if broken_previous:
        data[xref : xref + 4] = b"oops"
    return bytes(data)


def internal_compare_text_page(
    pdf_bytes: bytes, *, layout_modes: tuple[bool, ...] = (False, True)
) -> None:
    from io import BytesIO

    from core_pdf.api.compat import pdfplumber as compat_pdfplumber

    with (
        real_pdfplumber.open(BytesIO(pdf_bytes)) as expected,
        compat_pdfplumber.open(BytesIO(pdf_bytes)) as actual,
    ):
        expected_page, actual_page = expected.pages[0], actual.pages[0]
        assert (actual_page.width, actual_page.height) == (
            expected_page.width,
            expected_page.height,
        )
        keys = ("text", "x0", "top", "x1", "bottom", "size", "upright", "adv")
        assert [{key: char[key] for key in keys} for char in actual_page.chars] == words(
            [{key: char[key] for key in keys} for char in expected_page.chars]
        )
        assert actual_page.extract_text() == expected_page.extract_text()
        assert actual_page.extract_words() == words(expected_page.extract_words())

    from pdfminer.high_level import extract_pages as reference_pages

    from core_pdf.api.compat.pdfminer import extract_pages

    def layout_snapshot(item: Any, *, approximate: bool = False) -> tuple[Any, ...]:
        box = getattr(item, "bbox", None)
        size = getattr(item, "size", None)
        return (
            type(item).__name__,
            pytest.approx(box, abs=1e-5) if approximate and box is not None else box,
            pytest.approx(size, abs=1e-5) if approximate and size is not None else size,
            item.get_text() if hasattr(item, "get_text") else None,
            tuple(layout_snapshot(child, approximate=approximate) for child in item)
            if hasattr(item, "__iter__")
            else (),
        )

    expected_layout = layout_snapshot(next(reference_pages(BytesIO(pdf_bytes))), approximate=True)
    for unstructured_mode in layout_modes:
        assert (
            layout_snapshot(
                next(extract_pages(BytesIO(pdf_bytes), _unstructured_mode=unstructured_mode))
            )
            == expected_layout
        )


@pytest.mark.parametrize("rise", [-6, 0, 6])
@pytest.mark.parametrize("matrix", ["1 0 0 1", "0.8 0.2 -0.3 1.2", "0 1 -1 0"])
def test_text_rise_geometry_matches_reference(rise: int, matrix: str) -> None:
    content = (
        f"BT /F1 12 Tf {matrix} 100 200 Tm {rise} Ts (Raised) Tj 0 Ts ( normal) Tj ET"
    ).encode()
    internal_compare_text_page(internal_text_pdf(content))


@pytest.mark.parametrize("matrix", ["1 0 0 1", "0.9 0.1 -0.2 0.8"])
@pytest.mark.parametrize("spacing", ["", "2 Tc 80 Tz"])
def test_horizontal_text_resumes_vertical_cursor(matrix: str, spacing: str) -> None:
    content = (
        f"BT /F1 12 Tf {matrix} 100 200 Tm {spacing} (before) Tj "
        "/FV 12 Tf [<3042> 120 <3044>] TJ /F1 12 Tf (after) Tj ET"
    ).encode()
    internal_compare_text_page(internal_text_pdf(content))


@pytest.mark.parametrize("operator", ["'", '0 2 "'])
def test_quote_operator_cursor_matches_reference(operator: str) -> None:
    content = (
        f"BT /F1 12 Tf 20 TL 100 200 Td (first) Tj (second) {operator} T* (third) Tj ET"
    ).encode()
    # A double quote's two spacing operands precede its string operand.
    content = content.replace(b'(second) 0 2 "', b'0 2 (second) "')
    internal_compare_text_page(internal_text_pdf(content))


@pytest.mark.parametrize("free_generation", [65535, 65536])
def test_incremental_rotation_survives_free_list_generation(free_generation: int) -> None:
    internal_compare_text_page(
        internal_text_pdf(
            b"BT /F1 12 Tf 100 200 Td (latest revision) Tj ET",
            free_generation=free_generation,
            rotate_update=True,
        )
    )


def test_mixed_writing_mode_fixture_layout_matches_reference() -> None:
    internal_compare_text_page((FIXTURES_ROOT / "pdfminer.six/samples/simple3.pdf").read_bytes())


@pytest.mark.parametrize(
    ("source_page", "font_name", "content"),
    [
        (1, "CMMI10", b"<3B>"),
        (1, "CMSY10", b"<38395B>"),
        (33, "CMR10", b"<0B>"),
    ],
)
@pytest.mark.parametrize("explicit_encoding", [False, True])
def test_cff_encoding_fallback_matches_reference(
    source_page: int, font_name: str, content: bytes, explicit_encoding: bool
) -> None:
    from io import BytesIO

    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    source = PdfReader(
        FIXTURES_ROOT / "pypdf/sample-files/009-pdflatex-geotopo/GeoTopo-komprimiert.pdf"
    )
    source_resources: Any = source.pages[source_page]["/Resources"]
    source_fonts = source_resources["/Font"]
    writer = PdfWriter()
    font = next(
        value.get_object()
        for value in source_fonts.values()
        if str(value.get_object()["/BaseFont"]).endswith("+" + font_name)
    ).clone(writer)
    if explicit_encoding:
        font[NameObject("/Encoding")] = NameObject("/WinAnsiEncoding")
    page = writer.add_blank_page(width=300, height=400)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font.indirect_reference})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 100 200 Td " + content + b" Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    internal_compare_text_page(output.getvalue())


@pytest.mark.parametrize("matrix", ["1 0 0 1", "0 1 -1 0", "0.9 0.1 -0.2 0.8"])
@pytest.mark.parametrize("text", ["AABB", "BBAA", "BAAB", "AAAA"])
def test_empty_unicode_mapping_keeps_character_geometry(matrix: str, text: str) -> None:
    internal_compare_text_page(
        internal_text_pdf(
            f"BT /F1 12 Tf {matrix} 100 200 Tm ({text}) Tj ET".encode(),
            empty_mapping=True,
        )
    )


@pytest.mark.parametrize("cmap_name", [None, "CustomUnsupportedCMap"])
def test_embedded_cmap_dictionary_name_matches_reference(cmap_name: str | None) -> None:
    from io import BytesIO

    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    source = PdfReader(FIXTURES_ROOT / "PyMuPDF/tests/resources/test_3594.pdf")
    resources: Any = source.pages[0]["/Resources"]
    writer = PdfWriter()
    font = next(iter(resources["/Font"].values())).get_object().clone(writer)
    encoding = font["/Encoding"]
    if cmap_name is None:
        encoding.pop("/CMapName", None)
    else:
        encoding[NameObject("/CMapName")] = NameObject("/" + cmap_name)
    page = writer.add_blank_page(width=300, height=400)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font.indirect_reference})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 100 200 Td (AB) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    # Exercise PDFMiner's standard resource policy here. Unstructured's
    # resource recovery is compared with its own reference by its facade suite.
    internal_compare_text_page(output.getvalue(), layout_modes=(False,))


def test_incremental_rotation_survives_unreadable_older_xref() -> None:
    internal_compare_text_page(
        internal_text_pdf(
            b"BT /F1 12 Tf 100 200 Td (latest revision) Tj ET",
            rotate_update=True,
            broken_previous=True,
        )
    )


@pytest.mark.parametrize("info_alias", ["/Pages", "/Unrelated"])
def test_metadata_page_alias_matches_reference(info_alias: str) -> None:
    from io import BytesIO

    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DictionaryObject, NameObject

    reader = PdfReader(BytesIO(internal_text_pdf(b"BT /F1 12 Tf 100 200 Td (one page) Tj ET")))
    writer = PdfWriter(clone_from=reader)
    if info_alias == "/Pages":
        writer._info = writer._root_object.raw_get("/Pages")
    else:
        writer._info = writer._add_object(
            DictionaryObject({NameObject("/Title"): NameObject("/Example")})
        )
    output = BytesIO()
    writer.write(output)
    from core_pdf.api.compat import pdfplumber as compat_pdfplumber

    with real_pdfplumber.open(BytesIO(output.getvalue())) as expected:
        expected_texts = [page.extract_text() for page in expected.pages]
    with compat_pdfplumber.open(BytesIO(output.getvalue())) as actual:
        assert [page.extract_text() for page in actual.pages] == expected_texts
