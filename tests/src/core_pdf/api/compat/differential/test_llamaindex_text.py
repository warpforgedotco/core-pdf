from pathlib import Path
from typing import Any

import pytest

from .support import call_pair

real_pypdf = pytest.importorskip("pypdf")
real_reader = pytest.importorskip("llama_index.readers.file").PDFReader
pytestmark = pytest.mark.compat_differential


def internal_text_snapshot(path: Path, facade: str, *, reference: bool) -> object:
    if facade == "llamaindex":
        if reference:
            documents = real_reader().load_data(file=path)
        else:
            from core_pdf.api.compat.llamaindex import load_data

            documents = load_data(path)
        return [(document.text, document.metadata) for document in documents]
    from core_pdf.api.compat import pypdf as compat_pypdf

    reader = real_pypdf.PdfReader if reference else compat_pypdf.PdfReader
    with reader(path) as pdf:
        return [page.extract_text() for page in pdf.pages]


def internal_font_pdf(path: Path, font: Any, contents: bytes) -> None:
    generic = real_pypdf.generic
    name = generic.NameObject
    writer = real_pypdf.PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    other_font = generic.DictionaryObject(
        {
            name("/Type"): name("/Font"),
            name("/Subtype"): name("/Type1"),
            name("/BaseFont"): name("/Times-Roman"),
        }
    )
    page[name("/Resources")] = generic.DictionaryObject(
        {
            name("/Font"): generic.DictionaryObject(
                {name("/F1"): writer._add_object(font), name("/F2"): writer._add_object(other_font)}
            )
        }
    )
    stream = generic.DecodedStreamObject()
    stream.set_data(contents)
    page[name("/Contents")] = writer._add_object(stream)
    writer.write(path)


@pytest.mark.parametrize("facade", ["pypdf", "llamaindex"])
@pytest.mark.parametrize(
    "font_case",
    [
        "differences",
        "symbol",
        "zapf",
        "mapped-metrics",
        "zapf-metrics",
        "courier",
        "type1-program",
        "type3-metrics",
    ],
)
def test_legacy_font_projection_matches_reference(
    tmp_path: Path, facade: str, font_case: str
) -> None:
    generic = real_pypdf.generic
    name = generic.NameObject
    font = generic.DictionaryObject(
        {
            name("/Type"): name("/Font"),
            name("/Subtype"): name("/Type1"),
            name("/BaseFont"): name("/Helvetica"),
        }
    )
    data = b"\x01\x02\x03\x04\x05\x06"
    displacement = b"1"
    following = b" next"
    if font_case == "differences":
        font[name("/Encoding")] = generic.DictionaryObject(
            {
                name("/Differences"): generic.ArrayObject(
                    [
                        generic.NumberObject(1),
                        *[
                            name(glyph)
                            for glyph in ("/a76", "/a77", "/Bullet", "/;", "/cent", "/parenlefttp")
                        ],
                    ]
                )
            }
        )
    elif font_case == "type3-metrics":
        del font[name("/BaseFont")]
        font[name("/Subtype")] = name("/Type3")
        font[name("/FontMatrix")] = generic.ArrayObject(
            [generic.FloatObject(value) for value in (0.01204, 0, 0, 0.01204, 0, 0)]
        )
        font[name("/FontBBox")] = generic.ArrayObject(
            [generic.NumberObject(value) for value in (-2, -2, 86, 60)]
        )
        font[name("/FirstChar")] = generic.NumberObject(0)
        font[name("/LastChar")] = generic.NumberObject(0)
        font[name("/Widths")] = generic.ArrayObject([generic.FloatObject(83.06)])
        font[name("/Encoding")] = generic.DictionaryObject(
            {name("/Differences"): generic.ArrayObject([generic.NumberObject(0), name("/a0")])}
        )
        procedure = generic.DecodedStreamObject()
        procedure.set_data(b"83.06 0 d0")
        font[name("/CharProcs")] = generic.DictionaryObject({name("/a0"): procedure})
        data = b"\0"
    elif font_case == "type1-program":
        program = generic.DecodedStreamObject()
        program.set_data(
            b"%!PS-AdobeFont-1.0: Example 1.0\n/Encoding 256 array\n"
            b"dup 1 /uni20AC put\ndup 2 /a76 put\ndup 3 /unknownGlyph put\neexec\n"
        )
        font[name("/FontDescriptor")] = generic.DictionaryObject(
            {name("/FontFile"): program, name("/Flags"): generic.NumberObject(0)}
        )
        data = b"\x01\x02\x03"
    elif font_case in {"symbol", "zapf"}:
        font[name("/BaseFont")] = name("/Symbol" if font_case == "symbol" else "/ZapfDingbats")
        data = b"qba\xe3" if font_case == "symbol" else b"123ABC"
    elif font_case == "mapped-metrics":
        font[name("/BaseFont")] = name("/Symbol")
        font[name("/Encoding")] = generic.DictionaryObject(
            {
                name("/Differences"): generic.ArrayObject(
                    [generic.NumberObject(2), name("/arrowdblright")]
                )
            }
        )
        cmap = generic.DecodedStreamObject()
        cmap.set_data(
            b"1 begincodespacerange <00> <FF> endcodespacerange\n"
            b"1 beginbfchar <02> <21D2> endbfchar"
        )
        font[name("/ToUnicode")] = cmap
        data = b"\x02"
    else:
        font[name("/BaseFont")] = name(
            "/ZapfDingbats" if font_case == "zapf-metrics" else "/Courier"
        )
        data = b"!" if font_case == "zapf-metrics" else b"\x01"
        displacement = b"0.8" if font_case == "zapf-metrics" else b"0.9"
        following = b"next"
    contents = (
        b"BT /F1 1 Tf 10 0 0 10 10 100 Tm <"
        + data.hex().encode()
        + b"> Tj /F2 1 Tf "
        + displacement
        + b" 0 Td ("
        + following
        + b") Tj ET"
    )
    path = tmp_path / "font.pdf"
    internal_font_pdf(path, font, contents)
    assert internal_text_snapshot(path, facade, reference=False) == internal_text_snapshot(
        path, facade, reference=True
    )


@pytest.mark.parametrize(
    "prefix", [b"", b"Prefix", b"\xef\xbb\xbfPrefix", b"\xfe\xff\x00P\x00r\x00e"]
)
@pytest.mark.parametrize("numbered", [False, True])
def test_page_label_prefixes_match_reference(tmp_path: Path, prefix: bytes, numbered: bool) -> None:
    generic = real_pypdf.generic
    name = generic.NameObject
    writer = real_pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    label = generic.DictionaryObject({name("/P"): generic.ByteStringObject(prefix)})
    if numbered:
        label[name("/S")] = name("/r")
    writer.root_object[name("/PageLabels")] = generic.DictionaryObject(
        {name("/Nums"): generic.ArrayObject([generic.NumberObject(0), label])}
    )
    path = tmp_path / "labels.pdf"
    writer.write(path)
    assert internal_text_snapshot(path, "llamaindex", reference=False) == internal_text_snapshot(
        path, "llamaindex", reference=True
    )


@pytest.mark.parametrize("facade", ["pypdf", "llamaindex"])
@pytest.mark.parametrize("use_indirect_width", [False, True])
def test_indirect_cid_width_rejection_matches_reference(
    tmp_path: Path, facade: str, use_indirect_width: bool
) -> None:
    generic = real_pypdf.generic
    name = generic.NameObject
    writer = real_pypdf.PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    width = writer._add_object(generic.NumberObject(1000))
    descendant = generic.DictionaryObject(
        {
            name("/Type"): name("/Font"),
            name("/Subtype"): name("/CIDFontType2"),
            name("/BaseFont"): name("/Example"),
            name("/DW"): generic.NumberObject(1000),
            name("/W"): generic.ArrayObject(
                [generic.NumberObject(65), generic.ArrayObject([width])]
            ),
        }
    )
    font = generic.DictionaryObject(
        {
            name("/Type"): name("/Font"),
            name("/Subtype"): name("/Type0"),
            name("/BaseFont"): name("/Example"),
            name("/Encoding"): name("/Identity-H"),
            name("/DescendantFonts"): generic.ArrayObject([writer._add_object(descendant)]),
        }
    )
    page[name("/Resources")] = generic.DictionaryObject(
        {name("/Font"): generic.DictionaryObject({name("/F1"): writer._add_object(font)})}
    )
    stream = generic.DecodedStreamObject()
    stream.set_data(
        b"BT /F1 12 Tf 10 100 Td <" + (b"0041" if use_indirect_width else b"0042") + b"> Tj ET"
    )
    page[name("/Contents")] = writer._add_object(stream)
    path = tmp_path / "indirect-width.pdf"
    writer.write(path)
    snapshots = call_pair(
        lambda: internal_text_snapshot(path, facade, reference=True),
        lambda: internal_text_snapshot(path, facade, reference=False),
    )
    if use_indirect_width:
        assert snapshots is None
    else:
        assert snapshots is not None
        assert snapshots[0] == snapshots[1]


@pytest.mark.parametrize("facade", ["pypdf", "llamaindex"])
@pytest.mark.parametrize(
    "base_font",
    [
        "Arial",
        "Arial,Bold",
        "Arial,Italic",
        "Arial,BoldItalic",
        "TimesNewRoman",
        "TimesNewRoman,Bold",
        "TimesNewRoman,Italic",
        "TimesNewRoman,BoldItalic",
    ],
)
def test_standard_font_aliases_preserve_spacing(
    tmp_path: Path, facade: str, base_font: str
) -> None:
    g = real_pypdf.generic
    name = g.NameObject
    font = g.DictionaryObject(
        {
            name("/Type"): name("/Font"),
            name("/Subtype"): name("/TrueType"),
            name("/BaseFont"): name("/" + base_font),
            name("/Encoding"): name("/WinAnsiEncoding"),
        }
    )
    path = tmp_path / "font-alias.pdf"
    internal_font_pdf(
        path,
        font,
        b"BT /F1 12 Tf 20 100 Td [(A) -120 (B) -150 (C)] TJ /F2 12 Tf 60 0 Td (Next) Tj ET",
    )
    assert internal_text_snapshot(path, facade, reference=False) == internal_text_snapshot(
        path, facade, reference=True
    )
