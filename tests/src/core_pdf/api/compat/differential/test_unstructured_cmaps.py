from pathlib import Path

import pytest

from core_pdf.api.compat.unstructured import partition_pdf

real_pypdf = pytest.importorskip("pypdf")
real_partition = pytest.importorskip("unstructured.partition.pdf").partition_pdf
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("to_unicode", [False, True])
def test_embedded_cmap_preserves_character_advances(tmp_path: Path, to_unicode: bool) -> None:
    generic = real_pypdf.generic
    name = generic.NameObject
    number = generic.NumberObject
    writer = real_pypdf.PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    encoding = generic.DecodedStreamObject()
    encoding[name("/CMapName")] = name("/Example-CMap")
    encoding.set_data(
        b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> def\n"
        b"/CMapName /Example-CMap def /CMapType 1 def /WMode 0 def\n"
        b"1 begincodespacerange <0000> <ffff> endcodespacerange\n"
        b"1 begincidchar <0041> 65 endcidchar\nendcmap end end"
    )
    descendant = generic.DictionaryObject(
        {
            name("/Type"): name("/Font"),
            name("/Subtype"): name("/CIDFontType2"),
            name("/BaseFont"): name("/Example"),
            name("/CIDSystemInfo"): generic.DictionaryObject(
                {
                    name("/Registry"): generic.TextStringObject("Adobe"),
                    name("/Ordering"): generic.TextStringObject("Identity"),
                    name("/Supplement"): number(0),
                }
            ),
            name("/DW"): number(1000),
            name("/W"): generic.ArrayObject([number(65), generic.ArrayObject([number(600)])]),
        }
    )
    font = generic.DictionaryObject(
        {
            name("/Type"): name("/Font"),
            name("/Subtype"): name("/Type0"),
            name("/BaseFont"): name("/Example"),
            name("/Encoding"): writer._add_object(encoding),
            name("/DescendantFonts"): generic.ArrayObject([writer._add_object(descendant)]),
        }
    )
    if to_unicode:
        cmap = generic.DecodedStreamObject()
        cmap.set_data(
            b"1 begincodespacerange <0000> <ffff> endcodespacerange\n"
            b"1 beginbfchar <0041> <0041> endbfchar"
        )
        font[name("/ToUnicode")] = writer._add_object(cmap)
    page[name("/Resources")] = generic.DictionaryObject(
        {name("/Font"): generic.DictionaryObject({name("/F1"): writer._add_object(font)})}
    )
    contents = generic.DecodedStreamObject()
    contents.set_data(b"BT /F1 12 Tf 20 100 Td <00410041> Tj ET")
    page[name("/Contents")] = writer._add_object(contents)
    path = tmp_path / "embedded-cmap.pdf"
    writer.write(path)

    expected = real_partition(filename=str(path), strategy="fast")
    actual = partition_pdf(path)
    assert expected
    assert [(item.category, item.text) for item in actual] == [
        (item.category, item.text) for item in expected
    ]


@pytest.mark.parametrize("switches", [2, 16])
def test_named_cmap_font_switches_keep_unicode_maps_separate(tmp_path: Path, switches: int) -> None:
    g = real_pypdf.generic
    name = g.NameObject
    number = g.NumberObject
    writer = real_pypdf.PdfWriter()
    page = writer.add_blank_page(width=400, height=200)
    fonts = g.DictionaryObject()
    for index, character in enumerate(("A", "B")):
        cmap = g.DecodedStreamObject()
        cmap.set_data(
            (
                "1 begincodespacerange <0000> <ffff> endcodespacerange "
                # PDFMiner applies ToUnicode to the decoded CID, while native
                # PDF text also retains the original character code.
                f"2 beginbfchar <d6d0> <{ord(character):04x}> "
                f"<11cf> <{ord(character):04x}> endbfchar"
            ).encode()
        )
        descendant = g.DictionaryObject(
            {
                name("/Type"): name("/Font"),
                name("/Subtype"): name("/CIDFontType0"),
                name("/BaseFont"): name("/STSong-Light"),
                name("/DW"): number(1000),
                name("/CIDSystemInfo"): g.DictionaryObject(
                    {
                        name("/Registry"): g.TextStringObject("Adobe"),
                        name("/Ordering"): g.TextStringObject("GB1"),
                        name("/Supplement"): number(0),
                    }
                ),
            }
        )
        font = g.DictionaryObject(
            {
                name("/Type"): name("/Font"),
                name("/Subtype"): name("/Type0"),
                name("/BaseFont"): name("/STSong-Light"),
                name("/Encoding"): name("/GBK-EUC-H"),
                name("/ToUnicode"): writer._add_object(cmap),
                name("/DescendantFonts"): g.ArrayObject([writer._add_object(descendant)]),
            }
        )
        fonts[name(f"/F{index}")] = writer._add_object(font)
    page[name("/Resources")] = g.DictionaryObject({name("/Font"): fonts})
    stream = g.DecodedStreamObject()
    stream.set_data(
        (
            "BT 20 100 Td "
            + " ".join(f"/F{index % 2} 12 Tf <d6d0> Tj" for index in range(switches))
            + " ET"
        ).encode()
    )
    page[name("/Contents")] = writer._add_object(stream)
    path = tmp_path / "font-switches.pdf"
    writer.write(path)
    expected = real_partition(filename=str(path), strategy="fast")
    actual = partition_pdf(path)
    assert "".join(item.text for item in expected) == "AB" * (switches // 2)
    assert [(item.category, item.text) for item in actual] == [
        (item.category, item.text) for item in expected
    ]
