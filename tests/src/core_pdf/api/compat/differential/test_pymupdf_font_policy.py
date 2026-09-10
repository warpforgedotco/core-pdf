from io import BytesIO
from typing import Any, cast

import pytest

from core_pdf._vendor.fontTools.ttLib import TTFont
from core_pdf.api.compat import pymupdf as compat_pymupdf

from .support import FIXTURES_ROOT
from .test_pymupdf_geometry import internal_geometry_expected

real_pymupdf = pytest.importorskip("pymupdf")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("fixed_pitch", [0, 1])
@pytest.mark.parametrize("descriptor_flags", [0, 1])
def test_embedded_true_type_fixed_pitch_uses_font_program(
    fixed_pitch: int, descriptor_flags: int
) -> None:
    with real_pymupdf.open(FIXTURES_ROOT / "PyMuPDF/tests/resources/test_4670.pdf") as document:
        font_xref = document[0].get_fonts()[0][0]
        descendant = int(document.xref_get_key(font_xref, "DescendantFonts")[1][1:].split()[0])
        descriptor = int(document.xref_get_key(descendant, "FontDescriptor")[1].split()[0])
        font_file = int(document.xref_get_key(descriptor, "FontFile2")[1].split()[0])
        font = TTFont(BytesIO(document.xref_stream(font_file)))
        cast(Any, font["post"]).isFixedPitch = fixed_pitch
        output = BytesIO()
        font.save(output)
        document.update_stream(font_file, output.getvalue())
        document.xref_set_key(descriptor, "Flags", str(descriptor_flags))
        source = document.tobytes()
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        assert actual[0].get_text("rawdict") == internal_geometry_expected(
            expected[0].get_text("rawdict")
        )


@pytest.mark.parametrize("fontname", ["helv", "tiro", "cour"])
@pytest.mark.parametrize("first_char", [32, 33])
@pytest.mark.parametrize("missing_width", [0, 100])
@pytest.mark.parametrize("rotation", [0, 90])
def test_explicit_font_metrics_and_missing_glyph_widths(
    fontname: str, first_char: int, missing_width: int, rotation: int
) -> None:
    with real_pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((100, 200), "A B", fontname=fontname, rotate=rotation)
        font = page.get_fonts()[0][0]
        document.xref_set_key(font, "FirstChar", str(first_char))
        document.xref_set_key(font, "LastChar", "65")
        widths = " ".join("0" if code == 32 else "500" for code in range(first_char, 66))
        document.xref_set_key(font, "Widths", f"[{widths}]")
        document.xref_set_key(
            font,
            "FontDescriptor",
            "<< /Type /FontDescriptor /Ascent 718 /Descent -207 "
            "/FontBBox [-166 -225 1000 931] /Flags 32 /ItalicAngle 0 "
            f"/MissingWidth {missing_width} >>",
        )
        source = document.tobytes()
    with (
        real_pymupdf.open(stream=source) as expected,
        compat_pymupdf.open(stream=source) as actual,
    ):
        for kind in ("text", "words", "blocks", "rawdict"):
            assert actual[0].get_text(kind) == internal_geometry_expected(
                expected[0].get_text(kind)
            )


@pytest.mark.parametrize("weight", [b"Bold", b"Demi", b"Book"])
@pytest.mark.parametrize(
    "font_name",
    [
        "AAAAAA+EmbeddedWeight",
        "AAAAAA+LongEmbeddedFontNameForText",
        "CIDFont+F1",
        "abcdef+Name",
        "AB1DEF+Name",
    ],
)
def test_embedded_type1_weight_and_bounded_font_names(weight: bytes, font_name: str) -> None:
    with real_pymupdf.open(FIXTURES_ROOT / "PyMuPDF/tests/resources/cython.pdf") as document:
        page = document[2]
        font = next(font for font in page.get_fonts() if "NimbusRomNo9L-Medi" in font[3])
        descriptor = int(document.xref_get_key(font[0], "FontDescriptor")[1].split()[0])
        font_file = int(document.xref_get_key(descriptor, "FontFile")[1].split()[0])
        data = document.xref_stream(font_file).replace(
            b"/Weight (Bold)", b"/Weight (" + weight + b")"
        )
        document.update_stream(font_file, data)
        document.xref_set_key(font[0], "BaseFont", "/" + font_name)
        document.xref_set_key(descriptor, "FontName", "/" + font_name)
        content = page.get_contents()[0]
        page.set_contents(content)
        document.update_stream(
            content, f"BT /{font[4]} 12 Tf 50 700 Td (Compilation) Tj ET".encode()
        )
        document.select([2])
        source = document.tobytes()
    with (
        real_pymupdf.open(stream=source) as expected,
        compat_pymupdf.open(stream=source) as actual,
    ):
        assert actual[0].get_text("rawdict") == internal_geometry_expected(
            expected[0].get_text("rawdict")
        )


@pytest.mark.parametrize("width", [500.1, 500.49, 500.5, 500.51, 500.9])
def test_fractional_pdf_widths_round_before_cursor_advancement(width: float) -> None:
    with real_pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((100, 200), "AAAA")
        font = page.get_fonts()[0][0]
        document.xref_set_key(font, "FirstChar", "65")
        document.xref_set_key(font, "LastChar", "65")
        document.xref_set_key(font, "Widths", f"[{width}]")
        source = document.tobytes()
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        for kind in ("text", "words", "rawdict"):
            assert actual[0].get_text(kind) == internal_geometry_expected(
                expected[0].get_text(kind)
            )


@pytest.mark.parametrize("horizontal_scale", [50, 102, 160])
@pytest.mark.parametrize("rotation", [0, 90])
def test_short_font_metrics_expand_to_effective_size(horizontal_scale: int, rotation: int) -> None:
    with real_pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((100, 200), "Scaled text", fontsize=12, rotate=rotation)
        font = page.get_fonts()[0][0]
        document.xref_set_key(font, "FontDescriptor", "<< /Ascent 718 /Descent -207 /Flags 32 >>")
        for xref in page.get_contents():
            document.update_stream(
                xref,
                document.xref_stream(xref).replace(b"BT", f"BT {horizontal_scale} Tz".encode()),
            )
        source = document.tobytes()
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        assert actual[0].get_text("rawdict") == internal_geometry_expected(
            expected[0].get_text("rawdict")
        )


@pytest.mark.parametrize(
    "path",
    [
        "PyMuPDF/tests/resources/test_4716.pdf",
        "pdfminer.six/samples/contrib/issue-00352-hash-twos-complement.pdf",
    ],
)
def test_embedded_true_type_font_family_classification(path: str) -> None:
    with (
        real_pymupdf.open(FIXTURES_ROOT / path) as expected,
        compat_pymupdf.open(FIXTURES_ROOT / path) as actual,
    ):
        assert actual[0].get_text("rawdict") == internal_geometry_expected(
            expected[0].get_text("rawdict")
        )


@pytest.mark.parametrize("rise", [1, 2, 4])
@pytest.mark.parametrize("rotation", [0, 90])
def test_superscript_flags_follow_line_baseline(rise: int, rotation: int) -> None:
    with real_pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((100, 200), "A", fontsize=12, rotate=rotation)
        origin = (108, 200 - rise) if rotation == 0 else (100 - rise, 192)
        page.insert_text(origin, "2", fontsize=8, rotate=rotation)
        origin = (116, 200) if rotation == 0 else (100, 184)
        page.insert_text(origin, "B", fontsize=12, rotate=rotation)
        source = document.tobytes()
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        assert actual[0].get_text("rawdict") == internal_geometry_expected(
            expected[0].get_text("rawdict")
        )


@pytest.mark.parametrize("horizontal_scale", [50, 100, 160])
def test_short_embedded_font_ligature_continuation_geometry(horizontal_scale: int) -> None:
    with real_pymupdf.open(FIXTURES_ROOT / "PyMuPDF/tests/resources/test_4751.pdf") as document:
        page = document[0]
        font = next(font for font in page.get_fonts() if "NimbusRomNo9L-Regu" in font[3])
        content = page.get_contents()[0]
        page.set_contents(content)
        document.update_stream(
            content,
            f"BT /{font[4]} 12 Tf {horizontal_scale} Tz 100 500 Td (pro\\002ciency) Tj ET".encode(),
        )
        document.select([0])
        source = document.tobytes()
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        assert actual[0].get_text("rawdict") == internal_geometry_expected(
            expected[0].get_text("rawdict")
        )


@pytest.mark.parametrize("horizontal_scale", [50, 100, 160])
@pytest.mark.parametrize("metric", [0, 100])
def test_equal_ascender_and_descender_use_fallback_metrics(
    horizontal_scale: int, metric: int
) -> None:
    with real_pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((100, 200), "Fallback metrics", fontsize=12)
        font = page.get_fonts()[0][0]
        document.xref_set_key(
            font,
            "FontDescriptor",
            f"<< /Type /FontDescriptor /Ascent {metric} /Descent {metric} "
            "/FontBBox [0 0 1000 1000] /Flags 32 /ItalicAngle 0 >>",
        )
        content = page.get_contents()[0]
        document.update_stream(
            content, f"{horizontal_scale} Tz ".encode() + document.xref_stream(content)
        )
        source = document.tobytes()
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        assert actual[0].get_text("rawdict") == internal_geometry_expected(
            expected[0].get_text("rawdict")
        )


@pytest.mark.parametrize("font_name", ["Courier", "CourierNewPSMT"])
@pytest.mark.parametrize("descriptor_flags", [0, 34, 35])
def test_courier_substitute_font_flags(font_name: str, descriptor_flags: int) -> None:
    with real_pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((100, 200), "Courier text", fontname="cour")
        font = page.get_fonts()[0][0]
        document.xref_set_key(font, "BaseFont", "/" + font_name)
        document.xref_set_key(
            font,
            "FontDescriptor",
            f"<< /Type /FontDescriptor /Ascent 832 /Descent -300 /Flags {descriptor_flags} "
            f"/FontName /{font_name} /ItalicAngle 0 /FontBBox [-21 -680 638 1021] >>",
        )
        source = document.tobytes()
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        assert actual[0].get_text("rawdict") == internal_geometry_expected(
            expected[0].get_text("rawdict")
        )
