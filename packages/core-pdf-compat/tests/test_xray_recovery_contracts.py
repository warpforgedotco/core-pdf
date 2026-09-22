from types import SimpleNamespace
from typing import Any

import pytest

from core_pdf_compat import xray


@pytest.mark.parametrize("rectangle_seq", [0, 1, 2])
@pytest.mark.parametrize("allow_same_fill", [False, True])
@pytest.mark.parametrize("same_fill", [False, True])
def test_occlusion_requires_later_paint_or_explicit_same_color_rule(
    rectangle_seq, allow_same_fill, same_fill
):
    character = xray._Character((0, 0, 10, 10), "A", 1, (0.0,))
    rectangle = xray._Rectangle(
        (0, 0, 10, 10), rectangle_seq, (0.0,) if same_fill else (1.0,), allow_same_fill
    )
    assert xray._occluded(character, rectangle, 0.8) is (
        rectangle_seq > 1 or (allow_same_fill and same_fill)
    )


@pytest.mark.parametrize(("right", "expected"), [(0, False), (8, False), (8.01, True), (10, True)])
def test_occlusion_threshold_is_strictly_greater_than_eighty_percent(right, expected):
    character = xray._Character((0, 0, 10, 10), "A", 1, (0.0,))
    rectangle = xray._Rectangle((0, 0, right, 10), 2, (0.0,))
    assert xray._occluded(character, rectangle, 0.8) is expected


@pytest.mark.parametrize("name", ["F1", "F.test", "F+test"])
def test_raw_font_recovery_finds_unindexed_objects_and_decodes_widths(text_pdf_bytes, name):
    cmap = b"1 beginbfchar <41> <0041> endbfchar"
    suffix = (
        f"\n/{name} 90 0 R\n90 0 obj\n".encode()
        + b"<< /FirstChar 65 /Widths [500] /ToUnicode 91 0 R >>\nendobj\n"
        + f"91 0 obj\n<< /Length {len(cmap)} >>\nstream\n".encode()
        + cmap
        + b"\nendstream\nendobj\n"
    )
    with xray.XrayDocument(text_pdf_bytes + suffix) as document:
        font = xray._recover_font(SimpleNamespace(document=document), name)
        assert font is not None
        assert font.first_char == 65
        assert font.widths == (500.0,)
        assert font.cmap.decode(b"A") == "A"
        assert xray._recover_font(SimpleNamespace(document=document), "absent") is None


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"90 0 obj 42 endobj",
        b"90 0 obj << >> endobj",
        b"90 0 obj << /ToUnicode 91 0 R >> endobj",
        b"90 0 obj << /ToUnicode 91 0 R >> endobj 91 0 obj 42 endobj",
    ],
)
def test_unusable_raw_font_candidates_return_no_recovery(text_pdf_bytes, body):
    with xray.XrayDocument(text_pdf_bytes + b"\n/F1 90 0 R\n" + body) as document:
        assert xray._recover_font(SimpleNamespace(document=document), "F1") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b"<41><42> Tj", {b"A": b"B"}),
        (b"<41><42><43>Tj", {b"A": b"C"}),
        (b"<41> Tj", {}),
        (b"<41><42> TJ", {}),
        (b"<41><42>Tj <41><43>Tj", {b"A": b"C"}),
    ],
)
def test_multioperand_recovery_uses_final_hex_operand(raw, expected):
    assert xray._operand_overrides(raw) == expected


@pytest.mark.parametrize(
    ("pixels", "channels", "expected"),
    [
        (b"", 3, False),
        (b"abc", 0, False),
        (b"abcabc", 3, True),
        (b"abcabd", 3, False),
        (b"abcab", 3, False),
        (b"\0\0", 1, True),
    ],
)
def test_uniform_crop_checks_complete_pixels_and_all_channels(pixels, channels, expected):
    raster: Any = SimpleNamespace(
        rasterize=lambda box: SimpleNamespace(pixels=pixels, channels=channels)
    )
    assert xray._uniform(raster, (0, 0, 1, 1)) is expected


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        (b"%%EOF %PDF-1.7", "trailing"),
        (b"%%EOF 90 0 obj", "trailing"),
        (b"trailer []", "dictionary"),
        (b"trailer << /Root null >>", "root"),
        (b"\xff" * 32, "binary"),
    ],
)
def test_xray_rejects_malformed_top_level_structure(raw, error):
    document: Any = SimpleNamespace(raw_data=raw)
    with pytest.raises(xray.PdfUnsupportedError, match=error):
        xray._validate_mupdf_structure(document)


@pytest.mark.parametrize("raw", [b"trailer << /Root 1 0 R >>", b"%%EOF\n", b"trailer << >>", b""])
def test_xray_structure_checks_accept_valid_or_absent_optional_markers(raw):
    document: Any = SimpleNamespace(raw_data=raw)
    xray._validate_mupdf_structure(document)
