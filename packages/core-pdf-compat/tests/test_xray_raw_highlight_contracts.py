from types import SimpleNamespace

import pytest

from core_pdf_compat import xray


def source(data: bytes, content: bytes) -> bytes:
    cmap = b"2 beginbfchar <41> <0041> <42> <0042> endbfchar"
    return data.replace(b"BT /F1", b"   /F1") + (
        b"\n/F2 90 0 R\n90 0 obj\n<< /FirstChar 65 /Widths [500 600] /ToUnicode 91 0 R >>\nendobj\n"
        + f"91 0 obj\n<< /Length {len(cmap)} >>\nstream\n".encode()
        + cmap
        + b"\nendstream\nendobj\n"
        + content
    )


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (b"BT /F2 10 Tf 20 100 Td <4142> Tj ET", "AB"),
        (b"BT /F2 10 Tf 20 100 Td <41><42> Tj ET", "B"),
        (b"BT /F2 10 Tf 20 100 Td <004143> Tj ET", "A"),
        (b"BT /F2 10 Tf 20 100 Td <41> Tj /Other 10 Tf <42> Tj ET", "A"),
        (b"BT /F2 10 Tf 20 100 Td <41> Tj 0 -50 Td <42> Tj ET", "A"),
        (b"BT /F2 10 Tf 100 100 Td <4142> Tj ET", ""),
        (b"BT /F2 10 Tf 20 100 Td <43> Tj ET", ""),
        (b"/F2 10 Tf 20 100 Td <41> Tj ET", ""),
        (b"BT /F2 10 Tf 20 100 Td <41> Tj", ""),
    ],
)
def test_raw_highlight_recovery_tracks_font_position_and_width_evidence(
    text_pdf_bytes, content, expected
):
    with xray.XrayDocument(source(text_pdf_bytes, content)) as document:
        page = SimpleNamespace(
            document=document,
            resources={"Font": {"F2": None}},
            get_annotations=lambda: [SimpleNamespace(subtype="Highlight", rect=(20, 98, 40, 108))],
            crop_box=None,
            media_box=(0, 0, 200, 200),
            user_unit=1,
        )
        result = xray._raw_highlight_redactions(page, xray._DocumentRecovery(document))
        assert [item["text"] for item in result] == ([expected] if expected else [])
        if result:
            assert result[0]["bbox"] == pytest.approx((20, 92, 40, 102), abs=0.00001)
            box = result[0]["bbox"]
            assert isinstance(box, tuple)
            assert box[2] > 40


@pytest.mark.parametrize(
    "reason", ["no-highlight", "no-font-resources", "unselected-font", "unrecoverable-font"]
)
def test_raw_highlight_recovery_requires_annotation_and_font_evidence(text_pdf_bytes, reason):
    content = b"BT /F2 10 Tf 20 100 Td <41> Tj ET"
    data = source(text_pdf_bytes, content)
    if reason == "unrecoverable-font":
        data = data.replace(b"/ToUnicode 91 0 R", b"/ToUnicode 99 0 R")
    with xray.XrayDocument(data) as document:
        annotations = (
            []
            if reason == "no-highlight"
            else [SimpleNamespace(subtype="Highlight", rect=(20, 98, 40, 108))]
        )
        resources = {"Font": {"F2": None}}
        if reason == "no-font-resources":
            resources = {}
        elif reason == "unselected-font":
            resources = {"Font": {"Absent": None}}
        page = SimpleNamespace(
            document=document, resources=resources, get_annotations=lambda: annotations
        )
        assert xray._raw_highlight_redactions(page, xray._DocumentRecovery(document)) == []


@pytest.mark.parametrize("unit", [1, 2])
def test_raw_highlight_output_uses_crop_origin_and_user_units(text_pdf_bytes, unit):
    with xray.XrayDocument(
        source(text_pdf_bytes, b"BT /F2 10 Tf 20 100 Td <41> Tj ET")
    ) as document:
        page = SimpleNamespace(
            document=document,
            resources={"Font": {"F2": None}},
            get_annotations=lambda: [SimpleNamespace(subtype="Highlight", rect=(20, 98, 40, 108))],
            crop_box=(10, 20, 180, 150),
            media_box=(0, 0, 200, 200),
            user_unit=unit,
        )
        result = xray._raw_highlight_redactions(page, xray._DocumentRecovery(document))
        assert result[0]["text"] == "A"
        assert result[0]["bbox"] == pytest.approx(
            tuple(value * unit for value in (10, 42, 30, 52)), abs=0.00001
        )
