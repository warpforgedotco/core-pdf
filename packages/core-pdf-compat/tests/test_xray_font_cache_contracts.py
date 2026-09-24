from types import SimpleNamespace

import pytest

from core_pdf.impl.capture.program import CapturedProgram, PageProgram
from core_pdf.impl.capture.records import CapturedPath, CapturedSubpath
from core_pdf.impl.fonts.cmap_tounicode import ToUnicodeCMap
from core_pdf.impl.glyphs import GlyphObservation
from core_pdf_compat import xray


@pytest.mark.parametrize("recovered", [False, True])
@pytest.mark.parametrize("route", ["control", "override", "mixed"])
def test_font_recovery_cache_is_shared_across_glyphs_and_routes(monkeypatch, recovered, route):
    font = (
        xray._RecoveredFont(
            ToUnicodeCMap(b"2 beginbfchar <41> <0041> <42> <0042> endbfchar"), 65, (500, 500)
        )
        if recovered
        else None
    )
    calls = []

    def recover(page, name):
        calls.append(name)
        return font

    monkeypatch.setattr(xray, "_recover_font", recover)
    glyphs = tuple(
        GlyphObservation(
            "\x01",
            (index * 5, 0, index * 5 + 5, 10),
            (index * 5, 0, index * 5 + 5, 10),
            index,
            font_name="F1",
            char_code=65 + index % 2,
            code_bytes=bytes([65 + index % 2]),
            font_size=10,
        )
        for index in range(4)
    )
    program = PageProgram(CapturedProgram(glyphs=glyphs))
    path = CapturedPath([CapturedSubpath([(0, 0), (40, 0), (40, 20), (0, 20)], closed=True)])
    drawing = SimpleNamespace(
        kind="fill", fill=(0.0,), fill_opacity=1, path=path, seqno=10, rect=(0, 0, 40, 20)
    )
    page = SimpleNamespace(
        crop_box=(0, 0, 100, 100),
        media_box=(0, 0, 100, 100),
        user_unit=1,
        rotation=90,
        height=100,
        get_page_program=lambda: program,
        drawing_records=lambda drawings: [drawing],
        get_annotations=list,
    )
    overrides = (
        {}
        if route == "control"
        else {b"A": b"AB", **({b"B": b"BA"} if route == "override" else {})}
    )
    result = xray._page_redactions(page, {"overrides": overrides})
    assert calls == ["F1"]
    if recovered:
        assert result == [
            {
                "bbox": (0, 80, 40, 100),
                "text": {"control": "ABAB", "override": "ABBAABBA", "mixed": "ABBABB"}[route],
            }
        ]
    else:
        assert result == []
