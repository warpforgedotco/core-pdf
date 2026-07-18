from core_layout.impl.layout.geometry import RectBox
from core_ocr.impl.schematic import (
    vector_table_symbol_marks_from_drawings,
)

from core_pdf.impl.engine.spec.s_07_content.capture import (
    CapturedDrawing,
    CapturedPath,
    CapturedSubpath,
)


def symbol_drawing(index: int, *, kind: str = "fill") -> CapturedDrawing:
    x0 = 100.0
    y0 = float(index * 20)
    points = [
        (x0, y0),
        (x0 + 3.0, y0 + 5.0),
        (x0 + 5.0, y0 + 2.0),
        (x0 + 7.0, y0 + 7.0),
        (x0 + 12.0, y0 + 10.0),
        (x0 + 2.0, y0 + 8.0),
    ]
    return CapturedDrawing(
        seqno=index,
        fill=(0.0,),
        fill_opacity=1.0,
        kind=kind,
        path=CapturedPath([CapturedSubpath(points, closed=True)]),
        bbox=RectBox(x0, y0, x0 + 12.0, y0 + 10.0),
    )


def test_captured_graphics_detect_aligned_vector_table_symbols() -> None:
    marks = vector_table_symbol_marks_from_drawings(symbol_drawing(index) for index in range(6))

    assert len(marks) == 6
    assert all(mark.token == "✓" for mark in marks)


def test_captured_graphics_reject_stroked_or_insufficient_symbol_candidates() -> None:
    assert not vector_table_symbol_marks_from_drawings(
        symbol_drawing(index, kind="stroke") for index in range(6)
    )
    assert not vector_table_symbol_marks_from_drawings(symbol_drawing(index) for index in range(5))
