from __future__ import annotations

from types import SimpleNamespace

import pytest

from core_pdf import PdfContractError
from core_pdf.impl.spec.s_07_content.capture import CapturedDrawing, CapturedLine
from core_pdf.impl.spec.s_07_content.page_program import PageProgram


def test_page_program_filters_inline_image_drawings_and_keeps_lines() -> None:
    drawing = CapturedDrawing(1, None, None, kind="stroke")
    legacy_inline_image = CapturedDrawing(2, None, None, kind="inline-image")
    program = PageProgram.from_state(
        SimpleNamespace(
            runs=(),
            glyphs=(),
            drawings=(drawing, legacy_inline_image),
            inline_images=(),
            lines=(CapturedLine(1.0, 2.0, 3.0, 4.0, 0.5),),
        )
    )

    assert program.drawings == (drawing,)
    assert [(line.x0, line.y0, line.x1, line.y1, line.line_width) for line in program.lines] == [
        (1.0, 2.0, 3.0, 4.0, 0.5)
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("runs", ("numeric strings are not runs",), "text-run"),
        ("glyphs", (object(),), "glyph"),
        ("drawings", (object(),), "drawing"),
        ("inline_images", (object(),), "inline-image"),
        ("lines", (object(),), "line"),
    ],
)
def test_page_program_rejects_untyped_products(
    field: str,
    value: tuple[object, ...],
    message: str,
) -> None:
    state = SimpleNamespace(runs=(), glyphs=(), drawings=(), inline_images=(), lines=())
    setattr(state, field, value)

    with pytest.raises(PdfContractError, match=message):
        PageProgram.from_state(state)
