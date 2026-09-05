from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest

from core_pdf import PdfContractError
from core_pdf.impl.spec.s_07_content.capture import (
    CapturedDrawing,
    CapturedInlineImage,
    CapturedLine,
)
from core_pdf.impl.spec.s_07_content.page_program import PageProgram
from core_pdf.impl.spec.s_08_graphics.image_decode import ImageSource
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX


def test_page_program_normalizes_products_and_orders_commands() -> None:
    first = CapturedDrawing(1, None, None, kind="stroke")
    second = CapturedDrawing(2, None, None, kind="fill")
    inline_image = CapturedInlineImage(3, {}, b"", ImageSource(b"", {}), None, IDENTITY_MATRIX, 0)
    program = PageProgram(
        drawings=(second, first),
        inline_images=(inline_image,),
        lines=(CapturedLine(1.0, 2.0, 3.0, 4.0, 0.5),),
    )

    assert program.drawings == (second, first)
    assert [(line.x0, line.y0, line.x1, line.y1, line.line_width) for line in program.lines] == [
        (1.0, 2.0, 3.0, 4.0, 0.5)
    ]
    assert program.commands == (first, second, inline_image)
    assert replace(program, drawings=(second,)).commands == (second, inline_image)


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
    with pytest.raises(PdfContractError, match=message):
        PageProgram(**cast(Any, {field: value}))
