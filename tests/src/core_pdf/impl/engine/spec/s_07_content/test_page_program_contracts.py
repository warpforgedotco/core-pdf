from __future__ import annotations

from types import SimpleNamespace

import pytest

from core_pdf import PdfContractError
from core_pdf.impl.engine.spec.s_07_content.page_program import PageProducts


def test_page_program_products_accept_empty_typed_state() -> None:
    products = PageProducts.from_state(
        SimpleNamespace(
            runs=(),
            glyphs=(),
            drawings=(),
            inline_images=(),
            lines=(),
        )
    )

    assert products.runs == ()
    assert products.glyphs == ()
    assert products.drawings == ()
    assert products.inline_images == ()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("runs", ("numeric strings are not runs",), "text-run"),
        ("glyphs", (object(),), "glyph"),
        ("drawings", (object(),), "drawing"),
        ("inline_images", (object(),), "inline-image"),
    ],
)
def test_page_products_rejects_untyped_products(
    field: str,
    value: tuple[object, ...],
    message: str,
) -> None:
    state = SimpleNamespace(runs=(), glyphs=(), drawings=(), inline_images=(), lines=())
    setattr(state, field, value)

    with pytest.raises(PdfContractError, match=message):
        PageProducts.from_state(state)
