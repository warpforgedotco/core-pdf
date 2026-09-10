"""Core capture retains the same Indexed rounding as direct spec palette lookup."""

from types import SimpleNamespace

import pytest

from core_pdf.impl._impl.capture.interpreter import TextState
from core_pdf.impl._impl.capture.recovery import iter_content_operations
from core_pdf.impl._impl.document.recovery.lexer import PdfLexer
from core_pdf.impl._impl.document.recovery.resolver import ObjectResolver
from core_pdf_spec.s_08_graphics.color import indexed_color_components
from core_pdf_spec.s_08_graphics.color_spec import ColorSpace, parse_color_space
from core_pdf_spec.types import PdfName


@pytest.mark.parametrize(
    ("value", "index"),
    [(-10, 0), (0.49, 0), (0.5, 1), (0.51, 1), (1.5, 2), (2.49, 2), (2.5, 3), (10, 3)],
)
def test_reader_capture_preserves_indexed_quantization(value: float, index: int) -> None:
    resolver = ObjectResolver(b"", {}, {})
    state = TextState(SimpleNamespace(resolver=resolver, resolve=resolver.resolve))
    palette = bytes([0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255])
    state.resources = {
        "ColorSpace": {"Palette": [PdfName.of("Indexed"), PdfName.of("DeviceRGB"), 3, palette]}
    }
    lexer = PdfLexer(f"/Palette cs {value} sc 0 0 1 1 re f".encode("ascii"))
    try:
        for name, operands in iter_content_operations(lexer):
            assert state.execute_operation(name, operands, 0) is None
    finally:
        lexer.close()
    assert state.graphics.fill_color == (float(index),)
    spec = ColorSpace(
        "Indexed", ((0.0, float(3)),), base=parse_color_space("DeviceRGB"), hival=3, lookup=palette
    )
    assert state.drawings[-1].fill == indexed_color_components(spec, value)
