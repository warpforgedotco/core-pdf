# SPDX-License-Identifier: AGPL-3.0-only
"""Public content recovery remains outside the strict PDF operator parser."""

from types import SimpleNamespace

import pytest

from core_pdf import PdfDocument
from core_pdf.impl._impl.capture.interpreter import TextState
from core_pdf.impl._impl.capture.recovery import (
    iter_content_operations,
    recover_inline_image_position,
)
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_content.operations import iter_content_operations as strict_operations
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf.impl.types import PdfName, PdfReference
from tests.helpers.resolvers import IdentityResolver


def test_inline_image_recovery_accepts_registered_operator() -> None:
    data = b"damaged EI EMC"
    lexer = PdfLexer(data)

    position = recover_inline_image_position(lexer, 0, {b"EMC"}.__contains__)

    assert position == data.index(b"EMC")


def test_inline_image_recovery_supports_sliced_memoryview_with_false_candidates() -> None:
    content = b" EI unknown" * 20 + b" EI EMC"
    source = memoryview(b"prefix" + content + b"suffix")[len(b"prefix") : -len(b"suffix")]
    lexer = PdfLexer(source)

    position = recover_inline_image_position(lexer, 0, {b"EMC"}.__contains__)

    assert position == content.index(b"EMC")


def test_inline_image_recovery_supports_reversed_memoryview() -> None:
    content = b"damaged EI EMC"
    lexer = PdfLexer(memoryview(content[::-1])[::-1])

    position = recover_inline_image_position(lexer, 0, {b"EMC"}.__contains__)

    assert position == content.index(b"EMC")


def test_application_recovery_preserves_truncated_inline_image_behavior() -> None:
    source = b"BI /W 1 /H 1 /BPC 8 /CS /G ID missing"
    with pytest.raises(PdfParseError, match="unterminated inline image data"):
        list(strict_operations(PdfLexer(source)))
    assert list(iter_content_operations(PdfLexer(source))) == []


def test_application_resource_lookup_preserves_invalid_category_fallback() -> None:
    from tests.helpers.pdf_bytes import one_page_pdf

    with PdfDocument(one_page_pdf(b"")) as document:
        state = TextState(document)
        state.resources = {"Font": []}
        assert state.lookup_page_resource("Font", "F1") is None


@pytest.mark.parametrize("content", [b"Tc", b"(bad) Tc", b"J", b"/F1 (bad) Tf", b"[1 /bad] 0 d"])
def test_application_skips_invalid_numeric_operators_and_continues(content: bytes) -> None:
    state = TextState(SimpleNamespace(resolver=IdentityResolver()))
    state.consume_stream(PdfStream(raw_data=content + b" 17 0 m 18 0 l S"), {}, IDENTITY_MATRIX, 0)
    path = state.drawings[0].path
    assert path is not None
    assert path.subpaths[0].points == [(17, 0), (18, 0)]


def test_application_font_resolution_error_still_uses_fallback() -> None:
    class BrokenResolver(IdentityResolver):
        def resolve(self, ref: object) -> object:
            if isinstance(ref, PdfReference):
                raise PdfParseError("broken font object")
            return ref

    state = TextState(SimpleNamespace(resolver=BrokenResolver()))
    state.current_font = "F1"
    state.resources = {"Font": {"F1": PdfReference(7, 0)}}
    assert state.get_decoder().decode_glyphs(b"A")[0].unicode == "A"


def test_application_form_preserves_truncated_matrix_placement() -> None:
    state = TextState(SimpleNamespace(resolver=IdentityResolver()))
    form = PdfStream(
        {"Subtype": PdfName.of("Form"), "Matrix": [1, 0, 0, 1, 10, 20, 99]},
        b"0 0 m 1 1 l S",
    )
    state.consume_stream(
        PdfStream(raw_data=b"/Form Do"), {"XObject": {"Form": form}}, IDENTITY_MATRIX, 0
    )
    path = state.drawings[0].path
    assert path is not None
    assert path.subpaths[0].points == [(10, 20), (11, 21)]


def test_application_pattern_preserves_invalid_matrix_identity_fallback() -> None:
    state = TextState(SimpleNamespace(resolver=IdentityResolver()))
    state.resources = {
        "Pattern": {
            "P1": PdfStream(
                {
                    "PatternType": 1,
                    "PaintType": 1,
                    "BBox": [0, 0, 10, 10],
                    "XStep": 10,
                    "YStep": 10,
                    "Matrix": [1, 2],
                },
                b"0 0 10 10 re f",
            )
        }
    }
    pattern = state.resolve_pattern_color((PdfName.of("P1"),))
    from core_pdf.impl.spec.s_07_content.patterns import TilingPattern

    assert isinstance(pattern, TilingPattern)
    assert pattern.matrix == IDENTITY_MATRIX
