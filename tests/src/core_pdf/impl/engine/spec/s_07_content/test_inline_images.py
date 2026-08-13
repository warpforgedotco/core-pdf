# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl.engine.spec.s_07_content.inline_images import (
    parse_inline_image,
    recover_inline_image_position,
)
from core_pdf.impl.engine.spec.s_07_content.operations import validate_inline_images
from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.exceptions import PdfParseError


@pytest.mark.parametrize(
    ("filter_name", "image_data"),
    [
        (b"A85", b"abc EI def~>"),
        (b"AHx", b"12 EI 34>"),
        (b"DCT", b"\xff\xd8binary EI payload\xff\xd9"),
    ],
)
def test_filtered_inline_image_ignores_ei_before_filter_terminator(
    filter_name: bytes,
    image_data: bytes,
) -> None:
    data = b"/F /" + filter_name + b" ID " + image_data + b"\nEI Q"
    lexer = PdfLexer(data)

    image = parse_inline_image(lexer)

    assert image.data == image_data
    assert lexer.raw_data[lexer.pos : lexer.pos + 2] == b" Q"


def test_run_length_inline_image_skips_ei_inside_literal_run() -> None:
    literal = b"ab EI cd"
    image_data = bytes((len(literal) - 1,)) + literal + b"\x80"
    lexer = PdfLexer(b"/F /RL ID " + image_data + b"\nEI Q")

    image = parse_inline_image(lexer)

    assert image.data == image_data
    assert lexer.raw_data[lexer.pos : lexer.pos + 2] == b" Q"


def test_filtered_inline_image_without_terminator_keeps_boundary_fallback() -> None:
    lexer = PdfLexer(b"/F /A85 ID abc\nEI Q")

    image = parse_inline_image(lexer)

    assert image.data == b"abc"


def test_filtered_inline_image_hint_supports_sliced_memoryview() -> None:
    content = b"/F /A85 ID abc EI def~>\nEI Q"
    source = memoryview(b"prefix" + content + b"suffix")[len(b"prefix") : -len(b"suffix")]
    lexer = PdfLexer(source)

    image = parse_inline_image(lexer)

    assert image.data == b"abc EI def~>"
    assert lexer.raw_data[lexer.pos : lexer.pos + 2] == b" Q"


def test_filtered_inline_image_hint_supports_reversed_memoryview() -> None:
    content = b"/F /A85 ID abc EI def~>\nEI Q"
    source = memoryview(content[::-1])[::-1]
    lexer = PdfLexer(source)

    image = parse_inline_image(lexer)

    assert image.data == b"abc EI def~>"
    assert lexer.raw_data[lexer.pos : lexer.pos + 2] == b" Q"


@pytest.mark.parametrize("view_kind", ["bytes", "sliced", "reversed"])
def test_inline_image_search_rejects_undelimited_ei_after_filter_hint(view_kind: str) -> None:
    content = b"/F /A85 ID abc~> EIword\nEI Q"
    if view_kind == "sliced":
        data: bytes | memoryview = memoryview(b"prefix" + content + b"suffix")[6:-6]
    elif view_kind == "reversed":
        data = memoryview(content[::-1])[::-1]
    else:
        data = content
    lexer = PdfLexer(data)

    image = parse_inline_image(lexer)

    assert image.data == b"abc~> EIword"
    assert lexer.raw_data[lexer.pos : lexer.pos + 2] == b" Q"


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


def test_inline_image_validation_rejects_unterminated_data() -> None:
    with pytest.raises(PdfParseError, match="unterminated inline image data"):
        validate_inline_images(b"q BI /W 1 /H 1 /BPC 8 /CS /G ID missing")


def test_inline_image_validation_ignores_tokens_inside_strings() -> None:
    validate_inline_images(b"BT (BI /W 1 ID missing) Tj ET")
