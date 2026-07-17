# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl.engine.spec.s_07_content.inline_images import (
    parse_inline_image,
    recover_inline_image_position,
)
from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer


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
