# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl.engine.spec.s_07_content.operations import (
    content_stream_may_show_text,
    count_content_stream_operators,
    iter_content_operations,
)
from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer


@pytest.mark.parametrize(
    "data",
    [
        b"(Tj TJ Do ' \\\" (nested Tj))",
        b"<546a 544a 446f>",
        b"% Tj TJ Do ' \\\"\nq Q",
        b"/Tj /TJ /Do /' /\\\" gs",
        b"[Tj TJ Do ' \\\"] q Q",
        b"<< /Text Tj /XObject Do /Quote ' >> q Q",
        b"prefixTjsuffix prefixDosuffix",
        b"BI /F /A85 ID Tj TJ Do ' \\\"~>\nEI Q",
    ],
)
def test_content_stream_may_show_text_ignores_operand_and_image_bytes(data: bytes) -> None:
    assert not content_stream_may_show_text(data)


@pytest.mark.parametrize(
    "data",
    [
        b"(hello) Tj",
        b"[(hello)] TJ",
        b"/XObject Do",
        b"() '",
        b'0 0 () "',
    ],
)
def test_content_stream_may_show_text_accepts_delimited_operators(data: bytes) -> None:
    assert content_stream_may_show_text(data)


def test_content_stream_may_show_text_supports_sliced_memoryview() -> None:
    content = b"(embedded Tj) q Q"
    data = memoryview(b"prefix" + content + b"suffix")[len(b"prefix") : -len(b"suffix")]

    assert not content_stream_may_show_text(data)


def test_content_stream_may_show_text_supports_reversed_memoryview() -> None:
    content = b"(embedded Tj) q Q"

    assert not content_stream_may_show_text(memoryview(content[::-1])[::-1])


def test_content_stream_may_show_text_finds_operator_after_container() -> None:
    assert content_stream_may_show_text(b"[(embedded Tj)] TJ")


def test_content_operations_do_not_treat_vertical_tab_as_whitespace() -> None:
    operations = list(iter_content_operations(PdfLexer(b"Tj\vDo")))

    assert operations == [("Tj\vDo", ())]


def test_content_operations_treat_null_as_pdf_whitespace() -> None:
    operations = list(iter_content_operations(PdfLexer(b"Tj\0Do")))

    assert operations == [("Tj", ()), ("Do", ())]


def test_content_operations_support_reversed_memoryview() -> None:
    content = b"q Q"

    assert list(iter_content_operations(PdfLexer(memoryview(content[::-1])[::-1]))) == [
        ("q", ()),
        ("Q", ()),
    ]


@pytest.mark.parametrize("view_kind", ["bytes", "sliced", "reversed"])
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (b"q \t% fake Do\n\r % fake Tj\r\n Q", [("q", ()), ("Q", ())]),
        (b"q % fake Do", [("q", ())]),
    ],
)
def test_content_operations_skip_comments_after_whitespace(
    view_kind: str,
    content: bytes,
    expected: list[tuple[str, tuple[object, ...]]],
) -> None:
    if view_kind == "sliced":
        data: bytes | memoryview = memoryview(b"prefix" + content + b"suffix")[6:-6]
    elif view_kind == "reversed":
        data = memoryview(content[::-1])[::-1]
    else:
        data = content

    assert list(iter_content_operations(PdfLexer(data))) == expected


def test_content_operator_counts_ignore_operand_and_inline_image_bytes() -> None:
    data = b"(Tj) % Do\nBI /W 1 /H 1 /BPC 8 ID Tj Do S EI q Q"

    counts = count_content_stream_operators(data)

    assert counts.text == 0
    assert counts.image == 1
    assert counts.graphics_state == 2


def test_content_operator_counts_group_text_image_and_vector_operators() -> None:
    counts = count_content_stream_operators(
        b"BT /F1 12 Tf (hello) Tj ET q 0 0 10 10 re f /Im1 Do Q"
    )

    assert counts.text >= 4
    assert counts.image == 1
    assert counts.vector_path == 1
    assert counts.vector_paint == 1
    assert counts.graphics_state == 2
