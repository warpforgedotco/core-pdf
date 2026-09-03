# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl.primitives import PdfName, PdfString
from core_pdf.impl.spec.s_07_content.operations import (
    ContentOperands,
    content_stream_may_show_text,
    dispatch_operations,
    iter_content_operations,
)
from core_pdf.impl.spec.s_07_content.operator_tables import (
    OPERATOR_SPECS,
    TEXT_ONLY_SKIP_OPERATORS,
)
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax_primitives.tokens import PDF_CONTENT_OPERATOR_BYTES


def test_filter_operator_vocabulary_matches_content_operator_specs() -> None:
    expected = {name.encode("latin-1") for name in OPERATOR_SPECS} | {b"ID", b"EI"}

    assert expected == PDF_CONTENT_OPERATOR_BYTES


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


@pytest.mark.parametrize(
    ("view_kind", "content", "expected"),
    [
        ("bytes", b"q \t% fake Do\n\r % fake Tj\r\n Q", [("q", ()), ("Q", ())]),
        ("bytes", b"q % fake Do", [("q", ())]),
        ("sliced", b"q \t% fake Do\n\r % fake Tj\r\n Q", [("q", ()), ("Q", ())]),
        ("reversed", b"q % fake Do", [("q", ())]),
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


class internal_DispatchTarget:
    """Minimal target carrying the capture flags used by the dispatcher."""

    def __init__(
        self,
        *,
        capture_graphics: bool = True,
        capture_glyphs: bool = True,
        capture_clipping: bool = True,
    ) -> None:
        self.capture_graphics = capture_graphics
        self.capture_glyphs = capture_glyphs
        self.capture_clipping = capture_clipping


def internal_dispatch_with_handlers(
    data: bytes,
    target: internal_DispatchTarget | None = None,
) -> list[tuple[str, tuple[object, ...], int]]:
    """Dispatch through a complete recording handler map."""
    seen: list[tuple[str, tuple[object, ...], int]] = []
    if target is None:
        target = internal_DispatchTarget()

    def make(name: str):
        def handler(window: ContentOperands, depth: int) -> None:
            seen.append((name, tuple(window), depth))

        return handler

    handlers = {name: make(name) for name in OPERATOR_SPECS}
    dispatch_operations(PdfLexer(data), handlers.get, target, 0)
    return seen


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            b"5 w 1 J 2 j 10 M 1 2 m 3 4 l S f F q Q",
            [
                ("w", (5,), 0),
                ("J", (1,), 0),
                ("j", (2,), 0),
                ("M", (10,), 0),
                ("m", (1, 2), 0),
                ("l", (3, 4), 0),
                ("S", (), 0),
                ("f", (), 0),
                ("F", (), 0),
                ("q", (), 0),
                ("Q", (), 0),
            ],
            id="single-byte-operators",
        ),
        pytest.param(
            b"BT /F1 12 Tf 1 2 TD 5 Tc (hello) Tj [(hi) -20 (there)] TJ ET",
            [
                ("BT", (), 0),
                ("Tf", (PdfName.of("F1"), 12), 0),
                ("TD", (1, 2), 0),
                ("Tc", (5,), 0),
                ("Tj", (PdfString(b"hello"),), 0),
                ("TJ", ([PdfString(b"hi"), -20, PdfString(b"there")],), 0),
                ("ET", (), 0),
            ],
            id="text-operators",
        ),
        pytest.param(
            b"1 0 0 RG 1 0 0 rg 0 0 10 10 re 1 0 0 1 0 0 cm",
            [
                ("RG", (1, 0, 0), 0),
                ("rg", (1, 0, 0), 0),
                ("re", (0, 0, 10, 10), 0),
                ("cm", (1, 0, 0, 1, 0, 0), 0),
            ],
            id="graphics-operators",
        ),
        pytest.param(
            b"Tc 1 TD 1 0 RG 1 0 0 1 0 cm",
            [
                ("Tc", (), 0),
                ("TD", (1,), 0),
                ("RG", (1, 0), 0),
                ("cm", (1, 0, 0, 1, 0), 0),
            ],
            id="incomplete-operands",
        ),
    ],
)
def test_registered_handlers_receive_parsed_operand_windows(
    data: bytes,
    expected: list[tuple[str, tuple[object, ...], int]],
) -> None:
    assert internal_dispatch_with_handlers(data) == expected


# BI opens an inline image and consumes the bytes that follow, so it cannot
# be probed with a synthetic operand-only stream.
@pytest.mark.parametrize("operator", sorted(set(OPERATOR_SPECS) - {"BI"}))
def test_text_only_skip_matches_the_operator_table(operator: str) -> None:
    """Every operator is skipped in text-only mode iff its spec says so.

    This keeps the parser's compact skip tables aligned with OPERATOR_SPECS.
    """
    target = internal_DispatchTarget(
        capture_graphics=False, capture_glyphs=False, capture_clipping=False
    )
    fired = internal_dispatch_with_handlers(
        b"0 0 0 0 0 0 " + operator.encode("latin-1") + b"\n", target
    )
    skipped = not fired
    assert skipped == OPERATOR_SPECS[operator].text_only_skip


def test_synthetic_skip_entry_has_no_operator_spec() -> None:
    """`N` is skipped in text-only mode but is not a real operator.

    It is a damaged-producer no-op with no handler, so it cannot be covered by
    the table-driven test above.
    """
    assert "N" not in OPERATOR_SPECS
    assert b"N" in TEXT_ONLY_SKIP_OPERATORS
