# SPDX-License-Identifier: AGPL-3.0-only
"""Token identity, lexical policy, and strict scalar parsing regressions."""

from __future__ import annotations

import mmap
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.operations import parse_content_token
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax_primitives.numbers import (
    is_integer_token,
    is_number_token,
    parse_identifier_tokens,
    parse_integer_token,
    parse_real_token,
)
from core_pdf_spec.s_07_syntax_primitives.scanning import looks_like_indirect_object_header
from core_pdf_spec.s_07_syntax_primitives.tokens import LexicalRules, lexical_rules
from core_pdf_spec.standards import PdfVersion, SemanticContext
from core_pdf_spec.types import PdfName, PdfReference


@pytest.mark.parametrize(
    "make_buffer",
    [
        bytes,
        bytearray,
        memoryview,
        lambda raw: memoryview(bytearray(raw)),
        lambda raw: memoryview(b"x" + raw + b"x")[1:-1],
        lambda raw: memoryview(b"".join(bytes((byte, 255)) for byte in raw))[::2],
        lambda raw: memoryview(raw).cast("b"),
    ],
)
def test_word_scanning_returns_immutable_tokens_with_exact_positions(
    make_buffer: Callable[[bytes], bytes | bytearray | memoryview],
) -> None:
    lexer = PdfLexer(make_buffer(b" true/name "))
    try:
        assert lexer.scan_word() == (b"true", 5)
        assert lexer.pos == 0
        for position, expected in ((1, b"true"), (5, b"/"), (6, b"name")):
            token = lexer.scan_word_at(position, skip_ignored=False)
            assert token is not None
            assert type(token[0]) is bytes
            assert token[0] == expected
    finally:
        lexer.close()


def test_mapped_source_scanning_releases_borrowed_buffer(tmp_path: Path) -> None:
    path = tmp_path / "tokens.bin"
    path.write_bytes(b"true /Name")
    with path.open("rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:
        lexer = PdfLexer(data)
        try:
            token = lexer.scan_word()
            assert token == (b"true", 4)
            assert type(token[0]) is bytes
        finally:
            lexer.close()


@pytest.mark.parametrize(
    ("whitespace", "data", "expected"),
    [
        (b" ", b"[1 2]", [1, 2]),
        (b" ", b"[1\t2]", None),
        (b"\t\n\f\r |", b"[1|2]", [1, 2]),
        (b"\t\n\f\r ", b"[1\0 2]", None),
        (b"\t\n\f\r \0", b"[1\0 2]", [1, 2]),
        (b"\t\n\f\r \v", b"[1\v2]", [1, 2]),
        (b"\t\n\f\r ", b"[1\v2]", None),
    ],
)
def test_numeric_arrays_obey_custom_whitespace(
    whitespace: bytes, data: bytes, expected: list[int] | None
) -> None:
    class CustomLexer(PdfLexer):
        def select_lexical_rules(self, context: SemanticContext | None) -> LexicalRules:
            return LexicalRules(whitespace, name_escapes=True)

    lexer = CustomLexer(data + b" /Next")
    try:
        if expected is None:
            with pytest.raises(PdfParseError):
                lexer.parse_object()
        else:
            assert lexer.parse_object() == expected
            assert lexer.pos == len(data)
            assert lexer.parse_object() == PdfName.of("Next")
    finally:
        lexer.close()


@pytest.mark.parametrize("separator", [b"+", b"."])
def test_numeric_array_fast_path_respects_numeric_delimiters(separator: bytes) -> None:
    class CustomLexer(PdfLexer):
        def select_lexical_rules(self, context: SemanticContext | None) -> LexicalRules:
            rules = lexical_rules()
            return LexicalRules(
                rules.whitespace, name_escapes=True, delimiters=rules.delimiters + separator
            )

    for data in (separator + b"1", b"[" + separator + b"1]"):
        lexer = CustomLexer(data)
        try:
            with pytest.raises(PdfParseError):
                lexer.parse_object()
        finally:
            lexer.close()


def test_numeric_array_fast_path_respects_numeric_whitespace() -> None:
    class CustomLexer(PdfLexer):
        def select_lexical_rules(self, context: SemanticContext | None) -> LexicalRules:
            return LexicalRules(lexical_rules().whitespace + b"1", name_escapes=True)

    lexer = CustomLexer(b"12 [12] /Next")
    try:
        assert lexer.parse_object() == 2
        assert lexer.parse_object() == [2]
        assert lexer.parse_object() == PdfName.of("Next")
    finally:
        lexer.close()


@pytest.mark.parametrize("version", ["1.0", "1.1", "1.2", "1.3", "1.7", "2.0", None])
@pytest.mark.parametrize("brace", [b"{", b"}"])
def test_braces_follow_document_version(version: str | None, brace: bytes) -> None:
    # Adobe 1.2 4.5; Adobe 1.3 3.1.1; ISO 32000-1/2 7.2.3.
    context = None if version is None else SemanticContext(PdfVersion.parse(version))
    lexer = PdfLexer(b"/A" + brace + b"B", semantic_context=context)
    modern = version in (None, "2.0")
    try:
        assert lexer.parse_object() == PdfName.of(b"A" + brace + b"B" if modern else b"A")
        assert lexer.pos == (4 if modern else 2)
    finally:
        lexer.close()


def test_context_change_updates_brace_token_boundaries() -> None:
    lexer = PdfLexer(b"/A{B /A{B", semantic_context=SemanticContext(PdfVersion(2, 0)))
    try:
        assert lexer.parse_object() == PdfName.of("A{B")
        lexer.semantic_context = SemanticContext(PdfVersion(1, 7))
        assert lexer.parse_object() == PdfName.of("A")
    finally:
        lexer.close()


@pytest.mark.parametrize("version", ["1.2", "1.7", "2.0", None])
@pytest.mark.parametrize("identifier", [b"+1 +0", b"01 00", b"1 -0"])
@pytest.mark.parametrize("indirect", [False, True])
def test_object_identifiers_have_version_specific_grammar(
    version: str | None, identifier: bytes, indirect: bool
) -> None:
    # ISO-approved erratum 379 narrows 32000-2 7.3.10, not ordinary numbers.
    context = None if version is None else SemanticContext(PdfVersion.parse(version))
    lexer = PdfLexer(
        identifier + (b" obj true endobj" if indirect else b" R"), semantic_context=context
    )
    try:
        parse = lexer.parse_indirect_object if indirect else lexer.parse_object
        if version in (None, "2.0"):
            with pytest.raises(PdfParseError, match="identifier"):
                parse()
        else:
            assert parse() == (True if indirect else PdfReference(1))
    finally:
        lexer.close()


@pytest.mark.parametrize("identifier", [b"0 0", b"-1 0", b"1 -1", b"1 65536"])
@pytest.mark.parametrize("version", ["1.7", "2.0"])
def test_invalid_identifier_values_are_rejected(identifier: bytes, version: str) -> None:
    lexer = PdfLexer(
        identifier + b" R", semantic_context=SemanticContext(PdfVersion.parse(version))
    )
    try:
        with pytest.raises(PdfParseError, match="identifier"):
            lexer.parse_object()
    finally:
        lexer.close()


@pytest.mark.parametrize(
    "data", [b"<< /A 1 /A 2 >>", b"<< /A null /#41 2 >>", b"<< /D << /A 1 /A 2 >> >>"]
)
def test_strict_dictionaries_reject_duplicate_decoded_names(data: bytes) -> None:
    # ISO 32000-1/2 7.3.7: duplicate keys compare decoded name identities.
    lexer = PdfLexer(data)
    try:
        with pytest.raises(PdfParseError, match="duplicate dictionary key"):
            lexer.parse_object()
    finally:
        lexer.close()


def test_legacy_literal_name_escapes_are_distinct_keys() -> None:
    lexer = PdfLexer(b"<< /A null /#41 2 >>", semantic_context=SemanticContext(PdfVersion(1, 1)))
    try:
        assert lexer.parse_object() == {"A": None, "#41": 2}
    finally:
        lexer.close()


@pytest.mark.parametrize(
    ("prefix", "suffix"),
    [(b"", b""), (b"[", b"]"), (b"[%x\n", b"]"), (b"[(x) ", b"]"), (b"<< /N ", b" >>")],
)
def test_integer_limits_raise_parse_errors(prefix: bytes, suffix: bytes) -> None:
    limit = sys.get_int_max_str_digits()
    if not limit:
        pytest.skip("Python integer conversion limit is disabled")
    lexer = PdfLexer(prefix + b"9" * (limit + 1) + suffix)
    try:
        with pytest.raises(PdfParseError, match="integer exceeds implementation limits"):
            lexer.parse_object()
    finally:
        lexer.close()


def test_content_integer_limits_raise_parse_errors() -> None:
    limit = sys.get_int_max_str_digits()
    if not limit:
        pytest.skip("Python integer conversion limit is disabled")
    lexer = PdfLexer(b"9" * (limit + 1) + b" w")
    try:
        with pytest.raises(PdfParseError, match="integer exceeds implementation limits"):
            parse_content_token(lexer)
    finally:
        lexer.close()


@pytest.mark.parametrize("token", [b"+01", b"-0", b"123", b"000"])
def test_ordinary_integer_tokens_keep_signs_and_padding(token: bytes) -> None:
    assert is_integer_token(token)
    assert is_number_token(token)
    assert parse_integer_token(token) == int(token)
    assert parse_real_token(token) == float(token)


@pytest.mark.parametrize("token", [b".", b"+", b"1_000", b"1e2", b"1.2.3", b"", b" 1"])
def test_numeric_helpers_reject_non_pdf_spellings(token: bytes) -> None:
    assert not is_integer_token(token)
    assert not is_number_token(token)
    with pytest.raises(PdfParseError):
        parse_integer_token(token)
    with pytest.raises(PdfParseError):
        parse_real_token(token)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"1 0 obj", True),
        (b"1%a\n0%b\nobj/A", True),
        (b"1 0 object", False),
        (b"1 0 objx", False),
        (b"1 0 ob", False),
    ],
)
@pytest.mark.parametrize("strided", [False, True])
def test_header_recognition_uses_comments_and_complete_keywords(
    data: bytes, expected: bool, strided: bool
) -> None:
    view = (
        memoryview(b"".join(bytes((byte, 255)) for byte in data))[::2]
        if strided
        else memoryview(data)
    )
    try:
        assert looks_like_indirect_object_header(view, 0, len(view)) is expected
    finally:
        view.release()


def test_identifier_helper_and_header_recognition_share_legacy_rules() -> None:
    assert parse_identifier_tokens(b"01", b"+0", canonical=False) == (1, 0)
    raw = memoryview(b"01 +0 obj")
    try:
        assert not looks_like_indirect_object_header(raw, 0, len(raw))
        assert looks_like_indirect_object_header(
            raw, 0, len(raw), rules=lexical_rules(SemanticContext(PdfVersion(1, 7)))
        )
    finally:
        raw.release()


def test_dictionary_value_error_leaves_cursor_at_failure() -> None:
    # The strict lexer declines recovery; ``pos`` stays where parsing failed so
    # callers can report or resume from the offending value, not its start.
    lexer = PdfLexer(b"<< /A <GG> >>")
    with pytest.raises(PdfParseError, match="invalid hex string"):
        lexer.parse_object()
    assert lexer.pos == 10

    lexer = PdfLexer(b"<< /A (open >>")
    with pytest.raises(PdfParseError, match="unterminated string"):
        lexer.parse_object()
    assert lexer.pos == len(b"<< /A (open >>")
