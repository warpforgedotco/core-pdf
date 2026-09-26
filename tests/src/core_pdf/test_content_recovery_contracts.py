import pytest

from core_pdf.impl.capture_recovery import (
    CaptureRecovery,
    iter_content_operations,
    recover_inline_image_data,
    recover_inline_image_position,
)
from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.inline_images import InlineImage, InlineImageDataLengthError
from core_pdf_spec.s_07_syntax.lexer import PdfLexer


@pytest.mark.parametrize("view", [False, True])
@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"EI q", 3),
        (b"xEI q", None),
        (b" EIq", None),
        (b" EI", None),
        (b" EI ", 3),
        (b" EI nonsense EI BT", 16),
        (b" EI % ignored\n Q", 15),
        (b"no marker", None),
    ],
)
def test_inline_image_boundary_requires_separators_and_plausible_following_operator(
    data, expected, view
):
    lexer = PdfLexer(memoryview(data) if view else data)
    assert recover_inline_image_position(lexer, 0) == expected


@pytest.mark.parametrize("operator", [b"BT", b"ET", b"q", b"Q", b"cm", b"Do", b"BI"])
def test_inline_image_default_resume_operators_are_recognized(operator):
    data = b"garbage EI " + operator
    assert recover_inline_image_position(PdfLexer(data), 0) == 11


def test_custom_operator_policy_controls_boundary_acceptance():
    data = b" EI q EI custom"
    seen = []

    def accept(word):
        seen.append(word)
        return word == b"custom"

    assert recover_inline_image_position(PdfLexer(data), 0, accept) == 9
    assert seen == [b"q", b"custom"]


@pytest.mark.parametrize(
    ("payload", "length", "expected", "position"),
    [
        (b"abc EI Q", 3, b"abc", 6),
        (b"abcEIQ", 3, b"abc", 5),
        (b"abc EI Q", 20, b"abc", 6),
        (b"abc EI Q", 1, b"abc", 6),
    ],
)
def test_inline_image_length_recovery_preserves_payload_and_advances_past_marker(
    payload, length, expected, position
):
    lexer = PdfLexer(payload)
    dictionary = {}
    image = recover_inline_image_data(lexer, InlineImageDataLengthError(dictionary, 0, length))
    assert image.dictionary is dictionary
    assert image.data == expected
    assert lexer.pos == position


@pytest.mark.parametrize(
    ("kind", "message", "position", "expected"),
    [
        ("array", "unterminated array", 4, 4),
        ("dictionary", "unexpected end of PDF input", 0, 4),
        ("dictionary", "bad dictionary", 4, 4),
        ("token", "bad token", 2, 2),
        ("token", "bad token", 0, None),
        ("inline-image", "unknown failure", 2, None),
        ("inline-image", "unterminated inline image data", 0, 4),
        ("inline-image", "unterminated inline image", 0, None),
    ],
)
def test_resume_policy_distinguishes_eof_progress_and_unrecoverable_errors(
    kind, message, position, expected
):
    lexer = PdfLexer(b"data")
    lexer.pos = position
    assert CaptureRecovery().resume(lexer, PdfParseError(message), kind, 0) == expected


@pytest.mark.parametrize(
    "message",
    [
        "unterminated inline image",
        "unterminated inline image data",
        "inline image keys must be names",
        "expected inline image data separator",
    ],
)
def test_known_inline_image_errors_resume_at_the_next_content_operator(message):
    lexer = PdfLexer(b"bad EI Q")
    assert CaptureRecovery().resume(lexer, PdfParseError(message), "inline-image", 0) == 7


@pytest.mark.parametrize("delimiter", [b")", b">", b"]", b">>"])
def test_stray_delimiters_do_not_discard_pending_operands(delimiter):
    operations = list(iter_content_operations(PdfLexer(b"1 " + delimiter + b" 2 m Q")))
    assert operations == [("m", (1, 2)), ("Q", ())]


@pytest.mark.parametrize("structural", [b"R", b"obj", b"endobj", b"stream", b"endstream"])
def test_structural_tokens_consume_their_operands_without_becoming_operations(structural):
    assert list(iter_content_operations(PdfLexer(b"1 2 " + structural + b" 3 w"))) == [("w", (3,))]


def test_operand_limit_and_yield_cursor_prevent_replay():
    data = b" ".join(str(value).encode() for value in range(20)) + b" m 99 w"
    lexer = PdfLexer(data)
    operations = iter_content_operations(lexer)
    assert next(operations) == ("m", tuple(range(16)))
    assert lexer.pos == data.index(b" m") + 2
    assert list(operations) == [("w", (99,))]


@pytest.mark.parametrize("tail", [b"[1 2", b"<< /Key 1", b"[", b"<<"])
def test_unterminated_container_at_eof_keeps_completed_operations(tail):
    assert list(iter_content_operations(PdfLexer(b"q " + tail))) == [("q", ())]


@pytest.mark.parametrize("word", [b"{", b"}", b"unknown"])
def test_non_pdf_word_tokens_consume_operands_like_other_operators(word):
    assert list(iter_content_operations(PdfLexer(b"1 " + word + b" 2 m"))) == [
        (word.decode(), (1,)),
        ("m", (2,)),
    ]


def test_inline_image_search_at_eof_has_no_resume_position():
    assert recover_inline_image_position(PdfLexer(b"data"), 4) is None


def test_incomplete_array_without_cursor_progress_cannot_resume():
    assert (
        CaptureRecovery().resume(PdfLexer(b"data"), PdfParseError("bad array"), "array", 0) is None
    )


def test_bad_inline_image_discards_its_pending_operands_before_resuming():
    assert list(iter_content_operations(PdfLexer(b"9 BI bad EI Q"))) == [("Q", ())]


def test_known_length_inline_image_recovery_is_used_by_the_operation_iterator():
    operations = list(iter_content_operations(PdfLexer(b"BI /W 1 /H 1 /BPC 8 /CS /G ID xEI Q")))
    assert [name for name, _ in operations] == ["BI", "Q"]
    image = operations[0][1][0]
    assert isinstance(image, InlineImage)
    assert image.data == b"x"
    assert operations[1][1] == ()


def test_recovery_policy_can_reject_a_malformed_token(monkeypatch):
    monkeypatch.setattr(CaptureRecovery, "resume", lambda *args: None)
    with pytest.raises(PdfParseError, match="unterminated array"):
        list(iter_content_operations(PdfLexer(b"[")))


def test_inline_image_consumption_respects_a_full_operand_buffer():
    prefix = b" ".join(str(value).encode() for value in range(16))
    operations = list(
        iter_content_operations(PdfLexer(prefix + b" BI /W 1 /H 1 /BPC 8 /CS /G ID x EI Q"))
    )
    assert operations == [("BI", tuple(range(16))), ("Q", ())]


def parse_operations(data: bytes) -> list[tuple[str, tuple[object, ...]]]:
    return list(iter_content_operations(PdfLexer(data)))


def test_inline_fast_path_matches_token_parser_for_simple_operands() -> None:
    operations = parse_operations(
        b"1 -2.5 /Name true null (str) <41> [1 2] 12345678901234567 Tj Do"
    )
    assert len(operations) == 2
    name, operands = operations[0]
    assert name == "Tj"
    assert operands[:2] == (1, -2.5)
    assert str(operands[2]) == "Name"
    assert operands[3] is True
    assert operands[4] is None
    assert operands[5].data == b"str"  # ty: ignore[unresolved-attribute]
    assert operands[6].data == b"A"  # ty: ignore[unresolved-attribute]
    assert operands[7] == [1, 2]
    assert operands[8] == 12345678901234567
    assert operations[1] == ("Do", ())


def test_inline_fast_path_reuses_interned_names_and_caps_operands() -> None:
    ((_, operands),) = parse_operations(b"/F1 /F1 " + b"1 " * 40 + b"Tf")
    assert operands[0] is operands[1]
    assert len(operands) == 16


def test_inline_fast_path_filters_object_keywords_like_the_token_parser() -> None:
    assert parse_operations(b"1 0 obj 5 0 R endobj q Q stream endstream") == [
        ("q", ()),
        ("Q", ()),
    ]


def test_inline_fast_path_reports_unknown_delimiters_through_recovery() -> None:
    assert parse_operations(b"q >> Q") == [("q", ()), ("Q", ())]
