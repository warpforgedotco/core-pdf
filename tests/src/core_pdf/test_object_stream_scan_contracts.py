"""A compressed dictionary scanned in place reads as its own lexer would read it."""

import pytest

from core_pdf.impl.document.recovery.objects import PdfObjectStream
from core_pdf_spec.s_07_syntax.stream import PdfStream

BODIES = [
    pytest.param([b"<</A 1>>", b"<</B [2 3] /C (x)>>"], id="adjacent"),
    pytest.param([b"<</A 1>>\n \t", b"<</B /N>>\r\n"], id="trailing-whitespace"),
    pytest.param([b"<</A 1>> % note\n", b"<</B 2>>"], id="trailing-comment"),
    pytest.param([b"<</A 1>> 7 ", b"<</B 2>>"], id="trailing-data"),
    pytest.param([b"<</A 1>>stream\nxy", b"<</B 2>>"], id="stream-keyword"),
    pytest.param([b"<</A <</B 1>>strean>>", b"<</C 2>>"], id="nested-stream-misspelling"),
    pytest.param([b"<</A <</B 1", b">> >>", b"<</C 2>>"], id="span-cuts-dictionary"),
    pytest.param([b" <</A 1>>", b"<</B 2>>"], id="leading-whitespace"),
    pytest.param([b"<</A 1", b"<</B 2>>"], id="unterminated"),
    pytest.param([b"[1 2]", b"42", b"(s)"], id="not-dictionaries"),
]


def object_stream(parts: list[bytes]) -> PdfStream:
    offsets, position = [], 0
    for part in parts:
        offsets.append(position)
        position += len(part)
    header = b"".join(b"%d %d " % (10 + index, offset) for index, offset in enumerate(offsets))
    return PdfStream(
        {"Type": "ObjStm", "N": len(parts), "First": len(header)}, header + b"".join(parts)
    )


def outcome(container: PdfObjectStream, number: int) -> str:
    try:
        return repr(container.get(number))
    except Exception as error:  # noqa: BLE001 -- the comparison covers failures too
        return f"raised {type(error).__name__}: {error}"


class PerObjectLexer(PdfObjectStream):
    __slots__ = ()

    def scan_dictionary_at(self, offset: int, end: int) -> None:
        return None


@pytest.mark.parametrize("parts", BODIES)
def test_scanned_dictionaries_match_the_per_object_lexer(parts: list[bytes]) -> None:
    stream = object_stream(parts)
    scanned, parsed = PdfObjectStream(stream), PerObjectLexer(stream)
    for number in scanned.object_numbers:
        assert outcome(scanned, number) == outcome(parsed, number)


def test_only_a_dictionary_ending_its_span_is_scanned() -> None:
    container = PdfObjectStream(
        object_stream([b"<</A 1>>\n", b"<</B 2>> % c\n", b"<</C <</D 1", b">> >>"])
    )
    spans = sorted(container.ends.items())
    assert [container.scan_dictionary_at(start, end) is not None for start, end in spans] == [
        True,
        False,
        False,
        False,
    ]


def test_close_releases_the_body_scanner() -> None:
    container = PdfObjectStream(object_stream([b"<</A 1>>"]))
    container.get(10)
    assert container.body_lexer is not None
    container.close()
    assert container.body_lexer is None
