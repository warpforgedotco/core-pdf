import pytest

from core_pdf.impl.document_records import RawFormField
from core_pdf.impl.types import PdfName, PdfString


def field_with_options(*options: object) -> RawFormField:
    return RawFormField("choice", "Ch", None, "", None, {PdfName.of("Opt"): list(options)})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # UTF-16BE with its byte order mark (ISO 32000-2 7.9.2.2).
        ("﻿Café ✓".encode("utf-16-be"), "Café ✓"),
        # PDFDocEncoding: 0x80 is a bullet and 0xE9 is é, not UTF-8.
        (b"\x80 caf\xe9", "• café"),
        (b"plain", "plain"),
    ],
)
def test_options_decode_as_pdf_text_strings(raw: bytes, expected: str) -> None:
    assert field_with_options(PdfString(raw)).options == (expected,)


def test_undecodable_options_keep_the_lenient_reading() -> None:
    # A UTF-16BE mark before an odd number of bytes is not valid UTF-16.
    raw = b"\xfe\xff\x00A\x00"
    assert field_with_options(PdfString(raw)).options == (raw.decode("utf-8", errors="replace"),)


def test_text_options_pass_through_and_other_entries_are_ignored() -> None:
    assert field_with_options("text", 7, [b"export", b"display"]).options == ("text",)
