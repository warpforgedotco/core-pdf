# SPDX-License-Identifier: AGPL-3.0-only
"""Object-stream recovery does not weaken the strict header contract."""

import pytest

from core_pdf.impl._impl.document.recovery.objects import PdfObjectStream as RecoveryObjectStream
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_syntax.objects import PdfObjectStream
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream


@pytest.mark.parametrize(
    ("header", "count", "expected"),
    [
        (b"1 0 2 0 ", 2, {1: 17, 2: 17}),
        (b"0 0 ", 1, {0: 17}),
        (b"1 0 garbage ", 1, {1: 17}),
        (b"1 0 2 3 ", 1, {1: 17}),
    ],
)
def test_malformed_header_is_rejected_in_spec_and_preserved_by_adapter(
    header: bytes, count: int, expected: dict[int, int]
) -> None:
    stream = PdfStream(
        {"Type": "ObjStm", "N": count, "First": len(header)},
        decoded_data=header + b"17 29",
    )
    with pytest.raises(PdfParseError, match="object stream header"):
        PdfObjectStream(stream)
    recovered = RecoveryObjectStream(stream)
    assert {number: recovered.get(number) for number in expected} == expected


@pytest.mark.parametrize("tail", [b" \t\r\n", b" % trailing comment\r\n"])
def test_complete_header_allows_only_ignored_syntax_before_first(tail: bytes) -> None:
    header = b"1 0 2 3" + tail
    stream = PdfStream(
        {"Type": "ObjStm", "N": 2, "First": len(header)},
        decoded_data=header + b"17 29",
    )
    for parser in (PdfObjectStream, RecoveryObjectStream):
        objects = parser(stream)
        assert objects.get(1) == 17
        assert objects.get(2) == 29
