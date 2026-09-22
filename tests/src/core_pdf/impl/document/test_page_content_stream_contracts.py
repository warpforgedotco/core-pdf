from typing import Any, cast

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.capture.interpreter import TextState
from core_pdf.impl.exceptions import PdfParseError
from core_pdf_spec.s_07_syntax.stream import PdfStream


class RecordingExecutor:
    def __init__(self, failures: set[bytes]) -> None:
        self.failures = failures
        self.calls: list[bytes] = []

    def consume(self, stream: PdfStream, *_: object) -> None:
        self.calls.append(stream.data)
        if stream.data in self.failures:
            raise PdfParseError("injected stream failure")


def make_stream(data: bytes) -> PdfStream:
    return PdfStream(raw_data=data)


def test_content_streams_flatten_nested_arrays_in_source_order(text_pdf_bytes: bytes) -> None:
    with PdfDocument(text_pdf_bytes) as document:
        page = document.pages[0]
        page.contents = [make_stream(b"first"), [object(), make_stream(b"second")]]
        assert [stream.data for stream in page.content_streams] == [b"first", b"second"]


@pytest.mark.parametrize(
    ("streams", "recovered", "failures", "raises", "expected_calls"),
    [
        ([b"single"], False, {b"single"}, True, [b"single"]),
        ([b"single"], True, {b"single"}, False, [b"single"]),
        (
            [b"first", b"second"],
            False,
            {b"first\nsecond"},
            False,
            [b"first\nsecond", b"first", b"second"],
        ),
        (
            [b"first", b"second"],
            False,
            {b"first\nsecond", b"first"},
            False,
            [b"first\nsecond", b"first", b"second"],
        ),
    ],
)
def test_content_consumption_retries_or_skips_only_when_safe(
    text_pdf_bytes: bytes,
    streams: list[bytes],
    recovered: bool,
    failures: set[bytes],
    raises: bool,
    expected_calls: list[bytes],
) -> None:
    with PdfDocument(text_pdf_bytes) as document:
        document.xref_was_recovered = recovered
        page = document.pages[0]
        page.contents = [make_stream(data) for data in streams]
        state = TextState(document)
        executor = RecordingExecutor(failures)
        cast(Any, state).stream_executor = executor
        if raises:
            with pytest.raises(PdfParseError, match="injected"):
                page.consume_contents(state)
        else:
            page.consume_contents(state)
        assert executor.calls == expected_calls
