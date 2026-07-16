from core_pdf.impl.engine.spec.s_07_content.state import TextState
from core_pdf.impl.objects import PdfStream


def test_distinct_stream_slices_with_equal_lengths_have_distinct_execution_keys() -> None:
    source = memoryview(b"first second")
    first = PdfStream(raw_data=source[:5])
    second = PdfStream(raw_data=source[6:11])

    assert TextState.stream_execution_key(first) != TextState.stream_execution_key(second)
    assert TextState.stream_execution_key(first) == TextState.stream_execution_key(first)
