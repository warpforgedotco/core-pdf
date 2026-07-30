from types import SimpleNamespace
from typing import Any, cast

from core_pdf.impl.engine.spec.s_07_content.state import TextState
from core_pdf.impl.engine.spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf.impl.objects import PdfStream


def test_distinct_stream_slices_with_equal_lengths_have_distinct_execution_keys() -> None:
    source = memoryview(b"first second")
    first = PdfStream(raw_data=source[:5])
    second = PdfStream(raw_data=source[6:11])

    assert TextState.stream_execution_key(first) != TextState.stream_execution_key(second)
    assert TextState.stream_execution_key(first) == TextState.stream_execution_key(first)


def internal_capture_drawing_kinds(content: bytes) -> list[str]:
    resolver = SimpleNamespace(
        kw_cache={},
        resolve=lambda value: value,
        resolve_dict=lambda value: value if isinstance(value, dict) else None,
        resolve_name=lambda internal_value: None,
        resolve_str=lambda internal_value: None,
    )
    document = cast(Any, SimpleNamespace(resolver=resolver))
    state = TextState(document, {})
    state.consume_stream(PdfStream(raw_data=content), {}, IDENTITY_MATRIX, 0)
    assert not state.stack
    assert not state.clip_scope_stack
    return [drawing.kind for drawing in state.drawings]


def test_graphics_state_markers_are_emitted_only_for_clip_scopes() -> None:
    assert internal_capture_drawing_kinds(b"q 0 0 m 1 1 l S Q") == ["stroke"]
    assert internal_capture_drawing_kinds(b"q 0 0 10 10 re W n 0 0 m 1 1 l S Q") == [
        "state-push",
        "clip",
        "stroke",
        "state-pop",
    ]
