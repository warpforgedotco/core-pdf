"""Content-interpreter scaffolding for the spec tests; imports nothing but spec."""

from __future__ import annotations

from typing import Any

from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver


class NullSink:
    """A semantic sink whose callbacks accept anything and do nothing.

    Recording sinks subclass it and define only the callbacks they observe.
    """

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


def make_interpreter(
    sink: object = None,
    interpreter_class: type[ContentInterpreter] = ContentInterpreter,
) -> ContentInterpreter:
    """An interpreter over an empty document, sending events to sink, with no font service."""
    return interpreter_class(ObjectResolver(b"", {}), sink, None)  # ty: ignore[invalid-argument-type]
