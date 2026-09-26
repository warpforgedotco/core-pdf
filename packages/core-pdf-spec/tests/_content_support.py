from __future__ import annotations

from typing import Any

from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver


class NullSink:
    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


def make_interpreter(
    sink: object = None,
    interpreter_class: type[ContentInterpreter] = ContentInterpreter,
) -> ContentInterpreter:
    return interpreter_class(ObjectResolver(b"", {}), sink, None)  # ty: ignore[invalid-argument-type]
