from __future__ import annotations

import pytest
from _content_support import make_interpreter

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.color_spec import DEVICE_GRAY
from core_pdf_spec.types import PdfName


class RecoveringInterpreter(ContentInterpreter):
    """A reader that proceeds with each fallback, recording what it refused."""

    rejected: list[tuple[str, str]]

    def reject[T](self, error: Exception, context: str, fallback: T) -> T:
        self.rejected.append((context, str(error)))
        return fallback


def recovering() -> RecoveringInterpreter:
    state = make_interpreter(interpreter_class=RecoveringInterpreter)
    assert isinstance(state, RecoveringInterpreter)
    state.rejected = []
    return state


# 7.5.7: an object stream is never an XObject, whatever Subtype it claims.
OBJECT_STREAM = PdfStream({"Type": PdfName.of("ObjStm"), "Subtype": PdfName.of("Form")})


def test_the_default_hook_raises_the_rejected_error() -> None:
    state = make_interpreter()
    state.resources = {"XObject": {"X": OBJECT_STREAM}}
    with pytest.raises(PdfParseError, match="object stream"):
        state.append_xobject(PdfName.of("X"), 0)
    with pytest.raises(PdfParseError, match="unmatched Q"):
        state.op_Q((), 0)


def test_a_recovering_hook_proceeds_with_each_fallback() -> None:
    state = recovering()
    state.resources = {"XObject": {"X": OBJECT_STREAM}}
    assert state.append_xobject(PdfName.of("X"), 0) is None
    assert state.resolve_color_space(None) is DEVICE_GRAY
    state.op_Q((), 0)
    state.op_d(("not an array", 3), 0)
    assert state.graphics.dash_pattern == ((), 3.0)
    assert [context for context, _message in state.rejected] == [
        "xobject",
        "color-space",
        "graphics-state",
        "dash-pattern",
    ]


def test_rejected_coercions_keep_their_cause() -> None:
    state = make_interpreter()
    state.resources = {"ExtGState": {"G": {"AIS": 1}}}
    with pytest.raises(PdfParseError, match="alpha source") as caught:
        state.execute_operation("gs", (PdfName.of("G"),), 0)
    assert isinstance(caught.value.__cause__, ValueError)
    assert caught.value.__suppress_context__
