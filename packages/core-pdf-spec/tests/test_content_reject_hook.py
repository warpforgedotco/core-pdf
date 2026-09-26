from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from _content_support import make_interpreter

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_08_graphics.color_spec import DEVICE_GRAY, DEVICE_RGB, ColorSpace
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
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


@pytest.mark.parametrize(
    ("operator", "value", "message"),
    [("w", -1, "line width"), ("M", 0.5, "miter limit")],
)
def test_the_default_hook_refuses_out_of_range_line_parameters(
    operator: str, value: float, message: str
) -> None:
    # Called as a handler, past the operand check execute_operation makes first.
    handler = getattr(make_interpreter(), f"op_{operator}")
    with pytest.raises(PdfParseError, match=message):
        handler((value,), 0)


@pytest.mark.parametrize(
    ("operator", "value", "attribute", "expected", "rejected"),
    [
        ("w", -2.5, "line_width", 0.0, ["line-width"]),
        ("w", -0.0, "line_width", 0.0, []),
        ("w", 0, "line_width", 0.0, []),
        ("w", 3, "line_width", 3.0, []),
        ("M", 0.25, "miter_limit", 1.0, ["miter-limit"]),
        ("M", 1, "miter_limit", 1.0, []),
        ("M", 4.5, "miter_limit", 4.5, []),
    ],
)
def test_a_recovering_hook_clamps_line_parameters(
    operator: str, value: float, attribute: str, expected: float, rejected: list[str]
) -> None:
    state = recovering()
    getattr(state, f"op_{operator}")((value,), 0)
    result = getattr(state.graphics, attribute)
    assert result == expected
    # The clamp keeps the sign of a zero exact: recovery never stores -0.0.
    assert str(result) == str(expected)
    assert [context for context, _message in state.rejected] == rejected


def test_the_default_hook_refuses_unmatched_ex_and_a_curve_without_a_point() -> None:
    state = make_interpreter()
    with pytest.raises(PdfParseError, match="unmatched EX"):
        state.op_EX((), 0)
    with pytest.raises(PdfParseError, match="no current point"):
        state.append_cubic_curve(1, 2, 3, 4, 5, 6)
    assert state.current_point is None


def test_a_recovering_hook_ignores_unmatched_ex() -> None:
    state = recovering()
    state.op_BX((), 0)
    state.op_EX((), 0)
    state.op_EX((), 0)
    assert state.compatibility_depth == 0
    state.op_BX((), 0)
    assert state.compatibility_depth == 1
    assert [context for context, _message in state.rejected] == ["compatibility"]


def test_a_recovering_hook_moves_to_the_end_of_a_curve_without_a_point() -> None:
    state = recovering()
    state.append_cubic_curve(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert state.current_point == (5.0, 6.0)
    assert not state.current_path.ops
    assert [context for context, _message in state.rejected] == ["path"]


NUMERIC_OPERANDS = [
    (1.5, -2.0),
    (1, 2),
    (1, 2.5),
    [1.0, 2.0],
    (1.0, 2.0, 3.0),
    (1.0,),
    (float("nan"), 1.0),
    (1.0, float("inf")),
    (10**400, 1.0),
    (True, 1.0),
    ("1", 1.0),
]


@pytest.mark.parametrize("operands", NUMERIC_OPERANDS, ids=repr)
def test_as_floats_matches_as_float_or_refuses_as_it_does(operands: Any) -> None:
    state = make_interpreter()
    try:
        expected: object = tuple(state.as_float(value) for value in operands[:2])
        if len(operands) < 2:
            expected = PdfParseError
    except PdfParseError:
        expected = PdfParseError
    if expected is PdfParseError:
        with pytest.raises(PdfParseError, match="numeric operand"):
            state.as_floats(operands, 2)
        recovered = recovering()
        assert recovered.as_floats(operands, 2) is None
        assert [context for context, _message in recovered.rejected] == ["numeric-operands"]
    else:
        result = state.as_floats(operands, 2)
        assert result == expected
        assert result is not None
        assert all(type(value) is float for value in result)


@pytest.mark.parametrize(
    ("operands", "expected"), [((2, 9), 2), ((), None), ((True,), None), ((2.0,), None)]
)
def test_as_int_operand_accepts_only_an_integer(operands: Any, expected: int | None) -> None:
    state = recovering()
    assert state.as_int_operand(operands) == expected
    assert [context for context, _ in state.rejected] == (
        [] if expected is not None else ["integer-operand"]
    )
    if expected is None:
        with pytest.raises(PdfParseError, match="numeric operand|integer operand"):
            make_interpreter().as_int_operand(operands)


UNSUPPORTED_SPACE = ColorSpace("Unsupported", ())


def test_the_default_hook_refuses_bad_colors_and_resources() -> None:
    state = make_interpreter()
    with pytest.raises(PdfParseError, match="color component") as caught:
        state.normalize_color_components(DEVICE_RGB, (0.5, "x", 0.5))
    assert isinstance(caught.value.__cause__, ValueError)
    with pytest.raises(PdfParseError, match="unsupported color space"):
        state.initial_color_components(UNSUPPORTED_SPACE, stroke=False)
    with pytest.raises(PdfParseError, match="resource value"):
        state.resolve_resources(3)


class GuessingInterpreter(RecoveringInterpreter):
    def recover_color_components(self, components: Sequence[object]) -> tuple[float, ...] | None:
        return (0.25,) * len(components)


def test_a_recovering_hook_proceeds_with_color_and_resource_fallbacks() -> None:
    state = make_interpreter(interpreter_class=GuessingInterpreter)
    assert isinstance(state, GuessingInterpreter)
    state.rejected = []
    assert state.normalize_color_components(DEVICE_RGB, (0.5, "x", 0.5)) == (0.25, 0.25, 0.25)
    indexed = ColorSpace("Indexed", ((0.0, 3.0),), base=DEVICE_RGB, hival=3)
    assert state.normalize_color_components(indexed, ("x",)) is None
    state.graphics.fill_color = (0.5,)
    assert state.initial_color_components(UNSUPPORTED_SPACE, stroke=False) == (0.5,)
    assert state.resolve_resources(3) is None
    assert [context for context, _message in state.rejected] == [
        "color-components",
        "color-components",
        "color-space",
        "resources",
    ]


def test_the_base_recovery_has_no_color_guess() -> None:
    state = recovering()
    assert state.normalize_color_components(DEVICE_RGB, (0.5, "x", 0.5)) is None


class FontRecovery(RecoveringInterpreter):
    """Records the fonts it is handed and what decoder_for sees."""

    provided: list[object]
    decoded: list[tuple[object, object]]

    def decoder_for(self, font_reference: object, font: object, resources: PdfDict) -> Any:
        self.decoded.append((font_reference, font))
        return super().decoder_for(font_reference, font, resources)


def font_recovery(fonts: PdfDict) -> FontRecovery:
    provided: list[object] = []

    def provide(font: object, resources: object) -> object:
        provided.append(font)
        return ("decoder", len(provided))

    state = FontRecovery(ObjectResolver(b"", {}), None, provide)  # ty: ignore[invalid-argument-type]
    state.rejected = []
    state.provided = provided
    state.decoded = []
    state.resources = {"Font": fonts}
    return state


@pytest.mark.parametrize("font_name", [None, "", "Missing"])
def test_a_recovering_hook_decodes_an_unselected_font_as_empty_each_time(
    font_name: str | None,
) -> None:
    state = font_recovery({"F": {"BaseFont": PdfName.of("F")}})
    state.graphics.current_font = font_name
    first = state.get_decoder()
    assert state.get_decoder() != first
    assert state.graphics.current_decoder is None
    assert state.provided == [{}, {}]
    assert state.decoded == []
    context = "font-resource" if font_name else "font"
    assert [context for context, _ in state.rejected] == [context, context]


def test_a_recovering_hook_decodes_a_font_stream_by_its_dictionary() -> None:
    font = {"BaseFont": PdfName.of("F")}
    stream = PdfStream(font)
    state = font_recovery({"F": stream, "N": 7})
    state.graphics.current_font = "F"
    decoder = state.get_decoder()
    assert state.graphics.current_decoder is decoder
    assert state.decoded == [(stream, font)]
    state.graphics.current_decoder = None
    state.graphics.current_font = "N"
    state.get_decoder()
    assert state.decoded[-1] == (7, 7)
    assert state.provided[-1] == {}
    assert [context for context, _ in state.rejected] == ["font-resource", "font-resource"]


class MatrixRecovery(RecoveringInterpreter):
    def matrix_fallback(self, value: object, context: str) -> Matrix | None:
        return IDENTITY_MATRIX if context == "pattern" else None


def test_a_malformed_matrix_raises_without_a_fallback() -> None:
    with pytest.raises(ValueError):
        make_interpreter().matrix_operand([1, 0, 0, 1, 0], "pattern")
    state = recovering()
    with pytest.raises(ValueError):
        state.matrix_operand([1, 0, 0, 1, 0], "pattern")
    assert state.rejected == []


def test_a_recovering_hook_proceeds_with_the_matrix_fallback() -> None:
    state = make_interpreter(interpreter_class=MatrixRecovery)
    assert isinstance(state, MatrixRecovery)
    state.rejected = []
    assert state.matrix_operand([1, 0, 0, 1, 0], "pattern") is IDENTITY_MATRIX
    assert state.matrix_operand([2, 0, 0, 2, 0, 0], "form") == Matrix(2, 0, 0, 2, 0, 0)
    assert state.matrix_operand(None, "form") is IDENTITY_MATRIX
    with pytest.raises(ValueError):
        state.matrix_operand([1, 0, 0, 1, 0], "form")
    assert [context for context, _ in state.rejected] == ["pattern"]


@pytest.mark.parametrize("fonts", [{"F": PdfStream({})}, {"F": 7}, {}])
def test_the_default_hook_refuses_a_font_that_is_not_a_dictionary(fonts: PdfDict) -> None:
    state = make_interpreter()
    state.resources = {"Font": fonts}
    state.graphics.current_font = "F"
    with pytest.raises(PdfParseError, match="font resource must be a dictionary"):
        state.get_decoder()
