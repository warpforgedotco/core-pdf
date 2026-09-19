"""Reader recovery rejects malformed operands without corrupting paint state."""

from types import SimpleNamespace

import pytest

from core_pdf.impl._impl.capture.interpreter import TextState
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX


class RecordingRecovery(TextState):
    errors: list[tuple[type[Exception], str]]

    def handle_operand_error(self, error, context):
        self.errors.append((type(error), context))


@pytest.fixture
def state():
    resolver = ObjectResolver(b"", {})
    instance = RecordingRecovery(SimpleNamespace(resolver=resolver))
    instance.errors = []
    yield instance
    resolver.close()


@pytest.mark.parametrize("value", [True, "bad", None, float("nan"), float("inf"), 10**400])
@pytest.mark.parametrize(
    ("operator", "field", "initial"), [("w", "line_width", 7), ("M", "miter_limit", 8)]
)
def test_invalid_numeric_operands_leave_graphics_state_unchanged(
    state, value, operator, field, initial
):
    setattr(state.graphics, field, initial)
    state.execute_operation(operator, (value,), 0)
    assert getattr(state.graphics, field) == initial
    assert state.errors[-1][1] == "numeric-operands"


@pytest.mark.parametrize(
    ("operator", "field", "value", "expected"),
    [
        ("w", "line_width", -5, 0),
        ("w", "line_width", 2.5, 2.5),
        ("M", "miter_limit", -5, 1),
        ("M", "miter_limit", 2.5, 2.5),
    ],
)
def test_numeric_recovery_clamps_only_prescribed_reader_bounds(
    state, operator, field, value, expected
):
    state.execute_operation(operator, (value, 99), 0)
    assert getattr(state.graphics, field) == expected
    assert state.errors == []


@pytest.mark.parametrize("operands", [(), (True,), ("bad",), (None,)])
def test_invalid_integer_operands_report_errors(state, operands):
    assert state.as_int_operand(operands) is None
    assert state.errors[-1][1] == "integer-operand"


def test_valid_integer_and_missing_float_operands(state):
    assert state.as_int_operand((2, 9)) == 2
    assert state.as_floats((), 1) is None
    assert state.errors[-1][1] == "numeric-operands"


@pytest.mark.parametrize(
    ("operands", "expected", "error"),
    [
        ((), ((2, 3), 4), False),
        (([1],), ((2, 3), 4), False),
        (([1, 2], 3), ((1, 2), 3), False),
        (("bad", 3), ((), 3), False),
        (([True], 3), ((2, 3), 4), True),
        (([1], "bad"), ((2, 3), 4), True),
    ],
)
def test_dash_recovery_retains_previous_pattern_on_invalid_numbers(
    state, operands, expected, error
):
    state.graphics.dash_pattern = ((2, 3), 4)
    state.op_d(operands, 0)
    assert state.graphics.dash_pattern == expected
    assert bool(state.errors) is error


@pytest.mark.parametrize("operator", ["l", "v", "y"])
def test_unstarted_path_segments_are_ignored(state, operator):
    state.execute_operation(operator, (1, 2, 3, 4), 0)
    assert state.current_point is None
    assert state.errors == []


@pytest.mark.parametrize("operator", ["l", "v", "y"])
def test_incomplete_path_operands_preserve_current_point(state, operator):
    state.execute_operation("m", (1, 2), 0)
    state.execute_operation(operator, (), 0)
    assert state.current_point == (1, 2)
    assert state.errors[-1][1] == "numeric-operands"


def test_unbalanced_scope_endings_do_not_underflow(state):
    state.op_Q((), 0)
    state.op_EX((), 0)
    state.op_EMC((), 0)
    assert state.compatibility_depth == 0
    state.op_BX((), 0)
    state.op_BX((), 0)
    state.op_EX((), 0)
    assert state.compatibility_depth == 1
    assert state.execute_operation("unknown", (), 0) is None


@pytest.mark.parametrize("context", ["form", "pattern", "other"])
@pytest.mark.parametrize("value", [None, [1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0, 99], [1, 2]])
def test_matrix_recovery_is_limited_to_its_specific_context(state, context, value):
    valid = (
        value is None
        or len(value) == 6
        or context == "pattern"
        or (context == "form" and len(value) > 6)
    )
    if valid:
        assert state.matrix_operand(value, context) == IDENTITY_MATRIX
    else:
        with pytest.raises(ValueError):
            state.matrix_operand(value, context)


@pytest.mark.parametrize("value", ["bad", True, float("nan")])
def test_invalid_font_size_retains_recovery_error_context(state, value):
    assert state.parse_font_size(value) is None
    assert state.errors[-1][1] == "font-size"


@pytest.mark.parametrize("value", [2, 2.0, "2", b"2", "+2.0", "2e0"])
def test_valid_native_numeric_encodings_remain_accepted(state, value):
    assert state.as_float(value) == 2
    assert state.parse_font_size(value) == 2
    assert state.errors == []
