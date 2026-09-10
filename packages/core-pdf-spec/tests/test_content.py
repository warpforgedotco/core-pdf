# SPDX-License-Identifier: AGPL-3.0-only
from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content import operations
from core_pdf_spec.s_07_content.events import ContentSink
from core_pdf_spec.s_07_content.operations import (
    ContentOperands,
    ContentOperationState,
    dispatch_operations,
    iter_content_operations,
    parse_content_token,
)
from core_pdf_spec.s_07_content.state import TextState
from core_pdf_spec.s_07_content.stream_state import ContentStreamFrame
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from core_pdf_spec.s_09_fonts.service import DecodedFontGlyph
from core_pdf_spec.types import PdfName, PdfString


class EventSink:
    def __init__(self) -> None:
        self.colors: list[tuple[float, ...] | None] = []
        self.saved = 0

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None

    def save_graphics(self, state: TextState) -> None:
        self.saved += 1

    def restore_graphics(self, state: TextState) -> None:
        self.saved -= 1

    def paint_path(self, state: TextState, *args: Any) -> None:
        self.colors.append(state.fill_color)


class EmptyTextFont:
    is_type3 = False
    is_vertical = False
    ascent = 800.0
    descent = -200.0
    fast_widths = (500.0,) * 256

    def decode_glyphs(self, data: bytes) -> tuple[DecodedFontGlyph, ...]:
        return tuple(DecodedFontGlyph(bytes([code]), code, code, code, "", code) for code in data)

    def text_advance_vector(self, data: Any, **kwargs: Any) -> tuple[float, float]:
        return (5.0 * len(data), 0.0)

    def glyph_width(self, code: int) -> float:
        return 500.0


def state_with_sink() -> tuple[TextState, EventSink]:
    sink = EventSink()
    resolver = ObjectResolver(b"", {}, {})
    font: Any = EmptyTextFont()
    return TextState(
        SimpleNamespace(resolver=resolver), cast(ContentSink, sink), lambda *args: font
    ), sink


def tj_state_with_recording(
    monkeypatch: pytest.MonkeyPatch, *, vertical: bool = False
) -> tuple[TextState, list[tuple[bytes, float, float]]]:
    state, _ = state_with_sink()
    state.current_decoder = cast(Any, SimpleNamespace(is_vertical=vertical))
    state.tm_a, state.tm_b, state.tm_c, state.tm_d = 2.0, 3.0, 5.0, 7.0
    state.tm_e, state.tm_f = 11.0, 13.0
    state.font_size = 100
    state.horizontal_scale = 100
    state.update_text_scales()
    shown: list[tuple[bytes, float, float]] = []

    def append_text(*, data: bytes, decoder: object) -> None:
        shown.append((data, state.tm_e, state.tm_f))
        state.tm_e += 1.0
        state.tm_f += 2.0

    monkeypatch.setattr(state, "append_text", append_text)
    return state, shown


@pytest.mark.parametrize(
    ("vertical", "positions", "final"),
    [
        (False, [(11, 13), (10, 12), (11.5, 14.75)], (12.5, 16.75)),
        (True, [(11, 13), (7, 8), (9.25, 11.75)], (10.25, 13.75)),
    ],
)
def test_tj_array_preserves_text_chunks_and_numeric_adjustments(
    monkeypatch: pytest.MonkeyPatch,
    vertical: bool,
    positions: list[tuple[float, float]],
    final: tuple[float, float],
) -> None:
    # ISO 32000-1 Table 109: TJ numbers adjust the current writing direction.
    state, shown = tj_state_with_recording(monkeypatch, vertical=vertical)
    state.append_tj_array([PdfString(b"a"), b"b", 10, b"c", -2.5, b"d"])
    assert [data for data, _, _ in shown] == [b"ab", b"c", b"d"]
    assert [(x, y) for _, x, y in shown] == positions
    assert (state.tm_e, state.tm_f) == final


@pytest.mark.parametrize("array", [[], ()])
def test_empty_tj_array_does_not_require_a_font(array: object) -> None:
    state, _ = state_with_sink()
    state.append_tj_array(array)
    assert state.current_decoder is None


@pytest.mark.parametrize("array", [None, b"text", "text", 1])
def test_tj_requires_an_array(array: object) -> None:
    state, _ = state_with_sink()
    with pytest.raises(PdfParseError, match="TJ requires an array"):
        state.append_tj_array(array)


@pytest.mark.parametrize("item", ["text", True, None, PdfName.of("Invalid"), bytearray(b"x")])
def test_tj_rejects_non_pdf_entries_after_prior_text(
    monkeypatch: pytest.MonkeyPatch, item: object
) -> None:
    state, shown = tj_state_with_recording(monkeypatch)
    with pytest.raises(PdfParseError, match="TJ array entries must be strings or numbers"):
        state.append_tj_array([b"a", 10, b"b", item])
    assert shown == [(b"a", 11, 13)]
    # The pending adjustment and unflushed b have not yet committed their position.
    assert (state.tm_e, state.tm_f) == (12, 15)


@pytest.mark.parametrize(
    "content", [b"unknown", b"]", b">>", b"R", b"1 2 cm", b"(x) w", b"[1 /Bad] TJ", b"1 2", b"EX"]
)
def test_invalid_content_rejected(content: bytes) -> None:
    with pytest.raises(PdfParseError):
        list(iter_content_operations(PdfLexer(content)))


def test_compatibility_section_ignores_unknown_operators() -> None:
    operations = list(iter_content_operations(PdfLexer(b"BX 1 2 extension EX 3 w")))
    assert [name for name, _ in operations] == ["BX", "EX", "w"]


def test_indexed_components_remain_palette_indices() -> None:
    state, sink = state_with_sink()
    state.resources = {
        "ColorSpace": {
            "Palette": [
                PdfName.of("Indexed"),
                PdfName.of("DeviceRGB"),
                2,
                b"\0\0\0\xff\0\0\0\xff\0",
            ]
        }
    }
    dispatch_operations(
        PdfLexer(b"/Palette cs 2 sc 0 0 10 10 re f"), state.get_operation_handler, 0
    )
    assert state.fill_color == (2.0,)
    assert sink.colors == [(2.0,)]


def test_missing_font_resource_raises_without_substitution() -> None:
    state, _ = state_with_sink()
    with pytest.raises(PdfParseError, match="font resource"):
        dispatch_operations(PdfLexer(b"/Missing 12 Tf"), state.get_operation_handler, 0)


def test_failed_child_stream_restores_parent_graphics_state() -> None:
    state, sink = state_with_sink()
    child = PdfStream(
        raw_data=b"q 0.75 g invalid",
        dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 10, 10]},
    )
    resources: PdfDict = {"XObject": {"Child": child}}
    initial = state.capture_stream_state()
    with pytest.raises(PdfParseError, match="unknown content"):
        state.consume_stream(
            PdfStream(raw_data=b"0.25 g /Child Do", dictionary={}), resources, IDENTITY_MATRIX, 0
        )
    assert state.capture_stream_state() == initial
    assert sink.saved == 0
    assert not state.stream_executor.active_streams


def test_compatibility_scope_survives_nested_stream_suspension() -> None:
    state, _ = state_with_sink()
    child = PdfStream(
        raw_data=b"", dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 10, 10]}
    )
    state.consume_stream(
        PdfStream(raw_data=b"BX /Child Do extension EX", dictionary={}),
        {"XObject": {"Child": child}},
        IDENTITY_MATRIX,
        0,
    )


@pytest.mark.parametrize(
    "content", [b"q BX Q extension EX", b"BX q extension EX Q", b"BX BX EX EX"]
)
def test_compatibility_sections_are_independent_of_graphics_saves(content: bytes) -> None:
    # ISO 32000-1 7.8.2: compatibility sections are not graphics state.
    list(iter_content_operations(PdfLexer(content)))
    state, sink = state_with_sink()
    original = state.capture_stream_state()
    state.consume_stream(PdfStream(raw_data=content), {}, IDENTITY_MATRIX, 0)
    assert state.capture_stream_state() == original
    assert sink.saved == 0


@pytest.mark.parametrize("content", [b"BX q EX Q EX", b"q BX EX Q extension", b"BX q Q"])
def test_invalid_compatibility_sections_fail_in_both_entry_points(content: bytes) -> None:
    with pytest.raises(PdfParseError):
        list(iter_content_operations(PdfLexer(content)))
    state, sink = state_with_sink()
    original = state.capture_stream_state()
    with pytest.raises(PdfParseError):
        state.consume_stream(PdfStream(raw_data=content), {}, IDENTITY_MATRIX, 0)
    assert state.capture_stream_state() == original
    assert sink.saved == 0


@pytest.mark.parametrize("child_content", [b"EX", b"extension", b"BX"])
def test_child_compatibility_scope_cannot_use_or_leak_into_parent(child_content: bytes) -> None:
    state, sink = state_with_sink()
    original = state.capture_stream_state()
    child = PdfStream(
        raw_data=child_content,
        dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 10, 10]},
    )
    with pytest.raises(PdfParseError):
        state.consume_stream(
            PdfStream(raw_data=b"BX /Child Do extension EX"),
            {"XObject": {"Child": child}},
            IDENTITY_MATRIX,
            0,
        )
    assert state.capture_stream_state() == original
    assert sink.saved == 0
    assert not state.stream_executor.active_streams


def test_parent_scope_object_survives_child_suspension() -> None:
    state, _ = state_with_sink()
    scopes: list[tuple[ContentOperationState, int]] = []
    state.op_handlers["w"] = lambda operands, depth: scopes.append(
        (state.operation_state, state.compatibility_depth)
    )
    child = PdfStream(
        raw_data=b"3 w BX 4 w extension EX 5 w",
        dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 10, 10]},
    )
    state.consume_stream(
        PdfStream(raw_data=b"BX 1 w /Child Do 2 w extension EX"),
        {"XObject": {"Child": child}},
        IDENTITY_MATRIX,
        0,
    )
    assert [nesting for _, nesting in scopes] == [1, 0, 1, 0, 1]
    assert scopes[0][0] is scopes[-1][0]
    assert all(scope is scopes[1][0] for scope, _ in scopes[1:4])
    assert scopes[0][0] is not scopes[1][0]


def test_reentrant_stream_execution_restores_parent_scope() -> None:
    state, _ = state_with_sink()
    scopes: list[tuple[ContentOperationState, int]] = []

    def execute(operands: ContentOperands, depth: int) -> None:
        scopes.append((state.operation_state, state.compatibility_depth))
        if operands == (1,):
            state.consume_stream(PdfStream(raw_data=b"2 w BX extension EX"), {}, IDENTITY_MATRIX, 1)
            scopes.append((state.operation_state, state.compatibility_depth))

    state.op_handlers["w"] = execute
    state.consume_stream(PdfStream(raw_data=b"BX 1 w extension EX"), {}, IDENTITY_MATRIX, 0)
    assert [nesting for _, nesting in scopes] == [1, 0, 1]
    assert scopes[0][0] is scopes[2][0]
    assert scopes[0][0] is not scopes[1][0]


def test_custom_scope_callbacks_observe_scope_without_owning_it() -> None:
    state, _ = state_with_sink()
    depths: list[int] = []

    def callback(operands: ContentOperands, depth: int) -> None:
        depths.append(state.compatibility_depth)

    state.op_handlers.update(BX=callback, EX=callback)
    state.consume_stream(PdfStream(raw_data=b"BX extension EX"), {}, IDENTITY_MATRIX, 0)
    assert depths == [1, 0]


@pytest.mark.parametrize("name", ["BX", "EX"])
def test_direct_scope_handlers_validate_before_mutation(name: str) -> None:
    state, _ = state_with_sink()
    state.compatibility_depth = 1
    handler = state.get_operation_handler(name)
    assert handler is not None
    with pytest.raises(PdfParseError, match="requires 0 operands"):
        handler((1,), 0)
    assert state.compatibility_depth == 1
    handler((), 0)
    assert state.compatibility_depth == (2 if name == "BX" else 0)
    if name == "EX":
        with pytest.raises(PdfParseError, match="unmatched EX"):
            handler((), 0)


@pytest.mark.parametrize("name", [b"BX", b"EX"])
def test_standalone_scope_dispatch_validates_before_mutation(name: bytes) -> None:
    scope = ContentOperationState(1)
    with pytest.raises(PdfParseError, match="requires 0 operands"):
        dispatch_operations(
            PdfLexer(b"1 " + name),
            lambda name: lambda operands, depth: None,
            0,
            operation_state=scope,
        )
    assert scope.compatibility_depth == 1


def test_normal_execution_validates_each_fixed_signature_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core_pdf_spec.s_07_content import operations

    checked: list[str] = []
    validate = operations.validate_content_operands

    def record(name: str, operands: ContentOperands) -> None:
        checked.append(name)
        validate(name, operands)

    monkeypatch.setattr(operations, "validate_content_operands", record)
    state, _ = state_with_sink()
    state.consume_stream(PdfStream(raw_data=b"q BX 3 w extension EX Q"), {}, IDENTITY_MATRIX, 0)
    assert checked == ["q", "BX", "w", "EX", "Q"]


@pytest.mark.parametrize(("name", "category"), [("gs", "ExtGState"), ("sh", "Shading")])
@pytest.mark.parametrize("entry_point", ["stream", "handler"])
def test_resource_operator_looks_up_and_resolves_once(
    monkeypatch: pytest.MonkeyPatch, name: str, category: str, entry_point: str
) -> None:
    class Resolver(ObjectResolver):
        def __init__(self) -> None:
            super().__init__(b"", {}, {})
            self.dictionary_calls = 0

        def resolve_dict(self, value: object) -> PdfDict | None:
            self.dictionary_calls += 1
            return super().resolve_dict(value)

    state, sink = state_with_sink()
    resolver = Resolver()
    state.document = SimpleNamespace(resolver=resolver)
    lookups: list[tuple[str, str]] = []
    paints: list[PdfDict] = []
    resource: PdfDict = {"ca": 0.5} if name == "gs" else {"ShadingType": 2}

    def lookup(category: str, name: str) -> object:
        lookups.append((category, name))
        return resource

    monkeypatch.setattr(state, "lookup_page_resource", lookup)
    monkeypatch.setattr(sink, "paint_shading", lambda state, shading: paints.append(shading))
    if entry_point == "stream":
        state.consume_stream(
            PdfStream(raw_data=f"/Resource {name}".encode()), {}, IDENTITY_MATRIX, 0
        )
    else:
        handler = state.get_operation_handler(name)
        assert handler is not None
        handler((PdfName.of("Resource"),), 0)
        if name == "gs":
            assert state.fill_opacity == 0.5
    assert lookups == [(category, "Resource")]
    assert resolver.dictionary_calls == 1
    assert paints == ([resource] if name == "sh" else [])


@pytest.mark.parametrize("name", ["gs", "sh"])
@pytest.mark.parametrize("resource", [None, 1, [], PdfName.of("Invalid")])
def test_resource_handlers_reject_missing_or_invalid_resources(name: str, resource: object) -> None:
    state, _ = state_with_sink()
    category = "ExtGState" if name == "gs" else "Shading"
    state.resources = cast(PdfDict, {category: {"Resource": resource}})
    original = state.capture_stream_state()
    handler = state.get_operation_handler(name)
    assert handler is not None
    with pytest.raises(PdfParseError, match="resource"):
        handler((PdfName.of("Resource"),), 0)
    assert state.capture_stream_state() == original


def test_stream_snapshot_restores_fields_before_unwinding_graphics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, sink = state_with_sink()
    state.resources = {"Original": 1}
    state.resources_id = id(state.resources)
    state.text_matrix = Matrix(2, 0, 0, 3, 4, 5)
    state.line_matrix = Matrix(3, 0, 0, 2, 5, 4)
    state.xobject_depth = 7
    state.compatibility_depth = 2
    original = state.capture_stream_state()
    state.resources = {}
    state.resources_id = 42
    state.text_matrix = IDENTITY_MATRIX
    state.line_matrix = IDENTITY_MATRIX
    state.xobject_depth = 9
    state.operation_state = ContentOperationState(5)
    state.op_q((), 0)
    state.graphics_stack_floor = 1
    observed = []
    restore = sink.restore_graphics

    def observe(current: TextState) -> None:
        observed.append(
            (
                current.resources,
                current.resources_id,
                current.text_matrix,
                current.line_matrix,
                current.graphics_stack_floor,
                current.xobject_depth,
                current.operation_state,
            )
        )
        restore(current)

    monkeypatch.setattr(sink, "restore_graphics", observe)
    state.restore_stream_state(original)
    assert observed == [
        (
            original.resources,
            original.resources_id,
            original.text_matrix,
            original.line_matrix,
            original.graphics_stack_floor,
            original.xobject_depth,
            original.operation_state,
        )
    ]
    assert state.operation_state is original.operation_state
    assert state.capture_stream_state() == original
    assert sink.saved == 0


def test_failed_stream_decode_does_not_replace_parent_scope() -> None:
    class FailingStream(PdfStream):
        @property
        def data(self) -> bytes:
            raise PdfParseError("cannot decode child")

    state, sink = state_with_sink()
    state.compatibility_depth = 2
    original = state.capture_stream_state()
    frame = ContentStreamFrame(FailingStream(raw_data=b""), {}, IDENTITY_MATRIX, 1, None)
    with pytest.raises(PdfParseError, match="cannot decode child"):
        state.stream_executor.enter(frame)
    assert state.operation_state is original.operation_state
    assert state.capture_stream_state() == original
    assert sink.saved == 0


def test_stream_snapshot_restores_compatibility_depth_after_direct_execution() -> None:
    state, _ = state_with_sink()
    state.compatibility_depth = 2
    original = state.capture_stream_state()
    state.execute_operation("BX", (), 0)
    assert state.compatibility_depth == 3
    state.restore_stream_state(original)
    assert state.operation_state is original.operation_state
    assert state.compatibility_depth == 2
    state.execute_operation("EX", (), 0)
    assert state.compatibility_depth == 1
    state.restore_stream_state(original)
    assert state.operation_state is original.operation_state
    assert state.compatibility_depth == 2


@pytest.mark.parametrize("value", [float("inf"), float("nan"), 10**400, True])
@pytest.mark.parametrize("operator", ["w", "sc", "TJ"])
def test_content_numeric_validation_rejects_unrepresentable_values_without_mutation(
    value: Any, operator: str
) -> None:
    state, _ = state_with_sink()
    original = state.capture_stream_state()
    operands = ([value],) if operator == "TJ" else (value,)
    handler = state.get_operation_handler(operator)
    assert handler is not None
    with pytest.raises(PdfParseError):
        handler(operands, 0)
    assert state.capture_stream_state() == original


def test_content_real_token_rejects_conversion_overflow() -> None:
    with pytest.raises(PdfParseError, match="implementation limits"):
        parse_content_token(PdfLexer(b"9" * 400 + b".0"))
    token = parse_content_token(PdfLexer(b"12.5"))
    assert token is not None
    assert token.value == 12.5


def test_empty_unicode_still_advances_text() -> None:
    state, _ = state_with_sink()
    font: Any = EmptyTextFont()
    state.current_decoder = font
    state.append_text(data=b"ab")
    assert state.text_matrix.e == 10.0


def test_spec_has_no_reader_form_depth_limit() -> None:
    state, sink = state_with_sink()
    objects: PdfDict = {}
    for depth in range(12):
        following = f"/Child{depth + 1} Do".encode() if depth < 11 else b""
        objects[f"Child{depth}"] = PdfStream(
            dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 10, 10]},
            raw_data=b"0 0 10 10 re f " + following,
        )
    state.consume_stream(
        PdfStream(raw_data=b"/Child0 Do"), {"XObject": objects}, IDENTITY_MATRIX, 0
    )
    assert len(sink.colors) == 12


@pytest.mark.parametrize("content", [b"q", b"/Span BMC"])
def test_unbalanced_scopes_are_rejected_and_restored(content: bytes) -> None:
    state, sink = state_with_sink()
    with pytest.raises(PdfParseError, match="unbalanced"):
        state.consume_stream(PdfStream(raw_data=content), {}, IDENTITY_MATRIX, 0)
    assert not state.stack
    assert not state.marked_content_stack
    assert sink.saved == 0


def test_uncolored_type3_glyph_ignores_color_setting() -> None:
    state, _ = state_with_sink()
    state.type3_uncolored = True
    dispatch_operations(PdfLexer(b"1 0 0 rg /DeviceRGB cs"), state.get_operation_handler, 0)
    assert state.fill_color == (0.0,)
    assert state.fill_color_space == "DeviceGray"


@pytest.mark.parametrize(
    "content", [b"/Missing cs", b"/DeviceRGB cs 1 sc", b"/Pattern cs /Missing scn"]
)
def test_invalid_color_resources_and_operands_rejected(content: bytes) -> None:
    state, _ = state_with_sink()
    with pytest.raises(PdfParseError):
        dispatch_operations(PdfLexer(content), state.get_operation_handler, 0)


def test_inline_image_rejects_declared_length_mismatch() -> None:
    with pytest.raises(PdfParseError, match="data length"):
        list(iter_content_operations(PdfLexer(b"BI /W 3 /H 1 /BPC 8 /CS /G ID a EI")))


def test_lab_components_use_numeric_pdf_ranges() -> None:
    state, _ = state_with_sink()
    params: PdfDict = {"WhitePoint": [1, 1, 1], "Range": [-2, 2, -3, 3]}
    state.resources = {"ColorSpace": {"Test": [PdfName.of("Lab"), params]}}
    dispatch_operations(PdfLexer(b"/Test cs 150 4 -4 sc"), state.get_operation_handler, 0)
    assert state.fill_color == (100.0, 2.0, -3.0)
    params["Range"] = [PdfString(b"-2"), 2, -3, 3]
    with pytest.raises(PdfParseError, match="PDF number"):
        dispatch_operations(PdfLexer(b"/Test cs"), state.get_operation_handler, 0)


@pytest.mark.parametrize("scope_kind", ["shared", "separate", "implicit"])
def test_strict_dispatch_validates_once_and_advances_each_scope_once(
    monkeypatch: pytest.MonkeyPatch, scope_kind: str
) -> None:
    state, _ = state_with_sink()
    scope = state.operation_state if scope_kind == "shared" else ContentOperationState()
    checked: list[str] = []
    depths: list[tuple[int, int]] = []
    validate = operations.validate_content_operands

    def record(name: str, operands: ContentOperands) -> None:
        checked.append(name)
        validate(name, operands)

    def observe(operands: ContentOperands, depth: int) -> None:
        depths.append((state.compatibility_depth, scope.compatibility_depth))

    monkeypatch.setattr(operations, "validate_content_operands", record)
    state.op_handlers.update(BX=observe, EX=observe)
    dispatch_operations(
        PdfLexer(b"BX 3 w extension EX"),
        state.get_operation_handler,
        0,
        operation_state=None if scope_kind == "implicit" else scope,
    )
    assert checked == ["BX", "w", "EX"]
    assert depths == ([(1, 0), (0, 0)] if scope_kind == "implicit" else [(1, 1), (0, 0)])
    assert state.line_width == 3
    assert state.compatibility_depth == scope.compatibility_depth == 0


def test_retained_handlers_follow_stream_scope_and_keep_captured_callback() -> None:
    state, _ = state_with_sink()
    observed: list[ContentOperationState] = []
    state.op_handlers["BX"] = lambda operands, depth: observed.append(state.operation_state)
    begin = state.get_operation_handler("BX")
    end = state.get_operation_handler("EX")
    assert begin is not None
    assert end is not None
    state.op_handlers["BX"] = lambda operands, depth: pytest.fail("callback was replaced")
    parent = state.operation_state
    frame = ContentStreamFrame(PdfStream(raw_data=b""), {}, IDENTITY_MATRIX, 1, None)
    state.stream_executor.enter(frame)
    try:
        begin((), 1)
        assert frame.operation_state.compatibility_depth == 1
        assert parent.compatibility_depth == 0
        end((), 1)
    finally:
        state.stream_executor.exit(frame)
    begin((), 0)
    assert parent.compatibility_depth == 1
    end((), 0)
    assert observed == [frame.operation_state, parent]


@pytest.mark.parametrize(("dispatcher_depth", "handler_depth"), [(1, 0), (0, 1)])
def test_distinct_scope_failure_preserves_dispatcher_first_transition(
    dispatcher_depth: int, handler_depth: int
) -> None:
    state, _ = state_with_sink()
    state.compatibility_depth = handler_depth
    scope = ContentOperationState(dispatcher_depth)
    with pytest.raises(PdfParseError, match="unmatched EX"):
        dispatch_operations(PdfLexer(b"EX"), state.get_operation_handler, 0, operation_state=scope)
    assert scope.compatibility_depth == 0
    assert state.compatibility_depth == handler_depth


def test_remapped_strict_handler_retains_both_operator_validations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _ = state_with_sink()
    checked: list[str] = []
    validate = operations.validate_content_operands

    def record(name: str, operands: ContentOperands) -> None:
        checked.append(name)
        validate(name, operands)

    monkeypatch.setattr(operations, "validate_content_operands", record)
    with pytest.raises(PdfParseError, match="q requires 0 operands"):
        dispatch_operations(PdfLexer(b"1 w"), lambda name: state.get_operation_handler("q"), 0)
    assert checked == ["w", "q"]
    assert not state.stack


def test_dispatch_resumes_after_callback_suspension_without_replaying_scope() -> None:
    class Suspended(Exception):
        pass

    state, _ = state_with_sink()
    widths: list[ContentOperands] = []

    def suspend(operands: ContentOperands, depth: int) -> None:
        widths.append(operands)
        raise Suspended

    state.op_handlers["w"] = suspend
    lexer = PdfLexer(b"BX 7 w extension EX")
    with pytest.raises(Suspended):
        dispatch_operations(
            lexer, state.get_operation_handler, 0, operation_state=state.operation_state
        )
    assert lexer.pos == len(b"BX 7 w")
    assert state.compatibility_depth == 1
    dispatch_operations(
        lexer, state.get_operation_handler, 0, operation_state=state.operation_state
    )
    assert widths == [(7,)]
    assert state.compatibility_depth == 0


def test_mixed_generic_and_strict_callbacks_use_the_supplied_scope() -> None:
    state, _ = state_with_sink()
    widths: list[ContentOperands] = []

    def get_handler(name: str) -> operations.OperationHandler | None:
        if name == "w":
            return lambda operands, depth: widths.append(operands)
        return state.get_operation_handler(name)

    dispatch_operations(
        PdfLexer(b"BX 4 w extension EX"), get_handler, 0, operation_state=state.operation_state
    )
    assert widths == [(4,)]
    assert state.compatibility_depth == 0


@pytest.mark.parametrize(
    ("content", "message", "final_depth"),
    [
        (b"1 BX", "requires 0 operands", 0),
        (b"EX", "unmatched EX", 0),
        (b"BX", "unterminated compatibility", 1),
        (b"BX 1", "ends with operands", 1),
    ],
)
def test_composed_dispatch_retains_validation_and_eof_errors(
    content: bytes, message: str, final_depth: int
) -> None:
    state, _ = state_with_sink()
    with pytest.raises(PdfParseError, match=message):
        dispatch_operations(
            PdfLexer(content), state.get_operation_handler, 0, operation_state=state.operation_state
        )
    assert state.compatibility_depth == final_depth
