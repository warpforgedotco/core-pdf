# SPDX-License-Identifier: AGPL-3.0-only
from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content import interpreter
from core_pdf_spec.s_07_content.inline_images import validate_inline_images
from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_content.model import ContentSink
from core_pdf_spec.s_07_content.operations import (
    ContentOperands,
    iter_content_operations,
    parse_content_token,
)
from core_pdf_spec.s_07_content.streams import ContentStreamFrame
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

    def save_graphics(self, state: ContentInterpreter) -> None:
        self.saved += 1

    def restore_graphics(self, state: ContentInterpreter) -> None:
        self.saved -= 1

    def paint_path(self, state: ContentInterpreter, *args: Any) -> None:
        self.colors.append(state.graphics.fill_color)


class EmptyTextFont:
    is_type3 = False
    is_vertical = False

    def decode_glyphs(self, data: bytes) -> tuple[DecodedFontGlyph, ...]:
        return tuple(DecodedFontGlyph(bytes([code]), code, code, code, "", code) for code in data)

    def text_advance_vector(self, data: Any, **kwargs: Any) -> tuple[float, float]:
        return (5.0 * len(data), 0.0)


def state_with_sink() -> tuple[ContentInterpreter, EventSink]:
    sink = EventSink()
    resolver = ObjectResolver(b"", {})
    font: Any = EmptyTextFont()
    return ContentInterpreter(resolver, cast(ContentSink, sink), lambda *args: font), sink


def execute_content(state: ContentInterpreter, content: bytes, depth: int = 0) -> None:
    for name, operands in iter_content_operations(PdfLexer(content)):
        assert state.execute_operation(name, operands, depth) is None


def tj_state_with_recording(
    monkeypatch: pytest.MonkeyPatch, *, vertical: bool = False
) -> tuple[ContentInterpreter, list[tuple[bytes, float, float]]]:
    state, _ = state_with_sink()
    state.graphics.current_decoder = cast(Any, SimpleNamespace(is_vertical=vertical))
    state.text_matrix = Matrix(2, 3, 5, 7, 11, 13)
    state.graphics.font_size = 100
    state.graphics.horizontal_scale = 100
    shown: list[tuple[bytes, float, float]] = []

    def append_text(*, data: bytes, decoder: object) -> None:
        shown.append((data, state.text_matrix.e, state.text_matrix.f))
        state.text_matrix = state.text_matrix._replace(
            e=state.text_matrix.e + 1, f=state.text_matrix.f + 2
        )

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
    assert (state.text_matrix.e, state.text_matrix.f) == final


@pytest.mark.parametrize("array", [[], ()])
def test_empty_tj_array_does_not_require_a_font(array: object) -> None:
    state, _ = state_with_sink()
    state.append_tj_array(array)
    assert state.graphics.current_decoder is None


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
    assert (state.text_matrix.e, state.text_matrix.f) == (12, 15)


@pytest.mark.parametrize(
    "content", [b"unknown", b"]", b">>", b"R", b"1 2 cm", b"(x) w", b"[1 /Bad] TJ", b"1 2", b"EX"]
)
def test_invalid_content_rejected(content: bytes) -> None:
    state, _ = state_with_sink()
    with pytest.raises(PdfParseError):
        state.stream_executor.consume(PdfStream(raw_data=content), {}, IDENTITY_MATRIX, 0)


def test_compatibility_section_ignores_unknown_operators() -> None:
    state, _ = state_with_sink()
    execute_content(state, b"BX 1 2 extension EX 3 w")
    assert state.graphics.line_width == 3
    assert state.compatibility_depth == 0


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
    execute_content(state, b"/Palette cs 2 sc 0 0 10 10 re f", 0)
    assert state.graphics.fill_color == (2.0,)
    assert sink.colors == [(2.0,)]


def test_missing_font_resource_raises_without_substitution() -> None:
    state, _ = state_with_sink()
    with pytest.raises(PdfParseError, match="font resource"):
        execute_content(state, b"/Missing 12 Tf", 0)


def test_failed_child_stream_restores_parent_graphics_state() -> None:
    state, sink = state_with_sink()
    child = PdfStream(
        raw_data=b"q 0.75 g invalid",
        dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 10, 10]},
    )
    resources: PdfDict = {"XObject": {"Child": child}}
    initial = state.capture_stream_state()
    with pytest.raises(PdfParseError, match="unknown content"):
        state.stream_executor.consume(
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
    state.stream_executor.consume(
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
    state.stream_executor.consume(PdfStream(raw_data=content), {}, IDENTITY_MATRIX, 0)
    assert state.capture_stream_state() == original
    assert sink.saved == 0


@pytest.mark.parametrize("content", [b"BX q EX Q EX", b"q BX EX Q extension", b"BX q Q"])
def test_invalid_compatibility_sections_fail_and_restore_state(content: bytes) -> None:
    state, sink = state_with_sink()
    original = state.capture_stream_state()
    with pytest.raises(PdfParseError):
        state.stream_executor.consume(PdfStream(raw_data=content), {}, IDENTITY_MATRIX, 0)
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
        state.stream_executor.consume(
            PdfStream(raw_data=b"BX /Child Do extension EX"),
            {"XObject": {"Child": child}},
            IDENTITY_MATRIX,
            0,
        )
    assert state.capture_stream_state() == original
    assert sink.saved == 0
    assert not state.stream_executor.active_streams


def test_parent_scope_survives_child_suspension() -> None:
    state, _ = state_with_sink()
    scopes: list[tuple[ContentOperands, int, int]] = []
    state.operator_overrides["w"] = lambda operands, depth: scopes.append(
        (operands, depth, state.compatibility_depth)
    )
    child = PdfStream(
        raw_data=b"3 w BX 4 w extension EX 5 w",
        dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 10, 10]},
    )
    state.stream_executor.consume(
        PdfStream(raw_data=b"BX 1 w /Child Do 2 w extension EX"),
        {"XObject": {"Child": child}},
        IDENTITY_MATRIX,
        0,
    )
    assert scopes == [((1,), 0, 1), ((3,), 1, 0), ((4,), 1, 1), ((5,), 1, 0), ((2,), 0, 1)]


def test_reentrant_stream_execution_restores_parent_scope() -> None:
    state, _ = state_with_sink()
    scopes: list[tuple[int, int]] = []

    def execute(operands: ContentOperands, depth: int) -> None:
        scopes.append((depth, state.compatibility_depth))
        if operands == (1,):
            state.stream_executor.consume(
                PdfStream(raw_data=b"2 w BX extension EX"), {}, IDENTITY_MATRIX, 1
            )
            scopes.append((depth, state.compatibility_depth))

    state.operator_overrides["w"] = execute
    state.stream_executor.consume(
        PdfStream(raw_data=b"BX 1 w extension EX"), {}, IDENTITY_MATRIX, 0
    )
    assert scopes == [(0, 1), (1, 0), (0, 1)]


def test_custom_scope_callbacks_observe_scope_without_owning_it() -> None:
    state, _ = state_with_sink()
    depths: list[int] = []

    def callback(operands: ContentOperands, depth: int) -> None:
        depths.append(state.compatibility_depth)

    state.operator_overrides.update(BX=callback, EX=callback)
    state.stream_executor.consume(PdfStream(raw_data=b"BX extension EX"), {}, IDENTITY_MATRIX, 0)
    assert depths == [1, 0]


@pytest.mark.parametrize("name", ["BX", "EX"])
def test_direct_scope_execution_validates_before_mutation(name: str) -> None:
    state, _ = state_with_sink()
    state.compatibility_depth = 1
    with pytest.raises(PdfParseError, match="requires 0 operands"):
        state.execute_operation(name, (1,), 0)
    assert state.compatibility_depth == 1
    state.execute_operation(name, (), 0)
    assert state.compatibility_depth == (2 if name == "BX" else 0)
    if name == "EX":
        with pytest.raises(PdfParseError, match="unmatched EX"):
            state.execute_operation(name, (), 0)


def test_normal_execution_validates_each_fixed_signature_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    checked: list[str] = []
    validate = interpreter.validate_content_operands

    def record(name: str, operands: ContentOperands) -> None:
        checked.append(name)
        validate(name, operands)

    monkeypatch.setattr(interpreter, "validate_content_operands", record)
    state, _ = state_with_sink()
    state.stream_executor.consume(
        PdfStream(raw_data=b"q BX 3 w extension EX Q"), {}, IDENTITY_MATRIX, 0
    )
    assert checked == ["q", "BX", "w", "EX", "Q"]


@pytest.mark.parametrize(("name", "category"), [("gs", "ExtGState"), ("sh", "Shading")])
@pytest.mark.parametrize("entry_point", ["stream", "handler"])
def test_resource_operator_looks_up_and_resolves_once(
    monkeypatch: pytest.MonkeyPatch, name: str, category: str, entry_point: str
) -> None:
    class Resolver(ObjectResolver):
        def __init__(self) -> None:
            super().__init__(b"", {})
            self.dictionary_calls = 0

        def resolve_dict(self, value: object) -> PdfDict | None:
            self.dictionary_calls += 1
            return super().resolve_dict(value)

    state, sink = state_with_sink()
    resolver = Resolver()
    state.resolver = resolver
    lookups: list[tuple[str, str]] = []
    paints: list[PdfDict] = []
    resource: PdfDict = {"ca": 0.5} if name == "gs" else {"ShadingType": 2}

    def lookup(category: str, name: str) -> object:
        lookups.append((category, name))
        return resource

    monkeypatch.setattr(state, "lookup_page_resource", lookup)
    monkeypatch.setattr(sink, "paint_shading", lambda state, shading: paints.append(shading))
    if entry_point == "stream":
        state.stream_executor.consume(
            PdfStream(raw_data=f"/Resource {name}".encode()), {}, IDENTITY_MATRIX, 0
        )
    else:
        state.execute_operation(name, (PdfName.of("Resource"),), 0)
        if name == "gs":
            assert state.graphics.fill_opacity == 0.5
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
    with pytest.raises(PdfParseError, match="resource"):
        state.execute_operation(name, (PdfName.of("Resource"),), 0)
    assert state.capture_stream_state() == original


def test_stream_snapshot_restores_fields_before_unwinding_graphics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, sink = state_with_sink()
    state.resources = {"Original": 1}
    state.text_matrix = Matrix(2, 0, 0, 3, 4, 5)
    state.line_matrix = Matrix(3, 0, 0, 2, 5, 4)
    state.xobject_depth = 7
    state.compatibility_depth = 2
    original = state.capture_stream_state()
    state.resources = {}
    state.text_matrix = IDENTITY_MATRIX
    state.line_matrix = IDENTITY_MATRIX
    state.xobject_depth = 9
    state.compatibility_depth = 5
    state.op_q((), 0)
    state.graphics_stack_floor = 1
    observed = []
    restore = sink.restore_graphics

    def observe(current: ContentInterpreter) -> None:
        observed.append(
            (
                current.resources,
                current.text_matrix,
                current.line_matrix,
                current.graphics_stack_floor,
                current.xobject_depth,
                current.compatibility_depth,
            )
        )
        restore(current)

    monkeypatch.setattr(sink, "restore_graphics", observe)
    state.restore_stream_state(original)
    assert observed == [
        (
            original.resources,
            original.text_matrix,
            original.line_matrix,
            original.graphics_stack_floor,
            original.xobject_depth,
            original.compatibility_depth,
        )
    ]
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
    assert state.capture_stream_state() == original
    assert sink.saved == 0


def test_stream_snapshot_restores_compatibility_depth_after_direct_execution() -> None:
    state, _ = state_with_sink()
    state.compatibility_depth = 2
    original = state.capture_stream_state()
    state.execute_operation("BX", (), 0)
    assert state.compatibility_depth == 3
    state.restore_stream_state(original)
    assert state.compatibility_depth == 2
    state.execute_operation("EX", (), 0)
    assert state.compatibility_depth == 1
    state.restore_stream_state(original)
    assert state.compatibility_depth == 2


@pytest.mark.parametrize("value", [float("inf"), float("nan"), 10**400, True])
@pytest.mark.parametrize("operator", ["w", "sc", "TJ"])
def test_content_numeric_validation_rejects_unrepresentable_values_without_mutation(
    value: Any, operator: str
) -> None:
    state, _ = state_with_sink()
    original = state.capture_stream_state()
    operands = ([value],) if operator == "TJ" else (value,)
    with pytest.raises(PdfParseError):
        state.execute_operation(operator, operands, 0)
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
    state.graphics.current_decoder = font
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
    state.stream_executor.consume(
        PdfStream(raw_data=b"/Child0 Do"), {"XObject": objects}, IDENTITY_MATRIX, 0
    )
    assert len(sink.colors) == 12


@pytest.mark.parametrize("content", [b"q", b"/Span BMC"])
def test_unbalanced_scopes_are_rejected_and_restored(content: bytes) -> None:
    state, sink = state_with_sink()
    with pytest.raises(PdfParseError, match="unbalanced"):
        state.stream_executor.consume(PdfStream(raw_data=content), {}, IDENTITY_MATRIX, 0)
    assert not state.stack
    assert not state.marked_content_stack
    assert sink.saved == 0


def test_uncolored_type3_glyph_ignores_color_setting() -> None:
    state, _ = state_with_sink()
    state.type3_uncolored = True
    execute_content(state, b"1 0 0 rg /DeviceRGB cs", 0)
    assert state.graphics.fill_color == (0.0,)
    assert state.graphics.fill_space.kind == "DeviceGray"


@pytest.mark.parametrize(
    "content", [b"/Missing cs", b"/DeviceRGB cs 1 sc", b"/Pattern cs /Missing scn"]
)
def test_invalid_color_resources_and_operands_rejected(content: bytes) -> None:
    state, _ = state_with_sink()
    with pytest.raises(PdfParseError):
        execute_content(state, content, 0)


def test_inline_image_rejects_declared_length_mismatch() -> None:
    with pytest.raises(PdfParseError, match="data length"):
        list(iter_content_operations(PdfLexer(b"BI /W 3 /H 1 /BPC 8 /CS /G ID a EI")))


def test_lab_components_use_numeric_pdf_ranges() -> None:
    state, _ = state_with_sink()
    params: PdfDict = {"WhitePoint": [1, 1, 1], "Range": [-2, 2, -3, 3]}
    state.resources = {"ColorSpace": {"Test": [PdfName.of("Lab"), params]}}
    execute_content(state, b"/Test cs 150 4 -4 sc", 0)
    assert state.graphics.fill_color == (100.0, 2.0, -3.0)
    params["Range"] = [PdfString(b"-2"), 2, -3, 3]
    with pytest.raises(PdfParseError, match="Range"):
        execute_content(state, b"/Test cs", 0)


def test_frame_dispatch_returns_child_before_parsing_later_parent_tokens() -> None:
    state, sink = state_with_sink()
    child_stream = PdfStream(
        raw_data=b"0.5 g 0 0 1 1 re f",
        dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 1, 1]},
    )
    frame = ContentStreamFrame(
        PdfStream(raw_data=b"BX /Child Do ]"),
        {"XObject": {"Child": child_stream}},
        IDENTITY_MATRIX,
        0,
        None,
    )
    state.stream_executor.enter(frame)
    try:
        child = state.stream_executor.dispatch_frame(frame)
        assert child is not None
        assert child.stream is child_stream
        assert frame.lexer is not None
        assert frame.lexer.pos == len(b"BX /Child Do")
        assert state.compatibility_depth == 1
        assert sink.colors == []
        state.stream_executor.enter(child)
        try:
            assert state.compatibility_depth == 0
            assert state.stream_executor.dispatch_frame(child) is None
        finally:
            state.stream_executor.exit(child)
        assert sink.colors == [(0.5,)]
        assert state.compatibility_depth == 1
        with pytest.raises(PdfParseError, match="unexpected delimiter"):
            state.stream_executor.dispatch_frame(frame)
    finally:
        state.stream_executor.exit(frame)
    assert state.compatibility_depth == 0
    assert sink.saved == 0


@pytest.mark.parametrize(
    ("content", "message", "final_depth"),
    [
        (b"1 BX", "requires 0 operands", 0),
        (b"EX", "unmatched EX", 0),
        (b"BX", "unterminated compatibility", 1),
        (b"BX 1", "ends with operands", 1),
    ],
)
def test_frame_execution_retains_validation_and_eof_errors(
    content: bytes, message: str, final_depth: int
) -> None:
    state, _ = state_with_sink()
    frame = ContentStreamFrame(PdfStream(raw_data=content), {}, IDENTITY_MATRIX, 0, None)
    state.stream_executor.enter(frame)
    try:
        with pytest.raises(PdfParseError, match=message):
            state.stream_executor.dispatch_frame(frame)
        assert state.compatibility_depth == final_depth
    finally:
        state.stream_executor.exit(frame)


def test_operation_iterator_yields_before_later_errors_and_keeps_unknown_operators() -> None:
    lexer = PdfLexer(b"BX 1 extension EX ]")
    operations = iter_content_operations(lexer)
    assert lexer.pos == 0
    assert next(operations) == ("BX", ())
    assert lexer.pos == len(b"BX")
    assert next(operations) == ("extension", (1,))
    assert next(operations) == ("EX", ())
    with pytest.raises(PdfParseError, match="unexpected delimiter"):
        next(operations)


def test_operation_iterator_leaves_semantic_validation_to_execution() -> None:
    lexer = PdfLexer(b"1 2 cm")
    name, operands = next(iter_content_operations(lexer))
    assert (name, operands) == ("cm", (1, 2))
    state, _ = state_with_sink()
    with pytest.raises(PdfParseError, match="cm requires 6 operands"):
        state.execute_operation(name, operands, 0)


def test_operation_iterator_resumes_at_next_operation_without_prior_operands() -> None:
    lexer = PdfLexer(b"1 w 2 w")
    first = iter_content_operations(lexer)
    assert next(first) == ("w", (1,))
    assert lexer.pos == len(b"1 w")
    assert list(iter_content_operations(lexer)) == [("w", (2,))]


@pytest.mark.parametrize("suffix", [b"", b" BI /W 3 /H 1 /BPC 8 /CS /G ID a EI"])
def test_inline_only_validator_ignores_unrelated_content(suffix: bytes) -> None:
    # The invalid BI text in comments, names, strings, and containers is inert.
    prefix = b"% BI invalid\n /BI (BI invalid) <4249> [(BI) /BI] << /Key (BI) >> ] 1 unknown 1 2 cm"
    if suffix:
        with pytest.raises(PdfParseError, match="data length"):
            validate_inline_images(prefix + suffix)
    else:
        validate_inline_images(prefix + b" BI /W 1 /H 1 /BPC 8 /CS /G ID a EI")
