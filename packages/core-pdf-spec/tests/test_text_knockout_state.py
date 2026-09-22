# SPDX-License-Identifier: AGPL-3.0-only

from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_content.model import GraphicsState, PdfPath, TilingPattern
from core_pdf_spec.s_07_content.streams import StreamState
from core_pdf_spec.s_07_filters.errors import FilterUnsupportedError
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import CachedPdfObject, PdfDict
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf_spec.types import PdfName, PdfReference


class TextSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, bool, bool]] = []
        self.patterns: list[TilingPattern] = []

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None

    def text_boundary(self, state: ContentInterpreter, kind: str) -> None:
        self.events.append((kind, state.graphics.text_knockout, state.in_text_object))

    def paint_path(
        self, state: ContentInterpreter, path: PdfPath, kind: str, fill_rule: str
    ) -> None:
        self.events.append((kind, state.graphics.text_knockout, state.in_text_object))
        if isinstance(state.graphics.fill_pattern, TilingPattern):
            self.patterns.append(state.graphics.fill_pattern)


def make_state() -> tuple[ContentInterpreter, TextSink]:
    sink = TextSink()
    return ContentInterpreter(ObjectResolver(b"", {}), cast(Any, sink), cast(Any, None)), sink


def test_text_knockout_defaults_preserve_positional_snapshot_construction() -> None:
    graphics = GraphicsState()
    snapshot = StreamState(graphics, {}, IDENTITY_MATRIX, IDENTITY_MATRIX, 0, 0, 0, 0)
    assert graphics.text_knockout is True
    assert snapshot.initial_text_knockout is True
    assert snapshot.in_text_object is False


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("indirect", [False, True])
def test_text_knockout_resolves_booleans_and_preserves_q_Q(value: bool, indirect: bool) -> None:
    state, _ = make_state()
    state.graphics.text_knockout = not value
    state.op_q((), 0)
    cast(ObjectResolver, state.resolver).objects[key_for(1, 0)] = value
    state.apply_extgstate({"TK": PdfReference(1, 0) if indirect else value})
    assert state.graphics.text_knockout is value
    state.apply_extgstate({})
    state.apply_extgstate({"TK": None})
    assert state.graphics.text_knockout is value
    state.op_Q((), 0)
    assert state.graphics.text_knockout is not value


@pytest.mark.parametrize("value", [0, 1, 0.5, "true", PdfName.of("true"), [], {}])
@pytest.mark.parametrize("indirect", [False, True])
def test_text_knockout_rejects_nonbooleans_outside_text(value: object, indirect: bool) -> None:
    state, _ = make_state()
    cast(ObjectResolver, state.resolver).objects[key_for(1, 0)] = cast(CachedPdfObject, value)
    with pytest.raises(ValueError, match="invalid text knockout flag"):
        state.apply_extgstate({"ca": 0.4, "TK": PdfReference(1, 0) if indirect else value})
    assert state.graphics.text_knockout is True
    assert state.graphics.fill_opacity == 0.4


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("value", [False, True, 0, 1, 0.5, "true", PdfName.of("true"), [], {}])
def test_TK_is_ignored_inside_text_while_other_graphics_changes_persist(
    initial: bool, value: object
) -> None:
    state, _ = make_state()
    state.graphics.text_knockout = initial
    state.graphics.fill_opacity = 0.2
    state.graphics.blend_mode = "Screen"
    state.op_BT((), 0)
    assert state.graphics.fill_opacity == 0.2
    assert state.graphics.blend_mode == "Screen"
    state.apply_extgstate({"TK": value, "ca": 0.6, "BM": PdfName.of("Multiply"), "AIS": True})
    state.op_w((4,), 0)
    state.op_ET((), 0)
    assert state.graphics.text_knockout is initial
    assert state.graphics.fill_opacity == 0.6
    assert state.graphics.blend_mode == "Multiply"
    assert state.graphics.alpha_is_shape is True
    assert state.graphics.line_width == 4
    assert state.in_text_object is False
    state.apply_extgstate({"TK": not initial})
    assert state.graphics.text_knockout is not initial


def test_ignored_TK_does_not_resolve_indirect_values(monkeypatch: pytest.MonkeyPatch) -> None:
    state, _ = make_state()
    reference = PdfReference(1, 0)
    resolve = ObjectResolver.resolve

    def guarded_resolve(self: ObjectResolver, value: object) -> object:
        if value is reference:
            raise AssertionError("ignored TK was resolved")
        return resolve(self, value)

    monkeypatch.setattr(ObjectResolver, "resolve", guarded_resolve)
    state.resources = {"ExtGState": {"Ignored": {"TK": reference, "ca": 0.5}}}
    state.op_BT((), 0)
    state.op_gs((PdfName.of("Ignored"),), 0)
    state.op_ET((), 0)
    assert state.graphics.text_knockout is True
    assert state.graphics.fill_opacity == 0.5


def test_text_object_scope_is_not_part_of_q_Q_graphics_state() -> None:
    state, _ = make_state()
    state.op_q((), 0)
    state.op_BT((), 0)
    state.op_Q((), 0)
    assert state.in_text_object is True
    state.apply_extgstate({"TK": False})
    assert state.graphics.text_knockout is True
    state.op_q((), 0)
    state.op_ET((), 0)
    state.op_Q((), 0)
    assert state.in_text_object is False
    state.apply_extgstate({"TK": False})
    assert state.graphics.text_knockout is False


@pytest.mark.parametrize("failure", [False, True])
def test_child_stream_resets_and_restores_text_object_scope(failure: bool) -> None:
    state, sink = make_state()
    state.graphics.text_knockout = False
    state.op_BT((), 0)
    snapshot = state.capture_stream_state()
    child = PdfStream(
        raw_data=b"/True gs 0 0 1 1 re f BT /False gs 0 0 1 1 re f "
        + (b"unknown" if failure else b"ET")
    )
    resources: PdfDict = {"ExtGState": {"True": {"TK": True}, "False": {"TK": False}}}
    if failure:
        with pytest.raises(PdfParseError, match="unknown content operator"):
            state.stream_executor.consume(child, resources, IDENTITY_MATRIX, 1)
    else:
        state.stream_executor.consume(child, resources, IDENTITY_MATRIX, 1)
    assert [event for event in sink.events if event[0] == "fill"] == [
        ("fill", True, False),
        ("fill", True, True),
    ]
    assert state.capture_stream_state() == snapshot
    assert not state.stream_executor.active_streams


def test_failed_child_entry_preserves_parent_text_scope() -> None:
    state, sink = make_state()
    state.op_BT((), 0)
    snapshot = state.capture_stream_state()
    stream = PdfStream(raw_data=b"", spec={"Filter": PdfName.of("Unknown")})
    with pytest.raises(FilterUnsupportedError):
        state.stream_executor.consume(stream, {}, IDENTITY_MATRIX, 1)
    assert state.capture_stream_state() == snapshot
    assert sink.events == [("begin", True, True)]


def type3_font(stream: PdfStream, resources: PdfDict | None = None) -> Any:
    return SimpleNamespace(
        font={"CharProcs": {"A": stream, "B": stream}, "Resources": resources or {}},
        font_matrix=IDENTITY_MATRIX,
        glyph_name=lambda code: chr(code),
        glyph_advance_vector=lambda code, **kwargs: (5.0, 0.0),
    )


def test_type3_boundaries_enclose_all_commands_in_each_executed_glyph() -> None:
    state, sink = make_state()
    state.op_BT((), 0)
    glyph = PdfStream(raw_data=b"0 0 d0 /False gs 0 0 1 1 re f BT /True gs 0 0 1 1 re f ET")
    font = type3_font(glyph, {"ExtGState": {"False": {"TK": False}, "True": {"TK": True}}})
    state.render_type3_glyphs(b"ABC", font)
    assert sink.events == [("begin", True, True)] + 2 * [
        ("type3-glyph-begin", True, True),
        ("fill", False, False),
        ("begin", False, True),
        ("fill", False, True),
        ("end", False, True),
        ("stream-end", False, False),
        ("type3-glyph-end", True, True),
    ]
    assert state.text_matrix.e == 15
    assert state.in_text_object is True
    assert state.graphics.text_knockout is True


@pytest.mark.parametrize("failure", ["decode", "dispatch"])
def test_type3_boundary_closes_after_failed_CharProc_and_state_restoration(failure: str) -> None:
    state, sink = make_state()
    state.op_BT((), 0)
    state.type3_uncolored = True
    glyph = (
        PdfStream(spec={"Filter": PdfName.of("Unknown")})
        if failure == "decode"
        else PdfStream(raw_data=b"0 0 d0 /False gs BT unknown")
    )
    snapshot = state.capture_stream_state()
    with pytest.raises(FilterUnsupportedError if failure == "decode" else PdfParseError):
        state.render_type3_glyphs(b"A", type3_font(glyph, {"ExtGState": {"False": {"TK": False}}}))
    assert sink.events[1] == ("type3-glyph-begin", True, True)
    assert sink.events[-1] == ("type3-glyph-end", True, True)
    assert state.capture_stream_state() == snapshot
    assert state.type3_uncolored is True
    assert not state.stream_executor.active_streams


def test_invisible_type3_text_emits_no_glyph_boundaries() -> None:
    state, sink = make_state()
    state.graphics.render_mode = 3
    state.render_type3_glyphs(b"A", type3_font(PdfStream(raw_data=b"0 0 1 1 re f")))
    assert not sink.events


def test_pattern_retains_defining_stream_initial_TK_across_nested_and_later_changes() -> None:
    state, sink = make_state()
    pattern = PdfStream(
        dictionary={
            "PatternType": 1,
            "PaintType": 1,
            "TilingType": 1,
            "BBox": [0, 0, 1, 1],
            "XStep": 1,
            "YStep": 1,
            "Resources": {},
        }
    )
    form = PdfStream(
        raw_data=b"/True gs /Pattern cs /P scn 0 0 1 1 re f",
        dictionary={
            "Subtype": PdfName.of("Form"),
            "BBox": [0, 0, 1, 1],
            "Group": {"S": PdfName.of("Transparency"), "K": True},
            "Resources": {"Pattern": {"P": pattern}, "ExtGState": {"True": {"TK": True}}},
        },
    )
    resources: PdfDict = {
        "Pattern": {"P": pattern},
        "XObject": {"F": form},
        "ExtGState": {"False": {"TK": False}},
    }
    state.stream_executor.consume(
        PdfStream(raw_data=b"q /False gs /F Do /Pattern cs /P scn 0 0 1 1 re f Q"),
        resources,
        IDENTITY_MATRIX,
        0,
    )
    assert [pattern.text_knockout for pattern in sink.patterns] == [False, True]
    assert [event[1] for event in sink.events if event[0] == "fill"] == [True, False]
    assert state.initial_text_knockout is True
