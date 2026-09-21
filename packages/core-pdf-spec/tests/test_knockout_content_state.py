# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any, cast

import pytest

from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_content.model import PdfPath, TilingPattern
from core_pdf_spec.s_07_content.streams import ContentStreamFrame
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict, PdfObject
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf_spec.types import PdfName, PdfReference


class PaintSink:
    def __init__(self) -> None:
        self.states: list[tuple[bool, float, str | None]] = []
        self.patterns: list[TilingPattern] = []

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None

    def paint_path(
        self, state: ContentInterpreter, path: PdfPath, kind: str, fill_rule: str
    ) -> None:
        self.states.append(
            (state.graphics.alpha_is_shape, state.graphics.fill_opacity, state.graphics.blend_mode)
        )
        if isinstance(state.graphics.fill_pattern, TilingPattern):
            self.patterns.append(state.graphics.fill_pattern)


def internal_state() -> tuple[ContentInterpreter, PaintSink]:
    sink = PaintSink()
    return ContentInterpreter(ObjectResolver(b"", {}), cast(Any, sink), cast(Any, None)), sink


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("knockout", [None, False, True, 1, "true", PdfName.of("true")])
@pytest.mark.parametrize("indirect", [False, True])
def test_knockout_is_independent_boolean_group_attribute(
    isolated: bool, knockout: object, indirect: bool
) -> None:
    state, _ = internal_state()
    group: PdfDict = {"S": PdfName.of("Transparency"), "I": isolated}
    if knockout is not None:
        cast(ObjectResolver, state.resolver).objects[key_for(1, 0)] = cast(PdfObject, knockout)
        group["K"] = PdfReference(1, 0) if indirect else cast(PdfObject, knockout)
    frame = state.append_form_xobject(
        PdfStream(dictionary={"BBox": [0, 0, 1, 1], "Group": group}), 0
    )
    assert frame is not None
    assert frame.group_knockout is (knockout is True)
    assert frame.group_isolated is isolated


def test_ordinary_and_legacy_frames_do_not_gain_knockout() -> None:
    state, _ = internal_state()
    stream = PdfStream(dictionary={"BBox": [0, 0, 1, 1], "Group": {"K": True}})
    frame = state.append_form_xobject(stream, 0)
    assert frame is not None
    assert frame.group_alpha is None
    assert frame.group_knockout is False
    assert ContentStreamFrame(stream, {}, IDENTITY_MATRIX, 1, None, 0.5).group_knockout is False
    assert ContentStreamFrame(
        stream, {}, IDENTITY_MATRIX, 1, None, group_knockout=True
    ).group_knockout


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("indirect", [False, True])
def test_alpha_source_resolves_booleans_and_preserves_q_Q_state(
    value: bool, indirect: bool
) -> None:
    state, _ = internal_state()
    assert state.graphics.alpha_is_shape is False
    state.graphics.alpha_is_shape = not value
    state.op_q((), 0)
    cast(ObjectResolver, state.resolver).objects[key_for(1, 0)] = value
    state.apply_extgstate({"AIS": PdfReference(1, 0) if indirect else value})
    assert state.graphics.alpha_is_shape is value
    state.apply_extgstate({"ca": 0.4})
    state.apply_extgstate({"AIS": None})
    assert state.graphics.alpha_is_shape is value
    state.op_Q((), 0)
    assert state.graphics.alpha_is_shape is not value


@pytest.mark.parametrize("value", [0, 1, 0.5, "true", PdfName.of("true"), [], {}])
def test_alpha_source_rejects_nonboolean_values_without_changing_flag(value: object) -> None:
    state, _ = internal_state()
    state.graphics.alpha_is_shape = True
    with pytest.raises(ValueError, match="invalid alpha source flag"):
        state.apply_extgstate({"AIS": value})
    assert state.graphics.alpha_is_shape is True


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("knockout", [False, True])
def test_transparency_group_inherits_AIS_while_resetting_alpha_and_blend(
    initial: bool, knockout: bool
) -> None:
    state, sink = internal_state()
    state.graphics.alpha_is_shape = initial
    state.graphics.fill_opacity = 0.3
    state.graphics.blend_mode = "Multiply"
    form = PdfStream(
        raw_data=b"0 0 1 1 re f q /Other gs 0 0 1 1 re f Q 0 0 1 1 re f",
        dictionary={
            "Subtype": PdfName.of("Form"),
            "BBox": [0, 0, 1, 1],
            "Group": {"S": PdfName.of("Transparency"), "K": knockout},
            "Resources": {"ExtGState": {"Other": {"AIS": not initial, "ca": 0.6}}},
        },
    )
    state.stream_executor.consume(
        PdfStream(raw_data=b"/F Do 0 0 1 1 re f"), {"XObject": {"F": form}}, IDENTITY_MATRIX, 0
    )
    assert sink.states == [
        (initial, 1.0, None),
        (not initial, 0.6, None),
        (initial, 1.0, None),
        (initial, 0.3, "Multiply"),
    ]
    assert state.graphics.alpha_is_shape is initial
    assert state.initial_alpha_is_shape is False


def test_pattern_retains_defining_stream_initial_AIS_across_nested_and_later_state_changes() -> (
    None
):
    state, sink = internal_state()
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
        raw_data=b"/False gs /Pattern cs /P scn 0 0 1 1 re f",
        dictionary={
            "Subtype": PdfName.of("Form"),
            "BBox": [0, 0, 1, 1],
            "Resources": {"Pattern": {"P": pattern}, "ExtGState": {"False": {"AIS": False}}},
        },
    )
    resources: PdfDict = {
        "Pattern": {"P": pattern},
        "XObject": {"F": form},
        "ExtGState": {"True": {"AIS": True}},
    }
    state.stream_executor.consume(
        PdfStream(raw_data=b"/True gs /F Do /Pattern cs /P scn 0 0 1 1 re f"),
        resources,
        IDENTITY_MATRIX,
        0,
    )
    assert [pattern.alpha_is_shape for pattern in sink.patterns] == [True, False]
    assert [entry[0] for entry in sink.states] == [False, True]
