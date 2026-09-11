# SPDX-License-Identifier: AGPL-3.0-only
"""Preserve Form group state through the core capture/display boundary."""

from types import SimpleNamespace

import pytest

from core_pdf.impl._impl.capture.interpreter import TextState
from core_pdf.impl._impl.document.recovery.resolver import ObjectResolver
from core_pdf.impl._impl.render.display import DisplayList
from core_pdf.impl._impl.render.model import DisplayListItem
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf_spec.types import PdfName


def internal_state(form: PdfStream) -> TextState:
    resolver = ObjectResolver(b"", {})
    state = TextState(SimpleNamespace(resolver=resolver, resolve=resolver.resolve))
    state.resources = {"XObject": {"F": form}}
    return state


def internal_form(group: PdfDict | None = None, *, content: bytes = b"0 0 1 1 re B") -> PdfStream:
    dictionary: PdfDict = {"Subtype": PdfName.of("Form"), "BBox": [0, 0, 1, 1]}
    if group is not None:
        dictionary["Group"] = group
    return PdfStream(raw_data=content, dictionary=dictionary)


@pytest.mark.parametrize("isolated", [None, False, True])
def test_group_capture_resets_inner_scalar_mask_and_forwards_outer_composite_state(
    isolated: bool | None,
) -> None:
    # ISO 32000-1 11.6.6: invoking alpha, soft mask and blend apply to the group.
    group: PdfDict = {"S": PdfName.of("Transparency")}
    if isolated is not None:
        group["I"] = isolated
    state = internal_state(internal_form(group))
    state.graphics.fill_opacity = 0.3
    state.graphics.stroke_opacity = 0.6
    state.graphics.blend_mode = "Multiply"
    state.group_alpha = 0.7
    state.stream_executor.consume(PdfStream(raw_data=b"/F Do"), state.resources, IDENTITY_MATRIX, 0)
    begin, paint, end = state.drawings
    assert begin.kind == "group-begin"
    assert end.kind == "group-end"
    assert begin.group_isolated is end.group_isolated is (isolated is True)
    assert begin.fill_opacity == end.fill_opacity == 0.3
    assert begin.blend_mode == end.blend_mode == "Multiply"
    assert begin.soft_mask_alpha == end.soft_mask_alpha == 0.7
    assert paint.fill_opacity == paint.stroke_opacity == 1.0
    assert paint.blend_mode is paint.soft_mask_alpha is None
    assert state.graphics.fill_opacity == 0.3
    assert state.graphics.stroke_opacity == 0.6
    assert state.graphics.blend_mode == "Multiply"
    assert state.group_alpha == 0.7

    display = DisplayList(1, 1)
    for drawing in state.drawings:
        display.append_captured_drawing(drawing)
    for item in (display.items[0], display.items[-1]):
        assert isinstance(item, DisplayListItem)
        assert item.data["group_isolated"] is (isolated is True)
        assert item.data["fill_opacity"] == 0.3
        assert item.data["soft_mask_alpha"] == 0.7
        assert item.data["blend_mode"] == "Multiply"


def test_nested_groups_capture_independent_isolation_and_restore_outer_state() -> None:
    inner = internal_form({"S": PdfName.of("Transparency"), "I": True})
    outer = internal_form(
        {"S": PdfName.of("Transparency")}, content=b"/InnerState gs /Inner Do 0 0 1 1 re f"
    )
    outer.dictionary["Resources"] = {
        "XObject": {"Inner": inner},
        "ExtGState": {"InnerState": {"ca": 0.4, "CA": 0.6, "BM": PdfName.of("Screen")}},
    }
    state = internal_state(outer)
    state.graphics.fill_opacity = 0.3
    state.graphics.blend_mode = "Multiply"
    state.group_alpha = 0.7
    state.stream_executor.consume(PdfStream(raw_data=b"/F Do"), state.resources, IDENTITY_MATRIX, 0)
    outer_begin, inner_begin, inner_paint, inner_end, outer_paint, outer_end = state.drawings
    assert outer_begin.group_isolated is outer_end.group_isolated is False
    assert inner_begin.group_isolated is inner_end.group_isolated is True
    assert outer_begin.fill_opacity == outer_end.fill_opacity == 0.3
    assert outer_begin.soft_mask_alpha == outer_end.soft_mask_alpha == 0.7
    assert inner_begin.fill_opacity == inner_end.fill_opacity == 0.4
    assert inner_begin.blend_mode == inner_end.blend_mode == "Screen"
    assert inner_begin.soft_mask_alpha is inner_end.soft_mask_alpha is None
    assert inner_paint.fill_opacity == inner_paint.stroke_opacity == 1.0
    assert inner_paint.blend_mode is None
    assert outer_paint.fill_opacity == 0.4
    assert outer_paint.blend_mode == "Screen"
    assert state.group_alpha == 0.7


def test_ordinary_form_keeps_inherited_mask_alpha_and_blend_without_group_markers() -> None:
    state = internal_state(internal_form())
    state.graphics.fill_opacity = 0.3
    state.graphics.stroke_opacity = 0.6
    state.graphics.blend_mode = "Multiply"
    state.group_alpha = 0.7
    state.stream_executor.consume(PdfStream(raw_data=b"/F Do"), state.resources, IDENTITY_MATRIX, 0)
    (paint,) = state.drawings
    assert paint.kind == "fillstroke"
    assert paint.fill_opacity == 0.3
    assert paint.stroke_opacity == 0.6
    assert paint.blend_mode == "Multiply"
    assert paint.soft_mask_alpha == 0.7
