# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any, cast

import pytest

from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_content.model import PdfPath
from core_pdf_spec.s_07_content.streams import ContentStreamExecutor, ContentStreamFrame, StreamKey
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from core_pdf_spec.types import PdfName, PdfReference, Rectangle


class PaintSink:
    def __init__(self) -> None:
        self.paints: list[tuple[float, float, str | None]] = []

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None

    def paint_path(
        self, state: ContentInterpreter, path: PdfPath, kind: str, fill_rule: str
    ) -> None:
        self.paints.append(
            (state.graphics.fill_opacity, state.graphics.stroke_opacity, state.graphics.blend_mode)
        )


def internal_state() -> tuple[ContentInterpreter, PaintSink]:
    sink = PaintSink()
    return ContentInterpreter(ObjectResolver(b"", {}), cast(Any, sink), cast(Any, None)), sink


def internal_form(group: PdfDict | None = None) -> PdfStream:
    dictionary: PdfDict = {"Subtype": PdfName.of("Form"), "BBox": [0, 0, 1, 1]}
    if group is not None:
        dictionary["Group"] = group
    return PdfStream(raw_data=b"0 0 1 1 re B", dictionary=dictionary)


@pytest.mark.parametrize("isolated", [None, False, True])
@pytest.mark.parametrize("indirect", [False, True])
def test_transparency_group_resolves_isolation_with_nonisolated_default(
    isolated: bool | None, indirect: bool
) -> None:
    state, _ = internal_state()
    group: PdfDict = {"S": PdfName.of("Transparency")}
    if isolated is not None:
        group["I"] = PdfReference(1, 0) if indirect else isolated
        cast(ObjectResolver, state.resolver).objects[key_for(1, 0)] = isolated
    form = internal_form(group)
    if indirect:
        cast(ObjectResolver, state.resolver).objects[key_for(2, 0)] = group
        form.dictionary["Group"] = PdfReference(2, 0)
    frame = state.append_form_xobject(form, 0)
    assert frame is not None
    assert frame.group_alpha == 1.0
    assert frame.group_isolated is (isolated is True)


@pytest.mark.parametrize("isolated", [False, True])
def test_transparency_group_resets_child_alpha_and_blend_then_restores_caller(
    isolated: bool,
) -> None:
    state, sink = internal_state()
    state.graphics.fill_opacity = 0.3
    state.graphics.stroke_opacity = 0.6
    state.graphics.blend_mode = "Multiply"
    form = internal_form({"S": PdfName.of("Transparency"), "I": isolated})
    state.resources = {"XObject": {"F": form}}
    state.stream_executor.consume(
        PdfStream(raw_data=b"/F Do 0 0 1 1 re B"), state.resources, IDENTITY_MATRIX, 0
    )
    assert sink.paints == [(1.0, 1.0, None), (0.3, 0.6, "Multiply")]
    assert state.graphics.fill_opacity == 0.3
    assert state.graphics.stroke_opacity == 0.6
    assert state.graphics.blend_mode == "Multiply"


@pytest.mark.parametrize("group", [None, {}, {"S": PdfName.of("Other")}])
def test_ordinary_form_inherits_transparency_state(group: PdfDict | None) -> None:
    state, sink = internal_state()
    state.graphics.fill_opacity = 0.3
    state.graphics.stroke_opacity = 0.6
    state.graphics.blend_mode = "Multiply"
    state.resources = {"XObject": {"F": internal_form(group)}}
    frame = state.append_xobject(PdfName.of("F"), 0)
    assert frame is not None
    assert frame.group_alpha is None
    state.stream_executor.consume(PdfStream(raw_data=b"/F Do"), state.resources, IDENTITY_MATRIX, 0)
    assert sink.paints == [(0.3, 0.6, "Multiply")]


def test_existing_frame_and_queue_callers_keep_isolated_group_default() -> None:
    state, _ = internal_state()
    stream = PdfStream(raw_data=b"")
    frame = ContentStreamFrame(stream, {}, IDENTITY_MATRIX, 1, None, 0.5)
    assert frame.group_isolated is True
    queued = state.stream_executor.queue(stream, {}, IDENTITY_MATRIX, 1, group_alpha=0.5)
    assert queued is not None
    assert queued.group_isolated is True
    explicit = ContentStreamFrame(stream, {}, IDENTITY_MATRIX, 1, None, 0.5, group_isolated=False)
    assert explicit.group_isolated is False


def test_form_isolation_preserves_existing_executor_queue_override_signature() -> None:
    class ExistingExecutor(ContentStreamExecutor):
        def queue(
            self,
            stream: PdfStream,
            resources: PdfDict,
            ctm: Matrix,
            depth: int,
            *,
            clip_bbox: Rectangle | None = None,
            form_bbox_operand: object = None,
            group_alpha: float | None = None,
            stream_key: StreamKey | None = None,
        ) -> ContentStreamFrame | None:
            return super().queue(
                stream,
                resources,
                ctm,
                depth,
                clip_bbox=clip_bbox,
                form_bbox_operand=form_bbox_operand,
                group_alpha=group_alpha,
                stream_key=stream_key,
            )

    state, _ = internal_state()
    state.stream_executor = ExistingExecutor(state)
    frame = state.append_form_xobject(
        internal_form({"S": PdfName.of("Transparency"), "K": True}), 0
    )
    assert frame is not None
    assert frame.group_alpha == 1.0
    assert frame.group_isolated is False
    assert frame.group_knockout is True
