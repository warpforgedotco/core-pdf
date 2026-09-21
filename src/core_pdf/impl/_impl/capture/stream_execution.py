# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core_pdf.impl._impl.capture.tolerant_state import RecoveringTextState

from core_pdf.impl._impl.capture.recovery import iter_content_operations
from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.streams import ContentStreamExecutor, ContentStreamFrame, StreamKey
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_08_graphics.matrix import Matrix
from core_pdf_spec.types import Rectangle


class CaptureStreamExecutor(ContentStreamExecutor):
    state: RecoveringTextState

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
        if depth > 10 or (stream_key or self.execution_key(stream)) in self.active_streams:
            return None
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

    def enter(self, frame: ContentStreamFrame) -> bool:
        if (
            frame.depth > 10
            or (frame.stream_key or self.execution_key(frame.stream)) in self.active_streams
        ):
            return False
        return super().enter(frame)

    def dispatch_frame(self, frame: ContentStreamFrame) -> ContentStreamFrame | None:
        state = self.state
        assert frame.lexer is not None
        operator_names = frozenset(
            name.encode("latin-1")
            for name in (*state.internal_default_handlers, *state.operator_overrides)
        )
        for name, operands in iter_content_operations(
            frame.lexer,
            recovery=state.recovery,
            is_operator=operator_names.__contains__,
        ):
            child = state.execute_operation(name, operands, frame.depth)
            if child is not None:
                return child
        return None

    def handle_parse_error(self, frame: ContentStreamFrame, error: PdfParseError) -> None:
        if not frame.is_form:
            raise error
