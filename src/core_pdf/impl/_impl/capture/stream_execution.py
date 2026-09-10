# SPDX-License-Identifier: AGPL-3.0-only
"""Reader limits and error handling for nested content execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core_pdf.impl._impl.capture.tolerant_state import RecoveringTextState

from core_pdf.impl._impl.capture.recovery import dispatch_operations
from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.stream_execution import ContentStreamExecutor
from core_pdf_spec.s_07_content.stream_state import ContentStreamFrame, StreamKey
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
    ) -> None:
        if depth > 10 or (stream_key or self.execution_key(stream)) in self.active_streams:
            return
        super().queue(
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

    def dispatch_frame(self, frame: ContentStreamFrame) -> None:
        state = self.state
        assert frame.lexer is not None
        dispatch_operations(
            frame.lexer, state.op_handlers.get, frame.depth, recovery=state.recovery
        )

    def handle_parse_error(self, frame: ContentStreamFrame, error: PdfParseError) -> None:
        if not frame.is_form:
            raise error
