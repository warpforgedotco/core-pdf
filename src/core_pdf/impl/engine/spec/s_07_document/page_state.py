# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Protocol, cast

from core_pdf.impl.engine.spec.s_07_content import TextState
from core_pdf.impl.engine.spec.s_07_document.protocols import LayersDocumentProtocol
from core_pdf.impl.engine.spec.s_07_objects.object_cache import CachedPdfObject
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.objects import PdfStream
from core_pdf.impl.types import PdfDict

if TYPE_CHECKING:
    from core_pdf.impl.engine.spec.s_07_content.capture import CapturedLine
    from core_pdf.impl.engine.spec.s_07_document.document import PdfDocument


class PageStateHost(Protocol):
    document: PdfDocument
    page_dict: PdfDict
    page_number: int
    contents: CachedPdfObject | None
    state: TextState | None
    graphics: TextState | None
    grid_lines: list[CapturedLine] | None

    @property
    def content_streams(self) -> tuple[PdfStream, ...]: ...

    @property
    def cached_resources(self) -> PdfDict: ...

    def consume_contents(self, state: TextState) -> None: ...

    def get_state(self) -> TextState: ...

    def capture_text_state(self) -> TextState: ...

    def get_graphics(self) -> TextState: ...

    def get_text_and_graphics_state(self) -> TextState: ...


def page_hidden_layers(page: PageStateHost) -> frozenset[str]:
    return cast(LayersDocumentProtocol, page.document).oc_hidden_layers()


class PageStateMixin:
    state: TextState | None
    graphics: TextState | None
    grid_lines: list[CapturedLine] | None

    def collect_content_streams(self: PageStateHost) -> tuple[PdfStream, ...]:
        queue: deque[object] = deque()
        try:
            contents = self.document.resolver.resolve(self.contents)
        except PdfParseError:
            return ()
        if isinstance(contents, (list, tuple)):
            queue.extend(contents)
        elif contents is not None:
            queue.append(contents)
        streams: list[PdfStream] = []
        while queue:
            try:
                stream = self.document.resolver.resolve(queue.popleft())
            except PdfParseError:
                continue
            if isinstance(stream, (list, tuple)):
                queue.extendleft(reversed(stream))
                continue
            if isinstance(stream, PdfStream):
                stream = self.document.resolver.resolve_stream(stream)
                streams.append(stream)
        return tuple(streams)

    def iter_content_streams(self: PageStateHost) -> Iterator[PdfStream]:
        yield from self.content_streams

    def consume_contents(self: PageStateHost, state: TextState) -> None:
        if self.contents is None:
            return
        resources = self.cached_resources
        content_streams = self.content_streams
        try:
            contents_obj = self.document.resolver.resolve(self.contents)
        except PdfParseError:
            contents_obj = None
        can_skip_bad_stream = (
            len(content_streams) > 1
            or isinstance(contents_obj, (list, tuple))
            or (self.document.xref_was_recovered or self.document.page_tree_was_recovered)
        )
        if len(content_streams) > 1:
            try:
                data = b"\n".join(stream.data for stream in content_streams)
                state.consume_stream(
                    PdfStream(raw_data=data, decoded_data=data), resources, state.ctm, 0
                )
                return
            except PdfParseError:
                if not can_skip_bad_stream:
                    raise

        for stream in content_streams:
            try:
                state.consume_stream(stream, resources, state.ctm, 0)
            except PdfParseError:
                if can_skip_bad_stream:
                    continue
                raise

    def get_state(self: PageStateHost) -> TextState:
        if self.state is None:
            state = TextState(
                cast(Any, self.document),
                self.page_dict,
                capture_glyphs=True,
                capture_glyph_bitmaps=False,
                capture_graphics=True,
                hidden_layers=page_hidden_layers(self),
                decoder_cache=self.document.decoder_cache,
            )
            self.consume_contents(state)
            self.state = state
            self.graphics = state
        cached_state = self.state
        return cached_state

    def capture_text_state(self: PageStateHost) -> TextState:
        state = TextState(
            cast(Any, self.document),
            self.page_dict,
            capture_glyphs=True,
            capture_graphics=True,
            hidden_layers=page_hidden_layers(self),
            decoder_cache=self.document.decoder_cache,
        )
        self.consume_contents(state)
        return state

    def get_text_and_graphics_state(self: PageStateHost) -> TextState:
        if self.state is None and self.graphics is None:
            state = self.capture_text_state()
            self.state = state
            self.graphics = state
            return state
        state = self.get_state()
        if self.graphics is None:
            self.get_graphics()
        return state

    def get_graphics(self: PageStateHost) -> TextState:
        if self.graphics is None:
            state = TextState(
                cast(Any, self.document),
                self.page_dict,
                capture_graphics=True,
                hidden_layers=page_hidden_layers(self),
                decoder_cache=self.document.decoder_cache,
            )
            self.consume_contents(state)
            self.graphics = state
        graphics = self.graphics
        return graphics

    def get_grid_lines(self: PageStateHost) -> list[CapturedLine]:
        if self.grid_lines is None:
            self.grid_lines = self.get_graphics().lines
        grid_lines = self.grid_lines
        return grid_lines


__all__ = ("PageStateMixin",)
