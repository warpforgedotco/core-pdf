# SPDX-License-Identifier: AGPL-3.0-only
"""Consumer contract for semantic PDF content execution.

The interpreter supplies PDF state and source objects; consumers own recording,
projection, paint preparation, visibility policy and output products.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_08_graphics.color_spec import ImageColorSpec
from core_pdf.impl.spec.s_09_fonts.service import DecodedFontGlyph, FontService

if TYPE_CHECKING:
    from core_pdf.impl.spec.s_07_content.inline_images import InlineImage
    from core_pdf.impl.spec.s_07_content.marked_content import MarkedContentEntry
    from core_pdf.impl.spec.s_07_content.paths import PdfPath
    from core_pdf.impl.spec.s_07_content.state import TextState
    from core_pdf.impl.spec.s_07_content.stream_state import ContentStreamFrame


class ContentSink(Protocol):
    def show_text(
        self,
        state: TextState,
        text: str,
        data: bytes | memoryview,
        glyphs: tuple[DecodedFontGlyph, ...],
        decoder: FontService,
        adv_x: float,
        adv_y: float,
        /,
    ) -> None: ...

    def text_boundary(self, state: TextState, kind: str, /) -> None: ...

    def paint_path(self, state: TextState, path: PdfPath, kind: str, fill_rule: str, /) -> None: ...

    def clip_path(self, state: TextState, path: PdfPath, fill_rule: str, /) -> None: ...

    def paint_image(self, state: TextState, source: PdfStream, /) -> None: ...

    def paint_inline_image(self, state: TextState, image: InlineImage, /) -> None: ...

    def paint_shading(self, state: TextState, shading: PdfDict, /) -> None: ...

    def end_marked_content(self, state: TextState, entry: MarkedContentEntry, /) -> None: ...

    def save_graphics(self, state: TextState, /) -> None: ...

    def restore_graphics(self, state: TextState, /) -> None: ...

    def enter_stream(self, state: TextState, frame: ContentStreamFrame, /) -> None: ...

    def exit_stream(self, state: TextState, frame: ContentStreamFrame, /) -> None: ...

    def parse_color_space(self, value: object, /) -> ImageColorSpec: ...

    def convert_color(
        self, spec: ImageColorSpec, values: list[float], /
    ) -> tuple[float, ...] | None: ...

    def named_value(self, value: object, /, *, allow_text: bool = False) -> str | None: ...
