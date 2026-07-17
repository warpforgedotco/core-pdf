# SPDX-License-Identifier: AGPL-3.0-only
"""Lightweight entry points for page text extraction.

The extraction implementation includes OCR, rendering, and layout analysis. Import it
only when an extraction method is called so opening the public API does not initialize
the entire extraction stack.
"""

from __future__ import annotations

from typing import Any, cast

from core_pdf.impl.engine.extraction.common.page_content import PageContentMixin


def _implementation() -> type[Any]:
    from core_pdf.impl.engine.extraction.page_text.mixin import (
        PageExtractionMixin as Implementation,
    )

    return Implementation


class PageExtractionMixin(PageContentMixin):
    def get_page_profile(self) -> Any:
        return _implementation().get_page_profile(cast(Any, self))

    def get_text_lines(self) -> Any:
        return _implementation().get_text_lines(cast(Any, self))

    def get_drawings(self) -> list[dict[str, Any]]:
        return _implementation().get_drawings(cast(Any, self))

    def get_text_spans(self) -> Any:
        return _implementation().get_text_spans(cast(Any, self))

    def extract_text(self) -> str:
        return _implementation().extract_text(cast(Any, self))

    def extract_resolved_lines(self) -> list[dict[str, Any]]:
        return _implementation().extract_resolved_lines(cast(Any, self))

    def to_markdown(self) -> str:
        return _implementation().to_markdown(cast(Any, self))

    def render(self, options: Any = None) -> Any:
        return _implementation().render(cast(Any, self), options)


__all__ = ("PageExtractionMixin",)
