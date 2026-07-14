# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence

from core_pdf.impl.engine.extraction.tables.types import (
    ExtractTablesResult,
    Rect,
    TableCacheKey,
    TableExtractionResult,
)

if TYPE_CHECKING:
    from core_pdf.impl.engine.layout.models import TextRun
    from core_pdf.impl.engine.spec.s_07_content.capture import CapturedLine
    from core_pdf.impl.models import TextSpan


class PageTableHost(Protocol):
    grid_lines: list[CapturedLine] | None
    tables: dict[TableCacheKey, TableExtractionResult]

    @property
    def rotation(self) -> int: ...

    @property
    def width(self) -> float: ...

    @property
    def height(self) -> float: ...

    @property
    def chars(self) -> list[TextRun]: ...

    def get_text_spans(self) -> list[TextSpan]: ...

    def crop(self, bbox: tuple[float, float, float, float]) -> PageTableHost: ...

    def get_grid_lines(self) -> list[CapturedLine]: ...

    def display_text_span_chars(
        self,
    ) -> dict[int, list[tuple[str, float, float, float, float]]]: ...

    def extract_tables_core(self, **kwargs: object) -> TableExtractionResult: ...

    def extract_tables(self, **kwargs: object) -> ExtractTablesResult: ...

    def extract_table_bboxes(
        self, **kwargs: object
    ) -> list[Rect | None]: ...

    def table_extraction_payload(self, **kwargs: object) -> TableExtractionResult: ...

    def extract_table_rows(
        self,
        bbox: tuple[float, float, float, float],
        *,
        min_text_coverage: float = 0.85,
        flavor: str | Sequence[str] = ("lattice", "stream"),
        **extract_options: object,
    ) -> list[list[str]]: ...


__all__ = ("PageTableHost",)
