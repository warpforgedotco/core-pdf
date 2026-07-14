# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, cast

from core_pdf.impl.engine.extraction.common.page_content import PageContentMixin
from core_pdf.impl.engine.extraction.page_text.engine import (
    PageExtractionResult,
    build_page_extraction_result,
)
from core_pdf.impl.engine.extraction.page_text.mixin import PageExtractionMixin
from core_pdf.impl.engine.extraction.redactions import (
    RedactionAnalysis,
    RedactionAnalyzer,
    RedactionCandidate,
)
from core_pdf.impl.engine.extraction.tables.api import PageTableMixin
from core_pdf.impl.engine.spec.s_07_document.page import PdfPage as SpecPdfPage

if TYPE_CHECKING:
    from core_pdf.impl.engine.layout.models import TextRun


class PdfPage(PageExtractionMixin, PageTableMixin, SpecPdfPage):
    def extract(self) -> PageExtractionResult:
        return build_page_extraction_result(self, page_index=self.page_number - 1)

    def extract_text(self) -> str:
        return PageExtractionMixin.extract_text(cast(Any, self))

    def find_text_near(
        self,
        target_box: tuple[float, float, float, float],
        direction: str = "left",
        distance: float = 100.0,
    ) -> list[TextRun]:
        return PageContentMixin.find_text_near(self, target_box, direction, distance)

    def get_redaction_analysis(self) -> RedactionAnalysis:
        return RedactionAnalyzer().analyze(self)

    def iter_redaction_candidates(self) -> Iterator[RedactionCandidate]:
        return iter(self.get_redaction_analysis().candidates)


__all__ = ("PdfPage",)
