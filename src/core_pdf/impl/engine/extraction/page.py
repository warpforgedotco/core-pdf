# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, cast

from core_pdf.impl.engine.extraction.common.page_content import PageContentMixin
from core_pdf.impl.engine.extraction.page_text.api import PageExtractionMixin
from core_pdf.impl.engine.extraction.page_text.engine import (
    PageExtractionResult,
    build_page_extraction_result,
)
from core_pdf.impl.engine.extraction.tables.api import PageTableMixin
from core_pdf.impl.engine.spec.s_07_document.page import PdfPage as SpecPdfPage

if TYPE_CHECKING:
    from core_layout.impl.layout.models import TextRun

    from core_pdf.impl.engine.extraction.redactions import (
        RedactionAnalysis,
        RedactionCandidate,
    )


class PdfPage(PageExtractionMixin, PageTableMixin, SpecPdfPage):
    def extract(self) -> PageExtractionResult:
        return build_page_extraction_result(self)

    def extract_text(self) -> str:
        return PageExtractionMixin.extract_text(cast(Any, self))

    def text_rotation_correction(self, threshold: float = 0.95) -> int:
        """Return the counter-clockwise rotation needed to make dominant text upright.

        Text-run angles are evaluated in the displayed page frame and weighted by
        character count. Pages without a clear non-horizontal orientation return zero.
        """
        counts = {0: 0, 90: 0, 180: 0, 270: 0}
        total = 0
        for run in self.display_chars:
            angle = run.rotation_angle % 360
            bucket = min((0, 90, 180, 270, 360), key=lambda value: abs(value - angle)) % 360
            weight = max(1, len(run.text))
            counts[bucket] += weight
            total += weight

        if total == 0:
            return 0
        dominant = max(counts, key=lambda angle: counts[angle])
        if dominant == 0 or counts[dominant] / total < threshold:
            return 0
        return (360 - dominant) % 360

    def find_text_near(
        self,
        target_box: tuple[float, float, float, float],
        direction: str = "left",
        distance: float = 100.0,
    ) -> list[TextRun]:
        return PageContentMixin.find_text_near(self, target_box, direction, distance)

    def get_redaction_analysis(self) -> RedactionAnalysis:
        from core_pdf.impl.engine.extraction.redactions import RedactionAnalyzer

        return RedactionAnalyzer().analyze(self)

    def iter_redaction_candidates(self) -> Iterator[RedactionCandidate]:
        return iter(self.get_redaction_analysis().candidates)


__all__ = ("PdfPage",)
