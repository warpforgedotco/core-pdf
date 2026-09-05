# SPDX-License-Identifier: AGPL-3.0-only
"""Recognize and fuse page text through the shared native extraction stages."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

from core_pdf.impl._impl.extract.contracts import ObservationBatch
from core_pdf.impl._impl.extract.pipeline import (
    internal_PageExtraction as NativePageExtraction,
)
from core_pdf.impl._impl.extract.pipeline import (
    internal_PageProducts,
)
from core_pdf.impl._impl.output.model import Page
from core_pdf.impl._impl.runtime.execution import ExtractionScope
from core_pdf_ocr.impl.extract.block_layout import layout_blocks_with_evidence
from core_pdf_ocr.impl.extract.capture import capture_page, internal_STRUCTURE_UNSET
from core_pdf_ocr.impl.extract.contracts import PageAnalysis, RecognitionResult, WorkPlan
from core_pdf_ocr.impl.extract.emit import assemble_page
from core_pdf_ocr.impl.extract.observations import fuse_observations, plan_page
from core_pdf_ocr.impl.extract.table_detection import extract_tables

if TYPE_CHECKING:
    from core_pdf.impl._impl.extract.capture import internal_StructureUnset
    from core_pdf.impl.spec.s_07_document.page import PdfPage
    from core_pdf.impl.spec.s_07_document.records import RawFormField
    from core_pdf.impl.spec.s_14_structure.tree import PageStructure
    from core_pdf_ocr.impl.extract.ocr.strokes import StrokedTextProfile


class internal_PageExtraction(NativePageExtraction):
    """Selection-local recognition policy over native capture and metadata assembly."""

    internal_capture_page = staticmethod(capture_page)
    internal_assemble_page = staticmethod(assemble_page)

    @property
    def capture(self) -> PageAnalysis:
        return cast(PageAnalysis, self.internal_capture)

    @property
    def internal_route(self) -> str:
        return str(self.plan.route)

    def __init__(
        self,
        page: PdfPage,
        *,
        capture: PageAnalysis | None = None,
        plan: WorkPlan | None = None,
        recognition: RecognitionResult | None = None,
        fields: Iterable[RawFormField] | None = None,
        structure: PageStructure | None | internal_StructureUnset = internal_STRUCTURE_UNSET,
        hidden_layers: frozenset[str] | None = None,
        stroked_profile: StrokedTextProfile | None = None,
    ) -> None:
        super().__init__(
            page,
            capture=capture,
            fields=fields,
            structure=structure,
            hidden_layers=hidden_layers,
        )
        self.plan = plan if plan is not None else plan_page(self.capture)
        self.recognition_result = recognition
        self.internal_stroked_profile = stroked_profile

    @property
    def stroked_profile(self) -> StrokedTextProfile | None:
        """Materialize the immutable stroke geometry only for a trusted vector-text layer."""
        evidence = self.capture.evidence.stroked_vector_text
        if not evidence.trusted or not evidence.drawing_indexes:
            return None
        if self.internal_stroked_profile is None:
            from core_pdf_ocr.impl.extract.ocr.strokes import profile_stroked_text

            self.internal_stroked_profile = profile_stroked_text(
                self.capture.program.drawings, evidence.drawing_indexes
            )
        return self.internal_stroked_profile

    def recognize(self, context: ExtractionScope) -> RecognitionResult:
        if self.recognition_result is not None:
            return self.recognition_result
        plan = self.plan
        if plan.ocr_passes or plan.verify_hidden_text:
            from core_pdf_ocr.impl.extract.ocr.pipeline import recognize_page

            return recognize_page(self.capture, plan, context, stroked_profile=self.stroked_profile)
        return RecognitionResult(ObservationBatch.empty())

    def run(self, context: ExtractionScope) -> internal_PageProducts:
        context.raise_if_cancelled()
        observations = fuse_observations(
            self.capture.observations,
            self.recognize(context).observations,
            self.plan,
        )
        return self.internal_layout_products(
            observations,
            extract_tables(self.capture, observations),
            layout=layout_blocks_with_evidence,
        )


def extract_page(page: PdfPage, context: ExtractionScope) -> Page:
    """Extract one page, recognizing missing or untrusted native text when needed."""
    return internal_PageExtraction(page).assembled_page(context)
