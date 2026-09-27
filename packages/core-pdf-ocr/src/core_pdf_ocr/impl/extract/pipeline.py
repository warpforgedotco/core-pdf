# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Iterable
from copy import replace
from typing import TYPE_CHECKING, ClassVar, cast

from core_pdf.impl import extract_pipeline
from core_pdf.impl.exceptions import ExtractionScope
from core_pdf.impl.extract_capture import STRUCTURE_UNSET
from core_pdf.impl.extract_contracts import ObservationBatch, PageState
from core_pdf.impl.extract_pipeline import (
    NATIVE_PIPELINE,
    DetectTables,
    LayoutBlocks,
    PagePipeline,
)
from core_pdf.impl.output_model import Page
from core_pdf_ocr.impl.extract.block_layout import OCR_LAYOUT_HOOKS
from core_pdf_ocr.impl.extract.capture import capture_page
from core_pdf_ocr.impl.extract.contracts import PageAnalysis, RecognitionResult, WorkPlan
from core_pdf_ocr.impl.extract.observations import fuse_observations, plan_page
from core_pdf_ocr.impl.extract.table_detection import OCR_TABLES
from core_pdf_ocr.impl.extract.table_reconcile import remove_duplicate_tables

if TYPE_CHECKING:
    from core_pdf.impl.document_page import PdfPage
    from core_pdf.impl.document_records import RawFormField
    from core_pdf.impl.document_structure import PageStructure
    from core_pdf.impl.extract_capture import StructureUnset
    from core_pdf_ocr.impl.extract.ocr.strokes import StrokedTextProfile


class FuseRecognition:
    __slots__ = ()

    def __call__(
        self,
        state: PageState,
        extraction: extract_pipeline.PageExtraction,
        context: ExtractionScope,
        /,
    ) -> PageState:
        ocr_extraction = cast(PageExtraction, extraction)
        return replace(
            state,
            observations=fuse_observations(
                state.observations,
                ocr_extraction.recognize(context).observations,
                ocr_extraction.plan,
            ),
        )


class RemoveDuplicateTables:
    __slots__ = ()

    def __call__(
        self,
        state: PageState,
        _extraction: extract_pipeline.PageExtraction,
        _context: ExtractionScope,
        /,
    ) -> PageState:
        return replace(state, tables=remove_duplicate_tables(state.tables))


OCR_PIPELINE = (
    NATIVE_PIPELINE.replacing(DetectTables, DetectTables(OCR_TABLES))
    .replacing(LayoutBlocks, LayoutBlocks(OCR_LAYOUT_HOOKS))
    .inserting_before(DetectTables, FuseRecognition())
    .inserting_after(LayoutBlocks, RemoveDuplicateTables())
)


class PageExtraction(extract_pipeline.PageExtraction):
    capture_page_fn = staticmethod(capture_page)
    pipeline: ClassVar[PagePipeline] = OCR_PIPELINE

    @property
    def capture(self) -> PageAnalysis:
        return self.page_capture  # type: ignore[return-value]  # ty: ignore[invalid-return-type]

    @property
    def route_name(self) -> str:
        return str(self.plan.route)

    def __init__(
        self,
        page: PdfPage,
        *,
        capture: PageAnalysis | None = None,
        plan: WorkPlan | None = None,
        recognition: RecognitionResult | None = None,
        fields: Iterable[RawFormField] | None = None,
        structure: PageStructure | None | StructureUnset = STRUCTURE_UNSET,
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
        self.stroked_profile_of = stroked_profile

    @property
    def stroked_profile(self) -> StrokedTextProfile | None:
        evidence = self.capture.evidence.stroked_vector_text
        if not evidence.trusted or not evidence.drawing_indexes:
            return None
        if self.stroked_profile_of is None:
            from core_pdf_ocr.impl.extract.ocr.strokes import profile_stroked_text

            self.stroked_profile_of = profile_stroked_text(
                self.capture.program.drawings, evidence.drawing_indexes
            )
        return self.stroked_profile_of

    def recognize(self, context: ExtractionScope) -> RecognitionResult:
        if self.recognition_result is not None:
            return self.recognition_result
        plan = self.plan
        if plan.ocr_passes or plan.verify_hidden_text:
            from core_pdf_ocr.impl.extract.ocr.pipeline import recognize_page

            return recognize_page(self.capture, plan, context, stroked_profile=self.stroked_profile)
        return RecognitionResult(ObservationBatch.empty())


def extract_page(page: PdfPage, context: ExtractionScope) -> Page:
    return PageExtraction(page).assembled_page(context)
