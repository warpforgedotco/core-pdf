# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Iterable
from copy import replace
from typing import TYPE_CHECKING

from core_pdf.impl.execution import ExtractionScope
from core_pdf.impl.extract_capture import STRUCTURE_UNSET
from core_pdf.impl.extract_contracts import ObservationBatch
from core_pdf.impl.extract_pipeline import (
    PageExtraction as NativePageExtraction,
)
from core_pdf.impl.extract_pipeline import (
    PageProducts,
)
from core_pdf.impl.output_model import Page
from core_pdf_ocr.impl.extract.block_layout import layout_blocks_with_evidence
from core_pdf_ocr.impl.extract.capture import capture_page
from core_pdf_ocr.impl.extract.contracts import PageAnalysis, RecognitionResult, WorkPlan
from core_pdf_ocr.impl.extract.observations import fuse_observations, plan_page
from core_pdf_ocr.impl.extract.table_detection import extract_tables
from core_pdf_ocr.impl.extract.table_reconcile import remove_duplicate_tables

if TYPE_CHECKING:
    from core_pdf.impl.document_page import PdfPage
    from core_pdf.impl.document_records import RawFormField
    from core_pdf.impl.document_structure import PageStructure
    from core_pdf.impl.extract_capture import StructureUnset
    from core_pdf_ocr.impl.extract.ocr.strokes import StrokedTextProfile


class PageExtraction(NativePageExtraction):
    capture_page_fn = staticmethod(capture_page)

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

    def run(self, context: ExtractionScope) -> PageProducts:
        context.raise_if_cancelled()
        observations = fuse_observations(
            self.capture.observations,
            self.recognize(context).observations,
            self.plan,
        )
        products = self.layout_products(
            observations,
            extract_tables(self.capture, observations),
            layout=layout_blocks_with_evidence,
        )
        return replace(products, tables=remove_duplicate_tables(products.tables))


def extract_page(page: PdfPage, context: ExtractionScope) -> Page:
    return PageExtraction(page).assembled_page(context)
