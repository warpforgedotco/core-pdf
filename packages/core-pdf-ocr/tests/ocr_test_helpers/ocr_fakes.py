# SPDX-License-Identifier: AGPL-3.0-only
"""Stand-ins for the OCR stage's external boundaries: Tesseract and the task scope."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from core_pdf.impl.runtime.execution import ExtractionScope
from core_pdf_ocr.impl.extract.contracts import (
    ObservationBatch,
    ObservationSource,
    PageAnalysis,
    PagePlanReason,
    PageRoute,
    RecognitionResult,
    WorkPlan,
)
from core_pdf_ocr.impl.extract.ocr import pipeline as ocr
from core_pdf_ocr.impl.extract.ocr import raster as ocr_raster
from core_pdf_ocr.impl.extract.ocr import region_tasks as ocr_region_tasks
from core_pdf_ocr.impl.extract.ocr import regions as ocr_regions
from core_pdf_ocr.impl.extract.ocr import rescue as ocr_rescue
from core_pdf_ocr.impl.extract.ocr import session as ocr_pass_tasks
from core_pdf_ocr.impl.extract.ocr import tesseract as ocr_tesseract
from core_pdf_ocr.impl.extract.ocr import types as ocr_types
from core_pdf_ocr.impl.extract.ocr import vector as ocr_stroked_vector
from core_pdf_ocr.impl.extract.pipeline import internal_PageExtraction

Box = tuple[float, float, float, float]
PixelBox = tuple[int, int, int, int]
IteratorEntry = tuple[str, float, PixelBox]


def patch_ocr_helper(monkeypatch: pytest.MonkeyPatch, name: str, value: object) -> None:
    """Patch ``name`` in every OCR module that binds it.

    The stage is split across modules that each bind imported helpers into their
    own namespace, so patching one module intercepts only that module's calls.
    Tests mean "make this helper behave differently wherever it runs".
    """
    for module in (
        ocr,
        ocr_pass_tasks,
        ocr_raster,
        ocr_regions,
        ocr_region_tasks,
        ocr_rescue,
        ocr_stroked_vector,
        ocr_tesseract,
    ):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, value)


def patch_dominant_region(
    monkeypatch: pytest.MonkeyPatch,
    raster: ocr_types.internal_Raster | None,
    box: Box = (0.0, 0.0, 10.0, 10.0),
) -> None:
    """Make the dominant page image resolve to ``raster`` over ``box`` (or to nothing)."""
    region = None if raster is None else ocr_types.internal_RasterRegion(raster, box)
    patch_ocr_helper(
        monkeypatch,
        "internal_dominant_image_region",
        lambda internal_capture, **internal_kwargs: region,
    )


def inline_scope() -> ExtractionScope:
    return ExtractionScope()


class FakeResultIterator:
    """A tesserocr result iterator over fixed ``(text, confidence, bbox)`` entries.

    ``levels`` records the iteration level every accessor was asked for.
    """

    def __init__(
        self,
        entries: Sequence[IteratorEntry],
        *,
        line_starts: Sequence[bool] | None = None,
    ) -> None:
        self.entries = tuple(entries)
        self.line_starts = tuple(line_starts) if line_starts is not None else None
        self.index = 0
        self.levels: list[object] = []

    def IsAtBeginningOf(self, level: object) -> bool:
        return self.line_starts[self.index] if self.line_starts is not None else True

    def GetUTF8Text(self, level: object) -> str:
        self.levels.append(level)
        return self.entries[self.index][0]

    def Confidence(self, level: object) -> float:
        self.levels.append(level)
        return self.entries[self.index][1]

    def BoundingBox(self, level: object) -> PixelBox:
        self.levels.append(level)
        return self.entries[self.index][2]

    def Next(self, level: object) -> bool:
        self.levels.append(level)
        self.index += 1
        return self.index < len(self.entries)


class FakeTessApi:
    """The ``PyTessBaseAPI`` surface the OCR stage calls, recording what it receives.

    ``iterators`` are handed out in order by ``GetIterator``; when they run out
    the fake returns ``None`` (what Tesseract does after a failed recognition).
    ``rectangle_error`` makes ``SetRectangle`` raise, for tests asserting that a
    full-page task never sets one.
    """

    def __init__(
        self,
        iterators: Sequence[FakeResultIterator] = (),
        *,
        recognize: bool = True,
        rectangle_error: str | None = None,
    ) -> None:
        self.pending = list(iterators)
        self.recognize = recognize
        self.rectangle_error = rectangle_error
        self.image: bytes | None = None
        self.rectangle: PixelBox | None = None
        self.resolution: int | None = None
        self.mode: int | None = None
        self.recognitions = 0
        self.iterators = 0

    def SetVariable(self, name: str, value: str) -> None:
        pass

    def SetPageSegMode(self, mode: int) -> None:
        self.mode = mode

    def SetImageBytes(self, data: bytes, *internal_args: object) -> None:
        self.image = data

    def SetRectangle(self, left: int, top: int, width: int, height: int) -> None:
        if self.rectangle_error is not None:
            raise AssertionError(self.rectangle_error)
        self.rectangle = (left, top, width, height)

    def SetSourceResolution(self, resolution: int) -> None:
        self.resolution = resolution

    def Recognize(self, **internal_kwargs: object) -> bool:
        self.recognitions += 1
        return self.recognize

    def GetIterator(self) -> FakeResultIterator | None:
        self.iterators += 1
        return self.pending.pop(0) if self.pending else None

    def Clear(self) -> None:
        pass


def patch_engine(monkeypatch: pytest.MonkeyPatch, api: FakeTessApi | None = None) -> FakeTessApi:
    """Bind ``ocr_tesseract.internal_api`` to one fake so no Tesseract engine is built."""
    engine = api if api is not None else FakeTessApi()
    monkeypatch.setattr(ocr_tesseract, "internal_api", lambda internal_mode: engine)
    return engine


@dataclass(slots=True)
class FakeDocumentPage:
    """The page attributes document-level enrichment reads."""

    page_number: int


class RecordingExtraction(internal_PageExtraction):
    """A page extraction whose plan and recognition calls are recorded.

    ``recognition`` answers with one high-confidence ``seed`` observation and
    the learned stroked-vector alphabet.
    """

    def __init__(
        self,
        page: FakeDocumentPage,
        capture: PageAnalysis,
        *,
        alphabet: tuple[tuple[Any, str], ...],
        plan_calls: list[int],
        ocr_calls: list[int],
    ) -> None:
        super().__init__(
            page,
            capture=capture,
            plan=WorkPlan(PageRoute.OCR, reason=PagePlanReason.STROKED_VECTOR_TEXT),
        )
        self.internal_recognized_at: float | None = None
        self.alphabet = alphabet
        self.plan_calls = plan_calls
        self.ocr_calls = ocr_calls

    def recognize(self, context: object) -> RecognitionResult:
        self.ocr_calls.append(self.page.page_number)
        observations = ObservationBatch.from_columns(
            ("seed",),
            ((0.0, 0.0, 1.0, 1.0),),
            source=ObservationSource.OCR,
            confidence=(99.0,),
        )
        self.recognition_result = RecognitionResult(
            observations,
            stroked_vector_alphabet=self.alphabet,
        )
        return self.recognition_result


__all__ = (
    "FakeDocumentPage",
    "FakeResultIterator",
    "FakeTessApi",
    "RecordingExtraction",
    "inline_scope",
    "patch_dominant_region",
    "patch_engine",
    "patch_ocr_helper",
)
