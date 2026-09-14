"""Pipeline integration with deterministic raster and recognition boundaries."""

from dataclasses import replace

import pytest

from core_pdf.impl._impl.extract.contracts import ObservationBatch
from core_pdf.impl._impl.render.model import RasterImage
from core_pdf.impl._impl.runtime.execution import ExtractionScope
from core_pdf_ocr.impl.extract.contracts import (
    MAX_OCR_PIXELS,
    OcrPass,
    OcrPassScope,
    PageRoute,
    WorkPlan,
)
from core_pdf_ocr.impl.extract.ocr import pipeline
from core_pdf_ocr.impl.extract.ocr.session import internal_OcrPassTasks
from core_pdf_ocr.impl.extract.ocr.types import (
    internal_OcrTask,
    internal_Raster,
    internal_RasterRegion,
)
from core_pdf_ocr.impl.extract.quality import internal_candidate


def internal_result(text: str, *, height: float = 10, box=(0, 0, 20, 10), confidence=90):
    return internal_candidate(
        6,
        ObservationBatch.from_columns((text,), (box,), source=1, confidence=(confidence,)),
        median_text_height=height,
    )


def internal_task():
    return internal_OcrTask(
        6, RasterImage(bytes([255]) * 10000, 100, 100, 1), (0, 0, 100, 100), (0, 0, 100, 100), 72
    )


@pytest.mark.parametrize(
    ("height", "text", "retry_text", "retry_box", "available", "expected"),
    [
        (0, "abc", "better result", (0, 0, 20, 10), True, ("better result",)),
        (10, "abc", "extra", (40, 0, 60, 10), True, ("abc", "extra")),
        (10, "abc", "a", (0, 0, 20, 10), True, ("abc",)),
        (10, "abc", "extra", (40, 0, 60, 10), False, ("abc",)),
        (
            20,
            "a useful sentence containing several words",
            "extra",
            (40, 0, 60, 10),
            True,
            ("a useful sentence containing several words", "extra"),
        ),
        (
            20,
            "a useful sentence containing several words",
            "extra",
            (40, 0, 60, 10),
            False,
            ("a useful sentence containing several words",),
        ),
    ],
)
def test_adaptive_retry_selects_scope_and_preserves_best_content(
    monkeypatch: pytest.MonkeyPatch,
    ocr_capture,
    height,
    text,
    retry_text,
    retry_box,
    available,
    expected,
) -> None:
    primary = internal_result(text, height=height)
    retry = internal_result(retry_text, box=retry_box)
    source = internal_task()
    calls = []
    operation = OcrPass(
        "primary", OcrPassScope.PAGE, 2, (6,), adaptive_scale=True, pixel_budget=1000000
    )

    class Session:
        page_box = source.page_box

        def __init__(self, *args):
            self.recognitions = 0

        def materialize(self, ocr_pass, **kwargs):
            return internal_OcrPassTasks(ocr_pass, (source,))

        def recognize_tasks(self, tasks):
            self.recognitions += 1
            return (primary if self.recognitions == 1 else retry,)

        def render_raster(self, scale, *, max_pixels, include_native_text):
            calls.append("page")
            assert scale == (3 if height == 0 else 5)
            assert max_pixels == MAX_OCR_PIXELS
            assert not include_native_text
            return internal_Raster(source.image, 72) if available else None

        def internal_high_resolution_weak_region_tasks(self, tasks, retry_pass, observations):
            calls.append("weak")
            assert tasks == (source,)
            assert observations is primary.observations
            assert retry_pass.scope is OcrPassScope.WEAK_REGIONS
            assert retry_pass.scale == 3.2
            assert retry_pass.pixel_budget == MAX_OCR_PIXELS
            assert (retry_pass.tiles, retry_pass.region_columns, retry_pass.max_regions) == (
                6,
                3,
                8,
            )
            return (source,) if available else ()

    monkeypatch.setattr(pipeline, "internal_OcrSession", Session)
    result = pipeline.recognize_page(
        ocr_capture,
        WorkPlan(PageRoute.OCR, ocr_passes=(operation,)),
        ExtractionScope(),
        stroked_profile=None,
    )
    assert result.observations.text == expected
    assert calls == (["weak"] if height == 20 else ["page"])


@pytest.mark.parametrize("verified", [False, True])
def test_hidden_layer_verification_short_circuits_only_after_matching_preview(
    monkeypatch: pytest.MonkeyPatch, ocr_capture, verified: bool
) -> None:
    from core_pdf.impl._impl.capture.program import CapturedProgram, PageProgram
    from core_pdf.impl._impl.model.runs import TextRun

    texts = tuple(f"word{i}" for i in range(24))
    hidden = ObservationBatch.from_columns(
        texts,
        tuple((i * 20, 0, i * 20 + 15, 10) for i in range(24)),
        source=1,
        confidence=(100,) * 24,
    )
    # A failed preview must fall through to the regular OCR pass.
    preview = internal_candidate(11, hidden if verified else ObservationBatch.empty())
    source = internal_task()
    fallback = internal_result("fallback")
    seen = []
    runs = tuple(
        TextRun(text, i * 20, 0, i * 20 + 15, 10, i * 20, 10, 10, 2, 0, 0, 0, visible=False)
        for i, text in enumerate(texts)
    )
    capture = replace(
        ocr_capture,
        observations=hidden,
        program=PageProgram(CapturedProgram(runs=runs)),
        evidence=replace(ocr_capture.evidence, full_page_image=True, image_filters=("JPXDecode",)),
    )

    class Session:
        page_box = source.page_box

        def __init__(self, capture, plan, compact, context, profile):
            assert compact == "grayscale"

        def materialize(self, operation, **kwargs):
            seen.append("fallback")
            return internal_OcrPassTasks(operation, (source,))

        def recognize_tasks(self, tasks):
            if tasks[0].mode == 11:
                assert tasks[0].minimum_confidence == 80
                assert tasks[0].recognize_words
                seen.append("verify")
                return (preview,)
            return (fallback,)

    monkeypatch.setattr(pipeline, "internal_OcrSession", Session)
    monkeypatch.setattr(
        pipeline,
        "internal_dominant_image_region",
        lambda *a, **k: internal_RasterRegion(internal_Raster(source.image, 72), source.page_box),
    )
    plan = WorkPlan(
        PageRoute.OCR,
        verify_hidden_text=True,
        ocr_passes=(OcrPass("regular", OcrPassScope.PAGE, 1, (6,)),),
    )
    result = pipeline.recognize_page(capture, plan, ExtractionScope(), stroked_profile=None)
    assert result.observations.text == (texts if verified else ("fallback",))
    assert seen == (["verify"] if verified else ["verify", "fallback"])
