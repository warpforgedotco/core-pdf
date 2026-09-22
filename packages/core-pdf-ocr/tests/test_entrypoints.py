import importlib
import runpy
from copy import replace
from pathlib import Path
from typing import Any, cast

import pytest

from core_pdf.impl.capture.program import CapturedProgram, PageProgram
from core_pdf.impl.capture.records import CapturedDrawing, CapturedPath, CapturedSubpath
from core_pdf.impl.extract.contracts import ObservationBatch
from core_pdf.impl.runtime.execution import ExtractionScope
from core_pdf_ocr import PdfDocument, cli
from core_pdf_ocr.impl.extract import pipeline
from core_pdf_ocr.impl.extract.contracts import (
    OcrPass,
    OcrPassScope,
    PagePlanReason,
    PageRoute,
    RecognitionResult,
    WorkPlan,
)


def test_companion_page_api_extracts_native_text(text_pdf_bytes) -> None:
    with PdfDocument(text_pdf_bytes) as document:
        page = document.pages[0].extract()
        assert "Hello maintenance" in page.text
        assert page.page_number == 1


def test_companion_cli_uses_native_route_for_legible_page(tmp_path, capsys, text_pdf_bytes) -> None:
    path = tmp_path / "native.pdf"
    path.write_bytes(text_pdf_bytes)
    assert cli.main([str(path), "--print"]) == 0
    output = capsys.readouterr()
    assert "Hello maintenance" in output.out
    assert not output.err


@pytest.mark.parametrize("entrypoint", ["cli.py", "__main__.py"])
def test_module_entrypoint_runs_companion_cli(
    entrypoint, tmp_path, monkeypatch, capsys, text_pdf_bytes
) -> None:
    path = tmp_path / "native.pdf"
    path.write_bytes(text_pdf_bytes)
    monkeypatch.setattr("sys.argv", ["core-pdf-ocr", str(path), "--print"])
    module_path = Path(cli.__file__).with_name(entrypoint)
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(module_path), run_name="__main__")
    assert result.value.code == 0
    assert "Hello maintenance" in capsys.readouterr().out


def test_importing_module_entrypoint_does_not_execute_cli(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "main", lambda: calls.append(True))
    importlib.import_module("core_pdf_ocr.__main__")
    assert calls == []


def test_stroked_profile_is_lazily_built_and_reused(ocr_capture) -> None:
    drawing = CapturedDrawing(
        0,
        None,
        None,
        kind="stroke",
        path=CapturedPath([CapturedSubpath([(0, 0), (5, 10), (10, 0)])]),
        stroke_color=(0,),
        line_width=0.2,
    )
    capture = replace(
        ocr_capture,
        program=PageProgram(CapturedProgram(drawings=(drawing,))),
        evidence=replace(
            ocr_capture.evidence,
            stroked_vector_text=replace(
                ocr_capture.evidence.stroked_vector_text,
                trusted=True,
                drawing_indexes=(0,),
                candidate_paths=1,
            ),
        ),
    )
    extraction = pipeline.internal_PageExtraction(
        cast(Any, object()), capture=capture, plan=WorkPlan(PageRoute.OCR)
    )
    assert extraction.internal_stroked_profile is None
    profile = extraction.stroked_profile
    assert profile is not None
    assert extraction.stroked_profile is profile
    assert extraction.internal_stroked_profile is profile


def test_page_pipeline_dispatches_recognition_with_exact_context(ocr_capture, monkeypatch) -> None:
    from core_pdf_ocr.impl.extract.ocr import pipeline as recognition_pipeline

    context = ExtractionScope()
    recognition = RecognitionResult(ObservationBatch.empty())
    operation = OcrPass("page", OcrPassScope.PAGE, 1, (6,))
    plan = WorkPlan(PageRoute.OCR, ocr_passes=(operation,))
    extraction = pipeline.internal_PageExtraction(
        cast(Any, object()), capture=ocr_capture, plan=plan
    )

    def recognize(capture, requested_plan, requested_context, *, stroked_profile):
        assert capture is extraction.capture
        assert requested_plan is plan
        assert requested_context is context
        assert stroked_profile is None
        return recognition

    monkeypatch.setattr(recognition_pipeline, "recognize_page", recognize)
    assert extraction.recognize(context) is recognition


def test_legacy_reason_strings_normalize_and_vector_match_ratio_handles_empty_counts(
    ocr_capture,
) -> None:
    plan = WorkPlan(PageRoute.NATIVE, reason=cast(Any, "healthy-native-text"))
    assert plan.reason is PagePlanReason.HEALTHY_NATIVE_TEXT
    assert ocr_capture.evidence.vector_text_segment_coverage == 0
    evidence = replace(
        ocr_capture.evidence, vector_text_candidate_segments=20, vector_text_matched_segments=15
    )
    assert evidence.vector_text_segment_coverage == 0.75
