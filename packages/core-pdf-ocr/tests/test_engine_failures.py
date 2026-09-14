import subprocess
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core_pdf.impl._impl.extract.contracts import ObservationBatch
from core_pdf.impl._impl.render.model import RasterImage
from core_pdf.impl._impl.runtime.execution import ExtractionScope, internal_ExtractionCancelled
from core_pdf_ocr.impl.extract.contracts import (
    OcrPass,
    OcrPassScope,
    PageAnalysis,
    PageRoute,
    WorkPlan,
)
from core_pdf_ocr.impl.extract.ocr import pipeline, tesseract
from core_pdf_ocr.impl.extract.ocr.types import internal_OcrTask
from core_pdf_ocr.impl.extract.quality import internal_candidate


@pytest.fixture
def task() -> internal_OcrTask:
    return internal_OcrTask(
        3, RasterImage(bytes(100), 10, 10, 1), (0, 0, 10, 10), (0, 0, 100, 100), 300
    )


class Engine:
    def __init__(self, recognized: bool = False, error: Exception | None = None) -> None:
        self.recognized = recognized
        self.error = error
        self.calls: list[str] = []

    def SetPageSegMode(self, mode: int) -> None:
        self.calls.append("mode")

    def SetImageBytes(self, *args: Any) -> None:
        self.calls.append("image")

    def SetRectangle(self, *args: Any) -> None:
        self.calls.append("rectangle")

    def SetSourceResolution(self, resolution: int) -> None:
        pass

    def Recognize(self, **kwargs: Any) -> bool:
        self.calls.append("recognize")
        if self.error is not None:
            raise self.error
        return self.recognized

    def GetIterator(self) -> None:
        return None

    def ClearAdaptiveClassifier(self) -> None:
        self.calls.append("clear")

    def End(self) -> None:
        self.calls.append("end")


@pytest.fixture
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> Engine:
    engine = Engine()
    monkeypatch.setattr(tesseract, "internal_api", lambda mode: engine)
    monkeypatch.setattr(
        tesseract,
        "internal_import_tesserocr",
        lambda: SimpleNamespace(RIL=SimpleNamespace(WORD=1, TEXTLINE=2)),
    )
    monkeypatch.setattr(tesseract, "internal_suppress_c_stderr", nullcontext)
    return engine


@pytest.mark.parametrize(
    ("recognized", "elapsed", "status"),
    [(True, 0.1, "ok"), (False, 0.1, "failed"), (False, 999, "timeout")],
)
def test_engine_status_and_owned_resource_cleanup(
    task: internal_OcrTask,
    fake_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
    recognized: bool,
    elapsed: float,
    status: str,
) -> None:
    fake_engine.recognized = recognized
    times = iter((0, elapsed))
    monkeypatch.setattr(tesseract.time, "perf_counter", lambda: next(times))
    result = tesseract.internal_recognize(task)
    assert result.recognition_status == status
    assert not len(result.observations)
    assert fake_engine.calls[-1] == "end"
    assert fake_engine.calls.count("image") == 1


def test_engine_failure_propagates_and_releases_api(
    task: internal_OcrTask, fake_engine: Engine
) -> None:
    fake_engine.error = RuntimeError("engine crashed")
    with pytest.raises(RuntimeError, match="engine crashed"):
        tesseract.internal_recognize(task)
    assert fake_engine.calls[-1] == "end"


@pytest.mark.parametrize(
    ("text", "confidence", "expected"),
    [
        ("Good  text\n", 99, ("Good text",)),
        ("Good text", 19, ()),
        ("", 99, ()),
        ("aaaaaaaa", 99, ()),
        ("※", 99, ()),
    ],
)
def test_recognition_normalizes_filters_and_maps_engine_output(
    task: internal_OcrTask,
    fake_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    confidence: float,
    expected: tuple[str, ...],
) -> None:
    fake_engine.recognized = True
    iterator = SimpleNamespace(
        GetUTF8Text=lambda level: text,
        Confidence=lambda level: confidence,
        BoundingBox=lambda level: (1, 2, 5, 6),
        Next=lambda level: False,
    )
    monkeypatch.setattr(fake_engine, "GetIterator", lambda: iterator)
    candidate = tesseract.internal_recognize(task)
    assert candidate.observations.text == expected
    if expected:
        assert candidate.observations.bbox.tolist() == [[10, 40, 50, 80]]
        assert candidate.observations.line_break_before.tolist() == [True]
    assert fake_engine.calls[-1] == "end"


def test_iterator_failure_skips_bad_result_and_cleans_up(
    task: internal_OcrTask, fake_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_engine.recognized = True

    def text(level: int) -> str:
        raise RuntimeError("bad iterator entry")

    monkeypatch.setattr(
        fake_engine,
        "GetIterator",
        lambda: SimpleNamespace(GetUTF8Text=text, Next=lambda level: False),
    )
    assert not len(tesseract.internal_recognize(task).observations)
    assert fake_engine.calls[-1] == "end"


@pytest.mark.parametrize("as_bytes", [False, True])
def test_hocr_character_filter_preserves_entities_and_word_spacing(as_bytes: bool) -> None:
    hocr = (
        "<div>"
        '<span class="ocr_line" title="bbox 1 2 30 40">'
        '<span class="ocrx_word">'
        '<span class="ocrx_cinfo" title="x_conf 90">A&amp;</span>'
        '<span class="ocrx_cinfo" title="x_conf 10">X</span>'
        "</span>"
        '<span class="ocrx_word">'
        '<span class="ocrx_cinfo" title="x_conf 95">B</span>'
        "</span>"
        "</span>"
        "</div>"
    )
    api = SimpleNamespace(GetHOCRText=lambda page: hocr.encode() if as_bytes else hocr)
    assert tesseract.internal_hocr_filtered_lines(api, 80) == {(1, 2, 30, 40): "A& B"}
    assert tesseract.internal_hocr_filtered_lines(api, None) == {}
    assert tesseract.internal_hocr_filtered_lines(object(), 80) == {}


def test_task_group_reuses_image_and_releases_once(
    task: internal_OcrTask, fake_engine: Engine
) -> None:
    assert tesseract.internal_recognize_group(()) == ()
    result = tesseract.internal_recognize_group(
        (task, replace(task, mode=6, rectangle=(2, 2, 5, 5)))
    )
    assert len(result) == 2
    assert fake_engine.calls.count("image") == 1
    assert fake_engine.calls.count("recognize") == 2
    assert fake_engine.calls.count("rectangle") == 1
    assert fake_engine.calls.count("end") == 1


def test_missing_or_explicit_invalid_language_data_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path))
    with pytest.raises(RuntimeError, match="TESSDATA_PREFIX"):
        tesseract.internal_tessdata_path()
    (tmp_path / "eng.traineddata").write_bytes(b"test data")
    assert tesseract.internal_tessdata_path() == str(tmp_path.resolve())
    monkeypatch.delenv("TESSDATA_PREFIX")
    monkeypatch.setattr(
        tesseract,
        "internal_import_tesserocr",
        lambda: SimpleNamespace(get_languages=lambda: ("", ())),
    )
    monkeypatch.setattr(tesseract.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="English Tesseract data was not found"):
        tesseract.internal_tessdata_path()


@pytest.mark.parametrize(
    "error", [OSError("missing executable"), subprocess.TimeoutExpired("tesseract", 5)]
)
def test_language_probe_failure_does_not_hide_configuration_error(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    monkeypatch.setattr(
        tesseract,
        "internal_import_tesserocr",
        lambda: SimpleNamespace(get_languages=lambda: ("", ())),
    )
    monkeypatch.setattr(tesseract.shutil, "which", lambda name: "/fake/tesseract")

    def run(*args: Any, **kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(tesseract.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="English Tesseract data was not found"):
        tesseract.internal_tessdata_path()


def test_timeout_retry_preserves_page_geometry_and_only_replaces_recovered_text(
    task: internal_OcrTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tesseract, "OCR_TIMEOUT_RETRY_PIXELS", 4)
    cropped = replace(task, rectangle=(2, 2, 4, 4))
    retry = tesseract.internal_timeout_recovery_task(cropped)
    assert retry is not None
    assert retry.rectangle == (0, 0, 2, 2)
    assert retry.page_box == (20, 40, 60, 80)
    assert retry.resolution == 150
    assert tesseract.internal_timeout_recovery_task(retry) is None
    empty = internal_candidate(3, ObservationBatch.empty(), recognition_status="timeout")
    recovered = internal_candidate(
        3,
        ObservationBatch.from_columns(("Recovered",), ((0, 0, 5, 5),), source=1, confidence=(99,)),
    )
    failed = replace(empty, recognition_status="failed")
    requests: list[tuple[internal_OcrTask, ...]] = []

    def recognize(tasks: tuple[internal_OcrTask, ...]) -> Any:
        requests.append(tasks)
        return (recovered,)

    results = tesseract.internal_recover_timed_out_tasks(
        (cropped, cropped, cropped), (empty, recovered, failed), recognize
    )
    assert len(requests) == 1
    assert len(requests[0]) == 1
    assert results[0].recognition_status == "timeout-recovered"
    assert results[0].observations.text == ("Recovered",)
    assert results[1] is recovered
    assert results[2] is failed
    unchanged = (empty,)
    assert tesseract.internal_recover_timed_out_tasks((retry,), unchanged, recognize) is unchanged
    assert (
        tesseract.internal_recover_timed_out_tasks((cropped,), unchanged, lambda tasks: (empty,))
        == unchanged
    )


def test_cancelled_page_does_not_start_raster_or_engine(
    ocr_capture: PageAnalysis, monkeypatch: pytest.MonkeyPatch
) -> None:
    def recognize(*args: Any, **kwargs: Any) -> None:
        pytest.fail("cancelled extraction started recognition")

    monkeypatch.setattr(pipeline, "internal_recognize_page_with_reserved_raster", recognize)
    plan = WorkPlan(PageRoute.OCR, ocr_passes=(OcrPass("page", OcrPassScope.PAGE, 1, (3,)),))
    with pytest.raises(internal_ExtractionCancelled):
        pipeline.recognize_page(
            ocr_capture, plan, ExtractionScope(cancelled=lambda: True), stroked_profile=None
        )
    result = pipeline.recognize_page(
        ocr_capture, WorkPlan(PageRoute.NATIVE), ExtractionScope(), stroked_profile=None
    )
    assert not len(result.observations)
