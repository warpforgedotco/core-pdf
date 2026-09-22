import subprocess
from contextlib import nullcontext
from copy import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core_pdf.impl.extract.contracts import ObservationBatch
from core_pdf.impl.render.model import RasterImage
from core_pdf.impl.runtime.execution import ExtractionCancelled, ExtractionScope
from core_pdf_ocr.impl.extract.contracts import (
    OcrPass,
    OcrPassScope,
    PageAnalysis,
    PageRoute,
    WorkPlan,
)
from core_pdf_ocr.impl.extract.ocr import pipeline, tesseract
from core_pdf_ocr.impl.extract.ocr.types import OcrTask
from core_pdf_ocr.impl.extract.quality import internal_candidate


@pytest.fixture
def task() -> OcrTask:
    return OcrTask(3, RasterImage(bytes(100), 10, 10, 1), (0, 0, 10, 10), (0, 0, 100, 100), 300)


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
        "import_tesserocr",
        lambda: SimpleNamespace(RIL=SimpleNamespace(WORD=1, TEXTLINE=2)),
    )
    monkeypatch.setattr(tesseract, "suppress_c_stderr", nullcontext)
    return engine


@pytest.mark.parametrize(
    ("recognized", "elapsed", "status"),
    [(True, 0.1, "ok"), (False, 0.1, "failed"), (False, 999, "timeout")],
)
def test_engine_status_and_owned_resource_cleanup(
    task: OcrTask,
    fake_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
    recognized: bool,
    elapsed: float,
    status: str,
) -> None:
    fake_engine.recognized = recognized
    times = iter((0, elapsed))
    monkeypatch.setattr(tesseract.time, "perf_counter", lambda: next(times))
    result = tesseract.recognize(task)
    assert result.recognition_status == status
    assert not len(result.observations)
    assert fake_engine.calls[-1] == "end"
    assert fake_engine.calls.count("image") == 1


def test_engine_failure_propagates_and_releases_api(task: OcrTask, fake_engine: Engine) -> None:
    fake_engine.error = RuntimeError("engine crashed")
    with pytest.raises(RuntimeError, match="engine crashed"):
        tesseract.recognize(task)
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
    task: OcrTask,
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
    candidate = tesseract.recognize(task)
    assert candidate.observations.text == expected
    if expected:
        assert candidate.observations.bbox.tolist() == [[10, 40, 50, 80]]
        assert candidate.observations.line_break_before.tolist() == [True]
    assert fake_engine.calls[-1] == "end"


def test_iterator_failure_skips_bad_result_and_cleans_up(
    task: OcrTask, fake_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_engine.recognized = True

    def text(level: int) -> str:
        raise RuntimeError("bad iterator entry")

    monkeypatch.setattr(
        fake_engine,
        "GetIterator",
        lambda: SimpleNamespace(GetUTF8Text=text, Next=lambda level: False),
    )
    assert not len(tesseract.recognize(task).observations)
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
    assert tesseract.hocr_filtered_lines(api, 80) == {(1, 2, 30, 40): "A& B"}
    assert tesseract.hocr_filtered_lines(api, None) == {}
    assert tesseract.hocr_filtered_lines(object(), 80) == {}


def test_task_group_reuses_image_and_releases_once(task: OcrTask, fake_engine: Engine) -> None:
    assert tesseract.recognize_group(()) == ()
    result = tesseract.recognize_group((task, replace(task, mode=6, rectangle=(2, 2, 5, 5))))
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
        tesseract.tessdata_path()
    (tmp_path / "eng.traineddata").write_bytes(b"test data")
    assert tesseract.tessdata_path() == str(tmp_path.resolve())
    monkeypatch.delenv("TESSDATA_PREFIX")
    monkeypatch.setattr(
        tesseract,
        "import_tesserocr",
        lambda: SimpleNamespace(get_languages=lambda: ("", ())),
    )
    monkeypatch.setattr(tesseract.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="English Tesseract data was not found"):
        tesseract.tessdata_path()


@pytest.mark.parametrize(
    "error", [OSError("missing executable"), subprocess.TimeoutExpired("tesseract", 5)]
)
def test_language_probe_failure_does_not_hide_configuration_error(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    monkeypatch.setattr(
        tesseract,
        "import_tesserocr",
        lambda: SimpleNamespace(get_languages=lambda: ("", ())),
    )
    monkeypatch.setattr(tesseract.shutil, "which", lambda name: "/fake/tesseract")

    def run(*args: Any, **kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(tesseract.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="English Tesseract data was not found"):
        tesseract.tessdata_path()


def test_timeout_retry_preserves_page_geometry_and_only_replaces_recovered_text(
    task: OcrTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tesseract, "OCR_TIMEOUT_RETRY_PIXELS", 4)
    cropped = replace(task, rectangle=(2, 2, 4, 4))
    retry = tesseract.timeout_recovery_task(cropped)
    assert retry is not None
    assert retry.rectangle == (0, 0, 2, 2)
    assert retry.page_box == (20, 40, 60, 80)
    assert retry.resolution == 150
    assert tesseract.timeout_recovery_task(retry) is None
    empty = internal_candidate(3, ObservationBatch.empty(), recognition_status="timeout")
    recovered = internal_candidate(
        3,
        ObservationBatch.from_columns(("Recovered",), ((0, 0, 5, 5),), source=1, confidence=(99,)),
    )
    failed = replace(empty, recognition_status="failed")
    requests: list[tuple[OcrTask, ...]] = []

    def recognize(tasks: tuple[OcrTask, ...]) -> Any:
        requests.append(tasks)
        return (recovered,)

    results = tesseract.recover_timed_out_tasks(
        (cropped, cropped, cropped), (empty, recovered, failed), recognize
    )
    assert len(requests) == 1
    assert len(requests[0]) == 1
    assert results[0].recognition_status == "timeout-recovered"
    assert results[0].observations.text == ("Recovered",)
    assert results[1] is recovered
    assert results[2] is failed
    unchanged = (empty,)
    assert tesseract.recover_timed_out_tasks((retry,), unchanged, recognize) is unchanged
    assert (
        tesseract.recover_timed_out_tasks((cropped,), unchanged, lambda tasks: (empty,))
        == unchanged
    )


def test_cancelled_page_does_not_start_raster_or_engine(
    ocr_capture: PageAnalysis, monkeypatch: pytest.MonkeyPatch
) -> None:
    def recognize(*args: Any, **kwargs: Any) -> None:
        pytest.fail("cancelled extraction started recognition")

    monkeypatch.setattr(pipeline, "recognize_page_with_reserved_raster", recognize)
    plan = WorkPlan(PageRoute.OCR, ocr_passes=(OcrPass("page", OcrPassScope.PAGE, 1, (3,)),))
    with pytest.raises(ExtractionCancelled):
        pipeline.recognize_page(
            ocr_capture, plan, ExtractionScope(cancelled=lambda: True), stroked_profile=None
        )
    result = pipeline.recognize_page(
        ocr_capture, WorkPlan(PageRoute.NATIVE), ExtractionScope(), stroked_profile=None
    )
    assert not len(result.observations)


@pytest.mark.parametrize("same_image", [True, False])
def test_session_cancellation_between_tasks_closes_engine(
    task: OcrTask,
    fake_engine: Engine,
    ocr_capture: PageAnalysis,
    same_image: bool,
) -> None:
    from core_pdf_ocr.impl.extract.ocr.session import OcrSession

    second = task if same_image else replace(task, image=RasterImage(bytes(100), 10, 10, 1))
    context = ExtractionScope(cancelled=lambda: "recognize" in fake_engine.calls)
    session = OcrSession(ocr_capture, WorkPlan(PageRoute.OCR), True, context, None)
    with pytest.raises(ExtractionCancelled):
        session.recognize_tasks((task, second))
    assert fake_engine.calls.count("recognize") == 1
    assert fake_engine.calls.count("end") == 1


@pytest.mark.parametrize(("width", "height"), [(1, 1000), (1000, 1)])
def test_timeout_retry_budget_holds_for_narrow_rasters(
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    height: int,
) -> None:
    monkeypatch.setattr(tesseract, "OCR_TIMEOUT_RETRY_PIXELS", 4)
    task = OcrTask(
        3,
        RasterImage(bytes(width * height), width, height, 1),
        (0, 0, width, height),
        (10, 20, 30, 40),
        300,
    )
    retry = tesseract.timeout_recovery_task(task)
    assert retry is not None
    assert retry.image.width * retry.image.height <= 4
    assert retry.page_box == task.page_box


@pytest.mark.parametrize(
    ("text", "confidence", "minimum", "expected"),
    [
        (" \t\n", 99, 20, False),
        ("a\x00b", 99, 20, False),
        ("a!!!", 84, 20, False),
        ("a!!!", 85, 20, True),
        ("!!!!", 99, 20, False),
        ("!!!!", 99, 55, True),
        (".", 69, 20, False),
        (".", 70, 20, True),
        (".", 55, 55, True),
        ("AaAaAaAa", 99, 20, False),
        ("aaaaaaaB", 99, 20, True),
        ("hello", float("nan"), 20, False),
        ("hello", float("inf"), 20, False),
    ],
)
def test_acceptable_text_rejects_noise_and_nonfinite_confidence(
    text, confidence, minimum, expected
) -> None:
    assert tesseract.acceptable_text(text, confidence, minimum) is expected


@pytest.mark.parametrize("failure", [None, RuntimeError("hocr failed"), TypeError("wrong hocr")])
def test_hocr_unavailable_output_is_optional(failure) -> None:
    def hocr(page):
        if failure is not None:
            raise failure
        return ""

    assert tesseract.hocr_filtered_lines(SimpleNamespace(GetHOCRText=hocr), 80) == {}


def test_hocr_ignores_unrelated_attributes_and_lines_without_geometry() -> None:
    parser = tesseract.HocrCharacterParser(80)
    parser.feed(
        '<span id="ignored" class="ocr_line"><span class="ocrx_word">'
        '<span class="ocrx_cinfo">X</span></span></span>'
        '<span class="unrelated"></span><span class="ocrx_cinfo">orphan</span>'
    )
    assert parser.lines == {}


@pytest.mark.parametrize("reason", ["empty", "line-loss", "alnum-loss", "utility-loss", "accepted"])
def test_character_filter_must_preserve_recall_and_utility(reason) -> None:
    raw = internal_candidate(
        6,
        ObservationBatch.from_columns(
            ("word", "word"), ((0, 0, 10, 10), (0, 20, 10, 30)), source=1, confidence=(99, 99)
        ),
    )
    filtered = raw
    if reason == "empty":
        filtered = internal_candidate(6, ObservationBatch.empty())
    elif reason == "line-loss":
        filtered = replace(raw, metrics=replace(raw.metrics, line_count=1))
    elif reason == "alnum-loss":
        filtered = replace(raw, metrics=replace(raw.metrics, alphanumeric_characters=7))
    elif reason == "utility-loss":
        filtered = replace(raw, metrics=replace(raw.metrics, utility=raw.metrics.utility * 0.97))
    else:
        filtered = replace(raw, metrics=replace(raw.metrics, utility=raw.metrics.utility * 0.98))
    assert tesseract.select_character_filtered_candidate(raw, filtered) is (
        filtered if reason == "accepted" else raw
    )


def test_symbol_iteration_skips_bad_entries_and_maps_coordinates(task, monkeypatch) -> None:
    entries: list[Any] = [
        ("A", 90, (1, 2, 5, 6)),
        ("AB", 99, (1, 2, 5, 6)),
        ("B", float("nan"), (1, 2, 5, 6)),
        ("C", 90, None),
        ("\x00", 90, (1, 2, 5, 6)),
        RuntimeError("bad symbol"),
        ("7", 80, (2, 3, 6, 7)),
    ]

    class Iterator:
        index = 0

        def GetUTF8Text(self, level):
            entry = entries[self.index]
            if isinstance(entry, Exception):
                raise entry
            return entry[0]

        def Confidence(self, level):
            return entries[self.index][1]

        def BoundingBox(self, level):
            return entries[self.index][2]

        def Next(self, level):
            self.index += 1
            return self.index < len(entries)

    monkeypatch.setattr(
        tesseract,
        "import_tesserocr",
        lambda: SimpleNamespace(RIL=SimpleNamespace(SYMBOL=3)),
    )
    result = tesseract.recognized_symbols(SimpleNamespace(GetIterator=Iterator), task)
    assert result.text == ("A", "7")
    assert result.bbox.tolist() == [[10, 40, 50, 80], [20, 30, 60, 70]]
    assert result.confidence.tolist() == [90, 80]
    assert not len(tesseract.recognized_symbols(SimpleNamespace(GetIterator=lambda: None), task))


@pytest.mark.parametrize("beginning_method", [False, True])
def test_word_iteration_carries_line_break_across_rejected_word(
    task, fake_engine, monkeypatch, beginning_method
) -> None:
    class Iterator:
        index = 0

        def GetUTF8Text(self, level):
            return ("first", "", "second", "third")[self.index]

        def Confidence(self, level):
            return 90

        def BoundingBox(self, level):
            return (1, 2, 5, 6)

        def Next(self, level):
            self.index += 1
            return self.index < 4

    iterator = Iterator()
    if beginning_method:
        monkeypatch.setattr(
            iterator, "IsAtBeginningOf", lambda level: iterator.index == 1, raising=False
        )
    fake_engine.recognized = True
    monkeypatch.setattr(fake_engine, "GetIterator", lambda: iterator)
    result = tesseract.recognize(replace(task, recognize_words=True))
    assert result.observations.text == ("first", "second", "third")
    assert result.observations.line_break_before.tolist() == [True, beginning_method, False]


@pytest.mark.parametrize("filtered", ["", "hello", "hello world"])
def test_recognition_integrates_character_filtered_lines_without_losing_text(
    task, fake_engine, monkeypatch, filtered
) -> None:
    fake_engine.recognized = True
    iterator = SimpleNamespace(
        GetUTF8Text=lambda level: "hello world",
        Confidence=lambda level: 99,
        BoundingBox=lambda level: (1, 2, 5, 6),
        Next=lambda level: False,
    )
    monkeypatch.setattr(fake_engine, "GetIterator", lambda: iterator)
    monkeypatch.setattr(tesseract, "hocr_filtered_lines", lambda *args: {(1, 2, 5, 6): filtered})
    result = tesseract.recognize(replace(task, character_confidence_threshold=80))
    assert result.observations.text == ("hello world",)
    assert result.recognition_status == "ok"
    assert result.metrics.median_text_height == 4


def test_recognition_collects_symbols_and_tolerates_missing_optional_cleanup(
    task, fake_engine, monkeypatch
) -> None:
    fake_engine.recognized = True
    monkeypatch.setattr(
        tesseract,
        "import_tesserocr",
        lambda: SimpleNamespace(RIL=SimpleNamespace(WORD=1, TEXTLINE=2, SYMBOL=3)),
    )
    monkeypatch.setattr(
        fake_engine,
        "GetIterator",
        lambda: SimpleNamespace(
            GetUTF8Text=lambda level: "A",
            Confidence=lambda level: 90,
            BoundingBox=lambda level: (1, 2, 5, 6),
            Next=lambda level: False,
        ),
    )
    monkeypatch.setattr(fake_engine, "ClearAdaptiveClassifier", None)
    monkeypatch.setattr(fake_engine, "End", None)
    result = tesseract.recognize(replace(task, collect_symbols=True))
    assert result.observations.text == result.symbols.text == ("A",)
    assert result.symbols.bbox.tolist() == [[10, 40, 50, 80]]


@pytest.mark.parametrize("default_valid", [False, True])
def test_language_resolution_checks_reported_english_data_before_executable(
    tmp_path, monkeypatch, default_valid
) -> None:
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    if default_valid:
        (tmp_path / "eng.traineddata").write_bytes(b"fixture")
    monkeypatch.setattr(
        tesseract,
        "import_tesserocr",
        lambda: SimpleNamespace(get_languages=lambda: (str(tmp_path), ("eng",))),
    )
    monkeypatch.setattr(tesseract.shutil, "which", lambda name: None)
    resolved, error = tesseract.resolve_tessdata_path()
    assert resolved == (str(tmp_path.resolve()) if default_valid else None)
    assert bool(error) is not default_valid


@pytest.mark.parametrize("reported_directory", [False, True])
def test_language_resolution_handles_binding_failure_and_unusable_executable_output(
    monkeypatch, tmp_path, reported_directory
) -> None:
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)

    def languages():
        raise RuntimeError("no compiled-in data")

    monkeypatch.setattr(
        tesseract, "import_tesserocr", lambda: SimpleNamespace(get_languages=languages)
    )
    monkeypatch.setattr(tesseract.shutil, "which", lambda name: "/fixture/tesseract")
    monkeypatch.setattr(
        tesseract.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stdout=f'List of available languages in "{tmp_path}"'
            if reported_directory
            else "no language directory",
            stderr="",
        ),
    )
    assert tesseract.resolve_tessdata_path()[0] is None


def test_signal_initialization_rejects_first_use_from_worker_thread(monkeypatch) -> None:
    monkeypatch.setattr(tesseract, "OCR_SIGNALS_READY", False)
    monkeypatch.setattr(tesseract.threading, "current_thread", lambda: object())
    with pytest.raises(RuntimeError, match="main thread"):
        tesseract.prepare_ocr_signals()
    assert not tesseract.OCR_SIGNALS_READY


@pytest.mark.parametrize("open_succeeds", [False, True])
def test_stderr_suppression_yields_and_closes_partial_setup(monkeypatch, open_succeeds) -> None:
    closed = []

    def open_null(*args):
        if not open_succeeds:
            raise OSError("cannot open")
        return 12345

    def duplicate(fd):
        raise OSError("cannot duplicate")

    with monkeypatch.context() as patch:
        patch.setattr(tesseract.os, "open", open_null)
        patch.setattr(tesseract.os, "dup", duplicate)
        patch.setattr(tesseract.os, "close", closed.append)
        with tesseract.suppress_c_stderr():
            reached = True
    assert reached
    assert closed == ([12345] if open_succeeds else [])


def test_timeout_recovery_declines_outside_image_crop(task, monkeypatch) -> None:
    monkeypatch.setattr(tesseract, "OCR_TIMEOUT_RETRY_PIXELS", 4)
    assert tesseract.timeout_recovery_task(replace(task, rectangle=(20, 20, 10, 10))) is None
