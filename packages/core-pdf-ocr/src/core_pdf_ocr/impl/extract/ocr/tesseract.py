# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from copy import replace
from html.parser import HTMLParser
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy

from core_pdf.impl.array_views import contiguous_bytes, finite_median
from core_pdf.impl.extract_contracts import ObservationBatch
from core_pdf.impl.render_model import RasterImage
from core_pdf.impl.text import collapse_ws
from core_pdf_ocr.impl.extract.contracts import PRIMARY_OCR_PIXELS, ObservationSource
from core_pdf_ocr.impl.extract.ocr.resampling import resample_smooth
from core_pdf_ocr.impl.extract.ocr.types import (
    OcrTask,
    Raster,
    map_ocr_box,
    raster_rectangle_page_box,
)
from core_pdf_ocr.impl.extract.quality import Candidate, make_candidate

os.environ["OMP_THREAD_LIMIT"] = "1"

OCR_SIGNALS_READY = False
MAIN_THREAD_MESSAGE = (
    "core_pdf must initialize OCR on the main thread; import PdfDocument before starting OCR"
)


def prepare_ocr_signals() -> None:
    global OCR_SIGNALS_READY
    if OCR_SIGNALS_READY:
        return
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(MAIN_THREAD_MESSAGE)
    with suppress(ImportError):
        import_module("cysignals.signals")
    OCR_SIGNALS_READY = True


OCR_TIMEOUT_MILLISECONDS = 12_000
OCR_TIMEOUT_MILLISECONDS_PER_MEGAPIXEL = 2_000
OCR_TIMEOUT_MAX_MILLISECONDS = 30_000
OCR_TIMEOUT_RETRY_PIXELS = 4_000_000


def import_tesserocr() -> Any:
    if "tesserocr" not in sys.modules:
        prepare_ocr_signals()
    return import_module("tesserocr")


def valid_tessdata_path(path: str | os.PathLike[str]) -> Path | None:
    candidate = Path(path).expanduser()
    if (candidate / "eng.traineddata").is_file():
        return candidate.resolve()
    return None


def tessdata_path() -> str:
    resolved_path, error_message = resolve_tessdata_path()
    if resolved_path is None:
        raise RuntimeError(error_message)
    return resolved_path


def resolve_tessdata_path() -> tuple[str | None, str]:
    configured = os.environ.get("TESSDATA_PREFIX")
    if configured:
        resolved = valid_tessdata_path(configured)
        if resolved is None:
            return (
                None,
                "TESSDATA_PREFIX must name a tessdata directory containing eng.traineddata",
            )
        return str(resolved), ""

    try:
        default_path, languages = import_tesserocr().get_languages()
    except RuntimeError:
        default_path, languages = "", ()
    if "eng" in languages:
        resolved = valid_tessdata_path(default_path)
        if resolved is not None:
            return str(resolved), ""

    executable = shutil.which("tesseract")
    if executable is not None:
        try:
            completed = subprocess.run(
                [executable, "--list-langs"],
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
        except OSError, subprocess.TimeoutExpired:
            completed = None
        if completed is not None:
            output = f"{completed.stdout}\n{completed.stderr}"
            match = re.search(r'List of available languages in "([^"]+)"', output)
            if match is not None:
                resolved = valid_tessdata_path(match.group(1))
                if resolved is not None:
                    return str(resolved), ""

    return (
        None,
        "English Tesseract data was not found; set TESSDATA_PREFIX to a tessdata directory "
        "containing eng.traineddata",
    )


def open_tesseract_api(mode: int) -> Any:
    tesserocr = import_tesserocr()
    api = tesserocr.PyTessBaseAPI(
        path=tessdata_path(),
        psm=mode,
        oem=tesserocr.OEM.LSTM_ONLY,
    )
    api.SetVariable("preserve_interword_spaces", "0")
    api.SetVariable("textord_tablefind_recognize_tables", "0")
    api.SetVariable("textord_tabfind_find_tables", "0")
    return api


HOCR_BBOX_RE = re.compile(r"bbox (\d+) (\d+) (\d+) (\d+)")
HOCR_CONFIDENCE_RE = re.compile(r"(?:x_conf|x_wconf) (-?\d+(?:\.\d+)?)")


class HocrCharacterParser(HTMLParser):
    def __init__(self, threshold: float) -> None:
        super().__init__(convert_charrefs=True)
        self.threshold = threshold
        self.lines: dict[tuple[int, int, int, int], str] = {}
        self.line_box: tuple[int, int, int, int] | None = None
        self.words: list[str] = []
        self.make_chars: list[str] = []
        self.char_confidence = threshold
        self.in_char = False
        self.in_word = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "span":
            return
        class_value = title = ""
        for name, value in attrs:
            if name == "class":
                class_value = value or ""
            elif name == "title":
                title = value or ""
        classes = class_value.split()
        if "ocr_line" in classes:
            match = HOCR_BBOX_RE.search(title)
            self.line_box = None
            if match:
                left, top, right, bottom = (int(value) for value in match.groups())
                self.line_box = (left, top, right, bottom)
            self.words = []
        elif "ocrx_word" in classes:
            self.in_word = True
            self.make_chars = []
        elif "ocrx_cinfo" in classes and self.in_word:
            match = HOCR_CONFIDENCE_RE.search(title)
            self.char_confidence = float(match.group(1)) if match else 0.0
            self.in_char = True

    def handle_data(self, data: str) -> None:
        if self.in_char and self.char_confidence >= self.threshold:
            self.make_chars.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "span":
            return
        if self.in_char:
            self.in_char = False
        elif self.in_word:
            self.words.append("".join(self.make_chars))
            self.make_chars = []
            self.in_word = False
        elif self.line_box is not None:
            text = " ".join(word for word in self.words if word).strip()
            self.lines[self.line_box] = text
            self.line_box = None


def hocr_filtered_lines(api: Any, threshold: float | None) -> dict[tuple[int, int, int, int], str]:
    if threshold is None or not hasattr(api, "GetHOCRText"):
        return {}
    try:
        hocr = api.GetHOCRText(0)
    except RuntimeError, TypeError:
        return {}
    if not hocr:
        return {}
    parser = HocrCharacterParser(threshold)
    parser.feed(hocr.decode("utf-8", "replace") if isinstance(hocr, bytes) else hocr)
    return parser.lines


def acceptable_text(text: str, confidence: float, minimum_confidence: float = 20.0) -> bool:
    if not math.isfinite(confidence) or confidence < minimum_confidence or not text:
        return False
    stripped = collapse_ws(text)
    if not stripped:
        return False
    length = len(stripped)
    printable_count = 0
    nonspace_count = 0
    alphanumeric_count = 0
    first_char = ""
    same_char_count = 0
    for ch in stripped:
        if ch.isprintable():
            printable_count += 1
        if not ch.isspace():
            nonspace_count += 1
            if ch.isalnum():
                alphanumeric_count += 1
        ch_lower = ch.casefold()
        if not first_char:
            first_char = ch_lower
            same_char_count = 1
        elif ch_lower == first_char:
            same_char_count += 1

    if printable_count / length < 0.95:
        return False
    if nonspace_count >= 4:
        symbol_ratio = 1.0 - alphanumeric_count / nonspace_count
        if symbol_ratio >= 0.65 and confidence < 85.0:
            return False
        if alphanumeric_count == 0 and minimum_confidence != 55.0:
            return False
    if length == 1 and not stripped.isalnum():
        character = stripped[0]
        if not character.isascii() or (confidence < 70.0 and minimum_confidence != 55.0):
            return False
    return not (length >= 8 and same_char_count == length)


def select_character_filtered_candidate(
    raw: Candidate,
    filtered: Candidate,
) -> Candidate:
    raw_metrics = raw.metrics
    filtered_metrics = filtered.metrics
    if not len(filtered.observations):
        return raw
    if filtered_metrics.line_count < raw_metrics.line_count * 0.98:
        return raw
    if filtered_metrics.alphanumeric_characters < raw_metrics.alphanumeric_characters:
        return raw
    if filtered_metrics.utility < raw_metrics.utility * 0.98:
        return raw
    return filtered


def recognized_symbols(api: Any, task: OcrTask) -> ObservationBatch:
    iterator = api.GetIterator()
    if iterator is None:
        return ObservationBatch.empty()
    level = import_tesserocr().RIL.SYMBOL
    texts: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    confidences: list[float] = []
    while True:
        try:
            text = (iterator.GetUTF8Text(level) or "").strip()
            confidence = float(iterator.Confidence(level))
            bbox = iterator.BoundingBox(level)
        except RuntimeError:
            text = ""
            confidence = 0.0
            bbox = None
        if bbox is not None and len(text) == 1 and text.isprintable() and math.isfinite(confidence):
            texts.append(text)
            boxes.append(map_ocr_box(task, bbox))
            confidences.append(confidence)
        if not iterator.Next(level):
            break
    return ObservationBatch.from_columns(
        texts,
        boxes,
        source=ObservationSource.OCR,
        confidence=confidences,
        sequence=range(len(texts)),
    )


@contextmanager
def suppress_c_stderr() -> Iterator[None]:
    devnull_fd = None
    stderr_fd = None
    with suppress(OSError):
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        stderr_fd = os.dup(2)
        os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        if stderr_fd is not None:
            with suppress(OSError):
                os.dup2(stderr_fd, 2)
                os.close(stderr_fd)
        if devnull_fd is not None:
            with suppress(OSError):
                os.close(devnull_fd)


def recognition_timeout(task: OcrTask) -> int:
    pixels = max(1, task.rectangle[2] * task.rectangle[3])
    excess_megapixels = max(0, pixels - PRIMARY_OCR_PIXELS) / 1_000_000
    budget = OCR_TIMEOUT_MILLISECONDS + int(
        excess_megapixels * OCR_TIMEOUT_MILLISECONDS_PER_MEGAPIXEL
    )
    return min(OCR_TIMEOUT_MAX_MILLISECONDS, budget)


@contextmanager
def owned_api(mode: int) -> Iterator[Any]:
    api = open_tesseract_api(mode)
    try:
        yield api
    finally:
        end = getattr(api, "End", None)
        if callable(end):
            end()


def recognize(
    task: OcrTask,
    *,
    api_override: Any | None = None,
    image_prepared: bool = False,
) -> Candidate:
    if api_override is None:
        with owned_api(task.mode) as api:
            return recognize(task, api_override=api, image_prepared=image_prepared)
    tesserocr = import_tesserocr()
    api = api_override
    api.SetPageSegMode(task.mode)
    if not image_prepared:
        api.SetImageBytes(
            bytes(task.image.pixels),
            task.image.width,
            task.image.height,
            task.image.channels,
            task.image.stride,
        )
    raw_x, raw_y, rectangle_width, rectangle_height = task.rectangle
    right = max(raw_x, min(task.image.width, raw_x + rectangle_width))
    bottom = max(raw_y, min(task.image.height, raw_y + rectangle_height))
    x0 = max(0, min(task.image.width - 1, int(raw_x)))
    y0 = max(0, min(task.image.height - 1, int(raw_y)))
    w = max(1, int(right - x0))
    h = max(1, int(bottom - y0))
    if image_prepared or (x0, y0, w, h) != (0, 0, task.image.width, task.image.height):
        with suppress_c_stderr():
            api.SetRectangle(x0, y0, w, h)
    api.SetSourceResolution(task.resolution)
    timeout_milliseconds = recognition_timeout(task)
    recognition_started = time.perf_counter()
    with suppress_c_stderr():
        recognized = api.Recognize(timeout=timeout_milliseconds)
    recognition_elapsed = time.perf_counter() - recognition_started
    if recognized:
        recognition_status = "ok"
    elif recognition_elapsed >= timeout_milliseconds / 1000.0 * 0.9:
        recognition_status = "timeout"
    else:
        recognition_status = "failed"
    iterator = api.GetIterator() if recognized else None
    level = tesserocr.RIL.WORD if task.recognize_words else tesserocr.RIL.TEXTLINE
    filtered_lines = (
        {}
        if task.recognize_words
        else hocr_filtered_lines(api, task.character_confidence_threshold)
    )
    texts: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    confidences: list[float] = []
    filtered_texts: list[str] = []
    filtered_boxes: list[tuple[float, float, float, float]] = []
    filtered_confidences: list[float] = []
    text_heights: list[float] = []
    line_breaks: list[bool] = []
    filtered_line_breaks: list[bool] = []
    pending_line_break = True
    if iterator is not None:
        while True:
            if task.recognize_words:
                is_at_beginning = getattr(iterator, "IsAtBeginningOf", None)
                if callable(is_at_beginning):
                    pending_line_break |= bool(is_at_beginning(tesserocr.RIL.TEXTLINE))
            else:
                pending_line_break = True
            try:
                text = iterator.GetUTF8Text(level) or ""
                confidence = float(iterator.Confidence(level))
                bbox = iterator.BoundingBox(level)
            except RuntimeError:
                text = ""
                confidence = 0.0
                bbox = None
            text = collapse_ws(text)
            if bbox is not None and acceptable_text(
                text,
                confidence,
                task.minimum_confidence,
            ):
                x0, y0, x1, y1 = bbox
                bbox_key = (int(x0), int(y0), int(x1), int(y1))
                filtered = filtered_lines.get(bbox_key)
                filtered_text = collapse_ws(filtered) if filtered is not None else text
                texts.append(text)
                mapped_box = map_ocr_box(task, (x0, y0, x1, y1))
                boxes.append(mapped_box)
                confidences.append(confidence)
                line_breaks.append(pending_line_break)
                pending_line_break = False
                if acceptable_text(
                    filtered_text,
                    confidence,
                    task.minimum_confidence,
                ):
                    filtered_texts.append(filtered_text)
                    filtered_boxes.append(mapped_box)
                    filtered_confidences.append(confidence)
                    filtered_line_breaks.append(line_breaks[-1])
                text_heights.append(float(y1 - y0))
            if not iterator.Next(level):
                break
    symbols = (
        recognized_symbols(api, task)
        if recognized and task.collect_symbols
        else ObservationBatch.empty()
    )
    clear_adaptive = getattr(api, "ClearAdaptiveClassifier", None)
    if callable(clear_adaptive):
        clear_adaptive()
    median_text_height = (
        finite_median(numpy.asarray(text_heights, dtype=numpy.float64)) if text_heights else 0.0
    )
    observations = ObservationBatch.from_columns(
        texts,
        boxes,
        source=ObservationSource.OCR,
        confidence=confidences,
        sequence=range(len(texts)),
        line_break_before=line_breaks,
    )
    candidate = make_candidate(
        task.mode,
        observations,
        symbols=symbols,
        recognition_status=recognition_status,
        median_text_height=median_text_height,
    )
    if task.character_confidence_threshold is None:
        return candidate
    filtered_observations = ObservationBatch.from_columns(
        filtered_texts,
        filtered_boxes,
        source=ObservationSource.OCR,
        confidence=filtered_confidences,
        sequence=range(len(filtered_texts)),
        line_break_before=filtered_line_breaks,
    )
    filtered_candidate = make_candidate(
        task.mode,
        filtered_observations,
        symbols=symbols,
        recognition_status=recognition_status,
        median_text_height=median_text_height,
    )
    return select_character_filtered_candidate(candidate, filtered_candidate)


def recognize_group(
    tasks: tuple[OcrTask, ...],
    *,
    raise_if_cancelled: Callable[[], None] | None = None,
) -> tuple[Candidate, ...]:
    if not tasks:
        return ()
    if raise_if_cancelled is not None:
        raise_if_cancelled()
    first = tasks[0]
    with owned_api(first.mode) as api:
        candidates = [recognize(first, api_override=api)]
        for task in tasks[1:]:
            if raise_if_cancelled is not None:
                raise_if_cancelled()
            candidates.append(recognize(task, api_override=api, image_prepared=True))
        return tuple(candidates)


def timeout_recovery_task(task: OcrTask) -> OcrTask | None:
    x, y, width, height = task.rectangle
    pixels = max(1, width * height)
    if pixels <= OCR_TIMEOUT_RETRY_PIXELS:
        return None
    reduction = min(
        math.sqrt(OCR_TIMEOUT_RETRY_PIXELS / pixels),
        OCR_TIMEOUT_RETRY_PIXELS / max(1, width),
        OCR_TIMEOUT_RETRY_PIXELS / max(1, height),
    )
    target_width = max(1, int(width * reduction))
    target_height = max(1, int(height * reduction))
    source = task.image.array()[y : y + height, x : x + width]
    if source.shape[0] < 1 or source.shape[1] < 1:
        return None
    reduced = resample_smooth(source, target_height, target_width)
    image = RasterImage(
        contiguous_bytes(reduced),
        target_width,
        target_height,
        task.image.channels,
    )
    page_box = raster_rectangle_page_box(
        Raster(task.image, task.resolution),
        task.page_box,
        task.rectangle,
    )
    return replace(
        task,
        image=image,
        rectangle=(0, 0, target_width, target_height),
        page_box=page_box,
        resolution=max(70, int(round(task.resolution * reduction))),
    )


def recover_timed_out_tasks(
    tasks: tuple[OcrTask, ...],
    candidates: tuple[Candidate, ...],
    recognize: Callable[[tuple[OcrTask, ...]], tuple[Candidate, ...]],
) -> tuple[Candidate, ...]:
    retry_indexes: list[int] = []
    retry_tasks: list[OcrTask] = []
    for index, (task, candidate) in enumerate(zip(tasks, candidates, strict=False)):
        if candidate.recognition_status != "timeout" or len(candidate.observations):
            continue
        retry = timeout_recovery_task(task)
        if retry is not None:
            retry_indexes.append(index)
            retry_tasks.append(retry)
    if not retry_tasks:
        return candidates
    recovered = list(candidates)
    for index, candidate in zip(retry_indexes, recognize(tuple(retry_tasks)), strict=False):
        if len(candidate.observations):
            recovered[index] = replace(candidate, recognition_status="timeout-recovered")
    return tuple(recovered)
