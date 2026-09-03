# SPDX-License-Identifier: AGPL-3.0-only
"""The Tesseract binding: process setup, tessdata resolution, timeouts, and hOCR.

Everything that knows Tesseract exists lives here -- importing tesserocr, locating
trained data, budgeting per-task timeouts and recovering from them, and parsing the
hOCR that comes back. The rest of the OCR stage decides *what* to recognize and
consumes observations; only this module talks to the engine.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import replace
from html.parser import HTMLParser
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy

from core_pdf.impl.extract.contracts import (
    PRIMARY_OCR_PIXELS,
    ObservationBatch,
    ObservationSource,
)
from core_pdf.impl.extract.ocr.types import (
    internal_map_ocr_box,
    internal_OcrTask,
    internal_Raster,
    internal_raster_rectangle_page_box,
)
from core_pdf.impl.extract.quality import internal_Candidate, internal_candidate
from core_pdf.impl.render.model import RasterImage
from core_pdf.impl.runtime.array_views import contiguous_bytes, finite_median, resample_smooth
from core_pdf.impl.text import collapse_ws

# OCR already has an explicit worker limit. Prevent Tesseract's OpenMP kernels
# from creating another layer of workers on top of it.
os.environ["OMP_THREAD_LIMIT"] = "1"

internal_OCR_SIGNALS_READY = False
internal_MAIN_THREAD_MESSAGE = (
    "core_pdf must initialize OCR on the main thread; import PdfDocument before starting OCR"
)


def internal_prepare_ocr_signals() -> None:
    """Install cysignals' handlers from the main thread, once per process."""
    global internal_OCR_SIGNALS_READY
    if internal_OCR_SIGNALS_READY:
        return
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(internal_MAIN_THREAD_MESSAGE)
    # Not every tesserocr build depends on cysignals. Where it is absent there
    # is no signal handler to install and therefore no main-thread constraint.
    with suppress(ImportError):
        import_module("cysignals.signals")
    internal_OCR_SIGNALS_READY = True


OCR_TIMEOUT_MILLISECONDS = 12_000
# Recognition cost grows with the raster, so a flat budget starves exactly the
# large rasters the adaptive passes escalate to. Extend it per megapixel above
# the primary budget, but keep a ceiling so one page cannot stall a document.
OCR_TIMEOUT_MILLISECONDS_PER_MEGAPIXEL = 2_000
OCR_TIMEOUT_MAX_MILLISECONDS = 30_000
# A timed-out recognition yields nothing at all. Retry it once on a raster small
# enough to finish rather than letting the empty candidate win selection.
OCR_TIMEOUT_RETRY_PIXELS = 4_000_000

internal_OCR_TOKEN_TRANSLATION = str.maketrans(
    {
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
    }
)


def internal_import_tesserocr() -> Any:
    """Import tesserocr once cysignals' main-thread setup is in place."""
    if "tesserocr" not in sys.modules:
        internal_prepare_ocr_signals()
    return import_module("tesserocr")


def internal_ensure_tesserocr() -> Any:
    """Return the imported binding after its signal setup is ready."""
    return internal_import_tesserocr()


def internal_normalized_ocr_token_key(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(internal_OCR_TOKEN_TRANSLATION).casefold()


def internal_valid_tessdata_path(path: str | os.PathLike[str]) -> Path | None:
    candidate = Path(path).expanduser()
    if (candidate / "eng.traineddata").is_file():
        return candidate.resolve()
    return None


def internal_tessdata_path() -> str:
    """Resolve English traineddata without relying on wheel build prefixes.

    Success and failure are both memoized: ``functools.cache`` alone would not
    store a raised error, so a machine without tessdata re-ran the full
    resolution — directory probes plus a ``tesseract --list-langs``
    subprocess — on every OCR attempt.
    """
    resolved_path, error_message = internal_resolve_tessdata_path()
    if resolved_path is None:
        raise RuntimeError(error_message)
    return resolved_path


def internal_resolve_tessdata_path() -> tuple[str | None, str]:
    configured = os.environ.get("TESSDATA_PREFIX")
    if configured:
        resolved = internal_valid_tessdata_path(configured)
        if resolved is None:
            return (
                None,
                "TESSDATA_PREFIX must name a tessdata directory containing eng.traineddata",
            )
        return str(resolved), ""

    try:
        default_path, languages = internal_ensure_tesserocr().get_languages()
    except RuntimeError:
        default_path, languages = "", ()
    if "eng" in languages:
        resolved = internal_valid_tessdata_path(default_path)
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
        except (OSError, subprocess.TimeoutExpired):
            completed = None
        if completed is not None:
            output = f"{completed.stdout}\n{completed.stderr}"
            match = re.search(r'List of available languages in "([^"]+)"', output)
            if match is not None:
                resolved = internal_valid_tessdata_path(match.group(1))
                if resolved is not None:
                    return str(resolved), ""

    return (
        None,
        "English Tesseract data was not found; set TESSDATA_PREFIX to a tessdata directory "
        "containing eng.traineddata",
    )


def internal_api(mode: int) -> Any:
    tesserocr = internal_ensure_tesserocr()
    api = tesserocr.PyTessBaseAPI(
        path=internal_tessdata_path(),
        psm=mode,
        oem=tesserocr.OEM.LSTM_ONLY,
    )
    api.SetVariable("preserve_interword_spaces", "0")
    api.SetVariable("textord_tablefind_recognize_tables", "0")
    api.SetVariable("textord_tabfind_find_tables", "0")
    return api


internal_HOCR_BBOX_RE = re.compile(r"bbox (\d+) (\d+) (\d+) (\d+)")
internal_HOCR_CONFIDENCE_RE = re.compile(r"(?:x_conf|x_wconf) (-?\d+(?:\.\d+)?)")


class internal_HocrCharacterParser(HTMLParser):
    """Extract line text after dropping low-confidence hOCR characters."""

    def __init__(self, threshold: float) -> None:
        super().__init__(convert_charrefs=True)
        self.threshold = threshold
        self.lines: dict[tuple[int, int, int, int], str] = {}
        self.internal_line_box: tuple[int, int, int, int] | None = None
        self.internal_words: list[str] = []
        self.internal_chars: list[str] = []
        self.internal_char_confidence = threshold
        self.internal_in_char = False
        self.internal_in_word = False

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
            match = internal_HOCR_BBOX_RE.search(title)
            self.internal_line_box = None
            if match:
                left, top, right, bottom = (int(value) for value in match.groups())
                self.internal_line_box = (left, top, right, bottom)
            self.internal_words = []
        elif "ocrx_word" in classes:
            self.internal_in_word = True
            self.internal_chars = []
        elif "ocrx_cinfo" in classes and self.internal_in_word:
            match = internal_HOCR_CONFIDENCE_RE.search(title)
            self.internal_char_confidence = float(match.group(1)) if match else 0.0
            self.internal_in_char = True

    def handle_data(self, data: str) -> None:
        if self.internal_in_char and self.internal_char_confidence >= self.threshold:
            self.internal_chars.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "span":
            return
        if self.internal_in_char:
            self.internal_in_char = False
        elif self.internal_in_word:
            self.internal_words.append("".join(self.internal_chars))
            self.internal_chars = []
            self.internal_in_word = False
        elif self.internal_line_box is not None:
            text = " ".join(word for word in self.internal_words if word).strip()
            self.lines[self.internal_line_box] = text
            self.internal_line_box = None


def internal_hocr_filtered_lines(
    api: Any, threshold: float | None
) -> dict[tuple[int, int, int, int], str]:
    if threshold is None or not hasattr(api, "GetHOCRText"):
        return {}
    try:
        hocr = api.GetHOCRText(0)
    except (RuntimeError, TypeError):
        return {}
    if not hocr:
        return {}
    parser = internal_HocrCharacterParser(threshold)
    parser.feed(hocr.decode("utf-8", "replace") if isinstance(hocr, bytes) else hocr)
    return parser.lines


def internal_acceptable_text(
    text: str, confidence: float, minimum_confidence: float = 20.0
) -> bool:
    if confidence < minimum_confidence or not text:
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
        # A lone non-alphanumeric character as the entirety of an OCR
        # observation is almost always segmentation noise — Braille cells
        # misread from blank regions, decorative glyphs, or empty form-field
        # marks.  Non-ASCII symbols are rejected outright; ASCII punctuation
        # keeps the existing confidence gate so legitimate low-volume marks
        # (e.g. a lone period) are not silently dropped.
        character = stripped[0]
        if not character.isascii() or (confidence < 70.0 and minimum_confidence != 55.0):
            return False
    return not (length >= 8 and same_char_count == length)


def internal_select_character_filtered_candidate(
    raw: internal_Candidate,
    filtered: internal_Candidate,
) -> internal_Candidate:
    """Keep raw OCR unless filtering earns its recall cost.

    hOCR character confidence is useful for removing isolated noise, but treating
    every low-confidence character as false creates large recall losses on dense
    schematics and degraded scans.  The filtered candidate may give up only a small
    amount of local utility while retaining nearly all recovered content.
    """
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


def internal_recognized_symbols(api: Any, task: internal_OcrTask) -> ObservationBatch:
    """Read character boxes from an existing recognition without another OCR pass."""
    iterator = api.GetIterator()
    if iterator is None:
        return ObservationBatch.empty()
    level = internal_ensure_tesserocr().RIL.SYMBOL
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
            boxes.append(internal_map_ocr_box(task, bbox))
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
def internal_suppress_c_stderr() -> Iterator[None]:
    """Redirect C-level stderr (fd 2) to /dev/null during C library execution."""
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


def internal_recognition_timeout(task: internal_OcrTask) -> int:
    """Allow a large raster proportionally more time before it is abandoned."""
    pixels = max(1, task.rectangle[2] * task.rectangle[3])
    excess_megapixels = max(0, pixels - PRIMARY_OCR_PIXELS) / 1_000_000
    budget = OCR_TIMEOUT_MILLISECONDS + int(
        excess_megapixels * OCR_TIMEOUT_MILLISECONDS_PER_MEGAPIXEL
    )
    return min(OCR_TIMEOUT_MAX_MILLISECONDS, budget)


def internal_recognize(
    task: internal_OcrTask,
    *,
    api_override: Any | None = None,
    image_prepared: bool = False,
) -> internal_Candidate:
    if api_override is None:
        api = internal_api(task.mode)
        try:
            return internal_recognize(task, api_override=api, image_prepared=image_prepared)
        finally:
            api.End()
    tesserocr = internal_ensure_tesserocr()
    api = api_override
    api.SetPageSegMode(task.mode)
    if not image_prepared:
        api.SetImageBytes(
            task.image.tesseract_bytes(),
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
    if (
        w > 0
        and h > 0
        and (image_prepared or (x0, y0, w, h) != (0, 0, task.image.width, task.image.height))
    ):
        with internal_suppress_c_stderr():
            api.SetRectangle(x0, y0, w, h)
    api.SetSourceResolution(task.resolution)
    timeout_milliseconds = internal_recognition_timeout(task)
    recognition_started = time.perf_counter()
    with internal_suppress_c_stderr():
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
        else internal_hocr_filtered_lines(api, task.character_confidence_threshold)
    )
    texts: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    polygons: list[tuple[float, ...]] = []
    confidences: list[float] = []
    filtered_texts: list[str] = []
    filtered_boxes: list[tuple[float, float, float, float]] = []
    filtered_polygons: list[tuple[float, ...]] = []
    filtered_confidences: list[float] = []
    text_heights: list[float] = []
    line_breaks: list[bool] = []
    filtered_line_breaks: list[bool] = []
    pending_line_break = True
    if iterator is not None:
        sequence = 0
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
            if bbox is not None and internal_acceptable_text(
                text,
                confidence,
                task.minimum_confidence,
            ):
                x0, y0, x1, y1 = bbox
                bbox_key = (int(x0), int(y0), int(x1), int(y1))
                filtered = filtered_lines.get(bbox_key)
                filtered_text = collapse_ws(filtered) if filtered is not None else text
                texts.append(text)
                mapped_box = internal_map_ocr_box(task, (x0, y0, x1, y1))
                boxes.append(mapped_box)
                polygon = (
                    mapped_box[0],
                    mapped_box[1],
                    mapped_box[2],
                    mapped_box[1],
                    mapped_box[2],
                    mapped_box[3],
                    mapped_box[0],
                    mapped_box[3],
                )
                polygons.append(polygon)
                confidences.append(confidence)
                line_breaks.append(pending_line_break)
                pending_line_break = False
                if internal_acceptable_text(
                    filtered_text,
                    confidence,
                    task.minimum_confidence,
                ):
                    filtered_texts.append(filtered_text)
                    filtered_boxes.append(mapped_box)
                    filtered_polygons.append(polygon)
                    filtered_confidences.append(confidence)
                    filtered_line_breaks.append(line_breaks[-1])
                text_heights.append(float(y1 - y0))
                sequence += 1
            if not iterator.Next(level):
                break
    symbols = (
        internal_recognized_symbols(api, task)
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
        polygon=polygons,
        confidence=confidences,
        sequence=range(len(texts)),
        line_break_before=line_breaks,
    )
    candidate = internal_candidate(
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
        polygon=filtered_polygons,
        confidence=filtered_confidences,
        sequence=range(len(filtered_texts)),
        line_break_before=filtered_line_breaks,
    )
    filtered_candidate = internal_candidate(
        task.mode,
        filtered_observations,
        symbols=symbols,
        recognition_status=recognition_status,
        median_text_height=median_text_height,
    )
    return internal_select_character_filtered_candidate(candidate, filtered_candidate)


def internal_recognize_group(tasks: tuple[internal_OcrTask, ...]) -> tuple[internal_Candidate, ...]:
    """Recognize same-raster tasks while reusing Tesseract image setup."""
    if not tasks:
        return ()
    first = tasks[0]
    api = internal_api(first.mode)
    try:
        candidates = [internal_recognize(first, api_override=api)]
        for task in tasks[1:]:
            candidates.append(internal_recognize(task, api_override=api, image_prepared=True))
        return tuple(candidates)
    finally:
        api.End()


def internal_timeout_recovery_task(task: internal_OcrTask) -> internal_OcrTask | None:
    """Rebuild a timed-out task on a raster small enough to finish.

    Tesseract returns nothing at all when it runs out of time, and the empty
    candidate then wins selection because no later pass improves on it. Cropping
    to the task's own rectangle and reducing it keeps the page's text recoverable
    at lower fidelity instead of dropping the page entirely.
    """
    x, y, width, height = task.rectangle
    pixels = max(1, width * height)
    if pixels <= OCR_TIMEOUT_RETRY_PIXELS:
        return None
    reduction = math.sqrt(OCR_TIMEOUT_RETRY_PIXELS / pixels)
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
    page_box = internal_raster_rectangle_page_box(
        internal_Raster(task.image, task.resolution),
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


def internal_recover_timed_out_tasks(
    tasks: tuple[internal_OcrTask, ...],
    candidates: tuple[internal_Candidate, ...],
    recognize: Callable[[tuple[internal_OcrTask, ...]], tuple[internal_Candidate, ...]],
) -> tuple[internal_Candidate, ...]:
    """Re-run only the tasks whose recognition timed out without producing text.

    Recovery is best effort: a retry that still returns nothing leaves the original
    candidate in place rather than failing the page.
    """
    retry_indexes: list[int] = []
    retry_tasks: list[internal_OcrTask] = []
    for index, (task, candidate) in enumerate(zip(tasks, candidates, strict=False)):
        if candidate.recognition_status != "timeout" or len(candidate.observations):
            continue
        retry = internal_timeout_recovery_task(task)
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
