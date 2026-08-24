# SPDX-License-Identifier: AGPL-3.0-only
"""Tesseract integration: rasterization, recognition, and rescue passes."""

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
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cache
from html.parser import HTMLParser
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy

from core_pdf.impl.engine.array_views import (
    contiguous_bytes,
    resample_bilinear,
    resample_nearest,
    resample_smooth,
    uint8_image_view,
)
from core_pdf.impl.engine.execution import RUNTIME, TaskScope, WorkStage
from core_pdf.impl.engine.image_cache import ImageCacheKey
from core_pdf.impl.engine.layout.geometry import (
    bbox_union,
    overlap_ratio_min,
    overlap_ratio_min_exact,
    overlap_ratio_of,
    rect_tuple,
)
from core_pdf.impl.engine.layout.spatial import (
    SpatialIndex,
    bbox_intersection_area,
)
from core_pdf.impl.engine.parse.capture import (
    VECTOR_PAINT_KINDS,
    internal_promoted_hidden_observations,
)
from core_pdf.impl.engine.parse.fusion import (
    internal_text_tokens,
    maximum_candidate_coverage,
)
from core_pdf.impl.engine.parse.model import (
    HIDDEN_TEXT_VERIFY_MIN_CONFIDENCE,
    HIDDEN_TEXT_VERIFY_MIN_MATCHED_TOKENS,
    HIDDEN_TEXT_VERIFY_MIN_SPATIAL_OVERLAP,
    HIDDEN_TEXT_VERIFY_MIN_TOKEN_OVERLAP,
    HIDDEN_TEXT_VERIFY_PIXELS,
    MAX_OCR_PIXELS,
    MAX_OCR_RASTER_BYTES,
    OCR_PARALLEL_TILE_MIN_VECTOR_COMPLEXITY,
    OCR_PREFLIGHT_PIXELS,
    OCR_RESCUE_LARGE_TEXT_HEIGHT,
    OCR_RESCUE_MIN_CONFIDENCE,
    OCR_RESCUE_MIN_WEAK_INK_RATIO,
    OCR_RESCUE_SATURATED_MEAN_INK,
    PRIMARY_OCR_PIXELS,
    CapturedPage,
    ObservationBatch,
    ObservationSource,
    OcrPass,
    OcrPassScope,
    RecognitionReport,
    RecognitionResult,
    WorkPlan,
    internal_bbox_tuple,
    internal_Candidate,
    internal_candidate,
    internal_OCR_RESCUE_DENSE_MIN_CHARACTERS,
    internal_OCR_RESCUE_DENSE_MIN_CONFIDENCE,
)
from core_pdf.impl.engine.parse.ocr_bootstrap import internal_prepare_ocr_signals
from core_pdf.impl.engine.parse.route import (
    PSM_SPARSE_TEXT,
)
from core_pdf.impl.engine.parse.tables import (
    internal_axis_segments,
    internal_grid_components,
    internal_split_grid_component,
)
from core_pdf.impl.engine.render.display import (
    DisplayList,
    PathPaintItem,
    PathPaintKind,
    RenderOptions,
)
from core_pdf.impl.engine.render.kernels import (
    rasterize_packed_stroked_paths,
)
from core_pdf.impl.engine.render.page import RenderedPage, compose_page
from core_pdf.impl.engine.render.raster_image import RasterImage
from core_pdf.impl.engine.spec.s_07_content.page_program import line_coordinate_columns
from core_pdf.impl.engine.spec.s_08_graphics.image_decode import decode_pdf_image
from core_pdf.impl.engine.stroked_text import (
    StrokedTextDecode,
    StrokedTextObservation,
    StrokedTextProfile,
    StrokedTextRun,
    StrokedTextSeed,
    decode_stroked_text_profile,
    decode_stroked_text_profile_with_supplemental_seeds,
    profile_stroked_text,
    stroked_text_isolated_runs,
)
from core_pdf.impl.text import collapse_ws, search_key

# OCR already has an explicit worker limit. Prevent Tesseract's OpenMP kernels
# from creating another layer of workers on top of it.
os.environ["OMP_THREAD_LIMIT"] = "1"


def internal_import_tesserocr() -> Any:
    """Import tesserocr once cysignals' main-thread setup is in place."""
    if "tesserocr" not in sys.modules:
        internal_prepare_ocr_signals()
    return import_module("tesserocr")


# Importing tesserocr costs ~25 ms and drags in PIL, so it is bound on first use
# rather than at import time: a native-text document never needs either.
# parse.ocr_bootstrap has already installed cysignals' handlers on the main
# thread, so an OCR worker can safely be the one to import it.
internal_TESSEROCR_MODULE: Any | None = None


def internal_ensure_tesserocr() -> Any:
    """Bind tesserocr once, from whichever thread first needs recognition."""
    global internal_TESSEROCR_MODULE
    module = internal_TESSEROCR_MODULE
    if module is None:
        module = internal_import_tesserocr()
        internal_TESSEROCR_MODULE = module
    return module


internal_OCR_LOCAL = threading.local()
OCR_TIMEOUT_MILLISECONDS = 12_000
# Recognition cost grows with the raster, so a flat budget starves exactly the
# large rasters the adaptive passes escalate to. Extend it per megapixel above
# the primary budget, but keep a ceiling so one page cannot stall a document.
OCR_TIMEOUT_MILLISECONDS_PER_MEGAPIXEL = 2_000
OCR_TIMEOUT_MAX_MILLISECONDS = 30_000
# A timed-out recognition yields nothing at all. Retry it once on a raster small
# enough to finish rather than letting the empty candidate win selection.
OCR_TIMEOUT_RETRY_PIXELS = 4_000_000
# Tesseract's LSTM was trained near 300-400 DPI. Scans below that are enlarged to
# reach it; the gain comes from the resampling being smooth, not from pixel count,
# so enlarging past this target only costs recognition time.
DIRECT_OCR_TARGET_RESOLUTION = 400
DIRECT_OCR_MIN_UPSCALE = 1.05
DIRECT_OCR_WHOLE_SCALE_TOLERANCE = 0.06
OCR_BATCH_MAX_TASKS = 16
OCR_BATCH_MAX_PIXELS = 8_000_000


internal_OCR_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
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


@dataclass(slots=True)
class internal_RecognitionTrace:
    """Mutable stage-local collector finalized as one immutable recognition report."""

    passes: list[dict[str, object]]
    candidates: tuple[dict[str, object], ...] = ()
    candidate_analysis: tuple[dict[str, object], ...] = ()
    hidden_text_verification: dict[str, object] | None = None
    stroked_vector_decode: dict[str, object] | None = None
    stroked_vector_packed: dict[str, object] | None = None
    document_stroked_glyphs: dict[str, object] | None = None
    render_timings: dict[str, object] | None = None
    grid_cell_ocr: dict[str, object] | None = None
    render_error: str | None = None
    pending_stroked_decode: tuple[int, StrokedTextDecode, float] | None = None
    stroked_vector_alphabet: tuple[tuple[object, str], ...] = ()

    @classmethod
    def create(cls) -> internal_RecognitionTrace:
        return cls([])

    def report(self) -> RecognitionReport:
        return RecognitionReport(
            passes=tuple(self.passes),
            candidates=self.candidates,
            candidate_analysis=self.candidate_analysis,
            hidden_text_verification=self.hidden_text_verification or {},
            stroked_vector_decode=self.stroked_vector_decode or {},
            stroked_vector_packed=self.stroked_vector_packed or {},
            document_stroked_glyphs=self.document_stroked_glyphs or {},
            render_timings=self.render_timings or {},
            grid_cell_ocr=self.grid_cell_ocr or {},
            render_error=self.render_error,
            stroked_vector_alphabet=self.stroked_vector_alphabet,
        )


def internal_normalized_ocr_token_key(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(internal_OCR_TOKEN_TRANSLATION).casefold()


@dataclass(frozen=True, slots=True)
class internal_HiddenTextVerification:
    hidden_tokens: int
    preview_tokens: int
    matched_tokens: int
    spatially_matched_tokens: int
    token_overlap: float
    spatial_overlap: float
    accepted: bool
    reason: str

    def as_record(self) -> dict[str, int | float | bool | str]:
        return {
            "hidden_tokens": self.hidden_tokens,
            "preview_tokens": self.preview_tokens,
            "matched_tokens": self.matched_tokens,
            "spatially_matched_tokens": self.spatially_matched_tokens,
            "token_overlap": self.token_overlap,
            "spatial_overlap": self.spatial_overlap,
            "accepted": self.accepted,
            "reason": self.reason,
        }


def internal_hidden_text_verification(
    hidden: ObservationBatch,
    preview: ObservationBatch,
) -> internal_HiddenTextVerification:
    """Compare a word-level raster preview with hidden text and its page geometry."""
    hidden_by_token: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for text, raw_box in zip(hidden.text, hidden.bbox, strict=True):
        box = internal_bbox_tuple(raw_box)
        for token in internal_text_tokens(text):
            hidden_by_token[token].append(box)

    preview_entries = tuple(
        (token, internal_bbox_tuple(raw_box))
        for text, raw_box in zip(preview.text, preview.bbox, strict=True)
        for token in internal_text_tokens(text)
    )
    used: dict[str, set[int]] = defaultdict(set)
    matched = 0
    spatially_matched = 0
    for token, preview_box in preview_entries:
        candidates = hidden_by_token.get(token, ())
        available = (
            (index, box) for index, box in enumerate(candidates) if index not in used[token]
        )
        preview_center_x = (preview_box[0] + preview_box[2]) * 0.5
        preview_center_y = (preview_box[1] + preview_box[3]) * 0.5
        closest = min(
            available,
            key=lambda item: (
                ((item[1][0] + item[1][2]) * 0.5 - preview_center_x) ** 2
                + ((item[1][1] + item[1][3]) * 0.5 - preview_center_y) ** 2
            ),
            default=None,
        )
        if closest is None:
            continue
        index, hidden_box = closest
        used[token].add(index)
        matched += 1
        hidden_center_x = (hidden_box[0] + hidden_box[2]) * 0.5
        hidden_center_y = (hidden_box[1] + hidden_box[3]) * 0.5
        x_tolerance = max(
            12.0,
            (preview_box[2] - preview_box[0]) * 1.5,
            (hidden_box[2] - hidden_box[0]) * 1.5,
        )
        y_tolerance = max(
            6.0,
            (preview_box[3] - preview_box[1]) * 1.5,
            (hidden_box[3] - hidden_box[1]) * 1.5,
        )
        spatially_matched += int(
            abs(preview_center_x - hidden_center_x) <= x_tolerance
            and abs(preview_center_y - hidden_center_y) <= y_tolerance
        )

    preview_tokens = len(preview_entries)
    hidden_tokens = sum(len(boxes) for boxes in hidden_by_token.values())
    token_overlap = matched / max(1, preview_tokens)
    spatial_overlap = spatially_matched / max(1, preview_tokens)
    if matched < HIDDEN_TEXT_VERIFY_MIN_MATCHED_TOKENS:
        accepted = False
        reason = "insufficient-matched-tokens"
    elif token_overlap < HIDDEN_TEXT_VERIFY_MIN_TOKEN_OVERLAP:
        accepted = False
        reason = "low-token-overlap"
    elif spatial_overlap < HIDDEN_TEXT_VERIFY_MIN_SPATIAL_OVERLAP:
        accepted = False
        reason = "low-spatial-overlap"
    else:
        accepted = True
        reason = "semantic-and-spatial-match"
    return internal_HiddenTextVerification(
        hidden_tokens=hidden_tokens,
        preview_tokens=preview_tokens,
        matched_tokens=matched,
        spatially_matched_tokens=spatially_matched,
        token_overlap=token_overlap,
        spatial_overlap=spatial_overlap,
        accepted=accepted,
        reason=reason,
    )


@dataclass(frozen=True, slots=True)
class internal_Raster:
    image: RasterImage
    resolution: int
    render_report: dict[str, object] | None = None

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    @property
    def nbytes(self) -> int:
        return self.image.nbytes


@dataclass(frozen=True, slots=True)
class internal_RasterRegion:
    """A decoded raster coupled to the page-space area it actually represents."""

    raster: internal_Raster
    page_box: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class internal_StrokedTextCell:
    """A translated vector-text run in a packed OCR raster."""

    source_box: tuple[float, float, float, float]
    packed_box: tuple[float, float, float, float]
    drawing_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class internal_PackedStrokedTextRaster:
    """One compact raster plus the piecewise map back into PDF page space."""

    raster: internal_Raster
    packed_box: tuple[float, float, float, float]
    cells: tuple[internal_StrokedTextCell, ...]


@dataclass(frozen=True, slots=True)
class internal_OcrRegion:
    """A ranked page-space region selected before compositor rasterization."""

    page_box: tuple[float, float, float, float]
    score: float
    reasons: tuple[str, ...]

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.page_box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


@dataclass(frozen=True, slots=True)
class internal_OcrTask:
    mode: int
    image: RasterImage
    rectangle: tuple[int, int, int, int]
    page_box: tuple[float, float, float, float]
    resolution: int
    minimum_confidence: float = 20.0
    character_confidence_threshold: float | None = None
    recognize_words: bool = False
    collect_symbols: bool = False


def internal_valid_tessdata_path(path: str | os.PathLike[str]) -> Path | None:
    candidate = Path(path).expanduser()
    if (candidate / "eng.traineddata").is_file():
        return candidate.resolve()
    return None


@cache
def internal_tessdata_path() -> str:
    """Resolve English traineddata without relying on wheel build prefixes."""
    configured = os.environ.get("TESSDATA_PREFIX")
    if configured:
        resolved = internal_valid_tessdata_path(configured)
        if resolved is None:
            raise RuntimeError(
                "TESSDATA_PREFIX must name a tessdata directory containing eng.traineddata"
            )
        return str(resolved)

    try:
        default_path, languages = internal_ensure_tesserocr().get_languages()
    except RuntimeError:
        default_path, languages = "", ()
    if "eng" in languages:
        resolved = internal_valid_tessdata_path(default_path)
        if resolved is not None:
            return str(resolved)

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
                    return str(resolved)

    raise RuntimeError(
        "English Tesseract data was not found; set TESSDATA_PREFIX to a tessdata directory "
        "containing eng.traineddata"
    )


def internal_api(mode: int) -> Any:
    api = getattr(internal_OCR_LOCAL, "api", None)
    if api is None:
        psm = mode
        tesserocr = internal_ensure_tesserocr()
        api = tesserocr.PyTessBaseAPI(
            path=internal_tessdata_path(),
            psm=psm,
            oem=tesserocr.OEM.LSTM_ONLY,
        )
        api.SetVariable("preserve_interword_spaces", "0")
        api.SetVariable("textord_tablefind_recognize_tables", "0")
        api.SetVariable("textord_tabfind_find_tables", "0")
        internal_OCR_LOCAL.api = api
    api.SetPageSegMode(mode)
    return api


def internal_prepare_ocr() -> None:
    """Validate OCR startup and construct a reusable API on every worker.

    ``internal_api`` caches its ``PyTessBaseAPI`` in thread-local storage, so
    each worker that recognizes a page pays the Tesseract model load itself:
    measured at ~363 ms for the first build in a process and ~40 ms per
    additional thread once tessdata is in the page cache. Warming the pool here
    keeps that off the critical path of the first page each worker handles,
    which matters most for the single-page documents that make up the bulk of
    OCR work and cannot amortize it over later pages.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("prewarm_runtime() must be called on the main thread")
    internal_api(3)
    RUNTIME.run_on_each_worker(lambda: internal_api(3))


def prewarm_runtime() -> None:
    """Start shared workers and validate OCR during main-thread startup."""
    RUNTIME.prewarm()
    internal_prepare_ocr()


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
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        title = attributes.get("title") or ""
        if "ocr_line" in classes:
            match = re.search(r"bbox (\d+) (\d+) (\d+) (\d+)", title)
            self.internal_line_box = None
            if match:
                left, top, right, bottom = (int(value) for value in match.groups())
                self.internal_line_box = (left, top, right, bottom)
            self.internal_words = []
        elif "ocrx_word" in classes:
            self.internal_in_word = True
            self.internal_chars = []
        elif "ocrx_cinfo" in classes and self.internal_in_word:
            match = re.search(r"(?:x_conf|x_wconf) (-?\d+(?:\.\d+)?)", title)
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


def internal_observation_utility(text: str, confidence: float) -> float:
    """Estimate useful recovered content without rewarding punctuation noise."""
    nonspace_characters = [character for character in text if not character.isspace()]
    if not nonspace_characters:
        return 0.0
    alphanumeric = sum(character.isalnum() for character in nonspace_characters)
    symbols = len(nonspace_characters) - alphanumeric
    # Symbols are useful in forms and schematics, but an unlimited symbol reward lets
    # noisy segmentation beat a smaller, readable pass. Cap their contribution relative
    # to actual text while preserving short labels such as "+5V" and "R/C".
    symbol_credit = min(symbols, max(2.0, alphanumeric * 0.5)) * 0.30
    confidence_factor = 0.25 + 0.75 * min(100.0, max(0.0, confidence)) / 100.0
    repetition_penalty = 1.0
    if len(nonspace_characters) >= 6:
        dominant_ratio = max(
            Counter(character.casefold() for character in nonspace_characters).values()
        ) / len(nonspace_characters)
        if dominant_ratio > 0.60:
            repetition_penalty = max(0.20, 1.0 - (dominant_ratio - 0.60) * 2.0)
    return (alphanumeric + symbol_credit) * confidence_factor * repetition_penalty


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


def internal_map_ocr_box(
    task: internal_OcrTask,
    bbox: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    """Map one Tesseract pixel box into the task's PDF coordinate space."""
    x0, y0, x1, y1 = bbox
    page_x0, page_y0, page_x1, page_y1 = task.page_box
    page_width = page_x1 - page_x0
    page_height = page_y1 - page_y0
    return (
        page_x0 + x0 * page_width / task.image.width,
        page_y1 - y1 * page_height / task.image.height,
        page_x0 + x1 * page_width / task.image.width,
        page_y1 - y0 * page_height / task.image.height,
    )


internal_GRID_DARK_THRESHOLD = 160
internal_GRID_LINE_MIN_FRACTION = 0.30
internal_GRID_LINE_CLUSTER_PX = 5
internal_GRID_MIN_LINES = 4
internal_GRID_MIN_SPAN_FRACTION = 0.25
internal_GRID_CELL_MIN_PX = 6
internal_GRID_CELL_INSET_PX = 2
internal_GRID_CELL_MIN_INK = 0.003
internal_GRID_MAX_CELLS = 1200
internal_GRID_MIN_CELLS = 12
internal_GRID_CELL_MIN_CONFIDENCE = 50.0
internal_PSM_SINGLE_LINE = 7


internal_GRID_RULE_GAP_PX = 6


def internal_close_row_gaps(mask: numpy.ndarray, gap: int) -> numpy.ndarray:
    """Bridge horizontal gaps up to ``gap`` pixels inside each row.

    Scanned rulings drop out along their length, so a rule reads as many
    short runs. Closing (dilate then erode along the row) reconnects them
    without thickening genuine text into false rules.
    """
    if gap <= 0:
        return mask
    window = gap + 1
    padded = numpy.zeros((mask.shape[0], mask.shape[1] + 2 * window), dtype=numpy.int32)
    padded[:, window:-window] = mask
    sums = numpy.cumsum(padded, axis=1)
    dilated = (sums[:, 2 * window :] - sums[:, : -2 * window]) > 0
    padded2 = numpy.zeros_like(padded)
    padded2[:, window:-window] = dilated
    sums2 = numpy.cumsum(padded2, axis=1)
    return (sums2[:, 2 * window :] - sums2[:, : -2 * window]) >= (2 * window)


def internal_longest_true_runs(mask: numpy.ndarray) -> numpy.ndarray:
    """Return the longest consecutive True run per row of a boolean matrix."""
    height, width = mask.shape
    separated = numpy.zeros((height, width + 2), dtype=numpy.int8)
    separated[:, 1:-1] = mask
    flat = separated.ravel()
    deltas = numpy.diff(flat)
    starts = numpy.flatnonzero(deltas == 1)
    ends = numpy.flatnonzero(deltas == -1)
    longest = numpy.zeros(height, dtype=numpy.int64)
    if len(starts):
        numpy.maximum.at(longest, starts // (width + 2), ends - starts)
    return longest


def internal_cluster_line_positions(
    positions: numpy.ndarray,
    tolerance: int = internal_GRID_LINE_CLUSTER_PX,
) -> list[int]:
    clusters: list[list[int]] = []
    for position in positions.tolist():
        if clusters and position - clusters[-1][-1] <= tolerance:
            clusters[-1].append(position)
        else:
            clusters.append([position])
    return [int(round(sum(cluster) / len(cluster))) for cluster in clusters]


internal_GRID_STRIP_MIN_FRACTION = 0.5
internal_GRID_STRIP_GAP_PX = 12
internal_GRID_MAX_SKEW = 0.02
internal_GRID_DETECT_POOL = 3


def internal_estimate_ruling_skew(dark: numpy.ndarray) -> float:
    """Estimate page skew from horizontal rule offsets between page thirds.

    A fraction of a degree of scanner skew walks a page-wide rule across a
    dozen pixel rows, which breaks projection-based detection outright. The
    rules themselves measure the skew: detect them separately in the left
    and right thirds, pair them up, and read the slope off the median
    vertical offset.
    """
    height, width = dark.shape
    strip = width // 3
    if strip < 50:
        return 0.0
    gap = internal_GRID_STRIP_GAP_PX
    left_runs = internal_longest_true_runs(internal_close_row_gaps(dark[:, :strip], gap))
    right_runs = internal_longest_true_runs(internal_close_row_gaps(dark[:, -strip:], gap))
    minimum = strip * internal_GRID_STRIP_MIN_FRACTION
    left_lines = internal_cluster_line_positions(numpy.flatnonzero(left_runs >= minimum))
    right_lines = internal_cluster_line_positions(numpy.flatnonzero(right_runs >= minimum))
    if len(left_lines) < 3 or len(right_lines) < 3:
        return 0.0
    baseline = width - strip
    offsets = []
    for line in left_lines:
        nearest = min(right_lines, key=lambda candidate: abs(candidate - line))
        if abs(nearest - line) <= baseline * internal_GRID_MAX_SKEW:
            offsets.append(nearest - line)
    if len(offsets) < 3:
        return 0.0
    return float(numpy.median(offsets)) / baseline


def internal_vertical_shear(dark: numpy.ndarray, slope: float) -> numpy.ndarray:
    """Shift each column vertically by ``-slope * x`` to straighten h-rules."""
    height, width = dark.shape
    shifts = numpy.round(slope * numpy.arange(width)).astype(numpy.int64)
    rows = numpy.arange(height)[:, None] + shifts[None, :]
    numpy.clip(rows, 0, height - 1, out=rows)
    return numpy.take_along_axis(dark, rows, axis=0)


def internal_detect_ruling_grid(
    image: RasterImage,
) -> tuple[list[int], list[int], numpy.ndarray, float] | None:
    """Find a ruled table grid; return (x_edges, y_edges, source samples, skew).

    A ruling is a near-full-span dark run: project the longest dark run per
    row (and per column) after closing scan dropouts and straightening the
    measured skew, and keep positions exceeding a fraction of the page span,
    clustered so a thick or doubled rule counts once. Ruled ledgers and data
    books produce a dozen or more of each; ordinary prose and underline
    styled forms do not reach the minimum in both axes. Edges are reported
    in the sheared (straightened) frame; callers map cells back through the
    returned skew.
    """
    array = numpy.asarray(image.array())
    if array.ndim == 3 and array.shape[2] >= 3:
        color = array[:, :, :3]
    elif array.ndim == 3 and array.shape[2] == 1:
        color = array[:, :, 0]
    elif array.ndim == 2:
        color = array
    else:
        return None
    height, width = color.shape[:2]
    if height < 100 or width < 100:
        return None
    # Detection runs on every recognized page, so work on a 3x max-pooled
    # mask: rules survive any-pooling and the +-3px edge error disappears
    # into the cell insets, while the shears and run scans cost a ninth.
    pool = internal_GRID_DETECT_POOL
    pooled_height = height // pool
    pooled_width = width // pool
    cropped = color[: pooled_height * pool, : pooled_width * pool]
    if cropped.ndim == 3:
        pooled = (
            cropped.reshape(pooled_height, pool, pooled_width, pool, 3).min(axis=(1, 3, 4))
            < internal_GRID_DARK_THRESHOLD
        )
    else:
        pooled = (
            cropped.reshape(pooled_height, pool, pooled_width, pool).min(axis=(1, 3))
            < internal_GRID_DARK_THRESHOLD
        )
    pooled_height, pooled_width = pooled.shape
    slope = internal_estimate_ruling_skew(pooled)
    straight = internal_vertical_shear(pooled, slope) if slope else pooled
    straight_columns = internal_vertical_shear(pooled.T, -slope) if slope else pooled.T
    gap = max(1, internal_GRID_STRIP_GAP_PX // pool)
    row_runs = internal_longest_true_runs(internal_close_row_gaps(straight, gap))
    column_runs = internal_longest_true_runs(internal_close_row_gaps(straight_columns, gap))
    y_lines = internal_cluster_line_positions(
        numpy.flatnonzero(row_runs >= pooled_width * internal_GRID_LINE_MIN_FRACTION),
        tolerance=max(1, internal_GRID_LINE_CLUSTER_PX // pool),
    )
    x_lines = internal_cluster_line_positions(
        numpy.flatnonzero(column_runs >= pooled_height * internal_GRID_LINE_MIN_FRACTION),
        tolerance=max(1, internal_GRID_LINE_CLUSTER_PX // pool),
    )
    if len(y_lines) < internal_GRID_MIN_LINES or len(x_lines) < internal_GRID_MIN_LINES:
        return None
    if (
        y_lines[-1] - y_lines[0] < pooled_height * internal_GRID_MIN_SPAN_FRACTION
        or x_lines[-1] - x_lines[0] < pooled_width * internal_GRID_MIN_SPAN_FRACTION
    ):
        return None
    scaled_x = [line * pool + pool // 2 for line in x_lines]
    scaled_y = [line * pool + pool // 2 for line in y_lines]
    return scaled_x, scaled_y, color, slope


def internal_grid_cell_tasks(
    task: internal_OcrTask,
    x_lines: list[int],
    y_lines: list[int],
    source_samples: numpy.ndarray,
    slope: float,
) -> tuple[internal_OcrTask, ...]:
    """Build one single-line OCR task per populated ruled cell.

    Tesseract's own segmentation merges and splits ruled cells; giving it one
    cell at a time replaces its layout analysis with the grid geometry. Grid
    edges arrive in the straightened frame, so each cell shifts back through
    the measured skew at its own centre -- the residual rotation across one
    cell is under a pixel. Cell pixel boxes stay in full-image coordinates so
    the shared page-box mapping applies unchanged, and near-blank cells are
    skipped by ink density instead of being recognized into noise.
    """
    if (len(x_lines) - 1) * (len(y_lines) - 1) > internal_GRID_MAX_CELLS:
        return ()
    # Line positions mark rule centres, and a scanned rule is several pixels
    # thick: an inset smaller than the rule leaves box fragments inside every
    # crop, which single-line recognition reads as bars or rejects outright.
    inset = max(internal_GRID_CELL_INSET_PX, int(round(task.resolution / 40)))
    height, width = source_samples.shape[:2]
    tasks: list[internal_OcrTask] = []
    for row_start, row_end in zip(y_lines, y_lines[1:]):
        if row_end - row_start < internal_GRID_CELL_MIN_PX + 2 * inset:
            continue
        for column_start, column_end in zip(x_lines, x_lines[1:]):
            if column_end - column_start < internal_GRID_CELL_MIN_PX + 2 * inset:
                continue
            center_x = (column_start + column_end) * 0.5
            center_y = (row_start + row_end) * 0.5
            row_shift = int(round(slope * center_x))
            column_shift = -int(round(slope * center_y))
            top = max(0, row_start + inset + row_shift)
            bottom = min(height, row_end - inset + row_shift)
            left = max(0, column_start + inset + column_shift)
            right = min(width, column_end - inset + column_shift)
            if bottom - top < internal_GRID_CELL_MIN_PX or right - left < (
                internal_GRID_CELL_MIN_PX
            ):
                continue
            cell = source_samples[top:bottom, left:right]
            if cell.ndim == 3:
                ink_ratio = float(
                    numpy.count_nonzero(cell.min(axis=2) < internal_GRID_DARK_THRESHOLD)
                ) / (cell.shape[0] * cell.shape[1])
            else:
                ink_ratio = float(numpy.count_nonzero(cell < internal_GRID_DARK_THRESHOLD)) / (
                    cell.shape[0] * cell.shape[1]
                )
            if ink_ratio < internal_GRID_CELL_MIN_INK:
                continue
            tasks.append(
                internal_OcrTask(
                    mode=internal_PSM_SINGLE_LINE,
                    image=task.image,
                    rectangle=(left, top, right - left, bottom - top),
                    page_box=task.page_box,
                    resolution=task.resolution,
                    minimum_confidence=internal_GRID_CELL_MIN_CONFIDENCE,
                )
            )
    return tuple(tasks)


def internal_grid_region_page_box(
    task: internal_OcrTask,
    x_lines: list[int],
    y_lines: list[int],
) -> tuple[float, float, float, float]:
    page_x0, page_y0, page_x1, page_y1 = task.page_box
    page_width = page_x1 - page_x0
    page_height = page_y1 - page_y0
    return (
        page_x0 + x_lines[0] * page_width / task.image.width,
        page_y1 - y_lines[-1] * page_height / task.image.height,
        page_x0 + x_lines[-1] * page_width / task.image.width,
        page_y1 - y_lines[0] * page_height / task.image.height,
    )


internal_GRID_MIN_ROWS = 8
internal_GRID_MIN_COLUMNS = 5
internal_GRID_MAX_ROW_HEIGHT_DEVIATION = 0.4
internal_GRID_MAX_STRADDLE_RATIO = 0.25


def internal_grid_is_regular_table(
    grid: tuple[list[int], list[int], numpy.ndarray, float],
    prior: ObservationBatch,
    task: internal_OcrTask,
) -> bool:
    """Distinguish a data table's grid from a form's boxed fields.

    Boxed forms clear the line-count bar as easily as data tables, but
    slicing a form into cells chops label text that spans boxes. A data
    table has many rows of near-uniform height and its words respect the
    column rules; a form has neither, and the words the page pass already
    read betray it by straddling the vertical lines.
    """
    x_lines, y_lines, _source_samples, slope = grid
    if len(y_lines) - 1 < internal_GRID_MIN_ROWS or len(x_lines) - 1 < internal_GRID_MIN_COLUMNS:
        return False
    heights = numpy.diff(numpy.asarray(y_lines, dtype=numpy.float64))
    median_height = float(numpy.median(heights))
    if median_height <= 0.0:
        return False
    deviation = float(numpy.median(numpy.abs(heights - median_height))) / median_height
    if deviation > internal_GRID_MAX_ROW_HEIGHT_DEVIATION:
        return False
    if not len(prior):
        return True
    page_x0, _page_y0, page_x1, page_y1 = task.page_box
    page_width = page_x1 - page_x0
    scale = task.image.width / max(1e-6, page_width)
    interior = x_lines[1:-1]
    if not interior:
        return True
    grid_box = internal_grid_region_page_box(task, x_lines, y_lines)
    inside = 0
    straddling = 0
    slack = max(2.0, task.image.width * 0.002)
    for box in prior.bbox:
        center_x = float(box[0] + box[2]) * 0.5
        center_y = float(box[1] + box[3]) * 0.5
        if not (grid_box[0] <= center_x <= grid_box[2] and grid_box[1] <= center_y <= grid_box[3]):
            continue
        inside += 1
        # Straighten the word into the sheared frame the lines live in.
        pixel_y = (page_y1 - center_y) * scale
        pixel_x0 = (float(box[0]) - page_x0) * scale + slope * pixel_y
        pixel_x1 = (float(box[2]) - page_x0) * scale + slope * pixel_y
        if any(pixel_x0 < line - slack and pixel_x1 > line + slack for line in interior):
            straddling += 1
    if inside < 8:
        return True
    return straddling <= inside * internal_GRID_MAX_STRADDLE_RATIO


def internal_grid_row_observations(
    observations: ObservationBatch,
) -> ObservationBatch:
    """Merge cell reads into one observation per grid row, left to right.

    Cell observations are isolated boxes, and the column gutters between them
    invite the layout cut to emit the table column-major. The reference
    reading order for a data table is row-major, so pre-assemble each row
    into a single line observation the layout keeps whole.
    """
    if not len(observations):
        return observations
    heights = observations.bbox[:, 3] - observations.bbox[:, 1]
    tolerance = max(2.0, float(numpy.median(heights)) * 0.6)
    order = numpy.argsort(-(observations.bbox[:, 1] + observations.bbox[:, 3]) * 0.5)
    rows: list[list[int]] = []
    row_center = 0.0
    for index in order.tolist():
        center = float(observations.bbox[index, 1] + observations.bbox[index, 3]) * 0.5
        if rows and abs(row_center - center) <= tolerance:
            rows[-1].append(index)
        else:
            rows.append([index])
            row_center = center
    texts: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    confidences: list[float] = []
    for row in rows:
        ordered = sorted(row, key=lambda index: float(observations.bbox[index, 0]))
        texts.append(" ".join(observations.text[index].strip() for index in ordered))
        row_boxes = observations.bbox[ordered]
        boxes.append(
            (
                float(row_boxes[:, 0].min()),
                float(row_boxes[:, 1].min()),
                float(row_boxes[:, 2].max()),
                float(row_boxes[:, 3].max()),
            )
        )
        confidences.append(
            float(numpy.mean([float(observations.confidence[index]) for index in ordered]))
        )
    return ObservationBatch.from_columns(
        texts,
        boxes,
        source=ObservationSource.OCR,
        confidence=confidences,
        sequence=range(len(texts)),
        rotation=(0 for _ in texts),
        font_size=(box[3] - box[1] for box in boxes),
        line_break_before=(True for _ in texts),
    )


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
    tesserocr = internal_ensure_tesserocr()
    api_started = time.perf_counter()
    api = api_override if api_override is not None else internal_api(task.mode)
    api_seconds = time.perf_counter() - api_started if api_override is None else 0.0
    setup_started = time.perf_counter()
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
    setup_seconds = time.perf_counter() - setup_started
    timeout_milliseconds = internal_recognition_timeout(task)
    recognition_started = time.perf_counter()
    with internal_suppress_c_stderr():
        recognized = api.Recognize(timeout=timeout_milliseconds)
    recognition_seconds = time.perf_counter() - recognition_started
    if recognized:
        recognition_status = "ok"
    elif recognition_seconds >= timeout_milliseconds / 1000.0 * 0.9:
        recognition_status = "timeout"
    else:
        recognition_status = "failed"
    iterator_started = time.perf_counter()
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
    iterator_seconds = time.perf_counter() - iterator_started
    cleanup_started = time.perf_counter()
    clear_adaptive = getattr(api, "ClearAdaptiveClassifier", None)
    if callable(clear_adaptive):
        clear_adaptive()
    cleanup_seconds = time.perf_counter() - cleanup_started
    candidate_started = time.perf_counter()
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
        api_seconds=api_seconds,
        setup_seconds=setup_seconds,
        recognition_seconds=recognition_seconds,
        iterator_seconds=iterator_seconds,
        cleanup_seconds=cleanup_seconds,
        recognition_status=recognition_status,
        median_text_height=(float(numpy.median(text_heights)) if text_heights else 0.0),
    )
    candidate_seconds = time.perf_counter() - candidate_started
    candidate = replace(candidate, candidate_seconds=candidate_seconds)
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
        api_seconds=api_seconds,
        setup_seconds=setup_seconds,
        recognition_seconds=recognition_seconds,
        iterator_seconds=iterator_seconds,
        cleanup_seconds=cleanup_seconds,
        candidate_seconds=candidate_seconds,
        recognition_status=recognition_status,
        median_text_height=(float(numpy.median(text_heights)) if text_heights else 0.0),
    )
    return internal_select_character_filtered_candidate(candidate, filtered_candidate)


def internal_recognize_group(tasks: tuple[internal_OcrTask, ...]) -> tuple[internal_Candidate, ...]:
    """Recognize same-raster tasks while reusing Tesseract image setup."""
    if not tasks:
        return ()
    if len(tasks) == 1:
        return (internal_recognize(tasks[0]),)
    first = tasks[0]
    api = internal_api(first.mode)
    candidates = [internal_recognize(first, api_override=api)]
    for task in tasks[1:]:
        candidates.append(internal_recognize(task, api_override=api, image_prepared=True))
    return tuple(candidates)


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


def internal_ocr_task_groups(
    tasks: tuple[internal_OcrTask, ...],
) -> tuple[tuple[internal_OcrTask, ...], ...]:
    """Create ordered same-raster/mode batches without duplicating image setup."""
    groups: list[tuple[internal_OcrTask, ...]] = []
    current: list[internal_OcrTask] = []
    current_pixels = 0
    for task in tasks:
        pixels = task.rectangle[2] * task.rectangle[3]
        if current and (
            task.image is not current[0].image
            or task.mode != current[0].mode
            or len(current) >= OCR_BATCH_MAX_TASKS
            or current_pixels + pixels > OCR_BATCH_MAX_PIXELS
        ):
            groups.append(tuple(current))
            current = []
            current_pixels = 0
        current.append(task)
        current_pixels += pixels
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def internal_tile_tasks(
    raster: internal_Raster,
    page_box: tuple[float, float, float, float],
    ocr_pass: OcrPass,
    *,
    compact_image: bool | str = False,
) -> tuple[internal_OcrTask, ...]:
    if ocr_pass.preprocess == "binary-clean":
        raster = internal_adaptive_ocr_raster(raster)
    image = (
        internal_compact_ocr_image(raster.image, grayscale=compact_image == "grayscale")
        if compact_image
        else raster.image
    )
    requested_tiles = ocr_pass.tiles if ocr_pass.scope is OcrPassScope.TILES else 1
    tiles = max(1, min(requested_tiles, raster.height))
    if tiles == 1:
        return tuple(
            internal_OcrTask(
                mode=mode,
                image=image,
                rectangle=(0, 0, raster.width, raster.height),
                page_box=page_box,
                resolution=raster.resolution,
                minimum_confidence=ocr_pass.minimum_confidence,
                character_confidence_threshold=ocr_pass.character_confidence_threshold,
                recognize_words=ocr_pass.recognize_words,
                collect_symbols=ocr_pass.collect_symbols,
            )
            for mode in ocr_pass.modes
        )
    overlap = max(24, int(round(raster.resolution * 0.35)))
    base_height = math.ceil(raster.height / tiles)
    tasks = []
    for mode in ocr_pass.modes:
        for tile_index in range(tiles):
            y0 = max(0, tile_index * base_height - overlap)
            y1 = min(raster.height, (tile_index + 1) * base_height + overlap)
            tasks.append(
                internal_OcrTask(
                    mode=mode,
                    image=image,
                    rectangle=(0, y0, raster.width, y1 - y0),
                    page_box=page_box,
                    resolution=raster.resolution,
                    minimum_confidence=ocr_pass.minimum_confidence,
                    character_confidence_threshold=ocr_pass.character_confidence_threshold,
                    recognize_words=ocr_pass.recognize_words,
                    collect_symbols=ocr_pass.collect_symbols,
                )
            )
    return tuple(tasks)


def internal_raster_ink_grid(
    raster: internal_Raster, rows: int, columns: int
) -> numpy.ndarray[Any, Any]:
    """Measure visual ink per coarse region from a bounded zero-copy raster sample."""
    if rows <= 0 or columns <= 0:
        return numpy.zeros(max(0, rows * columns), dtype=numpy.float32)
    pixels = raster.image.array()
    y_step = max(1, raster.height // 512)
    x_step = max(1, raster.width // 512)
    sampled = pixels[::y_step, ::x_step]
    if raster.image.channels == 1:
        intensity = sampled[:, :, 0]
    else:
        intensity = numpy.min(sampled[:, :, :3], axis=2)
    ink = intensity < 245
    integral = numpy.pad(
        ink.cumsum(axis=0, dtype=numpy.int32).cumsum(axis=1, dtype=numpy.int32),
        ((1, 0), (1, 0)),
    )
    y_bounds = numpy.arange(rows + 1, dtype=numpy.intp) * len(ink) // rows
    x_bounds = numpy.arange(columns + 1, dtype=numpy.intp) * ink.shape[1] // columns
    y0, y1 = y_bounds[:-1], y_bounds[1:]
    x0, x1 = x_bounds[:-1], x_bounds[1:]
    sums = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    counts = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    grid_output = numpy.zeros((rows, columns), dtype=numpy.float32)
    numpy.divide(sums, counts, out=grid_output, where=counts != 0)
    return grid_output.reshape(-1)


def internal_estimated_text_height(raster: internal_Raster) -> float:
    """Estimate ordinary text-band height from a bounded raster preview.

    Horizontal projections are substantially cheaper than an exploratory OCR
    pass.  Sampling several vertical strips avoids letting table borders or a
    single illustration join otherwise independent text lines.
    """
    pixels = raster.image.array()
    sample_step = max(1, math.ceil(math.sqrt(raster.width * raster.height / 1_000_000)))
    sampled = pixels[::sample_step, ::sample_step]
    gray = sampled[:, :, 0] if raster.image.channels == 1 else numpy.min(sampled[:, :, :3], axis=2)
    background = float(numpy.percentile(gray, 90.0))
    threshold = max(80.0, min(225.0, background - 24.0))
    ink = gray < threshold
    if not numpy.any(ink):
        return 0.0
    strip_count = max(4, min(12, ink.shape[1] // 48))
    heights: list[int] = []
    for strip in numpy.array_split(ink, strip_count, axis=1):
        if strip.shape[1] < 4:
            continue
        required = max(2, int(math.ceil(strip.shape[1] * 0.01)))
        active = numpy.count_nonzero(strip, axis=1) >= required
        # Close a one-row break caused by ascenders, punctuation, or scan noise.
        if len(active) >= 3:
            active[1:-1] |= active[:-2] & active[2:]
        padded = numpy.pad(active.astype(numpy.int8), (1, 1))
        transitions = numpy.diff(padded)
        starts = numpy.flatnonzero(transitions == 1)
        ends = numpy.flatnonzero(transitions == -1)
        for height in ends - starts:
            if 2 <= height <= max(12, sampled.shape[0] // 12):
                heights.append(int(height))
    if len(heights) < 4:
        return 0.0
    values = numpy.asarray(heights, dtype=numpy.float32)
    lower = float(numpy.percentile(values, 25.0))
    upper = float(numpy.percentile(values, 85.0))
    typical = values[(values >= lower) & (values <= upper)]
    return float(numpy.median(typical if len(typical) else values)) * sample_step


def internal_observation_coverage_grid(
    observations: ObservationBatch,
    page_box: tuple[float, float, float, float],
    rows: int,
    columns: int,
) -> numpy.ndarray[Any, Any]:
    if not len(observations):
        return numpy.zeros(rows * columns, dtype=numpy.float32)
    x0, y0, x1, y1 = page_box
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    output = numpy.zeros((rows, columns), dtype=numpy.float32)
    for text, confidence, raw_box in zip(
        observations.text,
        observations.confidence,
        observations.bbox,
        strict=True,
    ):
        raw_x0, raw_y0, raw_x1, raw_y1 = internal_bbox_tuple(raw_box)
        box_x0 = max(x0, raw_x0)
        box_y0 = max(y0, raw_y0)
        box_x1 = min(x1, raw_x1)
        box_y1 = min(y1, raw_y1)
        box_width = box_x1 - box_x0
        box_height = box_y1 - box_y0
        if box_width <= 0.0 or box_height <= 0.0:
            continue
        utility = internal_observation_utility(text, float(confidence))
        if utility <= 0.0:
            continue
        column_start = max(0, min(columns - 1, int((box_x0 - x0) * columns / width)))
        column_end = max(
            column_start,
            min(columns - 1, math.ceil((box_x1 - x0) * columns / width) - 1),
        )
        row_start = max(0, min(rows - 1, int((y1 - box_y1) * rows / height)))
        row_end = max(
            row_start,
            min(rows - 1, math.ceil((y1 - box_y0) * rows / height) - 1),
        )
        box_area = box_width * box_height
        for row in range(row_start, row_end + 1):
            cell_y0 = y1 - (row + 1) * height / rows
            cell_y1 = y1 - row * height / rows
            overlap_y = max(0.0, min(box_y1, cell_y1) - max(box_y0, cell_y0))
            if overlap_y <= 0.0:
                continue
            for column in range(column_start, column_end + 1):
                cell_x0 = x0 + column * width / columns
                cell_x1 = x0 + (column + 1) * width / columns
                overlap_x = max(0.0, min(box_x1, cell_x1) - max(box_x0, cell_x0))
                if overlap_x > 0.0:
                    output[row, column] += utility * overlap_x * overlap_y / box_area
    return output.reshape(-1)


def internal_observation_utility_grid(
    observations: ObservationBatch,
    page_box: tuple[float, float, float, float],
    rows: int,
    columns: int,
) -> numpy.ndarray[Any, Any]:
    """Assign each observation to one cell for stable weak-region ranking."""
    if not len(observations):
        return numpy.zeros(rows * columns, dtype=numpy.float32)
    x0, y0, x1, y1 = page_box
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    centers_x = (observations.bbox[:, 0] + observations.bbox[:, 2]) * 0.5
    centers_y = (observations.bbox[:, 1] + observations.bbox[:, 3]) * 0.5
    inside = (centers_x >= x0) & (centers_x <= x1) & (centers_y >= y0) & (centers_y <= y1)
    if not numpy.any(inside):
        return numpy.zeros(rows * columns, dtype=numpy.float32)
    centers_x = centers_x[inside]
    centers_y = centers_y[inside]
    columns_by_observation = numpy.clip(
        ((centers_x - x0) * columns / width).astype(numpy.int64),
        0,
        columns - 1,
    )
    rows_by_observation = numpy.clip(
        ((y1 - centers_y) * rows / height).astype(numpy.int64),
        0,
        rows - 1,
    )
    utility = numpy.fromiter(
        (
            internal_observation_utility(text, float(confidence))
            for text, confidence in zip(
                (
                    text
                    for text, selected in zip(observations.text, inside, strict=True)
                    if selected
                ),
                observations.confidence[inside],
                strict=True,
            )
        ),
        dtype=numpy.float32,
        count=int(numpy.count_nonzero(inside)),
    )
    return numpy.bincount(
        rows_by_observation * columns + columns_by_observation,
        weights=utility,
        minlength=rows * columns,
    ).astype(numpy.float32, copy=False)


def internal_weak_region_grid_shape(
    raster: internal_Raster,
    ocr_pass: OcrPass,
    primary: ObservationBatch,
) -> tuple[int, int]:
    rows = max(1, min(ocr_pass.tiles, raster.height))
    columns = max(1, min(ocr_pass.region_columns, raster.width))
    if len(primary) >= 40:
        rows = min(rows, 6)
        columns = min(columns, 3)
    return rows, columns


def internal_weak_region_rectangles(
    raster: internal_Raster,
    page_box: tuple[float, float, float, float],
    ocr_pass: OcrPass,
    primary: ObservationBatch,
) -> tuple[tuple[int, int, int, int], ...]:
    """Find visually occupied cells where the primary OCR recovered little text."""
    rows, columns = internal_weak_region_grid_shape(raster, ocr_pass, primary)
    ink = internal_raster_ink_grid(raster, rows, columns)
    utility = internal_observation_utility_grid(primary, page_box, rows, columns)
    expected_utility = float(numpy.sum(utility)) / max(1, rows * columns)
    utility_limit = max(4.0, expected_utility * 0.45)
    eligible = numpy.flatnonzero((ink >= 0.01) & (utility < utility_limit))
    if not len(eligible):
        return ()
    priority = ink[eligible] / (1.0 + utility[eligible] * 0.05)
    region_limit = ocr_pass.max_regions
    if len(primary) >= 40:
        region_limit = max(1, region_limit // 2)
        region_limit = min(region_limit, 8)
    ranked = eligible[numpy.argsort(priority)[::-1][:region_limit]]
    # Tesseract's sparse-text layout pass scans connected components that can cross
    # the requested rectangle.  A narrow horizontal margin can therefore make
    # Leptonica reject a component as being outside the active rectangle.  Keep a
    # generous, resolution-scaled margin so region boundaries do not bisect glyphs
    # or text lines.
    overlap_x = min(
        max(48, int(round(raster.resolution * 0.20))),
        max(0, (raster.width // columns - 1) // 2),
    )
    overlap_y = min(
        max(48, int(round(raster.resolution * 0.20))),
        max(0, (raster.height // rows - 1) // 2),
    )
    rectangles: list[tuple[int, int, int, int]] = []
    for raw_cell in ranked:
        cell = int(raw_cell)
        row, column = divmod(cell, columns)
        cell_x0 = column * raster.width // columns
        cell_x1 = (column + 1) * raster.width // columns
        cell_y0 = row * raster.height // rows
        cell_y1 = (row + 1) * raster.height // rows
        rectangle_x0 = max(0, cell_x0 - overlap_x)
        rectangle_x1 = min(raster.width, cell_x1 + overlap_x)
        rectangle_y0 = max(0, cell_y0 - overlap_y)
        rectangle_y1 = min(raster.height, cell_y1 + overlap_y)
        rectangles.append(
            (
                rectangle_x0,
                rectangle_y0,
                rectangle_x1 - rectangle_x0,
                rectangle_y1 - rectangle_y0,
            )
        )
    return tuple(rectangles)


def internal_weak_region_tasks(
    raster: internal_Raster,
    page_box: tuple[float, float, float, float],
    ocr_pass: OcrPass,
    primary: ObservationBatch,
    *,
    compact_image: bool | str = False,
) -> tuple[internal_OcrTask, ...]:
    """Create OCR tasks for weak regions in an already materialized raster."""
    image = (
        internal_compact_ocr_image(raster.image, grayscale=compact_image == "grayscale")
        if compact_image
        else raster.image
    )
    return tuple(
        internal_OcrTask(
            mode=mode,
            image=image,
            rectangle=rectangle,
            page_box=page_box,
            resolution=raster.resolution,
            minimum_confidence=ocr_pass.minimum_confidence,
            character_confidence_threshold=ocr_pass.character_confidence_threshold,
            recognize_words=ocr_pass.recognize_words,
            collect_symbols=ocr_pass.collect_symbols,
        )
        for mode in ocr_pass.modes
        for rectangle in internal_weak_region_rectangles(raster, page_box, ocr_pass, primary)
    )


@dataclass(frozen=True, slots=True)
class internal_RescueCoverage:
    raster_count: int = 0
    cell_count: int = 0
    ink_cells: int = 0
    weak_cells: int = 0
    ink: float = 0.0
    weak_ink: float = 0.0

    @property
    def mean_ink(self) -> float:
        return self.ink / max(1, self.cell_count)

    @property
    def weak_ink_ratio(self) -> float:
        return self.weak_ink / max(1e-9, self.ink)

    def as_record(self) -> dict[str, int | float]:
        return {
            "raster_count": self.raster_count,
            "cell_count": self.cell_count,
            "ink_cells": self.ink_cells,
            "weak_cells": self.weak_cells,
            "mean_ink": self.mean_ink,
            "weak_ink_ratio": self.weak_ink_ratio,
        }


def internal_adaptive_rescue_coverage(
    source_tasks: tuple[internal_OcrTask, ...],
    ocr_pass: OcrPass,
    primary: ObservationBatch,
) -> internal_RescueCoverage:
    """Measure ink not spatially explained by the primary OCR observations."""
    raster_count = 0
    cell_count = 0
    ink_cells = 0
    weak_cells = 0
    total_ink = 0.0
    weak_ink = 0.0
    seen: set[tuple[int, tuple[float, float, float, float], int]] = set()
    for task in source_tasks:
        key = (id(task.image), task.page_box, task.resolution)
        if key in seen:
            continue
        seen.add(key)
        raster = internal_Raster(task.image, task.resolution)
        rows, columns = internal_weak_region_grid_shape(raster, ocr_pass, primary)
        ink = internal_raster_ink_grid(raster, rows, columns)
        coverage = internal_observation_coverage_grid(primary, task.page_box, rows, columns)
        utility_limit = max(4.0, float(numpy.sum(coverage)) / (rows * columns) * 0.45)
        occupied = ink >= 0.01
        weak = occupied & (coverage < utility_limit)
        raster_count += 1
        cell_count += rows * columns
        ink_cells += int(numpy.count_nonzero(occupied))
        weak_cells += int(numpy.count_nonzero(weak))
        total_ink += float(numpy.sum(ink, dtype=numpy.float64))
        weak_ink += float(numpy.sum(ink[weak], dtype=numpy.float64))
    return internal_RescueCoverage(
        raster_count=raster_count,
        cell_count=cell_count,
        ink_cells=ink_cells,
        weak_cells=weak_cells,
        ink=total_ink,
        weak_ink=weak_ink,
    )


def internal_primary_text_is_sufficient(candidate: internal_Candidate) -> bool:
    """Return whether a sparse primary result is already large and trustworthy.

    Resolution escalation cannot add detail to text that is already comfortably
    sampled. Keep this decision shared by the adaptive rescue and subsequent
    full-page fallbacks so the latter cannot repeat work the former rejected.
    """
    metrics = candidate.metrics
    return (
        metrics.characters < 32
        and metrics.median_text_height >= OCR_RESCUE_LARGE_TEXT_HEIGHT
        and metrics.mean_confidence >= OCR_RESCUE_MIN_CONFIDENCE
    )


def internal_adaptive_rescue_decision(
    candidate: internal_Candidate,
    source_tasks: tuple[internal_OcrTask, ...],
    ocr_pass: OcrPass,
) -> tuple[bool, dict[str, object]]:
    """Decide whether another raster pass has enough unresolved visual evidence."""
    metrics = candidate.metrics
    coverage_pass = replace(
        ocr_pass,
        scope=OcrPassScope.WEAK_REGIONS,
        tiles=max(6, ocr_pass.tiles),
        region_columns=max(3, ocr_pass.region_columns),
        max_regions=max(8, ocr_pass.max_regions),
    )
    coverage = internal_adaptive_rescue_coverage(
        source_tasks,
        coverage_pass,
        candidate.observations,
    )
    reason = "unresolved-ink"
    run = True
    if internal_primary_text_is_sufficient(candidate):
        run = False
        reason = "primary-text-already-large"
    elif coverage.mean_ink >= OCR_RESCUE_SATURATED_MEAN_INK and (
        (metrics.characters >= 1_000 and metrics.mean_confidence >= OCR_RESCUE_MIN_CONFIDENCE)
        or (
            metrics.characters >= internal_OCR_RESCUE_DENSE_MIN_CHARACTERS
            and metrics.mean_confidence >= internal_OCR_RESCUE_DENSE_MIN_CONFIDENCE
        )
    ):
        # A nearly solid source gives the coarse ink grid no useful localization
        # signal.  Reprocessing arbitrary cells cannot target missing text.
        run = False
        reason = "ink-map-saturated"
    elif (
        metrics.characters >= 300
        and metrics.mean_confidence >= OCR_RESCUE_MIN_CONFIDENCE
        and coverage.raster_count
        and coverage.weak_ink_ratio < OCR_RESCUE_MIN_WEAK_INK_RATIO
        and (metrics.characters >= 600 or coverage.weak_ink_ratio == 0.0)
    ):
        run = False
        reason = "primary-covers-ink"
    return run, {
        "run": run,
        "reason": reason,
        "characters": metrics.characters,
        "mean_confidence": metrics.mean_confidence,
        "median_text_height": metrics.median_text_height,
        **coverage.as_record(),
    }


def internal_compact_ocr_image(image: RasterImage, *, grayscale: bool = False) -> RasterImage:
    """Drop redundant channels before sending a scan to Tesseract."""
    if image.channels == 3 and grayscale:
        if image.width * image.height < 5_000_000:
            return image
        samples = image.array()
        # Tesseract converts RGB to grayscale internally. Do it once here so its
        # segmentation pass processes one third as many bytes for scan-heavy pages.
        gray = (
            samples[:, :, 0].astype(numpy.uint16) * 77
            + samples[:, :, 1].astype(numpy.uint16) * 150
            + samples[:, :, 2].astype(numpy.uint16) * 29
            + 128
        ) >> 8
        gray = gray.astype(numpy.uint8)
        return RasterImage(contiguous_bytes(gray), image.width, image.height, 1)
    if image.channels == 4 and image.width * image.height >= 1_000_000:
        return image
    if image.channels not in {2, 4}:
        return image
    samples = image.array()
    alpha_index = image.channels - 1
    if not numpy.all(samples[:, :, alpha_index] == 255):
        if image.channels == 2:
            # Tesseract accepts gray, RGB, and RGBA byte layouts, but not the
            # gray-plus-alpha layout produced by PDF soft masks. Composite it
            # onto the same white background used by page rendering.
            distance_from_white = numpy.multiply(
                255 - samples[:, :, 0],
                samples[:, :, 1],
                dtype=numpy.uint16,
            )
            distance_from_white += 127
            distance_from_white //= 255
            gray_alpha = 255 - distance_from_white.astype(numpy.uint8)
            return RasterImage(contiguous_bytes(gray_alpha), image.width, image.height, 1)
        return image
    if image.channels == 2:
        return RasterImage(contiguous_bytes(samples[:, :, 0]), image.width, image.height, 1)
    if numpy.array_equal(samples[:, :, 0], samples[:, :, 1]) and numpy.array_equal(
        samples[:, :, 1], samples[:, :, 2]
    ):
        return RasterImage(contiguous_bytes(samples[:, :, 0]), image.width, image.height, 1)
    return RasterImage(contiguous_bytes(samples[:, :, :3]), image.width, image.height, 3)


OCR_IMAGE_TEXT_SAMPLE_PIXELS = 300_000
OCR_IMAGE_TEXT_EDGE_DELTA = 24
OCR_IMAGE_TEXT_MIN_HORIZONTAL_EDGES = 0.015
OCR_IMAGE_TEXT_PHOTO_MAX_WHITE = 0.20
OCR_IMAGE_TEXT_PHOTO_MIN_ENTROPY = 3.0
OCR_IMAGE_TEXT_STRONG_HORIZONTAL_EDGES = 0.09
OCR_IMAGE_TEXT_MIN_HORIZONTAL_EDGE_SHARE = 0.85


@dataclass(frozen=True, slots=True)
class internal_RasterTextSignal:
    likely_text: bool
    reason: str
    sampled_pixels: int
    white_ratio: float
    grayscale_entropy: float
    horizontal_edge_ratio: float
    vertical_edge_ratio: float

    def as_record(self) -> dict[str, bool | float | int | str]:
        return {
            "likely_text": self.likely_text,
            "reason": self.reason,
            "sampled_pixels": self.sampled_pixels,
            "white_ratio": self.white_ratio,
            "grayscale_entropy": self.grayscale_entropy,
            "horizontal_edge_ratio": self.horizontal_edge_ratio,
            "vertical_edge_ratio": self.vertical_edge_ratio,
        }


def internal_raster_text_signal(image: RasterImage) -> internal_RasterTextSignal:
    """Reject obvious non-text image supplements using a bounded pixel sample.

    This gate is intentionally limited to image supplements on pages that already
    have native text.  Full-page scan OCR and compositor fallbacks never use it.
    Text and line art have frequent horizontal intensity transitions; continuous-
    tone photographs may also have many edges, but those edges are less strongly
    horizontal and occur without a light document background.
    """
    pixels = image.array()
    sample_step = max(
        1,
        math.ceil(math.sqrt(image.width * image.height / OCR_IMAGE_TEXT_SAMPLE_PIXELS)),
    )
    sampled = pixels[::sample_step, ::sample_step]
    if image.channels == 1:
        gray = sampled[:, :, 0]
    elif image.channels == 2:
        source = sampled[:, :, 0].astype(numpy.uint16)
        alpha = sampled[:, :, 1].astype(numpy.uint16)
        gray = (255 - ((255 - source) * alpha + 127) // 255).astype(numpy.uint8)
    else:
        colour = sampled[:, :, :3]
        if image.channels == 4:
            alpha = sampled[:, :, 3:].astype(numpy.uint16)
            colour = (255 - ((255 - colour.astype(numpy.uint16)) * alpha + 127) // 255).astype(
                numpy.uint8
            )
        gray = numpy.min(colour, axis=2)

    gray_16 = gray.astype(numpy.int16)
    horizontal_edges = (
        float(numpy.mean(numpy.abs(numpy.diff(gray_16, axis=1)) >= OCR_IMAGE_TEXT_EDGE_DELTA))
        if gray.shape[1] > 1
        else 0.0
    )
    vertical_edges = (
        float(numpy.mean(numpy.abs(numpy.diff(gray_16, axis=0)) >= OCR_IMAGE_TEXT_EDGE_DELTA))
        if gray.shape[0] > 1
        else 0.0
    )
    white_ratio = float(numpy.mean(gray >= 245))
    histogram = numpy.bincount((gray // 8).reshape(-1), minlength=32).astype(numpy.float64)
    histogram /= max(1.0, float(numpy.sum(histogram)))
    occupied = histogram > 0.0
    entropy = float(-numpy.sum(histogram[occupied] * numpy.log2(histogram[occupied])))

    likely_text = True
    reason = "text-structure"
    if horizontal_edges < OCR_IMAGE_TEXT_MIN_HORIZONTAL_EDGES:
        likely_text = False
        reason = "low-edge-density"
    else:
        horizontal_edge_share = horizontal_edges / max(1e-9, vertical_edges)
        strongly_structured = bool(
            horizontal_edges >= OCR_IMAGE_TEXT_STRONG_HORIZONTAL_EDGES
            and horizontal_edge_share >= OCR_IMAGE_TEXT_MIN_HORIZONTAL_EDGE_SHARE
        )
        if (
            white_ratio < OCR_IMAGE_TEXT_PHOTO_MAX_WHITE
            and entropy >= OCR_IMAGE_TEXT_PHOTO_MIN_ENTROPY
            and not strongly_structured
        ):
            likely_text = False
            reason = "continuous-tone-image"
    return internal_RasterTextSignal(
        likely_text=likely_text,
        reason=reason,
        sampled_pixels=int(gray.size),
        white_ratio=white_ratio,
        grayscale_entropy=entropy,
        horizontal_edge_ratio=horizontal_edges,
        vertical_edge_ratio=vertical_edges,
    )


def internal_adaptive_ocr_raster(raster: internal_Raster) -> internal_Raster:
    """Binarize faded scans against their local background for a fallback pass."""
    pixels = raster.image.array()
    gray = (
        pixels[:, :, 0] if raster.image.channels == 1 else numpy.min(pixels[:, :, :3], axis=2)
    ).astype(numpy.float32)
    radius = max(8, min(24, min(raster.width, raster.height) // 80))
    integral = numpy.pad(gray, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    y = numpy.arange(raster.height)
    x = numpy.arange(raster.width)
    y0 = numpy.maximum(0, y - radius)
    y1 = numpy.minimum(raster.height, y + radius + 1)
    x0 = numpy.maximum(0, x - radius)
    x1 = numpy.minimum(raster.width, x + radius + 1)
    local_sum = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    local_area = ((y1 - y0)[:, None] * (x1 - x0)[None, :]).astype(numpy.float32)
    threshold = local_sum / local_area - 9.0
    binary = numpy.where(gray <= threshold, numpy.uint8(0), numpy.uint8(255))
    return internal_Raster(
        RasterImage(contiguous_bytes(binary), raster.width, raster.height, 1),
        raster.resolution,
        raster.render_report,
    )


def internal_candidate_text_containment(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> bool:
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if not shorter or sum(len(token) for token in shorter) < 4:
        return False
    width = len(shorter)
    return any(longer[index : index + width] == shorter for index in range(len(longer) - width + 1))


def internal_merge_candidate_batches(
    candidates: tuple[internal_Candidate, ...],
) -> internal_Candidate:
    if not candidates:
        return internal_candidate(-1, ObservationBatch.empty())
    if len(candidates) == 1:
        return candidates[0]
    modes = {candidate.mode for candidate in candidates}
    merged_by_mode: list[internal_Candidate] = []
    for mode in sorted(modes):
        mode_candidates = tuple(candidate for candidate in candidates if candidate.mode == mode)
        combined = ObservationBatch.concatenate(
            *(candidate.observations for candidate in mode_candidates)
        )
        combined_symbols = ObservationBatch.concatenate(
            *(candidate.symbols for candidate in mode_candidates)
        )
        fuzzy_tile_deduplication = len(mode_candidates) > 1
        order = numpy.lexsort((combined.bbox[:, 0], -combined.bbox[:, 1]))
        normalized_text = tuple(search_key(text) for text in combined.text)
        normalized_tokens = tuple(
            tuple(
                internal_normalized_ocr_token_key(match.group(0))
                for match in internal_OCR_TOKEN.finditer(text)
                if internal_normalized_ocr_token_key(match.group(0))
            )
            for text in combined.text
        )
        observation_utility = numpy.fromiter(
            (
                internal_observation_utility(text, float(confidence))
                for text, confidence in zip(
                    combined.text,
                    combined.confidence,
                    strict=True,
                )
            ),
            dtype=numpy.float32,
            count=len(combined),
        )
        deduplicated: list[int] = []
        for raw_index in order:
            index = int(raw_index)
            duplicate_index = next(
                (
                    accepted_position
                    for accepted_position in range(
                        max(0, len(deduplicated) - 24), len(deduplicated)
                    )
                    if (
                        overlap_ratio_min(
                            combined.bbox[index],
                            combined.bbox[deduplicated[accepted_position]],
                        )
                        >= (
                            0.35
                            if internal_candidate_text_containment(
                                normalized_tokens[deduplicated[accepted_position]],
                                normalized_tokens[index],
                            )
                            else (
                                0.45
                                if normalized_text[deduplicated[accepted_position]]
                                == normalized_text[index]
                                else (0.70 if fuzzy_tile_deduplication else math.inf)
                            )
                        )
                    )
                ),
                None,
            )
            if duplicate_index is None:
                deduplicated.append(index)
                continue
            accepted_index = deduplicated[duplicate_index]
            containment = internal_candidate_text_containment(
                normalized_tokens[accepted_index],
                normalized_tokens[index],
            )
            if containment and len(normalized_text[index]) != len(normalized_text[accepted_index]):
                if len(normalized_text[index]) > len(normalized_text[accepted_index]):
                    deduplicated[duplicate_index] = index
            elif observation_utility[index] > observation_utility[accepted_index]:
                deduplicated[duplicate_index] = index
        heights = tuple(
            candidate.metrics.median_text_height
            for candidate in mode_candidates
            if candidate.metrics.median_text_height > 0.0
        )
        merged_by_mode.append(
            internal_candidate(
                mode,
                combined.take(deduplicated),
                symbols=combined_symbols,
                median_text_height=float(numpy.median(heights)) if heights else 0.0,
            )
        )
    return max(merged_by_mode, key=lambda candidate: candidate.metrics.utility)


def internal_augment_candidate(
    primary: internal_Candidate,
    supplement: internal_Candidate,
    *,
    minimum_confidence: float,
) -> tuple[internal_Candidate, int]:
    """Add only high-quality supplement observations absent from the primary pass."""
    if not len(supplement.observations):
        return primary, 0
    observations = supplement.observations
    confidence = observations.confidence
    informative = numpy.fromiter(
        (sum(character.isalnum() for character in text) >= 1 for text in observations.text),
        dtype=numpy.bool_,
        count=len(observations),
    )
    useful = numpy.fromiter(
        (
            (
                internal_observation_utility(text, float(value)) >= 2.0
                or (len(text.strip()) == 1 and text.strip().isalnum() and float(value) >= 85.0)
            )
            for text, value in zip(observations.text, confidence, strict=True)
        ),
        dtype=numpy.bool_,
        count=len(observations),
    )
    coverage = maximum_candidate_coverage(
        observations.bbox,
        primary.observations.bbox,
    )
    additions = (
        (confidence >= max(70.0, minimum_confidence)) & informative & useful & (coverage < 0.30)
    )
    added = int(numpy.count_nonzero(additions))
    if not added:
        return primary, 0
    combined = ObservationBatch.concatenate_selected(
        primary.observations,
        observations,
        additions,
    )
    return internal_candidate(primary.mode, combined, symbols=primary.symbols), added


def internal_record_candidates(
    candidates: tuple[tuple[str, internal_Candidate], ...],
    selected_name: str,
    trace: internal_RecognitionTrace,
) -> None:
    trace.candidates = tuple(
        {
            "name": name,
            "mode": candidate.mode,
            "selected": name == selected_name,
            **candidate.metrics.as_record(),
        }
        for name, candidate in candidates
    )
    if os.environ.get("CORE_PDF_CANDIDATE_ANALYSIS", "").casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        trace.candidate_analysis = tuple(
            {
                "name": name,
                "mode": candidate.mode,
                "selected": name == selected_name,
                "text": "\n".join(candidate.observations.text),
                **candidate.metrics.as_record(),
            }
            for name, candidate in candidates
        )


def internal_decoded_image_raster(
    image: Any,
    display_area: float,
    *,
    cache: Any | None = None,
    image_cache: Any | None = None,
    max_pixels: int = MAX_OCR_PIXELS,
    upscale: bool = True,
) -> internal_Raster | None:
    source = getattr(image, "image_source", None)
    source_key = getattr(source, "cache_key", None)
    if not isinstance(source_key, tuple):
        source_key = ("image", id(image))
    shared_key = ImageCacheKey(
        "ocr-raster",
        tuple(source_key),
        (float(display_area), int(max_pixels), upscale),
    )
    page_cache_key = (
        "decoded_ocr_image_v4",
        *source_key,
        float(display_area),
        max_pixels,
        upscale,
    )
    if image_cache is not None:
        cached = image_cache.get(shared_key)
        if isinstance(cached, internal_Raster):
            return cached
    if cache is not None:
        cached = cache.get(page_cache_key)
        if isinstance(cached, internal_Raster):
            return cached
    shared = source.decode() if source is not None and hasattr(source, "decode") else None
    samples: numpy.ndarray[Any, Any] | None
    data: bytes | memoryview | None
    if shared is not None:
        samples = shared.array
        data = None
        decoded_width = shared.width
        decoded_height = shared.height
        decoded_channels = shared.channels
    else:
        raw = getattr(image, "raw_data", None)
        dictionary = getattr(image, "dictionary", None)
        if not isinstance(raw, (bytes, bytearray, memoryview)) or not isinstance(dictionary, dict):
            return None
        decoded = decode_pdf_image(raw, dictionary)
        if decoded is None:
            return None
        if isinstance(decoded.data, numpy.ndarray):
            array = cast(numpy.ndarray[Any, Any], decoded.data)
            samples = array.reshape((decoded.height, decoded.width, decoded.channels))
            data = None
        elif isinstance(decoded.data, (bytes, memoryview)):
            samples = None
            data = decoded.data
        else:
            samples = None
            data = memoryview(decoded.data).cast("B")
        decoded_width = decoded.width
        decoded_height = decoded.height
        decoded_channels = decoded.channels
    pixels_per_point = math.sqrt(decoded_width * decoded_height / max(1.0, display_area))
    resolution = max(70, min(600, int(round(72.0 * pixels_per_point))))
    width = decoded_width
    height = decoded_height
    channels = decoded_channels
    if width * height > max_pixels:
        reduction = math.sqrt(max_pixels / (width * height)) * 0.999
        target_width = max(1, int(width * reduction))
        target_height = max(1, int(height * reduction))
        if samples is None:
            assert data is not None
            samples = uint8_image_view(data, (height, width, channels))
        samples = resample_nearest(samples, target_height, target_width)
        data = None
        resolution = max(70, int(round(resolution * target_width / width)))
        width = target_width
        height = target_height
    headroom = math.sqrt(max_pixels / max(1, width * height))
    scale = min(DIRECT_OCR_TARGET_RESOLUTION / max(1, resolution), headroom)
    if upscale and scale > DIRECT_OCR_MIN_UPSCALE:
        if samples is None:
            assert data is not None
            samples = uint8_image_view(data, (height, width, channels))
        # Tesseract's line classifier wants roughly 300-400 DPI. How to get there
        # depends on the factor: a whole-number enlargement is exact pixel
        # replication and keeps stems crisp, while interpolating one only blurs
        # them. Fractional factors have no such option, and there replication
        # staircases the strokes badly enough to change which glyph is read.
        whole_factor = round(scale)
        if whole_factor >= 1 and abs(scale - whole_factor) <= DIRECT_OCR_WHOLE_SCALE_TOLERANCE:
            target_width = width * whole_factor
            target_height = height * whole_factor
            samples = resample_nearest(samples, target_height, target_width)
        else:
            target_width = max(1, int(width * scale))
            target_height = max(1, int(height * scale))
            samples = resample_bilinear(samples, target_height, target_width)
        data = None
        resolution = max(70, int(round(resolution * target_width / width)))
        width = target_width
        height = target_height
    if data is None:
        assert samples is not None
        data = contiguous_bytes(samples)
    raster = internal_Raster(RasterImage(data, width, height, channels), resolution)
    if image_cache is not None:
        image_cache.put(shared_key, raster)
    elif cache is not None:
        cache[page_cache_key] = raster
    return raster


class DirectImageOrientation(StrEnum):
    IDENTITY = "identity"
    FLIP_X = "flip-x"
    FLIP_Y = "flip-y"
    FLIP_XY = "flip-xy"
    TRANSPOSE = "transpose"
    TRANSPOSE_FLIP_X = "transpose-flip-x"
    TRANSPOSE_FLIP_Y = "transpose-flip-y"
    TRANSPOSE_FLIP_XY = "transpose-flip-xy"


internal_DIRECT_IMAGE_ORIENTATIONS: dict[DirectImageOrientation, tuple[int, int, int, int]] = {
    DirectImageOrientation.IDENTITY: (0, 1, 2, 3),
    DirectImageOrientation.FLIP_X: (1, 0, 3, 2),
    DirectImageOrientation.FLIP_Y: (2, 3, 0, 1),
    DirectImageOrientation.FLIP_XY: (3, 2, 1, 0),
    DirectImageOrientation.TRANSPOSE: (0, 2, 1, 3),
    DirectImageOrientation.TRANSPOSE_FLIP_X: (2, 0, 3, 1),
    DirectImageOrientation.TRANSPOSE_FLIP_Y: (1, 3, 0, 2),
    DirectImageOrientation.TRANSPOSE_FLIP_XY: (3, 1, 2, 0),
}


def internal_direct_image_orientation(
    image: Any,
    *,
    maximum_axis_deviation: float = 1e-5,
) -> DirectImageOrientation | None:
    items = getattr(image, "items", ())
    quad = next(
        (
            value
            for kind, value in items
            if kind == "quad" and isinstance(value, (list, tuple)) and len(value) == 4
        ),
        None,
    )
    if quad is None:
        return None
    try:
        points = tuple((float(point[0]), float(point[1])) for point in quad)
    except (IndexError, TypeError, ValueError):
        return None
    x0 = min(point[0] for point in points)
    y0 = min(point[1] for point in points)
    x1 = max(point[0] for point in points)
    y1 = max(point[1] for point in points)
    if x1 <= x0 or y1 <= y0:
        return None
    target_corners = ((x0, y0), (x1, y0), (x0, y1), (x1, y1))
    tolerance = max(0.01, max(x1 - x0, y1 - y0) * maximum_axis_deviation)
    target_to_raw = [-1, -1, -1, -1]
    for raw_index, point in enumerate(points):
        target_index = min(
            range(4),
            key=lambda index: (
                abs(point[0] - target_corners[index][0]) + abs(point[1] - target_corners[index][1])
            ),
        )
        target = target_corners[target_index]
        if max(abs(point[0] - target[0]), abs(point[1] - target[1])) > tolerance:
            return None
        if target_to_raw[target_index] != -1:
            return None
        target_to_raw[target_index] = raw_index
    orientation_corners = tuple(target_to_raw)
    return next(
        (
            orientation
            for orientation, corners in internal_DIRECT_IMAGE_ORIENTATIONS.items()
            if corners == orientation_corners
        ),
        None,
    )


def internal_orient_direct_image_raster(
    image: Any,
    raster: internal_Raster,
    *,
    orientation: DirectImageOrientation | None = None,
) -> internal_Raster:
    orientation = orientation or internal_direct_image_orientation(image)
    if orientation in {None, DirectImageOrientation.IDENTITY}:
        return raster
    samples = raster.image.array()
    match orientation:
        case DirectImageOrientation.FLIP_X:
            oriented = samples[:, ::-1]
        case DirectImageOrientation.FLIP_Y:
            oriented = samples[::-1]
        case DirectImageOrientation.FLIP_XY:
            oriented = samples[::-1, ::-1]
        case DirectImageOrientation.TRANSPOSE:
            oriented = samples.transpose(1, 0, 2)
        case DirectImageOrientation.TRANSPOSE_FLIP_X:
            oriented = samples.transpose(1, 0, 2)[::-1]
        case DirectImageOrientation.TRANSPOSE_FLIP_Y:
            oriented = samples.transpose(1, 0, 2)[:, ::-1]
        case DirectImageOrientation.TRANSPOSE_FLIP_XY:
            oriented = samples.transpose(1, 0, 2)[::-1, ::-1]
        case _:
            return raster
    height, width, channels = oriented.shape
    return internal_Raster(
        RasterImage(contiguous_bytes(oriented), int(width), int(height), int(channels)),
        raster.resolution,
    )


def internal_page_image_regions(
    capture: CapturedPage,
    *,
    minimum_area_ratio: float,
    max_pixels: int = MAX_OCR_PIXELS,
    maximum_axis_deviation: float = 1e-5,
    upscale: bool = True,
) -> tuple[internal_RasterRegion, ...]:
    page_width = float(capture.page.width)
    page_height = float(capture.page.height)
    page_area = max(1.0, page_width * page_height)
    regions: list[internal_RasterRegion] = []
    for image in getattr(capture, "drawings", ()):
        if getattr(image, "kind", None) != "image":
            continue
        orientation = internal_direct_image_orientation(
            image,
            maximum_axis_deviation=maximum_axis_deviation,
        )
        if orientation is None:
            continue
        box = rect_tuple(getattr(image, "rect", None))
        if box is None:
            continue
        clipped = (
            max(0.0, box[0]),
            max(0.0, box[1]),
            min(page_width, box[2]),
            min(page_height, box[3]),
        )
        # A decoded source raster represents the full image. If the image is clipped by
        # the page, mapping that full raster onto the clipped rectangle would compress
        # its OCR coordinates. Let the page compositor produce the correct crop instead.
        clip_tolerance = max(2.0, max(page_width, page_height) * 0.005)
        if any(
            abs(float(original) - clipped_value) > clip_tolerance
            for original, clipped_value in zip(box, clipped, strict=True)
        ):
            continue
        display_area = max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1])
        if display_area / page_area < minimum_area_ratio:
            continue
        raster = internal_decoded_image_raster(
            image,
            display_area,
            cache=getattr(capture.page, "extraction_cache", None),
            image_cache=getattr(getattr(capture.page, "document", None), "image_cache", None),
            max_pixels=max_pixels,
            upscale=upscale,
        )
        if raster is not None:
            oriented = raster
            if orientation is not DirectImageOrientation.IDENTITY:
                source = getattr(image, "image_source", None)
                source_key = getattr(source, "cache_key", None)
                if not isinstance(source_key, tuple):
                    source_key = ("image", id(image))
                oriented_key = ImageCacheKey(
                    "ocr-oriented-raster",
                    tuple(source_key),
                    (orientation.value, float(display_area), int(max_pixels), upscale),
                )
                cache = getattr(getattr(capture.page, "document", None), "image_cache", None)
                cached_oriented = cache.get(oriented_key) if cache is not None else None
                if isinstance(cached_oriented, internal_Raster):
                    oriented = cached_oriented
                else:
                    oriented = internal_orient_direct_image_raster(
                        image,
                        raster,
                        orientation=orientation,
                    )
                    if cache is not None:
                        cache.put(oriented_key, oriented)
            regions.append(
                internal_RasterRegion(
                    oriented,
                    clipped,
                )
            )
    return tuple(regions)


def internal_dominant_image_region(
    capture: CapturedPage,
    *,
    max_pixels: int = MAX_OCR_PIXELS,
    upscale: bool = True,
) -> internal_RasterRegion | None:
    def box_area(box: tuple[float, float, float, float]) -> float:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    regions = internal_page_image_regions(
        capture,
        minimum_area_ratio=0.65,
        max_pixels=max_pixels,
        upscale=upscale,
    )
    substantial = tuple(
        region for region in regions if region.raster.width * region.raster.height >= 4_096
    )
    if not substantial:
        return None
    if len(substantial) > 1:
        largest = max(substantial, key=lambda region: box_area(region.page_box))
        largest_area = max(1.0, box_area(largest.page_box))
        overlapping = sum(
            max(
                0.0,
                min(region.page_box[2], largest.page_box[2])
                - max(region.page_box[0], largest.page_box[0]),
            )
            * max(
                0.0,
                min(region.page_box[3], largest.page_box[3])
                - max(region.page_box[1], largest.page_box[1]),
            )
            / largest_area
            >= 0.90
            for region in substantial
            if region is not largest
        )
        if overlapping:
            return None
    return max(substantial, key=lambda region: region.raster.width * region.raster.height)


OCR_REGION_INITIAL_COUNT = 8
OCR_REGION_MAX_COUNT = 16
OCR_REGION_INITIAL_AREA_RATIO = 0.25
OCR_REGION_MAX_AREA_RATIO = 0.60
OCR_DIRECT_REGION_MIN_COVERAGE = 0.65
# Small affine placement noise is cheaper to absorb in OCR coordinates than to
# recompose and rasterize the entire page around an otherwise usable source image.
OCR_IMAGE_REGIONS_MAX_AXIS_DEVIATION = 0.01


def internal_ocr_region_box(
    box: tuple[float, float, float, float],
    *,
    page_width: float,
    page_height: float,
    padding: float,
) -> tuple[float, float, float, float] | None:
    x0, y0, x1, y1 = box
    clipped = (
        max(0.0, x0 - padding),
        max(0.0, y0 - padding),
        min(page_width, x1 + padding),
        min(page_height, y1 + padding),
    )
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None


def internal_ocr_region_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    return overlap_ratio_min_exact(left, right)


def internal_ocr_region_coverage(
    target: tuple[float, float, float, float],
    candidate: tuple[float, float, float, float],
) -> float:
    """Return how much of a requested OCR target is covered by a candidate raster."""
    return overlap_ratio_of(target, candidate)


def internal_merge_ocr_regions(regions: list[internal_OcrRegion]) -> tuple[internal_OcrRegion, ...]:
    merged: list[internal_OcrRegion] = []
    merged_areas: list[float] = []
    for region in sorted(regions, key=lambda item: (-item.score, item.page_box)):
        region_box = region.page_box
        region_area = max(0.0, region_box[2] - region_box[0]) * max(
            0.0, region_box[3] - region_box[1]
        )
        match = None
        for index, existing in enumerate(merged):
            existing_box = existing.page_box
            smaller = min(merged_areas[index], region_area)
            if not smaller:
                continue
            intersection_width = max(
                0.0, min(existing_box[2], region_box[2]) - max(existing_box[0], region_box[0])
            )
            intersection_height = max(
                0.0, min(existing_box[3], region_box[3]) - max(existing_box[1], region_box[1])
            )
            if intersection_width * intersection_height >= smaller * 0.35:
                match = index
                break
        if match is None:
            merged.append(region)
            merged_areas.append(region_area)
            continue
        existing = merged[match]
        existing_box = existing.page_box
        merged_box = (
            min(existing_box[0], region_box[0]),
            min(existing_box[1], region_box[1]),
            max(existing_box[2], region_box[2]),
            max(existing_box[3], region_box[3]),
        )
        merged[match] = internal_OcrRegion(
            merged_box,
            max(existing.score, region.score) + min(existing.score, region.score) * 0.15,
            tuple(dict.fromkeys((*existing.reasons, *region.reasons))),
        )
        merged_areas[match] = (merged_box[2] - merged_box[0]) * (merged_box[3] - merged_box[1])
    return tuple(sorted(merged, key=lambda item: (-item.score, item.page_box)))


def internal_candidate_ocr_regions(capture: CapturedPage) -> tuple[internal_OcrRegion, ...]:
    """Select likely OCR areas using capture-time geometry only.

    This deliberately does not render a preview image.  Native text, image bounds,
    captured paths, and grid lines are already available from the canonical page IR.
    """
    cache = getattr(capture.page, "extraction_cache", None)
    cache_key = "ocr_candidate_regions_v1"
    if cache is not None:
        cached = cache.get(cache_key)
        if isinstance(cached, tuple) and all(
            isinstance(item, internal_OcrRegion) for item in cached
        ):
            return cached

    page_width = float(capture.page.width)
    page_height = float(capture.page.height)
    page_area = max(1.0, page_width * page_height)
    padding = max(6.0, min(36.0, min(page_width, page_height) * 0.01))
    candidates: list[internal_OcrRegion] = []

    for box in getattr(capture.evidence, "image_boxes", ()):
        image_box = cast(
            tuple[float, float, float, float],
            tuple(float(value) for value in box),
        )
        padded = internal_ocr_region_box(
            image_box,
            page_width=page_width,
            page_height=page_height,
            padding=padding,
        )
        if padded is not None:
            candidates.append(internal_OcrRegion(padded, 5.0, ("image",)))

    native = getattr(capture, "observations", ObservationBatch.empty())
    native_boxes = tuple(tuple(float(value) for value in box) for box in native.bbox)
    native_index = SpatialIndex.from_boxes(native_boxes) if native_boxes else None

    def native_overlap(box: tuple[float, float, float, float]) -> float:
        area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        if native_index is not None:
            return min(
                1.0,
                sum(
                    bbox_intersection_area(box, hit.bbox)
                    for hit in native_index.intersecting_hits(box)
                )
                / area,
            )
        return min(
            1.0,
            sum(
                max(0.0, min(box[2], other[2]) - max(box[0], other[0]))
                * max(0.0, min(box[3], other[3]) - max(box[1], other[1]))
                for other in native_boxes
            )
            / area,
        )

    for drawing in getattr(capture, "drawings", ()):
        if getattr(drawing, "kind", None) not in {"fill", "fillstroke", "stroke"}:
            continue
        box = rect_tuple(getattr(drawing, "rect", None))
        if box is None:
            continue
        drawing_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        if drawing_area <= 0.0 or drawing_area >= page_area * 0.80:
            continue
        uncovered = native_overlap(box) < 0.25
        if uncovered and getattr(drawing, "kind", None) in {"fill", "fillstroke"}:
            padded = internal_ocr_region_box(
                box,
                page_width=page_width,
                page_height=page_height,
                padding=padding,
            )
            if padded is not None:
                candidates.append(internal_OcrRegion(padded, 3.5, ("uncovered-vector",)))

    if hasattr(capture, "grid_lines"):
        horizontal, vertical = internal_axis_segments(capture)
    else:
        horizontal = numpy.empty((0, 3), dtype=numpy.float32)
        vertical = numpy.empty((0, 3), dtype=numpy.float32)
    for component_horizontal, component_vertical in internal_grid_components(horizontal, vertical):
        x0 = min(float(component_horizontal[:, 0].min()), float(component_vertical[:, 0].min()))
        y0 = min(float(component_horizontal[:, 2].min()), float(component_vertical[:, 1].min()))
        x1 = max(float(component_horizontal[:, 1].max()), float(component_vertical[:, 0].max()))
        y1 = max(float(component_horizontal[:, 2].max()), float(component_vertical[:, 2].max()))
        for split_horizontal, split_vertical in internal_split_grid_component(
            component_horizontal,
            component_vertical,
        ):
            split_box = (
                min(float(split_horizontal[:, 0].min()), float(split_vertical[:, 0].min())),
                min(float(split_horizontal[:, 2].min()), float(split_vertical[:, 1].min())),
                max(float(split_horizontal[:, 1].max()), float(split_vertical[:, 0].max())),
                max(float(split_horizontal[:, 2].max()), float(split_vertical[:, 2].max())),
            )
            padded = internal_ocr_region_box(
                split_box,
                page_width=page_width,
                page_height=page_height,
                padding=padding,
            )
            if padded is not None and (
                (padded[2] - padded[0]) * (padded[3] - padded[1]) < page_area * 0.45
            ):
                candidates.append(internal_OcrRegion(padded, 4.0, ("grid",)))
        if not component_horizontal.size or not component_vertical.size:
            continue
        component_box = (x0, y0, x1, y1)
        component_area = (x1 - x0) * (y1 - y0)
        if component_area < page_area * 0.45 and native_overlap(component_box) < 0.25:
            padded = internal_ocr_region_box(
                component_box,
                page_width=page_width,
                page_height=page_height,
                padding=padding,
            )
            if padded is not None:
                candidates.append(internal_OcrRegion(padded, 3.0, ("grid-labels",)))

    columns = 6
    rows = max(2, min(8, int(round(columns * page_height / max(1.0, page_width)))))
    vector_density = numpy.zeros(rows * columns, dtype=numpy.float32)
    for drawing in getattr(capture, "drawings", ()):
        if getattr(drawing, "kind", None) not in VECTOR_PAINT_KINDS:
            continue
        box = rect_tuple(getattr(drawing, "rect", None))
        if box is None:
            continue
        center_x = (box[0] + box[2]) * 0.5
        center_y = (box[1] + box[3]) * 0.5
        column = min(columns - 1, max(0, int(center_x * columns / max(1.0, page_width))))
        row = min(rows - 1, max(0, int(center_y * rows / max(1.0, page_height))))
        vector_density[row * columns + column] += 1.0
    grid_lines = getattr(capture, "grid_lines", ())
    if len(grid_lines):
        # Bin every grid line at once.  Iterating the capture would rebuild one
        # Python object per line just to read its four coordinates back out.
        line_x0, line_y0, line_x1, line_y1 = line_coordinate_columns(grid_lines)
        line_columns = numpy.clip(
            ((line_x0 + line_x1) * 0.5 * columns / max(1.0, page_width)).astype(numpy.int64),
            0,
            columns - 1,
        )
        line_rows = numpy.clip(
            ((line_y0 + line_y1) * 0.5 * rows / max(1.0, page_height)).astype(numpy.int64),
            0,
            rows - 1,
        )
        vector_density += (
            numpy.bincount(
                line_rows * columns + line_columns,
                minlength=rows * columns,
            ).astype(numpy.float32)
            * 0.5
        )

    native_counts = numpy.zeros(rows * columns, dtype=numpy.float32)
    for text, raw_box in zip(native.text, native.bbox, strict=True):
        box = internal_bbox_tuple(raw_box)
        center_x = (box[0] + box[2]) * 0.5
        center_y = (box[1] + box[3]) * 0.5
        column = min(columns - 1, max(0, int(center_x * columns / max(1.0, page_width))))
        row = min(rows - 1, max(0, int(center_y * rows / max(1.0, page_height))))
        native_counts[row * columns + column] += sum(not char.isspace() for char in text)

    for cell, density in enumerate(vector_density):
        if density <= 0.0:
            continue
        row, column = divmod(cell, columns)
        cell_box = (
            column * page_width / columns,
            row * page_height / rows,
            (column + 1) * page_width / columns,
            (row + 1) * page_height / rows,
        )
        sparse = native_counts[cell] < 8.0
        header_band = row in {0, rows - 1} and native_counts[cell] < 24.0
        if not sparse and not header_band:
            continue
        padded = internal_ocr_region_box(
            cell_box,
            page_width=page_width,
            page_height=page_height,
            padding=padding,
        )
        if padded is not None:
            reasons = ["vector-density"]
            if sparse:
                reasons.append("sparse-label")
            if header_band:
                reasons.append("header-band")
            candidates.append(
                internal_OcrRegion(
                    padded,
                    1.5 + min(2.0, float(density) / 8.0),
                    tuple(reasons),
                )
            )

    if (
        capture.evidence.vector_complexity >= 180
        and capture.evidence.text_coverage < 0.05
        and (not native_boxes or len(native_boxes) >= 8)
    ):
        # Component labels are often isolated from the larger paths they
        # annotate. Use finer cells for these vector-only pages so the region
        # budget can select several label clusters instead of one broad artwork
        # box. The existing coarse density pass remains responsible for larger
        # diagram areas.
        label_columns = 12
        label_rows = max(
            4,
            min(12, int(round(label_columns * page_height / max(1.0, page_width)))),
        )
        label_density = numpy.zeros(label_rows * label_columns, dtype=numpy.float32)
        label_boxes: list[list[tuple[float, float, float, float]]] = [
            [] for _ in range(label_rows * label_columns)
        ]
        for drawing in getattr(capture, "drawings", ()):
            if getattr(drawing, "kind", None) not in VECTOR_PAINT_KINDS:
                continue
            box = rect_tuple(getattr(drawing, "rect", None))
            if box is None:
                continue
            center_x = (box[0] + box[2]) * 0.5
            center_y = (box[1] + box[3]) * 0.5
            column = min(
                label_columns - 1,
                max(0, int(center_x * label_columns / max(1.0, page_width))),
            )
            row = min(
                label_rows - 1,
                max(0, int(center_y * label_rows / max(1.0, page_height))),
            )
            label_density[row * label_columns + column] += 1.0
            label_boxes[row * label_columns + column].append(box)

        for cell, density in enumerate(label_density):
            if density <= 0.0:
                continue
            row, column = divmod(cell, label_columns)
            cell_box = (
                column * page_width / label_columns,
                row * page_height / label_rows,
                (column + 1) * page_width / label_columns,
                (row + 1) * page_height / label_rows,
            )
            component_boxes = label_boxes[cell]
            optional_component_box = bbox_union(component_boxes)
            assert optional_component_box is not None
            component_box = optional_component_box
            component_area = max(0.0, component_box[2] - component_box[0]) * max(
                0.0, component_box[3] - component_box[1]
            )
            label_padding = max(
                padding,
                min(72.0, min(page_width, page_height) * 0.03),
            )
            candidate_box = component_box if component_area <= page_area * 0.08 else cell_box
            padded = internal_ocr_region_box(
                candidate_box,
                page_width=page_width,
                page_height=page_height,
                padding=label_padding if candidate_box == component_box else padding,
            )
            if padded is not None:
                candidates.append(
                    internal_OcrRegion(
                        padded,
                        1.0 + min(3.0, float(density) / 8.0),
                        ("vector-label-density", "vector-label-neighborhood")
                        if candidate_box == component_box
                        else ("vector-label-density",),
                    )
                )

    regions = internal_merge_ocr_regions(candidates)
    if not regions:
        regions = (
            internal_OcrRegion(
                (0.0, 0.0, page_width, page_height),
                0.0,
                ("page-fallback",),
            ),
        )
    if cache is not None:
        cache[cache_key] = regions
    return regions


def internal_has_distributed_outline_text(capture: CapturedPage) -> bool:
    """Detect pages whose text was converted into many small filled vector paths."""
    page_width = float(capture.page.width)
    page_height = float(capture.page.height)
    max_width = max(24.0, page_width * 0.04)
    max_height = max(24.0, page_height * 0.04)
    boxes = tuple(
        box
        for drawing in getattr(capture, "drawings", ())
        if getattr(drawing, "kind", None) in {"fill", "fillstroke"}
        and (box := rect_tuple(getattr(drawing, "rect", None))) is not None
        and 0.0 < box[2] - box[0] <= max_width
        and 0.0 < box[3] - box[1] <= max_height
    )
    if len(boxes) < 200:
        return False
    bounds = bbox_union(boxes)
    assert bounds is not None
    width_ratio = (bounds[2] - bounds[0]) / max(1.0, page_width)
    height_ratio = (bounds[3] - bounds[1]) / max(1.0, page_height)
    return width_ratio >= 0.60 and height_ratio >= 0.60


def internal_direct_scan_allowed(capture: CapturedPage, plan: WorkPlan) -> bool:
    """Decide whether a page-scope pass may OCR the decoded scan itself.

    Rendering a scanned page through the compositor resamples the scan a second
    time at whatever scale the pass chose, which is strictly worse than reading
    its own pixels.  The rendered page is still required whenever the page holds
    content the dominant image does not cover.
    """
    evidence = capture.evidence
    if not plan.allow_direct_image_ocr:
        return False
    if evidence.visible_native_characters >= 10 or not evidence.image_count:
        return True
    # No native text and one image covering the page: the image *is* the page, so
    # nothing is lost by reading it directly.  Any weaker signal keeps the render.
    return bool(evidence.full_page_image) and not evidence.visible_native_characters


def internal_rendered_page_raster(
    capture: CapturedPage,
    requested_scale: float,
    *,
    crop: tuple[float, float, float, float] | None = None,
    rendered: Any | None = None,
    cache: bool = True,
    max_pixels: int = MAX_OCR_PIXELS,
    include_native_text: bool = False,
    trace: internal_RecognitionTrace | None = None,
) -> internal_Raster | None:
    page = capture.page
    compose_started = time.perf_counter()
    if rendered is None:
        rendered = compose_page(
            page,
            RenderOptions(include_text=include_native_text),
            page_program=capture.program,
        )
    compose_seconds = time.perf_counter() - compose_started
    if crop is None:
        raster_area = max(1.0, float(page.width) * float(page.height))
    else:
        raster_area = max(1.0, (crop[2] - crop[0]) * (crop[3] - crop[1]))
    safe_scale = math.sqrt(max_pixels / raster_area) * 0.999
    scale = min(requested_scale, safe_scale)
    raster_started = time.perf_counter()
    width, height = rendered.raster_size(scale)
    try:
        data = rendered.rasterize(
            background=(255, 255, 255, 255),
            scale=scale,
            max_pixels=max_pixels,
            crop=crop,
            cache=cache,
        )
    except IndexError as error:
        # A malformed embedded image can produce a source sample outside its
        # decoded raster during compositing.  Keep native extraction usable and
        # let OCR continue without the rendered-page fallback.
        if trace is not None:
            trace.render_error = str(error)
        return None
    render_report: dict[str, object] = {
        "compose_seconds": compose_seconds,
        "rasterize_seconds": time.perf_counter() - raster_started,
        "raster_mode": "region" if crop is not None else "page",
        "crop": crop,
        "raster_pixels": width * height,
        "pixel_budget": max_pixels,
        "include_native_text": include_native_text,
        "image_timings": rendered.metadata.get("__core_pdf_raster_image_timings__", {}),
        "display_items": len(rendered.display_list.items),
        "display_item_kinds": dict(
            Counter(
                str(getattr(item, "kind", type(item).__name__))
                for item in rendered.display_list.items
            )
        ),
        "image_filters": tuple(
            str(((getattr(item, "data", None) or {}).get("dictionary") or {}).get("Filter"))
            for item in rendered.display_list.items
            if getattr(item, "kind", None) in {"image", "inline-image"}
        ),
    }
    if trace is not None:
        trace.render_timings = render_report
    return internal_Raster(
        data,
        max(70, int(round(72.0 * scale))),
        render_report,
    )


STROKED_VECTOR_PACK_WIDTH = 240.0
STROKED_VECTOR_PACK_HORIZONTAL_PADDING = 4.0
STROKED_VECTOR_PACK_VERTICAL_PADDING = 4.0
STROKED_VECTOR_PACK_DENSE_VERTICAL_PADDING = 2.0
STROKED_VECTOR_PACK_DENSE_MIN_CELLS = 96
STROKED_VECTOR_PACK_REMAP_TOLERANCE = 4.0
STROKED_VECTOR_PACK_MIN_ALIGNED_SEEDS = 12
STROKED_VECTOR_PACK_MIN_LEARNED_SIGNATURES = 16
STROKED_VECTOR_PACK_MIN_DECODED_RUNS = 16


@dataclass(frozen=True, slots=True)
class internal_CachedStrokedTextProfile:
    drawings: tuple[Any, ...]
    drawing_indexes: tuple[int, ...]
    profile: StrokedTextProfile


def internal_stroked_text_profile(capture: CapturedPage) -> StrokedTextProfile:
    """Return the single structural glyph profile shared by OCR and document reuse."""
    evidence = capture.evidence.stroked_vector_text
    cache = capture.page.extraction_cache
    cached = cache.get("_stroked_text_profile")
    if (
        isinstance(cached, internal_CachedStrokedTextProfile)
        and cached.drawings is capture.drawings
        and cached.drawing_indexes == evidence.drawing_indexes
    ):
        return cached.profile
    profile = profile_stroked_text(capture.drawings, evidence.drawing_indexes)
    cache["_stroked_text_profile"] = internal_CachedStrokedTextProfile(
        capture.drawings,
        evidence.drawing_indexes,
        profile,
    )
    return profile


def internal_pack_stroked_text_runs(
    runs: tuple[StrokedTextRun, ...],
    *,
    width: float = STROKED_VECTOR_PACK_WIDTH,
    horizontal_padding: float = STROKED_VECTOR_PACK_HORIZONTAL_PADDING,
    vertical_padding: float = STROKED_VECTOR_PACK_VERTICAL_PADDING,
) -> tuple[tuple[internal_StrokedTextCell, ...], float]:
    """Shelf-pack vector words without scaling their glyph geometry."""
    if not runs:
        return (), 0.0
    ordered = sorted(
        runs,
        key=lambda run: (
            -(run.bbox[3] - run.bbox[1]),
            -(run.bbox[2] - run.bbox[0]),
            run.drawing_indexes[0],
        ),
    )
    x = horizontal_padding
    y = vertical_padding
    row_height = 0.0
    cells: list[internal_StrokedTextCell] = []
    for run in ordered:
        source = run.bbox
        run_width = source[2] - source[0]
        run_height = source[3] - source[1]
        cell_width = run_width + horizontal_padding * 2.0
        cell_height = run_height + vertical_padding * 2.0
        if x > horizontal_padding and x + cell_width > width:
            y += row_height
            x = horizontal_padding
            row_height = 0.0
        packed = (
            x + horizontal_padding,
            y + vertical_padding,
            x + horizontal_padding + run_width,
            y + vertical_padding + run_height,
        )
        cells.append(
            internal_StrokedTextCell(
                source_box=source,
                packed_box=packed,
                drawing_indexes=run.drawing_indexes,
            )
        )
        x += cell_width
        row_height = max(row_height, cell_height)
    return tuple(cells), y + row_height + vertical_padding


def internal_stroked_vector_text_raster(
    capture: CapturedPage,
    requested_scale: float,
    *,
    max_pixels: int = MAX_OCR_PIXELS,
    variant: str = "seed",
    trace: internal_RecognitionTrace | None = None,
) -> internal_PackedStrokedTextRaster | None:
    """Pack vector words into a compact seed raster with piecewise page mapping."""
    evidence = capture.evidence.stroked_vector_text
    if not evidence.trusted or not evidence.drawing_indexes:
        return None
    profile = internal_stroked_text_profile(capture)
    runs = stroked_text_isolated_runs(profile) if variant == "isolated" else profile.seed_runs
    if variant == "isolated":
        # Glyph-sized cells sit closer together than the remap tolerance, so
        # observation centres would match several cells and be dropped. Keep
        # the cell separation comfortably above the tolerance.
        horizontal_padding = STROKED_VECTOR_PACK_REMAP_TOLERANCE * 1.5
        vertical_padding = STROKED_VECTOR_PACK_REMAP_TOLERANCE * 1.5
    else:
        horizontal_padding = STROKED_VECTOR_PACK_HORIZONTAL_PADDING
        vertical_padding = (
            STROKED_VECTOR_PACK_DENSE_VERTICAL_PADDING
            if len(runs) >= STROKED_VECTOR_PACK_DENSE_MIN_CELLS
            else STROKED_VECTOR_PACK_VERTICAL_PADDING
        )
    cells, packed_height = internal_pack_stroked_text_runs(
        runs,
        horizontal_padding=horizontal_padding,
        vertical_padding=vertical_padding,
    )
    if not cells or packed_height <= 0.0:
        return None
    packed_width = STROKED_VECTOR_PACK_WIDTH
    area = max(1.0, packed_width * packed_height)
    safe_scale = math.sqrt(max_pixels / area) * 0.999
    scale = min(requested_scale, safe_scale)
    if variant == "isolated":
        # Isolated glyphs are the smallest text on the sheet (pin numbers run
        # 1-2pt tall); at the seed scale they raster below OCR's working size.
        # The montage area is tiny, so trade unused pixel budget for scale
        # until the median glyph is comfortably readable.
        median_height = float(numpy.median([run.bbox[3] - run.bbox[1] for run in runs]))
        target_scale = 24.0 / max(0.5, median_height)
        scale = min(max(requested_scale, target_scale), safe_scale, 48.0)
    page = capture.page
    cache = getattr(page, "extraction_cache", None)
    cache_key = ("packed_stroked_vector_text_raster_v5", scale, max_pixels, variant)
    if cache is not None:
        cached = cache.get(cache_key)
        if isinstance(cached, internal_PackedStrokedTextRaster):
            if trace is not None and variant == "seed" and cached.raster.render_report is not None:
                trace.render_timings = cached.raster.render_report
            return cached

    compose_started = time.perf_counter()
    display_list = DisplayList(width=packed_width, height=packed_height)
    for cell in cells:
        tx = cell.packed_box[0] - cell.source_box[0]
        ty = cell.packed_box[1] - cell.source_box[1]
        # Hairline strokes raster as one-pixel skeletons, which OCR reads
        # confidently inside a word but misclassifies on a lone character.
        # Give isolated glyphs a stroke proportional to their own size so
        # they render solid at the boosted montage scale, and stand rotated
        # glyphs upright: pin numbers beside vertical wires are drawn
        # sideways, and OCR cannot classify a lone rotated character.
        rotation_matrix = None
        line_width_floor = 0.0
        if variant == "isolated":
            cell_width = cell.source_box[2] - cell.source_box[0]
            cell_height = cell.source_box[3] - cell.source_box[1]
            line_width_floor = min(cell_width, cell_height) * 0.14
            if cell_width > cell_height * 1.2:
                center_x = (cell.source_box[0] + cell.source_box[2]) * 0.5
                center_y = (cell.source_box[1] + cell.source_box[3]) * 0.5
                rotation_matrix = (
                    0.0,
                    -1.0,
                    1.0,
                    0.0,
                    center_x - center_y,
                    center_y + center_x,
                )
        for index in cell.drawing_indexes:
            drawing = capture.drawings[index]
            drawing_box = rect_tuple(getattr(drawing, "rect", None))
            path = getattr(drawing, "path", None)
            if drawing_box is None or path is None:
                continue
            if rotation_matrix is not None:
                path = path.transformed(rotation_matrix)
            # Packed cells preserve capture order and paint style. Reuse the
            # display-list stroke coalescer so thousands of tiny glyph paths do
            # not become thousands of renderer dispatches.
            display_list.append_captured_drawing(
                drawing.replace(
                    bbox=None,
                    path=path.translated(tx, ty),
                    fill_pattern=None,
                    stroke_pattern=None,
                    fill=(0.0, 0.0, 0.0),
                    stroke_color=(0.0, 0.0, 0.0),
                    line_width=max(float(drawing.line_width or 0.0), line_width_floor),
                )
            )
    rendered = RenderedPage(
        page_number=int(getattr(page, "page_number", 0)),
        width=packed_width,
        height=packed_height,
        rotate=0,
        display_list=display_list,
    )
    compose_seconds = time.perf_counter() - compose_started
    raster_started = time.perf_counter()
    # The Wu kernel draws one-pixel skeletons regardless of stroke width, so
    # the isolated variant must use the general renderer to honour the widened
    # strokes it just requested.
    fast_path = (
        variant != "isolated"
        and bool(display_list.items)
        and all(
            type(item) is PathPaintItem
            and item.paint_kind is PathPaintKind.STROKE
            and not (item.dash_pattern and item.dash_pattern[0])
            and item.blend_mode is None
            and item.soft_mask_alpha is None
            and (item.stroke_opacity is None or float(item.stroke_opacity) >= 0.999)
            and int(item.line_cap or 0) == 1
            and int(item.line_join or 0) == 1
            for item in display_list.items
        )
    )
    if fast_path:
        data = rasterize_packed_stroked_paths(
            tuple(display_list.items),
            packed_width,
            packed_height,
            scale,
        )
    else:
        data = rendered.rasterize(
            background=(255, 255, 255, 255),
            scale=scale,
            max_pixels=max_pixels,
            cache=False,
        )
    raster_seconds = time.perf_counter() - raster_started
    render_report: dict[str, object] = {
        "compose_seconds": compose_seconds,
        "rasterize_seconds": raster_seconds,
        "raster_mode": "packed-stroked-vector-text",
        "raster_kernel": "wu" if fast_path else "general",
        "crop": (0.0, 0.0, packed_width, packed_height),
        "raster_pixels": data.width * data.height,
        "pixel_budget": max_pixels,
        "include_native_text": False,
        "image_timings": {},
        "display_items": len(display_list.items),
        "display_item_kinds": {"compact-stroke": len(display_list.items)},
        "image_filters": (),
        "packed_cells": len(cells),
        "horizontal_padding": horizontal_padding,
        "vertical_padding": vertical_padding,
        "source_bbox": evidence.bbox,
    }
    raster = internal_Raster(
        data,
        max(70, int(round(72.0 * scale))),
        render_report,
    )
    packed = internal_PackedStrokedTextRaster(
        raster=raster,
        packed_box=(0.0, 0.0, packed_width, packed_height),
        cells=cells,
    )
    if cache is not None:
        cache[cache_key] = packed
    # The isolated-glyph supplement uses the general renderer by design. Keep
    # the pass-level timing tied to the seed atlas, matching the actual primary
    # recognition path and preserving the Wu-kernel performance invariant.
    if trace is not None and variant == "seed":
        trace.render_timings = render_report
    return packed


def internal_full_stroked_vector_text_raster(
    capture: CapturedPage,
    requested_scale: float,
    *,
    max_pixels: int = MAX_OCR_PIXELS,
    trace: internal_RecognitionTrace | None = None,
) -> internal_RasterRegion | None:
    """Render the full compact-stroke layer when packed seed OCR is insufficient."""
    evidence = capture.evidence.stroked_vector_text
    if not evidence.trusted or evidence.bbox is None or not evidence.drawing_indexes:
        return None
    page = capture.page
    page_width = float(page.width)
    page_height = float(page.height)
    padding = 4.0
    bbox = evidence.bbox
    crop = (
        max(0.0, bbox[0] - padding),
        max(0.0, bbox[1] - padding),
        min(page_width, bbox[2] + padding),
        min(page_height, bbox[3] + padding),
    )
    area = max(1.0, (crop[2] - crop[0]) * (crop[3] - crop[1]))
    safe_scale = math.sqrt(max_pixels / area) * 0.999
    scale = min(requested_scale, safe_scale)
    cache = getattr(page, "extraction_cache", None)
    cache_key = ("stroked_vector_text_raster_v1", scale, max_pixels)
    if cache is not None:
        cached = cache.get(cache_key)
        if isinstance(cached, internal_RasterRegion):
            if trace is not None and cached.raster.render_report is not None:
                trace.render_timings = cached.raster.render_report
            return cached

    compose_started = time.perf_counter()
    display_list = DisplayList(width=page_width, height=page_height)
    for index in evidence.drawing_indexes:
        drawing = capture.drawings[index]
        display_list.append(
            drawing.kind,
            drawing.seqno,
            bbox=drawing.rect,
            path=drawing.path,
            fill=(0.0, 0.0, 0.0),
            fill_opacity=drawing.fill_opacity,
            stroke_color=(0.0, 0.0, 0.0),
            stroke_opacity=drawing.stroke_opacity,
            line_width=drawing.line_width,
            line_cap=drawing.line_cap,
            line_join=drawing.line_join,
            dash_pattern=drawing.dash_pattern,
            fill_rule=drawing.fill_rule,
            blend_mode=drawing.blend_mode,
            soft_mask_alpha=drawing.soft_mask_alpha,
        )
    rendered = RenderedPage(
        page_number=int(getattr(page, "page_number", 0)),
        width=page_width,
        height=page_height,
        rotate=0,
        display_list=display_list,
    )
    compose_seconds = time.perf_counter() - compose_started
    raster_started = time.perf_counter()
    data = rendered.rasterize(
        background=(255, 255, 255, 255),
        scale=scale,
        max_pixels=max_pixels,
        crop=crop,
        cache=False,
    )
    render_report: dict[str, object] = {
        "compose_seconds": compose_seconds,
        "rasterize_seconds": time.perf_counter() - raster_started,
        "raster_mode": "stroked-vector-text-fallback",
        "crop": crop,
        "raster_pixels": data.width * data.height,
        "pixel_budget": max_pixels,
        "include_native_text": False,
        "image_timings": {},
        "display_items": len(display_list.items),
        "display_item_kinds": {"compact-stroke": len(display_list.items)},
        "image_filters": (),
    }
    raster = internal_Raster(
        data,
        max(70, int(round(72.0 * scale))),
        render_report,
    )
    region = internal_RasterRegion(raster, crop)
    if cache is not None:
        cache[cache_key] = region
    if trace is not None:
        trace.render_timings = render_report
    return region


def internal_remap_stroked_vector_observations(
    observations: ObservationBatch,
    packed: internal_PackedStrokedTextRaster,
) -> tuple[ObservationBatch, int]:
    """Translate montage observations back through their containing cells."""
    texts: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    polygons: list[tuple[float, ...]] = []
    confidences: list[float] = []
    sequences: list[int] = []
    references: list[Any | None] = []
    tolerance = STROKED_VECTOR_PACK_REMAP_TOLERANCE
    cell_index = SpatialIndex.from_items(
        packed.cells,
        bbox=lambda cell: (
            cell.packed_box[0] - tolerance,
            cell.packed_box[1] - tolerance,
            cell.packed_box[2] + tolerance,
            cell.packed_box[3] + tolerance,
        ),
    )
    for index, packed_box in enumerate(observations.bbox):
        box = internal_bbox_tuple(packed_box)
        center_x = (box[0] + box[2]) * 0.5
        center_y = (box[1] + box[3]) * 0.5
        # Broad-phase grid lookup narrows to spatially nearby cells; the exact
        # tolerance check below is unchanged so results are identical to a
        # full scan, just without touching every cell on the page per glyph.
        # The query box needs a positive area (SpatialIndex rejects a
        # degenerate point), so pad it by a fixed epsilon far smaller than
        # any real geometry difference -- it only widens the broad-phase
        # candidate set, never the final exact-tolerance result.
        query_box = (
            center_x - 1e-6,
            center_y - 1e-6,
            center_x + 1e-6,
            center_y + 1e-6,
        )
        cells = tuple(
            cell
            for cell in cell_index.candidates(query_box)
            if cell.packed_box[0] - tolerance <= center_x <= cell.packed_box[2] + tolerance
            and cell.packed_box[1] - tolerance <= center_y <= cell.packed_box[3] + tolerance
        )
        if len(cells) != 1:
            continue
        cell = cells[0]
        dx = cell.source_box[0] - cell.packed_box[0]
        dy = cell.source_box[1] - cell.packed_box[1]
        mapped = (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)
        texts.append(observations.text[index])
        boxes.append(mapped)
        polygons.append(
            (
                mapped[0],
                mapped[1],
                mapped[2],
                mapped[1],
                mapped[2],
                mapped[3],
                mapped[0],
                mapped[3],
            )
        )
        confidences.append(float(observations.confidence[index]))
        sequences.append(cell.drawing_indexes[0])
        references.append(observations.references[index])
    return (
        ObservationBatch.from_columns(
            texts,
            boxes,
            polygon=polygons,
            source=ObservationSource.OCR,
            confidence=confidences,
            sequence=sequences,
            rotation=(0 for _ in texts),
            font_size=(box[3] - box[1] for box in boxes),
            line_break_before=(True for _ in texts),
            references=references,
        ),
        len(observations) - len(texts),
    )


def internal_isolated_pin_label(text: str) -> bool:
    """Report whether an isolated-glyph OCR read looks like a pin label.

    The isolated montage inevitably contains junction dots and wire stubs
    alongside the real lone glyphs, and OCR renders those as stray letters
    and slashes. Genuine isolated labels on a schematic are short and carry
    a digit -- pin numbers, reference suffixes -- so keep only those.
    """
    stripped = text.strip()
    return 1 <= len(stripped) <= 4 and any(character.isdigit() for character in stripped)


def internal_remap_stroked_vector_candidate(
    candidate: internal_Candidate,
    packed: internal_PackedStrokedTextRaster,
    *,
    digit_bearing_only: bool = False,
) -> tuple[internal_Candidate, int]:
    """Translate montage OCR words and symbols into their source page cells."""
    remapped, unmapped = internal_remap_stroked_vector_observations(
        candidate.observations,
        packed,
    )
    if digit_bearing_only:
        remapped = remapped.take(
            tuple(
                index
                for index, text in enumerate(remapped.text)
                if internal_isolated_pin_label(text)
            )
        )
    remapped_symbols, _unmapped_symbols = internal_remap_stroked_vector_observations(
        candidate.symbols,
        packed,
    )
    return (
        internal_candidate(
            candidate.mode,
            remapped,
            symbols=remapped_symbols,
            api_seconds=candidate.api_seconds,
            setup_seconds=candidate.setup_seconds,
            recognition_seconds=candidate.recognition_seconds,
            iterator_seconds=candidate.iterator_seconds,
            cleanup_seconds=candidate.cleanup_seconds,
            candidate_seconds=candidate.candidate_seconds,
            recognition_status=candidate.recognition_status,
            median_text_height=candidate.metrics.median_text_height,
        ),
        unmapped,
    )


def internal_safe_image_crop(capture: CapturedPage) -> tuple[float, float, float, float] | None:
    """Return a useful crop when OCR is known to be image-dominated.

    A crop is only safe when the image coverage is substantial.  Sparse images
    must not hide page text outside the image bounds from the page OCR path.
    """
    evidence = capture.evidence
    if not evidence.image_boxes or not (
        evidence.full_page_image or evidence.image_area_ratio >= 0.65
    ):
        return None
    page_width = float(capture.page.width)
    page_height = float(capture.page.height)
    bounds = bbox_union(evidence.image_boxes)
    assert bounds is not None
    x0, y0, x1, y1 = bounds
    crop = (max(0.0, x0), max(0.0, y0), min(page_width, x1), min(page_height, y1))
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        return None
    crop_area = (crop[2] - crop[0]) * (crop[3] - crop[1])
    if crop_area >= max(1.0, page_width * page_height * 0.90):
        return None
    return crop


def internal_ocr_region_batch(
    regions: tuple[internal_OcrRegion, ...],
    ocr_pass: OcrPass,
    *,
    expanded: bool,
    page_area: float,
) -> tuple[internal_OcrRegion, ...]:
    count_limit = max(
        ocr_pass.max_regions,
        OCR_REGION_MAX_COUNT if expanded else OCR_REGION_INITIAL_COUNT,
    )
    area_limit = OCR_REGION_MAX_AREA_RATIO if expanded else OCR_REGION_INITIAL_AREA_RATIO
    selected: list[internal_OcrRegion] = []
    area = 0.0
    page_area = max(1.0, page_area)
    if page_area <= 0.0:
        return ()
    for region in regions:
        if len(selected) >= count_limit:
            break
        if selected and area + region.area > page_area * area_limit:
            continue
        selected.append(region)
        area += region.area
    return tuple(selected)


def internal_candidate_region_tasks(
    capture: CapturedPage,
    regions: tuple[internal_OcrRegion, ...],
    ocr_pass: OcrPass,
    *,
    rendered: Any | None,
    compact_image: bool | str,
    trace: internal_RecognitionTrace | None = None,
) -> tuple[
    tuple[internal_OcrTask, ...], int, Any | None, tuple[tuple[float, float, float, float], ...]
]:
    direct_regions = internal_page_image_regions(
        capture,
        minimum_area_ratio=0.02,
        max_pixels=ocr_pass.pixel_budget,
    )
    if not direct_regions:
        dominant = internal_dominant_image_region(
            capture,
            max_pixels=ocr_pass.pixel_budget,
        )
        if dominant is not None:
            direct_regions = (dominant,)
    tasks: list[internal_OcrTask] = []
    raster_pixels = 0
    rendered_boxes: list[tuple[float, float, float, float]] = []
    region_pass = replace(ocr_pass, scope=OcrPassScope.TILES, tiles=1)
    direct_region_index = (
        SpatialIndex(((index, region.page_box) for index, region in enumerate(direct_regions)))
        if len(direct_regions) > 4
        else None
    )
    for region in regions:
        raster: internal_Raster | None
        direct_candidates = (
            (direct_regions[index] for index in direct_region_index.intersecting(region.page_box))
            if direct_region_index is not None
            else iter(direct_regions)
        )
        matching_direct = tuple(
            candidate
            for candidate in direct_candidates
            # Region proposals include padding, so a source image need not cover the
            # entire box. It must still cover most of the requested target: otherwise
            # a narrow banner can incorrectly replace a broad compositor render.
            if internal_ocr_region_coverage(region.page_box, candidate.page_box)
            >= OCR_DIRECT_REGION_MIN_COVERAGE
        )
        layered_scan = any(
            internal_ocr_region_overlap(left.page_box, right.page_box) >= 0.90
            for index, left in enumerate(matching_direct)
            for right in matching_direct[:index]
        )
        direct = (
            None
            if layered_scan
            else max(
                matching_direct,
                key=lambda candidate: candidate.raster.width * candidate.raster.height,
                default=None,
            )
        )
        if direct is not None:
            raster = direct.raster
            raster_box = direct.page_box
        else:
            if rendered is None:
                rendered = compose_page(
                    capture.page,
                    RenderOptions(include_text=ocr_pass.include_native_text),
                    page_program=capture.program,
                )
            raster = internal_rendered_page_raster(
                capture,
                ocr_pass.scale,
                crop=region.page_box,
                rendered=rendered,
                cache=True,
                max_pixels=ocr_pass.pixel_budget,
                include_native_text=ocr_pass.include_native_text,
                trace=trace,
            )
            raster_box = region.page_box
        if raster is None:
            continue
        rendered_boxes.append(raster_box)
        raster_pixels += raster.width * raster.height
        full_page_region = (
            ocr_pass.scope is OcrPassScope.PAGE
            and len(regions) == 1
            and region.area
            >= getattr(
                getattr(capture, "evidence", None),
                "page_area",
                float(capture.page.width) * float(capture.page.height),
            )
            * 0.75
            and internal_ocr_region_coverage(
                region.page_box,
                (0.0, 0.0, float(capture.page.width), float(capture.page.height)),
            )
            >= 0.90
            and getattr(getattr(capture, "evidence", None), "vector_complexity", 0)
            >= OCR_PARALLEL_TILE_MIN_VECTOR_COMPLEXITY
            and 4_000_000 <= raster.width * raster.height <= PRIMARY_OCR_PIXELS
        )
        tile_count = ocr_pass.parallel_tiles if full_page_region else 1
        task_pass = (
            replace(
                region_pass,
                tiles=max(1, tile_count),
                recognize_words=True,
            )
            if layered_scan
            else replace(region_pass, tiles=max(1, tile_count))
        )
        tasks.extend(
            internal_tile_tasks(
                raster,
                raster_box,
                task_pass,
                compact_image=compact_image,
            )
        )
    return tuple(tasks), raster_pixels, rendered, tuple(rendered_boxes)


def internal_raster_rectangle_page_box(
    raster: internal_Raster,
    page_box: tuple[float, float, float, float],
    rectangle: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    """Map a top-left raster rectangle into bottom-left PDF page space."""
    x, y, width, height = rectangle
    page_x0, page_y0, page_x1, page_y1 = page_box
    page_width = page_x1 - page_x0
    page_height = page_y1 - page_y0
    return (
        page_x0 + x * page_width / raster.width,
        page_y1 - (y + height) * page_height / raster.height,
        page_x0 + (x + width) * page_width / raster.width,
        page_y1 - y * page_height / raster.height,
    )


def internal_high_resolution_weak_region_tasks(
    capture: CapturedPage,
    source_tasks: tuple[internal_OcrTask, ...],
    ocr_pass: OcrPass,
    primary: ObservationBatch,
    *,
    rendered: Any | None,
    compact_image: bool | str,
    trace: internal_RecognitionTrace | None = None,
) -> tuple[
    tuple[internal_OcrTask, ...], int, Any | None, tuple[tuple[float, float, float, float], ...]
]:
    """Rasterize only weak cells at rescue resolution instead of the whole page."""
    source_rasters: dict[tuple[int, tuple[float, float, float, float], int], internal_Raster] = {}
    for task in source_tasks:
        source_rasters.setdefault(
            (id(task.image), task.page_box, task.resolution),
            internal_Raster(task.image, task.resolution),
        )
    weak_regions: list[internal_OcrRegion] = []
    for (_, page_box, _), source_raster in source_rasters.items():
        for rectangle in internal_weak_region_rectangles(
            source_raster,
            page_box,
            ocr_pass,
            primary,
        ):
            weak_regions.append(
                internal_OcrRegion(
                    internal_raster_rectangle_page_box(source_raster, page_box, rectangle),
                    1.0,
                    ("adaptive-weak-region",),
                )
            )
    regions = internal_merge_ocr_regions(weak_regions)
    if not regions:
        return (), 0, rendered, ()
    if rendered is None:
        rendered = compose_page(
            capture.page,
            RenderOptions(include_text=ocr_pass.include_native_text),
            page_program=capture.program,
        )
    region_pass = replace(ocr_pass, scope=OcrPassScope.TILES, tiles=1)
    tasks: list[internal_OcrTask] = []
    raster_pixels = 0
    boxes: list[tuple[float, float, float, float]] = []
    for region in regions:
        raster = internal_rendered_page_raster(
            capture,
            ocr_pass.scale,
            crop=region.page_box,
            rendered=rendered,
            cache=True,
            max_pixels=ocr_pass.pixel_budget,
            include_native_text=ocr_pass.include_native_text,
            trace=trace,
        )
        if raster is None:
            continue
        boxes.append(region.page_box)
        raster_pixels += raster.width * raster.height
        tasks.extend(
            internal_tile_tasks(
                raster,
                region.page_box,
                region_pass,
                compact_image=compact_image,
            )
        )
    return tuple(tasks), raster_pixels, rendered, tuple(boxes)


STROKED_VECTOR_DECODE_MIN_OVERLAP = 0.55
STROKED_VECTOR_MULTI_EDIT_MIN_OVERLAP = 0.90
STROKED_VECTOR_MULTI_EDIT_MAX_CONFIDENCE = 85.0


def internal_stroked_vector_decoded_batch(
    observations: tuple[StrokedTextObservation, ...],
) -> ObservationBatch:
    boxes = tuple(observation.bbox for observation in observations)
    return ObservationBatch.from_columns(
        (observation.text for observation in observations),
        boxes,
        polygon=((box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]) for box in boxes),
        source=ObservationSource.STRUCTURE,
        confidence=(observation.confidence for observation in observations),
        sequence=(observation.first_drawing for observation in observations),
        rotation=(0 for _ in observations),
        font_size=(max(0.0, box[3] - box[1]) for box in boxes),
        line_break_before=(True for _ in observations),
    )


def internal_single_character_substitution(left: str, right: str) -> bool:
    return len(left) == len(right) and sum(a != b for a, b in zip(left, right, strict=True)) == 1


def internal_bounded_edit_distance(left: str, right: str, maximum: int) -> int:
    """Return a small Levenshtein distance, stopping once the bound is exceeded."""
    if abs(len(left) - len(right)) > maximum:
        return maximum + 1
    previous = list(range(len(right) + 1))
    for row, left_character in enumerate(left, 1):
        current = [row]
        for column, right_character in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_character != right_character),
                )
            )
        if min(current) > maximum:
            return maximum + 1
        previous = current
    return previous[-1]


def internal_stroked_vector_substitution(
    recognized: str,
    decoded: str,
    *,
    confidence: float,
    overlap: float,
) -> bool:
    if internal_single_character_substitution(recognized, decoded):
        return True
    left = recognized.casefold()
    right = decoded.casefold()
    return bool(
        confidence < STROKED_VECTOR_MULTI_EDIT_MAX_CONFIDENCE
        and overlap >= STROKED_VECTOR_MULTI_EDIT_MIN_OVERLAP
        and 2 <= len(left) <= 12
        and 2 <= len(right) <= 12
        and abs(len(left) - len(right)) <= 1
        and left[0] == right[0]
        and any(character.isalnum() for character in left)
        and any(character.isalnum() for character in right)
        and internal_bounded_edit_distance(left, right, 2) <= 2
    )


def internal_stroked_vector_symbol_seeds(
    capture: CapturedPage,
    symbols: ObservationBatch,
) -> tuple[StrokedTextSeed, ...]:
    """Join character boxes only when they exactly fill one known vector run."""
    if not len(symbols):
        return ()
    runs_by_sequence = {
        run.drawing_indexes[0]: run for run in internal_stroked_text_profile(capture).seed_runs
    }
    grouped: dict[
        int,
        list[tuple[float, str, float]],
    ] = defaultdict(list)
    for text, raw_box, confidence, raw_sequence in zip(
        symbols.text,
        symbols.bbox,
        symbols.confidence,
        symbols.sequence,
        strict=True,
    ):
        character = text.strip()
        sequence = int(raw_sequence)
        if len(character) != 1 or sequence not in runs_by_sequence:
            continue
        grouped[sequence].append((internal_bbox_tuple(raw_box)[0], character, float(confidence)))

    seeds: list[StrokedTextSeed] = []
    for sequence, items in grouped.items():
        run = runs_by_sequence[sequence]
        if len(items) != run.glyph_count:
            continue
        ordered = sorted(items)
        seeds.append(
            StrokedTextSeed(
                text="".join(character for ignored_x, character, ignored_confidence in ordered),
                bbox=run.bbox,
                confidence=min(confidence for ignored_x, ignored_character, confidence in ordered),
                sequence=sequence,
            )
        )
    return tuple(seeds)


def internal_decode_stroked_vector_text(
    capture: CapturedPage,
    ocr: ObservationBatch,
    symbols: ObservationBatch | None = None,
) -> StrokedTextDecode:
    evidence = capture.evidence.stroked_vector_text
    if not evidence.trusted or not evidence.drawing_indexes or not len(ocr):
        return StrokedTextDecode()
    profile = internal_stroked_text_profile(capture)
    word_seeds = tuple(
        StrokedTextSeed(
            text=text,
            bbox=internal_bbox_tuple(box),
            confidence=float(confidence),
            sequence=int(sequence),
        )
        for text, box, confidence, sequence in zip(
            ocr.text,
            ocr.bbox,
            ocr.confidence,
            ocr.sequence,
            strict=True,
        )
    )
    symbol_seeds = internal_stroked_vector_symbol_seeds(
        capture,
        symbols if symbols is not None else ObservationBatch.empty(),
    )
    if not symbol_seeds:
        return decode_stroked_text_profile(profile, word_seeds)
    return decode_stroked_text_profile_with_supplemental_seeds(
        profile,
        word_seeds,
        symbol_seeds,
    )


def internal_packed_stroked_vector_decode_gate(
    decoded: StrokedTextDecode,
    cell_count: int,
) -> tuple[bool, dict[str, int | bool]]:
    """Require enough learned geometry before skipping the full-layer OCR fallback."""
    aligned_required = min(
        STROKED_VECTOR_PACK_MIN_ALIGNED_SEEDS,
        max(4, cell_count // 4),
    )
    learned_required = min(
        STROKED_VECTOR_PACK_MIN_LEARNED_SIGNATURES,
        max(8, cell_count // 3),
    )
    decoded_required = min(
        STROKED_VECTOR_PACK_MIN_DECODED_RUNS,
        max(8, cell_count // 3),
    )
    accepted = bool(
        decoded.aligned_seeds >= aligned_required
        and decoded.learned_signatures >= learned_required
        and len(decoded.observations) >= decoded_required
    )
    return accepted, {
        "accepted": accepted,
        "cells": cell_count,
        "aligned_seeds": decoded.aligned_seeds,
        "aligned_required": aligned_required,
        "learned_signatures": decoded.learned_signatures,
        "learned_required": learned_required,
        "decoded_runs": len(decoded.observations),
        "decoded_required": decoded_required,
    }


def internal_recover_stroked_vector_text(
    capture: CapturedPage,
    ocr: ObservationBatch,
    trace: internal_RecognitionTrace | None = None,
) -> ObservationBatch:
    """Augment one OCR pass with text decoded from repeated vector glyphs."""
    evidence = capture.evidence.stroked_vector_text
    if not evidence.trusted or not evidence.drawing_indexes or not len(ocr):
        return ocr
    trace = trace or internal_RecognitionTrace.create()
    started = time.perf_counter()
    cached_decode = trace.pending_stroked_decode
    if (
        isinstance(cached_decode, tuple)
        and len(cached_decode) == 3
        and cached_decode[0] == id(ocr)
        and isinstance(cached_decode[1], StrokedTextDecode)
    ):
        decoded = cached_decode[1]
        prior_decode_seconds = float(cached_decode[2])
    else:
        decoded = internal_decode_stroked_vector_text(capture, ocr)
        prior_decode_seconds = 0.0
    ocr_index = SpatialIndex.from_boxes(ocr.bbox)
    replacements: set[int] = set()
    accepted: list[StrokedTextObservation] = []
    additions = 0
    corrections = 0
    for observation in decoded.observations:
        candidate_area = max(
            0.01,
            (observation.bbox[2] - observation.bbox[0])
            * (observation.bbox[3] - observation.bbox[1]),
        )
        overlaps: list[tuple[float, int]] = []
        for hit in ocr_index.intersecting_hits(observation.bbox):
            hit_area = max(0.01, (hit.bbox[2] - hit.bbox[0]) * (hit.bbox[3] - hit.bbox[1]))
            overlap = bbox_intersection_area(observation.bbox, hit.bbox) / min(
                candidate_area, hit_area
            )
            if overlap >= STROKED_VECTOR_DECODE_MIN_OVERLAP:
                overlaps.append((overlap, int(hit.item)))
        if not overlaps:
            accepted.append(observation)
            additions += 1
            continue
        best_overlap, best_index = max(overlaps)
        recognized_text = ocr.text[best_index].strip()
        if recognized_text == observation.text:
            continue
        if best_index not in replacements and internal_stroked_vector_substitution(
            recognized_text,
            observation.text,
            confidence=float(ocr.confidence[best_index]),
            overlap=best_overlap,
        ):
            replacements.add(best_index)
            accepted.append(observation)
            corrections += 1

    trace.stroked_vector_decode = {
        "seconds": prior_decode_seconds + time.perf_counter() - started,
        "eligible_seeds": decoded.eligible_seeds,
        "aligned_seeds": decoded.aligned_seeds,
        "accepted_seeds": decoded.accepted_seeds,
        "initial_signatures": decoded.initial_signatures,
        "learned_signatures": decoded.learned_signatures,
        "approximate_signatures": decoded.approximate_signatures,
        "candidate_runs": decoded.candidate_runs,
        "decoded_candidate_runs": decoded.decoded_candidate_runs,
        "candidate_run_coverage": decoded.candidate_run_coverage,
        "candidate_glyph_coverage": decoded.candidate_glyph_coverage,
        "decoded_runs": len(decoded.observations),
        "additions": additions,
        "corrections": corrections,
    }
    trace.stroked_vector_alphabet = tuple(decoded.alphabet)
    if not accepted:
        return ocr
    retained = ocr.take(tuple(index for index in range(len(ocr)) if index not in replacements))
    return ObservationBatch.concatenate(
        retained,
        internal_stroked_vector_decoded_batch(tuple(accepted)),
    )


def recognize_page(
    capture: CapturedPage,
    plan: WorkPlan,
    context: TaskScope,
) -> RecognitionResult:
    trace = internal_RecognitionTrace.create()
    if not plan.ocr_passes:
        return RecognitionResult(ObservationBatch.empty(), trace.report())
    with context.reserve_raster(MAX_OCR_RASTER_BYTES):
        context.raise_if_cancelled()
        observations = internal_recognize_page_with_reserved_raster(
            capture,
            plan,
            context,
            trace=trace,
        )
    observations = internal_recover_stroked_vector_text(capture, observations, trace)
    return RecognitionResult(observations, trace.report())


def internal_recognize_page_with_reserved_raster(
    capture: CapturedPage,
    plan: WorkPlan,
    context: TaskScope,
    *,
    trace: internal_RecognitionTrace | None = None,
) -> ObservationBatch:
    trace = trace or internal_RecognitionTrace.create()
    page = capture.page
    page_box = (0.0, 0.0, float(page.width), float(page.height))
    compact_image: bool | str = True
    if capture.evidence.full_page_image:
        image_filters = capture.evidence.image_filters
        if any("JPX" in str(filter_name).upper() for filter_name in image_filters):
            compact_image = "grayscale"
    dominant_regions: dict[int, internal_RasterRegion | None] = {}
    rendered_rasters: dict[tuple[float, int, bool], internal_Raster | None] = {}
    rendered_page: Any | None = None
    candidate_regions: tuple[internal_OcrRegion, ...] | None = None
    candidates: list[tuple[str, internal_Candidate]] = []
    pass_diagnostics = trace.passes
    selected_name = ""
    selected: internal_Candidate | None = None
    selected_tasks: tuple[internal_OcrTask, ...] = ()
    previous_region_additions = 0
    seeded_region_selected = False
    adaptive_rescue_used = False

    def recognize_batch(tasks: tuple[internal_OcrTask, ...]) -> tuple[internal_Candidate, ...]:
        groups = internal_ocr_task_groups(tasks)
        results = context.map_ordered(internal_recognize_group, groups, stage=WorkStage.OCR)
        return tuple(candidate for group in results for candidate in group)

    def recognize_tasks(tasks: tuple[internal_OcrTask, ...]) -> tuple[internal_Candidate, ...]:
        candidates = recognize_batch(tasks)
        if any(candidate.recognition_status == "timeout" for candidate in candidates):
            context.raise_if_cancelled()
            candidates = internal_recover_timed_out_tasks(tasks, candidates, recognize_batch)
        return candidates

    if plan.verify_hidden_text:
        context.raise_if_cancelled()
        started = time.perf_counter()
        verification_pass = OcrPass(
            "hidden-text-verification",
            OcrPassScope.PAGE,
            1.0,
            (PSM_SPARSE_TEXT,),
            minimum_confidence=HIDDEN_TEXT_VERIFY_MIN_CONFIDENCE,
            pixel_budget=HIDDEN_TEXT_VERIFY_PIXELS,
            recognize_words=True,
            region_first=False,
        )
        verification_region = internal_dominant_image_region(
            capture,
            max_pixels=HIDDEN_TEXT_VERIFY_PIXELS,
        )
        verification_tasks = (
            internal_tile_tasks(
                verification_region.raster,
                verification_region.page_box,
                verification_pass,
                compact_image=compact_image,
            )
            if verification_region is not None
            else ()
        )
        verification_candidates = recognize_tasks(verification_tasks)
        verification_candidate = internal_merge_candidate_batches(verification_candidates)
        verification = internal_hidden_text_verification(
            capture.observations,
            verification_candidate.observations,
        )
        raster_pixels = (
            verification_region.raster.width * verification_region.raster.height
            if verification_region is not None
            else 0
        )
        verification_record: dict[str, object] = {
            "name": verification_pass.name,
            "scope": verification_pass.scope.value,
            "scale": verification_pass.scale,
            "modes": verification_pass.modes,
            "recognize_words": verification_pass.recognize_words,
            "character_confidence_threshold": None,
            "task_count": len(verification_tasks),
            "raster_pixels": raster_pixels,
            "region_stage": "dominant-image-preview",
            "region_boxes": (
                (verification_region.page_box,) if verification_region is not None else ()
            ),
            "full_page_fallback": False,
            "elapsed_seconds": time.perf_counter() - started,
            "render_timings": trace.render_timings or {},
            "recognition_seconds": sum(
                candidate.recognition_seconds for candidate in verification_candidates
            ),
            "setup_seconds": sum(candidate.setup_seconds for candidate in verification_candidates),
            "api_seconds": sum(candidate.api_seconds for candidate in verification_candidates),
            "iterator_seconds": sum(
                candidate.iterator_seconds for candidate in verification_candidates
            ),
            "cleanup_seconds": sum(
                candidate.cleanup_seconds for candidate in verification_candidates
            ),
            "candidate_seconds": sum(
                candidate.candidate_seconds for candidate in verification_candidates
            ),
            "recognition_statuses": tuple(
                candidate.recognition_status for candidate in verification_candidates
            ),
            "accepted_additions": 0,
            "adaptive_retry_scale": None,
            "adaptive_preflight": None,
            "adaptive_rescue_decision": None,
            "adaptive_rescue": None,
            "pixel_budget": verification_pass.pixel_budget,
            "rectangles": tuple(task.rectangle for task in verification_tasks),
            "selected": verification.accepted,
            **verification_candidate.metrics.as_record(),
            **verification.as_record(),
        }
        pass_diagnostics.append(verification_record)
        trace.hidden_text_verification = {
            "raster_pixels": raster_pixels,
            **verification.as_record(),
        }
        if verification.accepted:
            return internal_promoted_hidden_observations(capture)

    for ocr_pass in plan.ocr_passes:
        if (
            selected is not None
            and ocr_pass.scope is OcrPassScope.PAGE
            and ocr_pass.run_if_characters_below is not None
            and internal_primary_text_is_sufficient(selected)
        ):
            continue
        if (
            selected is not None
            and ocr_pass.run_if_characters_below is not None
            and selected.metrics.characters >= ocr_pass.run_if_characters_below
        ):
            continue
        if (
            selected is not None
            and ocr_pass.scope is OcrPassScope.IMAGE_REGIONS
            and ocr_pass.run_if_characters_below is not None
            and selected.metrics.characters >= 28
            and selected.metrics.mean_confidence >= 97.0
        ):
            continue
        if (
            selected is not None
            and ocr_pass.scope is OcrPassScope.IMAGE_REGIONS
            and ocr_pass.run_if_characters_below is not None
            and selected.metrics.characters >= 1500
            and selected.metrics.mean_confidence >= 98.0
        ):
            continue
        if (
            ocr_pass.run_if_additions_below is not None
            and previous_region_additions >= ocr_pass.run_if_additions_below
        ):
            continue
        if (
            ocr_pass.scope is OcrPassScope.PAGE
            and ocr_pass.run_if_additions_below is not None
            and previous_region_additions == 0
            and selected is None
            and capture.evidence.visible_native_characters >= 3_000
        ):
            continue
        if (
            ocr_pass.scope is OcrPassScope.WEAK_REGIONS
            and ocr_pass.run_if_additions_below is not None
            and previous_region_additions == 0
            and selected is not None
            and selected.metrics.characters >= 32
            and selected.metrics.mean_confidence >= 90.0
        ):
            continue
        if (
            ocr_pass.scope is OcrPassScope.PAGE
            and seeded_region_selected
            and ocr_pass.run_if_additions_below is not None
        ):
            selected = None
            selected_name = ""
            selected_tasks = ()
            seeded_region_selected = False
        context.raise_if_cancelled()
        started = time.perf_counter()
        adaptive_preflight: dict[str, object] | None = None
        vector_preview = bool(
            capture.evidence.image_count == 0
            and capture.evidence.vector_complexity >= 100_000
            and capture.evidence.text_coverage < 0.05
        )
        if (
            ocr_pass.adaptive_scale
            and ocr_pass.scope is OcrPassScope.PAGE
            and ocr_pass.pixel_budget == PRIMARY_OCR_PIXELS
            and (capture.evidence.full_page_image or vector_preview)
        ):
            preview_raster: internal_Raster | None = None
            if capture.evidence.full_page_image:
                if OCR_PREFLIGHT_PIXELS not in dominant_regions:
                    # The preview only measures text height, so enlarging it would
                    # cost time and shift the projection this decision depends on.
                    dominant_regions[OCR_PREFLIGHT_PIXELS] = internal_dominant_image_region(
                        capture,
                        max_pixels=OCR_PREFLIGHT_PIXELS,
                        upscale=False,
                    )
                preview_region = dominant_regions[OCR_PREFLIGHT_PIXELS]
                preview_raster = preview_region.raster if preview_region is not None else None
            else:
                if rendered_page is None:
                    rendered_page = compose_page(
                        capture.page,
                        RenderOptions(include_text=ocr_pass.include_native_text),
                        page_program=capture.program,
                    )
                preview_raster = internal_rendered_page_raster(
                    capture,
                    ocr_pass.scale,
                    rendered=rendered_page,
                    cache=True,
                    max_pixels=OCR_PREFLIGHT_PIXELS,
                    include_native_text=ocr_pass.include_native_text,
                    trace=trace,
                )
            if preview_raster is not None:
                preview_height = internal_estimated_text_height(preview_raster)
                projected_height = preview_height * math.sqrt(
                    ocr_pass.pixel_budget / max(1, preview_raster.width * preview_raster.height)
                )
                projected_limit = 22.0 if vector_preview else 20.0
                if 12.0 <= projected_height < projected_limit:
                    original_scale = ocr_pass.scale
                    ocr_pass = replace(
                        ocr_pass,
                        scale=min(
                            8.0,
                            max(
                                original_scale + 0.5,
                                original_scale * 32.0 / projected_height,
                            ),
                        ),
                        pixel_budget=MAX_OCR_PIXELS,
                    )
                    adaptive_preflight = {
                        "preview_pixels": preview_raster.width * preview_raster.height,
                        "preview_text_height": preview_height,
                        "projected_primary_text_height": projected_height,
                        "selected_scale": ocr_pass.scale,
                        "source": "vector-render" if vector_preview else "dominant-image",
                    }
        tasks: tuple[internal_OcrTask, ...]
        packed_stroked: internal_PackedStrokedTextRaster | None = None
        raster_pixels = 0
        skipped_raster_pixels = 0
        image_text_preflight: tuple[dict[str, object], ...] = ()
        skipped_region_boxes: tuple[tuple[float, float, float, float], ...] = ()
        region_stage = "page"
        region_boxes: tuple[tuple[float, float, float, float], ...] = ()
        if (
            ocr_pass.region_first
            and ocr_pass.scope in {OcrPassScope.PAGE, OcrPassScope.WEAK_REGIONS}
            and (
                ocr_pass.scope is not OcrPassScope.WEAK_REGIONS
                or selected is not None
                or ocr_pass.seed_with_native
            )
        ):
            if candidate_regions is None:
                candidate_regions = internal_candidate_ocr_regions(capture)
            distributed_outline_text = bool(
                ocr_pass.scope is OcrPassScope.PAGE
                and internal_has_distributed_outline_text(capture)
            )
            region_batch = (
                (
                    internal_OcrRegion(
                        page_box,
                        float("inf"),
                        ("distributed-outline-text",),
                    ),
                )
                if distributed_outline_text
                else internal_ocr_region_batch(
                    candidate_regions,
                    ocr_pass,
                    expanded=False,
                    page_area=max(1.0, float(page.width) * float(page.height)),
                )
            )
            tasks, raster_pixels, rendered_page, region_boxes = internal_candidate_region_tasks(
                capture,
                region_batch,
                ocr_pass,
                rendered=rendered_page,
                compact_image=compact_image,
                trace=trace,
            )
            region_stage = (
                "distributed-outline-page" if distributed_outline_text else "initial-regions"
            )
            if len(region_batch) == 1 and "page-fallback" in region_batch[0].reasons:
                region_stage = "page"
        elif ocr_pass.scope is OcrPassScope.WEAK_REGIONS:
            if selected is None and not ocr_pass.seed_with_native:
                continue
            if selected is not None and selected_tasks:
                tasks, raster_pixels, rendered_page, region_boxes = (
                    internal_high_resolution_weak_region_tasks(
                        capture,
                        selected_tasks,
                        ocr_pass,
                        selected.observations,
                        rendered=rendered_page,
                        compact_image=compact_image,
                        trace=trace,
                    )
                )
                region_stage = "weak-region-crops"
            else:
                if ocr_pass.pixel_budget not in dominant_regions:
                    dominant_regions[ocr_pass.pixel_budget] = internal_dominant_image_region(
                        capture,
                        max_pixels=ocr_pass.pixel_budget,
                    )
                direct_region = dominant_regions[ocr_pass.pixel_budget]
                raster = direct_region.raster if direct_region is not None else None
                raster_page_box = direct_region.page_box if direct_region is not None else page_box
                if raster is None:
                    raster_key = (
                        ocr_pass.scale,
                        ocr_pass.pixel_budget,
                        ocr_pass.include_native_text,
                    )
                    if raster_key not in rendered_rasters:
                        rendered_rasters[raster_key] = internal_rendered_page_raster(
                            capture,
                            ocr_pass.scale,
                            max_pixels=ocr_pass.pixel_budget,
                            include_native_text=ocr_pass.include_native_text,
                            trace=trace,
                        )
                    raster = rendered_rasters[raster_key]
                    raster_page_box = page_box
                tasks = (
                    internal_weak_region_tasks(
                        raster,
                        raster_page_box,
                        ocr_pass,
                        selected.observations if selected is not None else capture.observations,
                        compact_image=compact_image,
                    )
                    if raster is not None
                    else ()
                )
                raster_pixels = (
                    sum(task.rectangle[2] * task.rectangle[3] for task in tasks)
                    if raster is not None
                    else 0
                )
        elif ocr_pass.scope is OcrPassScope.STROKED_VECTOR_TEXT:
            packed_stroked = internal_stroked_vector_text_raster(
                capture,
                ocr_pass.scale,
                max_pixels=ocr_pass.pixel_budget,
                trace=trace,
            )
            if packed_stroked is not None:
                region_stage = "packed-stroked-vector-text"
                region_boxes = (
                    (capture.evidence.stroked_vector_text.bbox,)
                    if capture.evidence.stroked_vector_text.bbox is not None
                    else ()
                )
                tasks = internal_tile_tasks(
                    packed_stroked.raster,
                    packed_stroked.packed_box,
                    replace(ocr_pass, recognize_words=True, collect_symbols=True),
                    compact_image=compact_image,
                )
                raster_pixels = packed_stroked.raster.width * packed_stroked.raster.height
            else:
                fallback_region = internal_full_stroked_vector_text_raster(
                    capture,
                    ocr_pass.scale,
                    max_pixels=ocr_pass.pixel_budget,
                    trace=trace,
                )
                region_stage = "stroked-vector-text-fallback"
                region_boxes = (fallback_region.page_box,) if fallback_region is not None else ()
                tasks = (
                    internal_tile_tasks(
                        fallback_region.raster,
                        fallback_region.page_box,
                        ocr_pass,
                        compact_image=compact_image,
                    )
                    if fallback_region is not None
                    else ()
                )
                raster_pixels = (
                    fallback_region.raster.width * fallback_region.raster.height
                    if fallback_region is not None
                    else 0
                )
                trace.stroked_vector_packed = {
                    "accepted": False,
                    "cells": 0,
                    "raster_pixels": 0,
                    "unmapped_observations": 0,
                    "fallback_used": bool(tasks),
                }
        elif ocr_pass.scope is OcrPassScope.IMAGE_REGIONS:
            regions = internal_page_image_regions(
                capture,
                minimum_area_ratio=0.02,
                max_pixels=ocr_pass.pixel_budget,
                maximum_axis_deviation=OCR_IMAGE_REGIONS_MAX_AXIS_DEVIATION,
            )
            if regions:
                region_signals = tuple(
                    (region, internal_raster_text_signal(region.raster.image)) for region in regions
                )
                image_text_preflight = tuple(
                    {
                        "page_box": region.page_box,
                        "raster_pixels": region.raster.width * region.raster.height,
                        **signal.as_record(),
                    }
                    for region, signal in region_signals
                )
                eligible_regions = tuple(
                    region
                    for region, signal in region_signals
                    if signal.likely_text
                    or (
                        signal.horizontal_edge_ratio >= 0.035
                        and sum(len(t.strip()) for t in capture.observations.text) < 15
                    )
                )
                skipped_regions = tuple(
                    region for region, signal in region_signals if not signal.likely_text
                )
                skipped_raster_pixels = sum(
                    region.raster.width * region.raster.height for region in skipped_regions
                )
                skipped_region_boxes = tuple(region.page_box for region in skipped_regions)
                region_boxes = tuple(region.page_box for region in eligible_regions)
                region_stage = "direct-image-regions"
                tasks = tuple(
                    task
                    for region in eligible_regions
                    for task in internal_tile_tasks(
                        region.raster,
                        region.page_box,
                        ocr_pass,
                        compact_image=compact_image,
                    )
                )
                raster_pixels = sum(
                    region.raster.width * region.raster.height for region in eligible_regions
                )
            else:
                fallback_scale = max(2.0, ocr_pass.scale)
                image_crop = internal_safe_image_crop(capture)
                raster = internal_rendered_page_raster(
                    capture,
                    fallback_scale,
                    crop=image_crop,
                    max_pixels=ocr_pass.pixel_budget,
                    include_native_text=ocr_pass.include_native_text,
                    trace=trace,
                )
                raster_page_box = image_crop or page_box
                tasks = (
                    internal_tile_tasks(
                        raster,
                        raster_page_box,
                        ocr_pass,
                        compact_image=compact_image,
                    )
                    if raster is not None
                    else ()
                )
                raster_pixels = raster.width * raster.height if raster is not None else 0
        else:
            if internal_direct_scan_allowed(capture, plan):
                if ocr_pass.pixel_budget not in dominant_regions:
                    dominant_regions[ocr_pass.pixel_budget] = internal_dominant_image_region(
                        capture,
                        max_pixels=ocr_pass.pixel_budget,
                    )
                direct_region = dominant_regions.get(ocr_pass.pixel_budget)
            else:
                direct_region = None
            raster = direct_region.raster if direct_region is not None else None
            raster_page_box = direct_region.page_box if direct_region is not None else page_box
            if raster is None:
                raster_key = (
                    ocr_pass.scale,
                    ocr_pass.pixel_budget,
                    ocr_pass.include_native_text,
                )
                if raster_key not in rendered_rasters:
                    rendered_rasters[raster_key] = internal_rendered_page_raster(
                        capture,
                        ocr_pass.scale,
                        max_pixels=ocr_pass.pixel_budget,
                        include_native_text=ocr_pass.include_native_text,
                        trace=trace,
                    )
                raster = rendered_rasters[raster_key]
                raster_page_box = page_box
            task_raster = (
                internal_adaptive_ocr_raster(raster)
                if raster is not None and ocr_pass.name == "adaptive-page"
                else raster
            )
            tasks = (
                internal_tile_tasks(
                    task_raster,
                    raster_page_box,
                    ocr_pass,
                    compact_image=compact_image,
                )
                if task_raster is not None
                else ()
            )
            raster_pixels = raster.width * raster.height if raster is not None else 0
        if not tasks:
            if not image_text_preflight:
                continue
            region_stage = "image-text-preflight"

        candidate_source_tasks = tasks
        task_candidates = recognize_tasks(tasks)
        if packed_stroked is not None:
            remapped_with_counts = tuple(
                internal_remap_stroked_vector_candidate(candidate, packed_stroked)
                for candidate in task_candidates
            )
            task_candidates = tuple(item[0] for item in remapped_with_counts)
            unmapped_observations = sum(item[1] for item in remapped_with_counts)
            packed_candidate = internal_merge_candidate_batches(task_candidates)
            decode_started = time.perf_counter()
            packed_decode = internal_decode_stroked_vector_text(
                capture,
                packed_candidate.observations,
                packed_candidate.symbols,
            )
            decode_seconds = time.perf_counter() - decode_started
            packed_accepted, packed_gate = internal_packed_stroked_vector_decode_gate(
                packed_decode,
                len(packed_stroked.cells),
            )
            packed_pixels = raster_pixels
            fallback_used = False
            if packed_accepted:
                # Seed packing only rasterizes multi-glyph runs, so isolated
                # glyphs (pin numbers, lone digits) are never shown to OCR when
                # the packed decode gate passes. Recognize them from their own
                # high-scale montage as a supplement.
                isolated_packed = internal_stroked_vector_text_raster(
                    capture,
                    ocr_pass.scale,
                    max_pixels=ocr_pass.pixel_budget,
                    variant="isolated",
                    trace=trace,
                )
                isolated_tasks = (
                    internal_tile_tasks(
                        isolated_packed.raster,
                        isolated_packed.packed_box,
                        replace(
                            ocr_pass,
                            recognize_words=True,
                            collect_symbols=True,
                            minimum_confidence=50.0,
                        ),
                        compact_image=compact_image,
                    )
                    if isolated_packed is not None
                    else ()
                )
                if isolated_tasks and isolated_packed is not None:
                    isolated_remapped = tuple(
                        internal_remap_stroked_vector_candidate(
                            candidate,
                            isolated_packed,
                            digit_bearing_only=True,
                        )
                        for candidate in recognize_tasks(isolated_tasks)
                    )
                    isolated_candidates = tuple(item[0] for item in isolated_remapped)
                    packed_gate["isolated_cells"] = len(isolated_packed.cells)
                    packed_gate["isolated_observations"] = sum(
                        len(item[0].observations) for item in isolated_remapped
                    )
                    task_candidates = (*task_candidates, *isolated_candidates)
                    candidate_source_tasks = (*candidate_source_tasks, *isolated_tasks)
                    tasks = (*tasks, *isolated_tasks)
                    packed_candidate = internal_merge_candidate_batches(task_candidates)
                    raster_pixels += isolated_packed.raster.width * isolated_packed.raster.height
                trace.pending_stroked_decode = (
                    id(packed_candidate.observations),
                    packed_decode,
                    decode_seconds,
                )
            else:
                fallback_region = internal_full_stroked_vector_text_raster(
                    capture,
                    ocr_pass.scale,
                    max_pixels=ocr_pass.pixel_budget,
                    trace=trace,
                )
                fallback_tasks = (
                    internal_tile_tasks(
                        fallback_region.raster,
                        fallback_region.page_box,
                        replace(ocr_pass, recognize_words=False),
                        compact_image=compact_image,
                    )
                    if fallback_region is not None
                    else ()
                )
                if fallback_tasks:
                    fallback_used = True
                    fallback_candidates = recognize_tasks(fallback_tasks)
                    task_candidates = (*task_candidates, *fallback_candidates)
                    candidate_source_tasks = (*candidate_source_tasks, *fallback_tasks)
                    tasks = (*tasks, *fallback_tasks)
                    packed_candidate = internal_merge_candidate_batches(fallback_candidates)
                    raster_pixels += (
                        fallback_region.raster.width * fallback_region.raster.height
                        if fallback_region is not None
                        else 0
                    )
                    region_stage = "stroked-vector-text-fallback"
                    region_boxes = (
                        (fallback_region.page_box,) if fallback_region is not None else region_boxes
                    )
            trace.stroked_vector_packed = {
                **packed_gate,
                "raster_pixels": packed_pixels,
                "unmapped_observations": unmapped_observations,
                "symbol_observations": len(packed_candidate.symbols),
                "fallback_used": fallback_used,
            }
            candidate = packed_candidate
        else:
            candidate = internal_merge_candidate_batches(task_candidates)
        if (
            selected is not None
            and plan.augment_page_candidates
            and ocr_pass.scope is OcrPassScope.PAGE
            and not capture.evidence.vector_complexity >= 180
        ):
            candidate, _ = internal_augment_candidate(
                selected,
                candidate,
                minimum_confidence=70.0,
            )
        adaptive_retry_scale: float | None = None
        adaptive_rescue: dict[str, object] | None = None
        adaptive_rescue_decision: dict[str, object] | None = None
        median_height = candidate.metrics.median_text_height
        rescue_eligible = bool(
            ocr_pass.adaptive_scale
            and ocr_pass.scope is OcrPassScope.PAGE
            and ocr_pass.pixel_budget < MAX_OCR_PIXELS
            and not adaptive_rescue_used
            and candidate.metrics.characters >= ocr_pass.minimum_characters_for_rescue
            and (candidate.metrics.characters < 32 or 0.0 < median_height < 24.0)
        )
        run_rescue = False
        if rescue_eligible:
            adaptive_rescue_used = True
            run_rescue, adaptive_rescue_decision = internal_adaptive_rescue_decision(
                candidate,
                candidate_source_tasks,
                ocr_pass,
            )
        if run_rescue:
            factor = 1.5 if median_height <= 0.0 else min(2.5, max(1.25, 32.0 / median_height))
            adaptive_retry_scale = min(8.0, max(ocr_pass.scale + 0.5, ocr_pass.scale * factor))
            retry_pass = replace(
                ocr_pass,
                name="adaptive-rescue",
                scale=adaptive_retry_scale,
                pixel_budget=MAX_OCR_PIXELS,
                region_first=False,
            )
            retry_scope = (
                "page"
                if candidate.metrics.characters < 32 or median_height < 18.0
                else "weak-regions"
            )
            retry_boxes: tuple[tuple[float, float, float, float], ...] = ()
            if retry_scope == "page":
                retry_raster = internal_rendered_page_raster(
                    capture,
                    adaptive_retry_scale,
                    max_pixels=MAX_OCR_PIXELS,
                    include_native_text=ocr_pass.include_native_text,
                    trace=trace,
                )
                retry_tasks = (
                    internal_tile_tasks(
                        retry_raster,
                        page_box,
                        retry_pass,
                        compact_image=compact_image,
                    )
                    if retry_raster is not None
                    else ()
                )
                rescue_pixels = (
                    retry_raster.width * retry_raster.height if retry_raster is not None else 0
                )
            else:
                retry_pass = replace(
                    retry_pass,
                    scope=OcrPassScope.WEAK_REGIONS,
                    tiles=max(6, retry_pass.tiles),
                    region_columns=max(3, retry_pass.region_columns),
                    max_regions=max(8, retry_pass.max_regions),
                )
                retry_tasks, rescue_pixels, rendered_page, retry_boxes = (
                    internal_high_resolution_weak_region_tasks(
                        capture,
                        tasks,
                        retry_pass,
                        candidate.observations,
                        rendered=rendered_page,
                        compact_image=compact_image,
                        trace=trace,
                    )
                )
            if retry_tasks:
                candidate_source_tasks = (*candidate_source_tasks, *retry_tasks)
                retry_candidates = recognize_tasks(retry_tasks)
                retry_candidate = internal_merge_candidate_batches(retry_candidates)
                augmented_candidate, rescue_additions = internal_augment_candidate(
                    candidate,
                    retry_candidate,
                    minimum_confidence=ocr_pass.minimum_confidence,
                )
                if retry_candidate.metrics.utility > augmented_candidate.metrics.utility * 1.05:
                    candidate = retry_candidate
                elif augmented_candidate.metrics.utility > candidate.metrics.utility:
                    candidate = augmented_candidate
                task_candidates = (*task_candidates, *retry_candidates)
                raster_pixels += rescue_pixels
                adaptive_rescue = {
                    "scope": retry_scope,
                    "scale": adaptive_retry_scale,
                    "raster_pixels": rescue_pixels,
                    "task_count": len(retry_tasks),
                    "accepted_additions": rescue_additions,
                    "region_boxes": retry_boxes,
                }
        additions = 0
        if ocr_pass.scope is OcrPassScope.WEAK_REGIONS:
            used_native_seed = selected is None
            if selected is not None:
                candidate, additions = internal_augment_candidate(
                    selected,
                    candidate,
                    minimum_confidence=ocr_pass.minimum_confidence,
                )
            else:
                additions = len(candidate.observations)
        candidates.append((ocr_pass.name, candidate))
        elapsed = time.perf_counter() - started
        pass_diagnostics.append(
            {
                "name": ocr_pass.name,
                "scope": ocr_pass.scope.value,
                "scale": ocr_pass.scale,
                "modes": ocr_pass.modes,
                "recognize_words": any(task.recognize_words for task in tasks),
                "character_confidence_threshold": ocr_pass.character_confidence_threshold,
                "task_count": len(tasks),
                "raster_pixels": raster_pixels,
                "skipped_raster_pixels": skipped_raster_pixels,
                "image_text_preflight": image_text_preflight,
                "region_stage": region_stage,
                "region_boxes": region_boxes,
                "skipped_region_boxes": skipped_region_boxes,
                "full_page_fallback": (
                    region_stage == "page" and ocr_pass.scope is OcrPassScope.PAGE
                ),
                "elapsed_seconds": elapsed,
                "render_timings": trace.render_timings or {},
                "recognition_seconds": sum(
                    task_candidate.recognition_seconds for task_candidate in task_candidates
                ),
                "setup_seconds": sum(
                    task_candidate.setup_seconds for task_candidate in task_candidates
                ),
                "api_seconds": sum(
                    task_candidate.api_seconds for task_candidate in task_candidates
                ),
                "iterator_seconds": sum(
                    task_candidate.iterator_seconds for task_candidate in task_candidates
                ),
                "cleanup_seconds": sum(
                    task_candidate.cleanup_seconds for task_candidate in task_candidates
                ),
                "candidate_seconds": sum(
                    task_candidate.candidate_seconds for task_candidate in task_candidates
                ),
                "recognition_statuses": tuple(
                    task_candidate.recognition_status for task_candidate in task_candidates
                ),
                "accepted_additions": additions,
                "adaptive_retry_scale": adaptive_retry_scale,
                "adaptive_preflight": adaptive_preflight,
                "adaptive_rescue_decision": adaptive_rescue_decision,
                "adaptive_rescue": adaptive_rescue,
                "pixel_budget": ocr_pass.pixel_budget,
                "rectangles": tuple(task.rectangle for task in tasks),
                "selected": False,
                **candidate.metrics.as_record(),
            }
        )
        if not tasks:
            continue
        if ocr_pass.scope is OcrPassScope.WEAK_REGIONS:
            previous_region_additions = additions
            if additions:
                selected_name = ocr_pass.name
                selected = candidate
                selected_tasks = (*selected_tasks, *candidate_source_tasks)
                seeded_region_selected = used_native_seed and ocr_pass.seed_with_native
            continue
        if selected is None or candidate.metrics.utility > (
            selected.metrics.utility * ocr_pass.minimum_utility_gain
        ):
            selected_name = ocr_pass.name
            selected = candidate
            selected_tasks = candidate_source_tasks

    if selected is None:
        internal_record_candidates(tuple(candidates), selected_name, trace)
        return ObservationBatch.empty()
    for diagnostic in pass_diagnostics:
        diagnostic["selected"] = diagnostic["name"] == selected_name
    internal_record_candidates(tuple(candidates), selected_name, trace)
    if selected_tasks:
        # Ruled scanned tables defeat Tesseract's page segmentation; when the
        # page raster shows a full ruling grid, re-recognize cell by cell and
        # let the grid text replace the page-segmented text inside the grid.
        source_task = max(
            selected_tasks,
            key=lambda task: task.rectangle[2] * task.rectangle[3],
        )
        grid = internal_detect_ruling_grid(source_task.image)
        if grid is not None and internal_grid_is_regular_table(
            grid, selected.observations, source_task
        ):
            x_lines, y_lines, source_samples, slope = grid
            cell_tasks = internal_grid_cell_tasks(
                source_task, x_lines, y_lines, source_samples, slope
            )
            if len(cell_tasks) >= internal_GRID_MIN_CELLS:
                cell_candidate = internal_merge_candidate_batches(recognize_tasks(cell_tasks))
                cell_observations = internal_grid_row_observations(cell_candidate.observations)
                if len(cell_observations):
                    grid_box = internal_grid_region_page_box(source_task, x_lines, y_lines)
                    prior = selected.observations
                    centers_x = (prior.bbox[:, 0] + prior.bbox[:, 2]) * 0.5
                    centers_y = (prior.bbox[:, 1] + prior.bbox[:, 3]) * 0.5
                    outside = ~(
                        (centers_x >= grid_box[0])
                        & (centers_x <= grid_box[2])
                        & (centers_y >= grid_box[1])
                        & (centers_y <= grid_box[3])
                    )
                    replaced_alnum = sum(
                        sum(character.isalnum() for character in prior.text[index])
                        for index in numpy.flatnonzero(~outside)
                    )
                    cell_alnum = sum(
                        sum(character.isalnum() for character in text)
                        for text in cell_observations.text
                    )
                    if cell_alnum < replaced_alnum * 0.8:
                        # The page-segmented reads carried more content than
                        # the cell reads; this grid's cells recognize worse
                        # than whole-page OCR, so keep the original.
                        return selected.observations
                    retained = prior.take(numpy.flatnonzero(outside))
                    trace.grid_cell_ocr = {
                        "cells": len(cell_tasks),
                        "cell_observations": len(cell_observations),
                        "replaced_observations": int(numpy.count_nonzero(~outside)),
                        "grid_box": grid_box,
                        "columns": len(x_lines) - 1,
                        "rows": len(y_lines) - 1,
                    }
                    return ObservationBatch.concatenate(retained, cell_observations)
    return selected.observations
