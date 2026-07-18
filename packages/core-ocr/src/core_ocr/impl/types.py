# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TypeAlias

OcrRow: TypeAlias = dict[str, object]
OcrAffineTransform: TypeAlias = tuple[float, float, float, float, float, float]


def ocr_int_value(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str, bytes, bytearray)):
        return int(value)
    raise TypeError(f"expected int-compatible OCR value, got {type(value).__name__}")


def ocr_float_value(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str, bytes, bytearray)):
        return float(value)
    raise TypeError(f"expected float-compatible OCR value, got {type(value).__name__}")


TESSERACT_OEM_LSTM_ONLY = 1
TESSERACT_RIL_BLOCK = 0
TESSERACT_RIL_PARA = 1
TESSERACT_RIL_TEXTLINE = 2
TESSERACT_RIL_WORD = 3
TESSERACT_RIL_SYMBOL = 4
TESSERACT_LIBRARY_NAMES = ("tesseract", "libtesseract")
LEPTONICA_LIBRARY_NAMES = ("lept", "liblept", "leptonica", "libleptonica")
TESSERACT_LIBRARY_FILENAMES = (
    "libtesseract.dylib",
    "libtesseract.so",
    "libtesseract.so.5",
)
LEPTONICA_LIBRARY_FILENAMES = (
    "libleptonica.dylib",
    "liblept.dylib",
    "libleptonica.so",
    "liblept.so",
)
TESSERACT_DEFAULT_VARIABLES: Mapping[str, str] = {
    "classify_bln_numeric_mode": "0",
    "load_freq_dawg": "1",
    "load_system_dawg": "1",
    "preserve_interword_spaces": "0",
    "tessedit_char_blacklist": "",
    "tessedit_char_unblacklist": "",
    "tessedit_char_whitelist": "",
    "textord_tablefind_recognize_tables": "0",
    "textord_tabfind_find_tables": "0",
    "thresholding_kfactor": "0.34",
    "thresholding_method": "0",
    "thresholding_window_size": "0.33",
}
LEPTONICA_PIX_MAX_BYTES = (1 << 31) - 1
LEPTONICA_PIX_COLOR_BYTES_PER_PIXEL = 4
BMP_MAX_BYTES = (1 << 32) - 1


def leptonica_pix_estimated_bytes(
    width: int,
    height: int,
    bytes_per_pixel: int = LEPTONICA_PIX_COLOR_BYTES_PER_PIXEL,
) -> int:
    if width <= 0 or height <= 0 or bytes_per_pixel <= 0:
        return 0
    return width * height * bytes_per_pixel


def leptonica_pix_size_is_supported(
    width: int,
    height: int,
    bytes_per_pixel: int = LEPTONICA_PIX_COLOR_BYTES_PER_PIXEL,
) -> bool:
    estimated = leptonica_pix_estimated_bytes(width, height, bytes_per_pixel)
    return 0 < estimated < LEPTONICA_PIX_MAX_BYTES


def raw_ocr_image_size_is_supported(
    width: int,
    height: int,
    bytes_per_pixel: int,
    bytes_per_line: int,
) -> bool:
    if width <= 0 or height <= 0 or bytes_per_pixel <= 0 or bytes_per_line <= 0:
        return False
    estimated = (height - 1) * bytes_per_line + width * bytes_per_pixel
    return 0 < estimated < LEPTONICA_PIX_MAX_BYTES


@dataclass(frozen=True)
class OcrImage:
    data: bytes
    width: int
    height: int
    bytes_per_pixel: int
    bytes_per_line: int
    encoded: bytes | None = None
    source: str = "rgba"
    cache_key: Any | None = None
    target_width: int | None = None
    target_height: int | None = None
    resolution: int | None = None
    clockwise_quarter_turns: int = 0
    page_bbox: tuple[float, float, float, float] | None = None
    page_clockwise_quarter_turns: int = 0


@dataclass(frozen=True)
class OcrComponentBox:
    level: int
    index: int
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class OcrTextChoice:
    text: str
    confidence: int | None


@dataclass(frozen=True)
class OcrObservation:
    text: str
    level: int
    confidence: int | None
    bbox: tuple[int, int, int, int]
    baseline: tuple[int, int, int, int] | None = None
    page_bbox: tuple[float, float, float, float] | None = None
    page_baseline: tuple[float, float, float, float] | None = None
    source: str = ""
    image_width: int | None = None
    image_height: int | None = None
    image_resolution: int | None = None
    page_num: int = 1
    block_num: int = 1
    par_num: int = 1
    line_num: int = 1
    word_num: int = 0
    symbol_num: int = 0
    token_type: str | None = None
    choices: tuple[OcrTextChoice, ...] = ()
    provenance: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class OcrDeskewInfo:
    source: str
    angle_degrees: float | None
    confidence: float | None
    applied: bool
    reason: str
    image_width: int
    image_height: int
    source_to_deskew: OcrAffineTransform
    deskew_to_source: OcrAffineTransform


@dataclass(frozen=True)
class OcrTextResult:
    text: str
    confidence: int | None
    line_rows: tuple[OcrRow, ...] = ()
    word_rows: tuple[OcrRow, ...] = ()
    symbol_rows: tuple[OcrRow, ...] = ()
    component_boxes: tuple[OcrComponentBox, ...] = ()
    observations: tuple[OcrObservation, ...] = ()
    deskew_info: OcrDeskewInfo | None = None


@dataclass(frozen=True)
class OcrIteratorLayout:
    textline_rows: list[OcrRow]
    word_rows: list[OcrRow]
    symbol_rows: list[OcrRow]
    text: str = ""
    confidence: int | None = None


def ocr_observation_from_row(
    row: Mapping[str, object],
    *,
    source: str = "",
    image_width: int | None = None,
    image_height: int | None = None,
    image_resolution: int | None = None,
    page_bbox: tuple[float, float, float, float] | None = None,
    page_baseline: tuple[float, float, float, float] | None = None,
    token_type: str | None = None,
) -> OcrObservation | None:
    text = str(row.get("text", "")).strip()
    if not text:
        return None
    try:
        left = ocr_int_value(row["left"])
        top = ocr_int_value(row["top"])
        width = ocr_int_value(row["width"])
        height = ocr_int_value(row["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    confidence = row.get("conf")
    try:
        confidence_value = (
            max(0, min(100, ocr_int_value(confidence))) if confidence is not None else None
        )
    except (TypeError, ValueError):
        confidence_value = None
    baseline = row.get("baseline")
    baseline_value: tuple[int, int, int, int] | None = None
    if isinstance(baseline, (list, tuple)) and len(baseline) == 4:
        try:
            baseline_value = (
                ocr_int_value(baseline[0]),
                ocr_int_value(baseline[1]),
                ocr_int_value(baseline[2]),
                ocr_int_value(baseline[3]),
            )
        except (TypeError, ValueError):
            baseline_value = None
    if page_bbox is None:
        row_page_bbox = row.get("page_bbox")
        if isinstance(row_page_bbox, (list, tuple)) and len(row_page_bbox) == 4:
            try:
                page_bbox = (
                    ocr_float_value(row_page_bbox[0]),
                    ocr_float_value(row_page_bbox[1]),
                    ocr_float_value(row_page_bbox[2]),
                    ocr_float_value(row_page_bbox[3]),
                )
            except (TypeError, ValueError):
                page_bbox = None
    if page_baseline is None:
        row_page_baseline = row.get("page_baseline")
        if isinstance(row_page_baseline, (list, tuple)) and len(row_page_baseline) == 4:
            try:
                page_baseline = (
                    ocr_float_value(row_page_baseline[0]),
                    ocr_float_value(row_page_baseline[1]),
                    ocr_float_value(row_page_baseline[2]),
                    ocr_float_value(row_page_baseline[3]),
                )
            except (TypeError, ValueError):
                page_baseline = None
    if token_type is None:
        row_token_type = row.get("token_type")
        token_type = str(row_token_type) if row_token_type else None
    raw_choices = row.get("choices", ())
    choice_rows = raw_choices if isinstance(raw_choices, (list, tuple)) else ()
    choices = tuple(choice for choice in choice_rows if isinstance(choice, OcrTextChoice))
    return OcrObservation(
        text=text,
        level=ocr_int_value(row.get("level", TESSERACT_RIL_WORD)),
        confidence=confidence_value,
        bbox=(left, top, left + width, top + height),
        baseline=baseline_value,
        page_bbox=page_bbox,
        page_baseline=page_baseline,
        source=source,
        image_width=image_width,
        image_height=image_height,
        image_resolution=image_resolution,
        page_num=ocr_int_value(row.get("page_num", 1)),
        block_num=ocr_int_value(row.get("block_num", 1)),
        par_num=ocr_int_value(row.get("par_num", 1)),
        line_num=ocr_int_value(row.get("line_num", 1)),
        word_num=ocr_int_value(row.get("word_num", 0)),
        symbol_num=ocr_int_value(row.get("symbol_num", 0)),
        token_type=token_type,
        choices=choices,
    )


def ocr_observations_from_rows(
    rows: list[OcrRow] | tuple[OcrRow, ...],
    *,
    source: str = "",
    image_width: int | None = None,
    image_height: int | None = None,
    image_resolution: int | None = None,
) -> tuple[OcrObservation, ...]:
    observations: list[OcrObservation] = []
    for row in rows:
        observation = ocr_observation_from_row(
            row,
            source=source,
            image_width=image_width,
            image_height=image_height,
            image_resolution=image_resolution,
        )
        if observation is not None:
            observations.append(observation)
    return tuple(observations)
