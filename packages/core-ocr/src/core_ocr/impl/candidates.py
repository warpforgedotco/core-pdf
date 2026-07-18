# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core_ocr.impl.types import OcrImage, OcrTextResult


@dataclass(frozen=True)
class OcrCandidate:
    name: str
    result: OcrTextResult
    bbox: tuple[int, int, int, int] | None = None
    region_count: int = 0
    image_width: int | None = None
    image_height: int | None = None
    image_resolution: int | None = None
    page_width: float | None = None
    page_height: float | None = None
    page_bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class OcrPageTextResult:
    text: str
    candidate: OcrCandidate | None = None
    candidates: tuple[OcrCandidate, ...] = ()
    output_lines: tuple[Any, ...] = ()
    selected_output_lines: tuple[Any, ...] = ()
    verification_candidates: tuple[OcrCandidate, ...] = ()
    preserve_raw_text: bool = False


def full_page_ocr_candidate_name(image: OcrImage, index: int) -> str:
    if index == 0:
        return "full_page_image"
    suffix = re.sub(r"[^a-z0-9]+", "_", image.source.casefold()).strip("_")
    suffix = suffix.removeprefix("full_page_") or f"variant_{index}"
    return f"full_page_image_{index}_{suffix}"
