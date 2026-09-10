# SPDX-License-Identifier: AGPL-3.0-only
"""Passive pattern selections defined by PDF pattern dictionaries."""

from dataclasses import dataclass
from typing import TypeAlias

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_08_graphics.matrix import Matrix
from core_pdf_spec.types import Rectangle


@dataclass(frozen=True, slots=True)
class ShadingPattern:
    dictionary: PdfDict


@dataclass(frozen=True, slots=True)
class TilingPattern:
    bbox: Rectangle
    x_step: float
    y_step: float
    stream: PdfStream
    resources: PdfDict
    matrix: Matrix
    paint_type: int
    base_color: tuple[float, ...] | None


PatternPaint: TypeAlias = ShadingPattern | TilingPattern


__all__ = ("PatternPaint", "ShadingPattern", "TilingPattern")
