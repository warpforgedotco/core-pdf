# SPDX-License-Identifier: AGPL-3.0-only
"""Passive soft-mask selection retained by the extended graphics state."""

from dataclasses import dataclass

from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_08_graphics.matrix import Matrix


@dataclass(frozen=True, slots=True)
class SoftMaskSelection:
    dictionary: PdfDict
    ctm: Matrix
    resources: PdfDict
    source_key: tuple[str, int, int]
