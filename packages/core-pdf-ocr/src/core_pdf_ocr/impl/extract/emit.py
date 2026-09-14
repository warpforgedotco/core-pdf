# SPDX-License-Identifier: AGPL-3.0-only
"""Compose recognized text without inferring missing or spurious tokens."""

from __future__ import annotations

from core_pdf.impl._impl.capture.records import CapturedDrawing
from core_pdf.impl._impl.extract import emit as native_emit
from core_pdf.impl._impl.extract.contracts import ParsedBlock
from core_pdf.impl._impl.output.model import Figure, Page, Table
from core_pdf_ocr.impl.extract.table_reconcile import internal_remove_duplicate_tables


def assemble_page(
    blocks: tuple[ParsedBlock, ...],
    *,
    page_number: int,
    width: float,
    height: float,
    rotation: int,
    route: str,
    tables: tuple[Table, ...] = (),
    figures: tuple[Figure, ...] = (),
    diagnostics: tuple[str, ...] = (),
    full_page_image: bool = False,
    drawings: tuple[CapturedDrawing, ...] = (),
) -> Page:
    """Assemble the page natively after removing recognized chart table copies."""
    return native_emit.assemble_page(
        blocks,
        page_number=page_number,
        width=width,
        height=height,
        rotation=rotation,
        route=route,
        tables=internal_remove_duplicate_tables(tables),
        figures=figures,
        diagnostics=diagnostics,
        full_page_image=full_page_image,
        drawings=drawings,
    )
