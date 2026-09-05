# SPDX-License-Identifier: AGPL-3.0-only
"""Compose recognized text without inferring missing or spurious tokens."""

from __future__ import annotations

from core_pdf.impl._impl.extract import emit as native_emit
from core_pdf.impl._impl.extract.contracts import ParsedBlock
from core_pdf.impl._impl.output.model import Figure, Page, Table
from core_pdf.impl.spec.s_07_content.capture import CapturedDrawing
from core_pdf_ocr.impl.extract.table_reconcile import internal_project_text_and_tables


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
    normalized_blocks = native_emit.internal_normalized_blocks(blocks, drawings)
    normalized_blocks = native_emit.internal_remove_off_page_blocks(
        normalized_blocks,
        width,
        height,
    )
    normalized_blocks, projected_tables = internal_project_text_and_tables(
        normalized_blocks, tables
    )
    return native_emit.internal_compose_page(
        blocks,
        normalized_blocks,
        projected_tables,
        page_number=page_number,
        width=width,
        height=height,
        rotation=rotation,
        route=route,
        figures=figures,
        diagnostics=diagnostics,
        full_page_image=full_page_image,
    )
