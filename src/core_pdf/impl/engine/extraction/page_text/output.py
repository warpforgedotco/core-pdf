# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from core_ocr.impl import postprocess as ocr_postprocess
from core_ocr.impl.vector_text import VectorStrokeOcrResult

from core_pdf.impl.engine.extraction.common import observation_resolver
from core_pdf.impl.engine.extraction.common.render import render_resolved_text_lines

if TYPE_CHECKING:
    from core_ocr.impl.candidates import OcrPageTextResult

    from core_pdf.impl.engine.extraction.page_text.mixin import PageExtractionHost


def append_resolved_supplement_lines(
    text: str,
    current_lines: tuple[observation_resolver.ResolvedTextLine, ...],
    supplement_lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> tuple[str, tuple[observation_resolver.ResolvedTextLine, ...]]:
    if not supplement_lines:
        return text, current_lines
    supplemented_text = text.rstrip() + "\n" + "\n".join(line.text for line in supplement_lines)
    if current_lines and text == render_resolved_text_lines(current_lines):
        return supplemented_text, (*current_lines, *supplement_lines)
    return supplemented_text, ()


def insert_resolved_figure_supplement_lines(
    page: PageExtractionHost,
    text: str,
    current_lines: tuple[observation_resolver.ResolvedTextLine, ...],
    supplement_lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> tuple[str, tuple[observation_resolver.ResolvedTextLine, ...]]:
    if not supplement_lines:
        return text, current_lines
    text_lines = text.splitlines()
    insert_index = ocr_postprocess.figure_caption_insert_index(page, text_lines)
    if insert_index is None:
        return append_resolved_supplement_lines(text, current_lines, supplement_lines)
    supplemented_text = ocr_postprocess.insert_figure_supplemental_lines_near_caption(
        page,
        text,
        [line.text for line in supplement_lines],
    )
    if not current_lines or text != render_resolved_text_lines(current_lines):
        return supplemented_text, ()
    nonempty_before = sum(1 for line in text_lines[:insert_index] if line.strip())
    inserted_lines = with_first_resolved_line_break(supplement_lines, break_before=1)
    return (
        supplemented_text,
        (
            *current_lines[:nonempty_before],
            *inserted_lines,
            *current_lines[nonempty_before:],
        ),
    )


def ocr_result_output_lines(
    page: PageExtractionHost,
    ocr_result: OcrPageTextResult,
    text: str,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    output_lines = ocr_postprocess.ocr_page_result_resolved_lines(page, ocr_result)
    return best_effort_resolved_text_lines(
        text,
        output_lines,
        getattr(ocr_result, "selected_output_lines", ()),
    )


def vector_stroke_result_output_lines(
    vector_result: VectorStrokeOcrResult,
    text: str,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    output_lines = ocr_postprocess.resolved_text_lines_from_geometry_lines(
        vector_result.lines,
        source="vector_stroke",
        kind="vector_text_line",
    )
    return best_effort_resolved_text_lines(text, output_lines)


def best_effort_resolved_text_lines(
    text: str,
    *candidate_sets: tuple[observation_resolver.ResolvedTextLine, ...],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    for lines in candidate_sets:
        if lines and text == render_resolved_text_lines(lines):
            return lines
    text_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not text_lines:
        return ()
    for lines in candidate_sets:
        if not lines or len(lines) != len(text_lines):
            continue
        return replace_resolved_line_text(lines, tuple(text_lines))
    return ()


def replace_resolved_line_text(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    text_lines: tuple[str, ...],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    replaced_lines: list[observation_resolver.ResolvedTextLine] = []
    for line, text in zip(lines, text_lines, strict=True):
        observation = replace(line.observation, text=text)
        replaced_lines.append(
            replace(
                line,
                text=text,
                observation=observation,
                contributing_observations=(observation,),
            )
        )
    return tuple(replaced_lines)


def with_first_resolved_line_break(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    break_before: int,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if not lines:
        return ()
    first, *rest = lines
    return (replace(first, break_before=break_before), *rest)


__all__ = (
    "append_resolved_supplement_lines",
    "best_effort_resolved_text_lines",
    "insert_resolved_figure_supplement_lines",
    "ocr_result_output_lines",
    "replace_resolved_line_text",
    "vector_stroke_result_output_lines",
    "with_first_resolved_line_break",
)
