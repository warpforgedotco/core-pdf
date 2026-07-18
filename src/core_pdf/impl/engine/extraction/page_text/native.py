# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import re
from typing import Any

from core_pdf.impl.engine.extraction.common import page_profile
from core_pdf.impl.engine.extraction.common.render import (
    render_page_observation_lines,
    render_resolved_text_lines,
    resolved_text_lines_for_output,
)

MIN_VISIBLE_GLYPH_COVERAGE = 0.75


def native_text_runs_for_extraction(runs: list[Any]) -> list[Any]:
    """Return painted PDF text runs, excluding inactive or empty runs."""
    extractable = [
        run
        for run in runs
        if getattr(run, "text", "")
        and (getattr(run, "visible", True) or text_run_uses_invisible_render_mode(run))
        and text_run_is_inside_active_clip(run)
    ]
    invisible = [run for run in extractable if text_run_uses_invisible_render_mode(run)]
    painted = [run for run in extractable if not text_run_uses_invisible_render_mode(run)]
    if invisible and painted and invisible_text_duplicates_painted_text(invisible, painted):
        return painted
    return extractable


def invisible_text_duplicates_painted_text(invisible: list[Any], painted: list[Any]) -> bool:
    invisible_text = normalized_run_text(invisible)
    painted_text = normalized_run_text(painted)
    if not invisible_text or not painted_text:
        return False
    if invisible_text == painted_text:
        return True
    shorter, longer = sorted((invisible_text, painted_text), key=len)
    return len(shorter) >= len(longer) * 0.8 and shorter in longer


def normalized_run_text(runs: list[Any]) -> str:
    return " ".join(
        token.casefold()
        for run in runs
        for token in re.findall(r"\w+", str(getattr(run, "text", "")))
    )


def text_run_uses_invisible_render_mode(run: Any) -> bool:
    return getattr(run, "visible", True) is False and dict(getattr(run, "provenance", ())).get(
        "text_render_mode"
    ) in (3, "3")


def text_run_is_inside_active_clip(run: Any) -> bool:
    return bool(getattr(run, "inside_active_clip", True))


def native_text_runs_inside_page_bounds(
    runs: list[Any],
    media_box: tuple[float, float, float, float] | None,
    *,
    rotate: int = 0,
) -> list[Any]:
    if not runs or media_box is None or rotate % 360 != 0:
        return runs
    page_x0, page_y0, page_x1, page_y1 = media_box
    filtered: list[Any] = []
    for run in runs:
        x0, y0, x1, y1 = run.x0, run.y0, run.x1, run.y1
        width = max(0.0, x1 - x0)
        height = max(0.0, y1 - y0)
        if width * height == 0:
            if page_x0 <= x0 <= page_x1 and page_y0 <= y0 <= page_y1:
                filtered.append(run)
            continue
        intersection = max(0.0, min(x1, page_x1) - max(x0, page_x0)) * max(
            0.0, min(y1, page_y1) - max(y0, page_y0)
        )
        if intersection / (width * height) >= MIN_VISIBLE_GLYPH_COVERAGE:
            filtered.append(run)
            continue
        vertical_coverage = max(0.0, min(y1, page_y1) - max(y0, page_y0)) / height
        horizontally_anchored = page_x0 <= x0 <= page_x1 or page_x0 <= x1 <= page_x1
        if (
            vertical_coverage >= MIN_VISIBLE_GLYPH_COVERAGE
            and horizontally_anchored
            and width >= (page_x1 - page_x0) * 0.5
            and sum(character.isalnum() for character in str(run.text)) >= 8
        ):
            filtered.append(run)
    return filtered


def extract_native_text(page: Any) -> tuple[str, tuple[Any, ...]]:
    profile = page_profile.get_page_profile(page)
    runs = native_text_runs_for_extraction(list(page.chars))
    runs = native_text_runs_inside_page_bounds(runs, page.media_box, rotate=page.rotation)
    lines = render_page_observation_lines(
        runs,
        rotate=page.rotation,
        media_box=page.media_box,
        layout=True,
    )
    output_lines = resolved_text_lines_for_output(lines)
    text = render_resolved_text_lines(output_lines)
    cache = getattr(page, "extraction_cache", None)
    if cache is not None:
        cache["native_text_profile"] = profile
        cache["native_output_lines"] = output_lines
    return text, output_lines
