# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from typing import TYPE_CHECKING, Any

from core_pdf.impl.engine.extraction.cache import ExtractionCacheMapping
from core_pdf.impl.engine.extraction.common import page_profile
from core_pdf.impl.engine.extraction.common.ordering import LayoutAnalyzer
from core_pdf.impl.engine.extraction.common.render import (
    render_page_observation_lines,
    render_page_text,
    render_resolved_text_lines,
)
from core_pdf.impl.engine.extraction.ocr import (
    page_analysis as ocr_page_analysis,
)
from core_pdf.impl.engine.extraction.ocr import (
    postprocess as ocr_postprocess,
)
from core_pdf.impl.engine.extraction.ocr import (
    rendering as ocr_rendering,
)
from core_pdf.impl.engine.extraction.ocr import (
    schematic as ocr_schematic,
)
from core_pdf.impl.engine.extraction.ocr import (
    text_analysis as ocr_text_analysis,
)
from core_pdf.impl.engine.extraction.ocr.glyph_recognizer import (
    repair_text_runs_with_glyph_bitmaps,
    text_runs_from_rendered_glyphs,
)
from core_pdf.impl.engine.extraction.ocr.text_analysis import (
    extracted_text_token_count,
    sparse_text_looks_noisy,
    text_ocr_quality_score,
    uninterpretable_char_count,
)
from core_pdf.impl.engine.extraction.page_text.policy import classify_page_region
from core_pdf.impl.engine.layout.geometry_quality import (
    LayoutGeometrySummary,
    layout_geometry_summary_record,
    page_layout_geometry_summary,
    text_run_has_repairable_glyph_geometry_issue,
)

if TYPE_CHECKING:
    from core_pdf.impl.engine.layout.models import TextRun


def try_extract_native_text_fast(
    page: Any,
    profile: page_profile.PageProfile,
    cache: ExtractionCacheMapping,
) -> str | None:
    if ocr_postprocess.ocr_is_enabled():
        return None
    if page.rotation % 360 != 0:
        return None
    if profile.has_xobject_ops:
        xobject_reason = native_text_fast_xobject_fallback_reason(profile)
        if xobject_reason is not None:
            return None
    if profile.has_inline_images:
        return None

    chars = native_text_runs_for_extraction(page.chars)
    chars = native_text_runs_inside_page_bounds(
        chars,
        page.media_box,
        rotate=page.rotation,
    )
    chars = native_text_runs_inside_visible_row_bands(
        chars,
        page.media_box,
        page,
    )
    native_output_lines = render_page_observation_lines(
        chars,
        rotate=page.rotation,
        media_box=page.media_box,
        layout=True,
    )
    text = render_resolved_text_lines(native_output_lines)
    selected_text = select_native_text_layout(
        chars,
        text,
        rotate=page.rotation,
        media_box=page.media_box,
    )
    if selected_text != text:
        native_output_lines = ()
    text = selected_text
    path_reason = native_text_fast_path_fallback_reason(
        profile,
        chars=chars,
        text=text,
        media_box=page.media_box,
    )
    if path_reason is not None:
        return None
    if should_try_rendered_glyph_repair(chars, text):
        chars, text, native_output_lines = apply_rendered_glyph_repair_to_native_text(
            page,
            chars,
            text,
            native_output_lines,
        )

    native_geometry_summary = native_layout_geometry_summary_for_runs(chars)
    cache["native_layout_geometry_summary"] = layout_geometry_summary_record(
        native_geometry_summary
    )
    cache["page_region_classification"] = classify_page_region(
        text,
        page=page,
        native_runs=chars,
        media_box=page.media_box,
        include_dominant_image=False,
    )
    text = ocr_text_analysis.repair_formula_control_delimiters(text)
    return text


def apply_rendered_glyph_repair_to_native_text(
    page: Any,
    chars: list[TextRun],
    text: str,
    native_output_lines: tuple[Any, ...],
) -> tuple[list[TextRun], str, tuple[Any, ...]]:
    rendered = None
    try:
        rendered = ocr_rendering.rendered_page_for_ocr_render(
            page,
            source="glyph_repair",
        )
        chars = repair_text_runs_with_glyph_bitmaps(
            chars,
            rendered,
            repair_contextual_punctuation=(text_ocr_quality_score(text) >= 0.4),
        )
        native_output_lines = render_page_observation_lines(
            chars,
            rotate=page.rotation,
            media_box=page.media_box,
            layout=True,
        )
        text = render_resolved_text_lines(native_output_lines)
        selected_text = select_native_text_layout(
            chars,
            text,
            rotate=page.rotation,
            media_box=page.media_box,
        )
        if selected_text != text:
            native_output_lines = ()
        text = selected_text
    except Exception:
        rendered = None
    if rendered is not None and should_try_rendered_glyph_text(text):
        glyph_chars = text_runs_from_rendered_glyphs(rendered)
        if glyph_chars:
            glyph_output_lines = render_page_observation_lines(
                glyph_chars,
                rotate=page.rotation,
                media_box=page.media_box,
                layout=True,
            )
            glyph_text = render_resolved_text_lines(glyph_output_lines)
            if should_use_rendered_glyph_candidate(
                text,
                chars,
                glyph_text,
                glyph_chars,
            ):
                text = glyph_text
                native_output_lines = glyph_output_lines
    return chars, text, native_output_lines


def native_text_fast_xobject_fallback_reason(
    profile: page_profile.PageProfile,
) -> str | None:
    resources = profile.resources
    if resources.form_xobject_count:
        return "form_xobject_ops"
    if resources.unknown_xobject_count:
        return "unknown_xobject_ops"
    if resources.image_xobject_count:
        return None
    return "xobject_ops"


def native_text_fast_path_fallback_reason(
    profile: page_profile.PageProfile,
    *,
    chars: list[TextRun],
    text: str,
    media_box: tuple[float, float, float, float] | None,
) -> str | None:
    if not profile.has_path_ops:
        return None
    total_path_ops = sum(stream.path_ops for stream in profile.content_streams)
    total_clip_ops = sum(stream.clip_ops for stream in profile.content_streams)
    if total_clip_ops:
        if total_clip_ops == 1 and total_path_ops <= 2:
            return None
        return "complex_path_ops"
    if profile.likely_table_page or total_path_ops >= 12:
        return "table_or_vector_path_ops"
    if ocr_schematic.should_try_vector_table_symbol_supplement(
        text,
        chars,
        media_box,
    ):
        return "table_or_vector_path_ops"
    if total_path_ops > 6:
        return "complex_path_ops"
    return None


def select_native_text_layout(
    chars: list[TextRun],
    layout_text: str,
    *,
    rotate: int,
    media_box: Any,
) -> str:
    if not layout_text:
        return layout_text
    layout_tokens = extracted_text_token_count(layout_text)
    if layout_tokens < 120:
        return layout_text
    layout_quality = text_ocr_quality_score(layout_text)
    if layout_quality < 0.22:
        return layout_text
    linear_text = render_page_text(
        chars,
        rotate=rotate,
        media_box=media_box,
        layout=False,
    )
    if not linear_text:
        return layout_text
    linear_tokens = extracted_text_token_count(linear_text)
    if linear_tokens < max(80, int(layout_tokens * 0.65)):
        return layout_text
    linear_quality = text_ocr_quality_score(linear_text)
    if linear_quality + 0.01 < layout_quality:
        return linear_text
    return layout_text


def native_text_runs_inside_page_lower_bound(
    runs: list[TextRun],
    media_box: tuple[float, float, float, float] | None,
    *,
    rotate: int = 0,
) -> list[TextRun]:
    if not runs or media_box is None or rotate % 360 != 0:
        return runs
    page_height = media_box[3] - media_box[1]
    if page_height <= 0:
        return runs
    lower_bound = media_box[1] - max(page_height * 0.05, 12.0)
    if not any(run.y1 < lower_bound for run in runs):
        return runs
    return [run for run in runs if run.y1 >= lower_bound]


def native_text_runs_inside_page_bounds(
    runs: list[TextRun],
    media_box: tuple[float, float, float, float] | None,
    *,
    rotate: int = 0,
) -> list[TextRun]:
    """Drop text whose glyph box is substantially outside the declared page."""
    if not runs or media_box is None or rotate % 360 != 0:
        return runs
    page_x0, page_y0, page_x1, page_y1 = media_box
    if page_x1 <= page_x0 or page_y1 <= page_y0:
        return runs
    filtered: list[TextRun] = []
    for run in runs:
        width = max(0.0, run.x1 - run.x0)
        height = max(0.0, run.y1 - run.y0)
        area = width * height
        if area <= 0.0:
            continue
        intersection_width = max(0.0, min(run.x1, page_x1) - max(run.x0, page_x0))
        intersection_height = max(0.0, min(run.y1, page_y1) - max(run.y0, page_y0))
        if intersection_width * intersection_height / area >= 0.80:
            filtered.append(run)
    return filtered


def native_text_runs_for_extraction(runs: list[TextRun]) -> list[TextRun]:
    """Select painted text and non-duplicative invisible OCR text layers."""
    cleaned = ocr_page_analysis.native_text_runs_without_corrupt_control_text(runs)
    extractable = [
        run
        for run in cleaned
        if (run.visible or text_run_uses_invisible_render_mode(run))
        and text_run_is_inside_active_clip(run)
    ]
    if not extractable:
        return []
    invisible = [run for run in extractable if text_run_uses_invisible_render_mode(run)]
    if not invisible:
        return extractable
    painted = [run for run in extractable if not text_run_uses_invisible_render_mode(run)]
    if not painted or not invisible_text_layer_duplicates_painted_text(invisible, painted):
        return extractable
    return painted


def native_invisible_text_layer_is_trustworthy(runs: list[TextRun], text: str) -> bool:
    """Return whether a substantive invisible text layer should outrank fresh OCR."""
    if not runs or not text.strip():
        return False
    all_tokens = native_run_token_counter(runs)
    token_count = sum(all_tokens.values())
    if token_count < 100:
        return False
    invisible_tokens = native_run_token_counter(
        [run for run in runs if text_run_uses_invisible_render_mode(run)]
    )
    invisible_token_count = sum(invisible_tokens.values())
    if invisible_token_count / token_count < 0.80:
        return False
    invisible_run_count = sum(text_run_uses_invisible_render_mode(run) for run in runs)
    if invisible_token_count / invisible_run_count > 4.0:
        return False
    if ocr_text_analysis.text_ocr_quality_score(text) > 0.20:
        return False
    return ocr_text_analysis.scanned_ocr_artifact_score(text) <= 0.10


def text_run_uses_invisible_render_mode(run: TextRun) -> bool:
    for name, value in run.provenance:
        if name == "text_render_mode":
            return value == 3
    return False


def text_run_is_inside_active_clip(run: TextRun) -> bool:
    clip_bbox: tuple[float, float, float, float] | None = None
    for name, value in run.provenance:
        if name != "clip_bbox" or not isinstance(value, tuple) or len(value) != 4:
            continue
        x0, y0, x1, y1 = value
        if (
            isinstance(x0, int | float)
            and isinstance(y0, int | float)
            and isinstance(x1, int | float)
            and isinstance(y1, int | float)
        ):
            clip_bbox = (float(x0), float(y0), float(x1), float(y1))
            break
    if clip_bbox is None:
        return True
    width = max(0.0, run.x1 - run.x0)
    height = max(0.0, run.y1 - run.y0)
    area = width * height
    if area <= 0.0:
        return False
    intersection_width = max(0.0, min(run.x1, clip_bbox[2]) - max(run.x0, clip_bbox[0]))
    intersection_height = max(0.0, min(run.y1, clip_bbox[3]) - max(run.y0, clip_bbox[1]))
    return intersection_width * intersection_height / area >= 0.80


def invisible_text_layer_duplicates_painted_text(
    invisible: list[TextRun], painted: list[TextRun]
) -> bool:
    invisible_tokens = native_run_token_counter(invisible)
    painted_tokens = native_run_token_counter(painted)
    smaller_count = min(sum(invisible_tokens.values()), sum(painted_tokens.values()))
    if smaller_count < 12:
        return False
    common_count = sum((invisible_tokens & painted_tokens).values())
    return common_count >= 8 and common_count / smaller_count >= 0.20


def native_run_token_counter(runs: list[TextRun]) -> Counter[str]:
    tokens: Counter[str] = Counter()
    for run in runs:
        tokens.update(ocr_text_analysis.normalized_text_tokens(run.text))
    return tokens


def native_text_runs_inside_visible_row_bands(
    runs: list[TextRun],
    media_box: tuple[float, float, float, float] | None,
    page: Any,
) -> list[TextRun]:
    if not runs or media_box is None:
        return runs
    page_height = media_box[3] - media_box[1]
    page_width = media_box[2] - media_box[0]
    if page_height <= 0 or page_width <= 0:
        return runs
    upper_bound = media_box[3] + max(page_height * 0.05, 12.0)
    if not any(run.y0 > upper_bound for run in runs):
        return runs
    row_bands = visible_gray_row_bands(page, media_box)
    if len(row_bands) < 2:
        return runs
    active_candidates = [band for band in row_bands if band[1] <= media_box[3] - page_height * 0.1]
    if len(active_candidates) < 2:
        return runs
    active_top = max(band[1] for band in active_candidates)
    active_tolerance = max(4.0, page_height * 0.01)
    row_bands.sort(key=lambda band: band[0])
    band_seqnos = [band[0] for band in row_bands]
    filtered: list[TextRun] = []
    for run in runs:
        if run.y0 <= upper_bound or is_likely_mispositioned_page_number(run, media_box):
            filtered.append(run)
            continue
        band = nearest_visible_row_band(run.seqno, row_bands, band_seqnos)
        if band is not None and band[1] <= active_top + active_tolerance:
            filtered.append(run)
    if len(filtered) < len(runs):
        cache = getattr(page, "extraction_cache", None)
        if isinstance(cache, dict):
            cache["native_visible_row_band_filter_applied"] = True
    return filtered


def visible_gray_row_bands(
    page: Any, media_box: tuple[float, float, float, float]
) -> list[tuple[int, float]]:
    page_height = media_box[3] - media_box[1]
    page_width = media_box[2] - media_box[0]
    try:
        drawings = page.get_graphics().drawings
    except Exception:
        return []
    bands: list[tuple[int, float]] = []
    min_width = page_width * 0.6
    min_y = media_box[1] - page_height * 0.1
    max_y = media_box[3] + page_height * 0.5
    for drawing in drawings:
        rect = getattr(drawing, "rect", None)
        fill = getattr(drawing, "fill", None)
        if rect is None or not is_gray_row_fill(fill):
            continue
        if rect.x1 - rect.x0 < min_width:
            continue
        if rect.y0 < min_y or rect.y1 > max_y:
            continue
        bands.append((getattr(drawing, "seqno", -1), rect.y1))
    return bands


def is_gray_row_fill(fill: Any) -> bool:
    return type(fill) is tuple and len(fill) == 1 and 0.65 <= fill[0] <= 0.99


def nearest_visible_row_band(
    seqno: int, row_bands: list[tuple[int, float]], band_seqnos: list[int]
) -> tuple[int, float] | None:
    index = bisect_left(band_seqnos, seqno)
    best: tuple[int, tuple[int, float]] | None = None
    for candidate_index in (index - 1, index, index + 1):
        if not 0 <= candidate_index < len(row_bands):
            continue
        band = row_bands[candidate_index]
        distance = abs(band[0] - seqno)
        if distance > 96:
            continue
        if best is None or distance < best[0]:
            best = (distance, band)
    return best[1] if best is not None else None


def is_likely_mispositioned_page_number(
    run: TextRun, media_box: tuple[float, float, float, float]
) -> bool:
    text = run.text.strip()
    if run.seqno > 24 or not text or len(text) > 16:
        return False
    page_width = media_box[2] - media_box[0]
    if run.x0 < media_box[2] - page_width * 0.28:
        return False
    if sum(1 for ch in text if ch.isdigit()) < 1:
        return False
    return all(
        ch.isdigit() or ch.isalpha() or ch.isspace() or ch in "./-\u2013\u2014" for ch in text
    )


def should_try_rendered_glyph_repair(runs: list[TextRun], text: str) -> bool:
    if not text.strip():
        return False
    if should_try_rendered_glyph_text(text):
        return True
    return any(text_run_has_repairable_glyph_geometry_issue(run) for run in runs)


def native_layout_geometry_summary_for_runs(
    runs: list[TextRun],
) -> LayoutGeometrySummary:
    lines = LayoutAnalyzer.cluster_into_lines([run for run in runs if run.text])
    return page_layout_geometry_summary(lines)


def should_try_rendered_glyph_text(text: str) -> bool:
    tokens = extracted_text_token_count(text)
    if tokens < 20:
        return False
    return uninterpretable_char_count(text) > 0 or sparse_text_looks_noisy(text)


def should_use_rendered_glyph_text(current: str, glyph_text: str) -> bool:
    if not glyph_text:
        return False
    current_tokens = extracted_text_token_count(current)
    glyph_tokens = extracted_text_token_count(glyph_text)
    if glyph_tokens < max(20, current_tokens):
        return False
    current_quality = text_ocr_quality_score(current)
    glyph_quality = text_ocr_quality_score(glyph_text)
    current_fragmentation = ocr_text_analysis.rendered_ocr_fragmentation_score(current)
    glyph_fragmentation = ocr_text_analysis.rendered_ocr_fragmentation_score(glyph_text)
    current_garbled_ratio = garbled_alpha_token_ratio(current)
    glyph_garbled_ratio = garbled_alpha_token_ratio(glyph_text)
    if glyph_garbled_ratio > current_garbled_ratio + 0.04 and glyph_garbled_ratio >= 0.08:
        return False
    if (
        glyph_quality > current_quality + 0.06
        and glyph_fragmentation > current_fragmentation + 0.04
    ):
        return False
    current_noise = uninterpretable_char_count(current)
    glyph_noise = uninterpretable_char_count(glyph_text)
    if current_noise and glyph_noise < current_noise:
        return True
    if sparse_text_looks_noisy(current) and not sparse_text_looks_noisy(glyph_text):
        if glyph_quality <= current_quality + 0.02:
            return True
        if (
            glyph_tokens >= current_tokens * 1.8
            and glyph_quality <= current_quality + 0.06
            and glyph_fragmentation <= current_fragmentation + 0.04
        ):
            return True
    return False


def should_use_rendered_glyph_candidate(
    current: str,
    current_runs: list[Any],
    glyph_text: str,
    glyph_runs: list[Any],
) -> bool:
    if not should_use_rendered_glyph_text(current, glyph_text):
        return False
    current_score = text_run_geometry_quality_score(current_runs)
    glyph_score = text_run_geometry_quality_score(glyph_runs)
    current_tokens = extracted_text_token_count(current)
    glyph_tokens = extracted_text_token_count(glyph_text)
    if glyph_score + 0.12 < current_score and glyph_tokens <= current_tokens * 1.25:
        return False
    glyph_orphans = orphan_punctuation_run_ratio(glyph_runs)
    current_orphans = orphan_punctuation_run_ratio(current_runs)
    return not glyph_orphans > max(0.18, current_orphans + 0.12)


def text_run_geometry_quality_score(runs: list[Any]) -> float:
    visible_runs = [
        run
        for run in runs
        if getattr(run, "visible", True) and str(getattr(run, "text", "")).strip()
    ]
    if not visible_runs:
        return 0.0
    lines = LayoutAnalyzer.cluster_into_lines(visible_runs)
    line_count = len(lines)
    if line_count <= 0:
        return 0.0
    text_chars = sum(len(str(getattr(run, "text", "")).strip()) for run in visible_runs)
    avg_line_chars = text_chars / max(1, line_count)
    tiny_runs = 0
    for run in visible_runs:
        width = max(0.0, float(getattr(run, "x1", 0.0)) - float(getattr(run, "x0", 0.0)))
        height = max(0.0, float(getattr(run, "y1", 0.0)) - float(getattr(run, "y0", 0.0)))
        if width <= max(0.75, height * 0.12):
            tiny_runs += 1
    tiny_ratio = tiny_runs / len(visible_runs)
    line_coverage = min(1.0, line_count / 24.0)
    char_density = min(1.0, avg_line_chars / 48.0)
    run_density = min(1.0, text_chars / max(1, len(visible_runs)) / 8.0)
    return line_coverage * 0.35 + char_density * 0.45 + run_density * 0.2 - tiny_ratio * 0.25


def garbled_alpha_token_ratio(text: str) -> float:
    tokens = ocr_text_analysis.normalized_text_tokens(text)
    alpha_tokens = [token for token in tokens if any(ch.isalpha() for ch in token)]
    if not alpha_tokens:
        return 0.0
    garbled = sum(
        1 for token in alpha_tokens if ocr_text_analysis.alpha_token_looks_ocr_garbled(token)
    )
    return garbled / len(alpha_tokens)


def orphan_punctuation_run_ratio(runs: list[Any]) -> float:
    visible_runs = [
        run
        for run in runs
        if getattr(run, "visible", True) and str(getattr(run, "text", "")).strip()
    ]
    if not visible_runs:
        return 0.0
    orphan_count = 0
    for run in visible_runs:
        text = str(getattr(run, "text", "")).strip()
        if text and all(not ch.isalnum() for ch in text):
            orphan_count += 1
    return orphan_count / len(visible_runs)


__all__ = (
    "apply_rendered_glyph_repair_to_native_text",
    "native_invisible_text_layer_is_trustworthy",
    "native_text_runs_for_extraction",
    "native_text_runs_inside_page_bounds",
    "native_layout_geometry_summary_for_runs",
    "native_text_fast_path_fallback_reason",
    "native_text_fast_xobject_fallback_reason",
    "native_text_runs_inside_page_lower_bound",
    "native_text_runs_inside_visible_row_bands",
    "select_native_text_layout",
    "should_try_rendered_glyph_repair",
    "try_extract_native_text_fast",
)
