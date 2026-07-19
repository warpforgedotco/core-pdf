# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import re
from dataclasses import replace
from statistics import median
from typing import Any

from core_pdf.impl.engine.extraction.common import page_profile
from core_pdf.impl.engine.extraction.common.render import (
    render_page_observation_lines,
    render_resolved_text_lines,
    resolved_text_lines_for_output,
)

MIN_VISIBLE_GLYPH_COVERAGE = 0.75
WORD_TOKEN_RE = re.compile(r"\w+")


def native_text_runs_for_extraction(runs: list[Any]) -> list[Any]:
    """Return painted, active PDF text runs without duplicate text layers."""
    extractable: list[Any] = []
    invisible: list[Any] = []
    painted: list[Any] = []
    for run in runs:
        if not getattr(run, "text", "") or not text_run_is_inside_active_clip(run):
            continue
        invisible_render_mode = text_run_uses_invisible_render_mode(run)
        if not getattr(run, "visible", True) and not invisible_render_mode:
            continue
        extractable.append(run)
        (invisible if invisible_render_mode else painted).append(run)
    if (
        invisible
        and painted
        and (
            invisible_text_is_suspicious(invisible)
            or invisible_text_duplicates_painted_text(invisible, painted)
        )
    ):
        return normalize_misdecoded_space_runs(normalize_checkbox_runs(painted))
    return normalize_misdecoded_space_runs(
        normalize_checkbox_runs(
            discard_alternate_clipped_runs(discard_overlapping_nested_xobject_runs(extractable))
        )
    )


def discard_overlapping_nested_xobject_runs(runs: list[Any]) -> list[Any]:
    """Drop nested-form text that duplicates page-level painted geometry."""
    runs_by_depth: dict[int, list[Any]] = {}
    runs_with_depth: list[tuple[Any, int]] = []
    for run in runs:
        depth = xobject_depth(run)
        runs_by_depth.setdefault(depth, []).append(run)
        runs_with_depth.append((run, depth))
    page_runs = runs_by_depth.get(0, [])
    if not page_runs:
        return runs
    if len(runs_by_depth) == 1:
        return runs
    page_text = normalized_run_text(page_runs)
    nested_text_by_depth = {
        depth: normalized_run_text(depth_runs)
        for depth, depth_runs in runs_by_depth.items()
        if depth > 0
    }
    duplicate_nested_depths = {
        depth
        for depth, nested_text in nested_text_by_depth.items()
        if nested_layer_duplicates_page_text(nested_text, page_text)
    }
    alternate_nested_depths = {
        depth
        for depth, nested_text in nested_text_by_depth.items()
        if nested_layer_duplicates_page_text(nested_text, page_text, minimum_overlap=0.6)
    }
    filtered: list[Any] = []
    for run, depth in runs_with_depth:
        if depth == 0 or (
            depth not in duplicate_nested_depths
            and depth not in alternate_nested_depths
            and not nested_run_overlaps_page_text(run, page_runs)
        ):
            filtered.append(run)
    return filtered


def nested_layer_duplicates_page_text(
    nested_text: str, page_text: str, *, minimum_overlap: float = 0.8
) -> bool:
    """Return whether a nested layer repeats most of the page-layer tokens."""
    nested_tokens = nested_text.split()
    page_tokens = page_text.split()
    if len(nested_tokens) < 24 or len(page_tokens) < len(nested_tokens):
        return False
    page_counts: dict[str, int] = {}
    for token in page_tokens:
        page_counts[token] = page_counts.get(token, 0) + 1
    matched = 0
    nested_counts: dict[str, int] = {}
    for token in nested_tokens:
        nested_counts[token] = nested_counts.get(token, 0) + 1
    for token, count in nested_counts.items():
        matched += min(count, page_counts.get(token, 0))
    return matched / len(nested_tokens) >= minimum_overlap


def discard_alternate_clipped_runs(runs: list[Any]) -> list[Any]:
    """Keep the largest clipped text layer when smaller layers repeat it."""
    groups: dict[tuple[float, float, float, float], list[Any]] = {}
    run_bboxes: list[tuple[Any, tuple[float, float, float, float] | None]] = []
    for run in runs:
        bbox = clip_bbox_for_run(run)
        run_bboxes.append((run, bbox))
        if bbox is not None:
            groups.setdefault(bbox, []).append(run)
    if len(groups) < 2:
        return runs
    primary_bbox, primary_runs = max(groups.items(), key=lambda item: clip_bbox_area(item[0]))
    primary_text = normalized_run_text(primary_runs)
    if len(primary_text.split()) < 24:
        return runs
    alternate_bboxes = {
        bbox
        for bbox, group in groups.items()
        if bbox != primary_bbox
        and len(normalized_run_text(group).split()) >= 24
        and layer_text_overlap(normalized_run_text(group), primary_text) >= 0.6
    }
    if not alternate_bboxes:
        return runs
    return [run for run, bbox in run_bboxes if bbox not in alternate_bboxes]


def clip_bbox_for_run(run: Any) -> tuple[float, float, float, float] | None:
    provenance = run.provenance
    if not provenance:
        return None
    value: object | None = None
    for key, candidate in reversed(provenance):
        if key == "clip_bbox":
            value = candidate
            break
    else:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if isinstance(value, tuple):
        x0, y0, x1, y1 = value
        if (
            isinstance(x0, (int, float))
            and isinstance(y0, (int, float))
            and isinstance(x1, (int, float))
            and isinstance(y1, (int, float))
        ):
            return value if x1 > x0 and y1 > y0 else None
    try:
        bbox = tuple(float(part) for part in value)
    except (TypeError, ValueError):
        return None
    x0, y0, x1, y1 = bbox
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def clip_bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def layer_text_overlap(left: str, right: str) -> float:
    left_tokens = left.split()
    right_tokens = right.split()
    if not left_tokens or not right_tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in right_tokens:
        counts[token] = counts.get(token, 0) + 1
    matched = 0
    left_counts: dict[str, int] = {}
    for token in left_tokens:
        left_counts[token] = left_counts.get(token, 0) + 1
    for token, count in left_counts.items():
        matched += min(count, counts.get(token, 0))
    return matched / min(len(left_tokens), len(right_tokens))


def xobject_depth(run: Any) -> int:
    try:
        return int(getattr(run, "xobject_depth", 0))
    except (TypeError, ValueError):
        return 0


def nested_run_overlaps_page_text(run: Any, page_runs: list[Any]) -> bool:
    run_x0, run_y0, run_x1, run_y1 = (
        float(getattr(run, "x0", 0.0)),
        float(getattr(run, "y0", 0.0)),
        float(getattr(run, "x1", 0.0)),
        float(getattr(run, "y1", 0.0)),
    )
    run_width = run_x1 - run_x0
    run_height = run_y1 - run_y0
    if run_width <= 0.0 or run_height <= 0.0:
        return False
    run_area = run_width * run_height
    for candidate in page_runs:
        candidate_x0 = float(getattr(candidate, "x0", 0.0))
        candidate_y0 = float(getattr(candidate, "y0", 0.0))
        candidate_x1 = float(getattr(candidate, "x1", 0.0))
        candidate_y1 = float(getattr(candidate, "y1", 0.0))
        intersection_width = min(run_x1, candidate_x1) - max(run_x0, candidate_x0)
        intersection_height = min(run_y1, candidate_y1) - max(run_y0, candidate_y0)
        if intersection_width <= 0.0 or intersection_height <= 0.0:
            continue
        if (
            intersection_width * intersection_height / run_area >= 0.45
            and intersection_height / run_height >= 0.55
        ):
            return True
    return False


def normalize_checkbox_runs(runs: list[Any]) -> list[Any]:
    """Expose checkbox glyphs consistently when a PDF supplies them as text."""
    normalized: list[Any] = []
    for run in runs:
        text = str(getattr(run, "text", ""))
        if text in {"☒", "☐"} and hasattr(run, "replace"):
            run = run.replace(text="[x]" if text == "☒" else "[]")
        elif (
            len(text) >= 2
            and "\xa0" in text
            and all(character in " \xa0" for character in text)
            and is_checkbox_blank_run(run)
            and hasattr(run, "replace")
        ):
            run = run.replace(text="[]")
        elif text == "X" and is_checkbox_mark_run(run) and hasattr(run, "replace"):
            run = run.replace(text="[x]")
        normalized.append(run)
    return normalized


def is_checkbox_blank_run(run: Any) -> bool:
    text = str(getattr(run, "text", ""))
    if (
        not text
        or len(text) < 2
        or "\xa0" not in text
        or not all(character in " \xa0" for character in text)
    ):
        return False
    coords = getattr(run, "coords", None)
    if coords is None:
        x0, y0, x1, y1 = (
            float(getattr(run, "x0", 0.0)),
            float(getattr(run, "y0", 0.0)),
            float(getattr(run, "x1", 0.0)),
            float(getattr(run, "y1", 0.0)),
        )
    else:
        x0, y0, x1, y1 = coords[0], coords[1], coords[2], coords[3]
    width = max(0.0, float(x1) - float(x0))
    height = max(0.0, float(y1) - float(y0))
    return 5.0 <= width <= 7.0 and 5.0 <= height <= 7.0


def is_checkbox_mark_run(run: Any) -> bool:
    text = str(getattr(run, "text", ""))
    if text != "X":
        return False
    coords = getattr(run, "coords", None)
    if coords is None:
        x0, y0, x1, y1 = (
            float(getattr(run, "x0", 0.0)),
            float(getattr(run, "y0", 0.0)),
            float(getattr(run, "x1", 0.0)),
            float(getattr(run, "y1", 0.0)),
        )
    else:
        x0, y0, x1, y1 = coords[0], coords[1], coords[2], coords[3]
    width = max(0.0, float(x1) - float(x0))
    height = max(0.0, float(y1) - float(y0))
    return 0.0 < width <= 10.0 and 4.5 <= height <= 10.0


def normalize_misdecoded_space_runs(runs: list[Any]) -> list[Any]:
    """Use repeated font metrics to distinguish a space from a narrow glyph."""
    profiles: dict[tuple[object, str], tuple[list[float], list[float]]] = {}
    for run in runs:
        text = str(getattr(run, "text", ""))
        if text not in {"a", "P"}:
            continue
        space_width = max(0.0, float(getattr(run, "space_width", 0.0)))
        if space_width <= 0.0:
            continue
        coords = getattr(run, "coords", None)
        if coords is None:
            x0 = float(getattr(run, "x0", 0.0))
            x1 = float(getattr(run, "x1", 0.0))
        else:
            x0, x1 = coords[0], coords[2]
        width = max(0.0, float(x1) - float(x0))
        ratio = width / space_width
        narrow, wide = profiles.setdefault((getattr(run, "font_name", None), text), ([], []))
        (narrow if ratio <= 1.15 else wide).append(ratio)
    space_profiles = {
        key
        for key, (narrow, wide) in profiles.items()
        if len(narrow) >= 3 and wide and median(narrow) <= median(wide) * 0.75
    }
    if not space_profiles:
        return runs
    return [
        run.replace(text=" ")
        if (getattr(run, "font_name", None), str(getattr(run, "text", ""))) in space_profiles
        and is_misdecoded_space_run(run)
        and hasattr(run, "replace")
        else run
        for run in runs
    ]


def is_misdecoded_space_run(run: Any) -> bool:
    text = str(getattr(run, "text", ""))
    if text not in {"a", "P"}:
        return False
    coords = getattr(run, "coords", None)
    if coords is None:
        x0 = float(getattr(run, "x0", 0.0))
        x1 = float(getattr(run, "x1", 0.0))
    else:
        x0, x1 = coords[0], coords[2]
    width = max(0.0, float(x1) - float(x0))
    space_width = max(0.0, float(getattr(run, "space_width", 0.0)))
    return space_width > 0.0 and width <= space_width * 1.15


def invisible_text_duplicates_painted_text(invisible: list[Any], painted: list[Any]) -> bool:
    invisible_text = normalized_run_text(invisible)
    painted_text = normalized_run_text(painted)
    if not invisible_text or not painted_text:
        return False
    shorter, longer = sorted((invisible_text, painted_text), key=len)
    return invisible_text == painted_text or (
        len(shorter) >= len(longer) * 0.8 and shorter in longer
    )


def invisible_text_is_suspicious(runs: list[Any]) -> bool:
    text = "".join(str(getattr(run, "text", "")) for run in runs)
    meaningful = [character for character in text if not character.isspace()]
    if len(meaningful) < 24:
        return False
    counts: dict[str, int] = {}
    for character in meaningful:
        counts[character] = counts.get(character, 0) + 1
    dominant = max(counts.values(), default=0)
    dominant_character = max(counts, key=lambda character: counts[character])
    return dominant / len(meaningful) >= 0.7 and not dominant_character.isalnum()


def normalized_run_text(runs: list[Any]) -> str:
    return " ".join(
        token.casefold()
        for run in runs
        for token in WORD_TOKEN_RE.findall(str(getattr(run, "text", "")))
    )


def text_run_uses_invisible_render_mode(run: Any) -> bool:
    if run.visible is not False:
        return False
    for key, value in reversed(run.provenance):
        if key == "text_render_mode":
            return value in (3, "3")
    return False


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
        coords = run.coords
        x0, y0, x1, y1 = coords[0], coords[1], coords[2], coords[3]
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


def normalize_fullwidth_ascii_text(lines: tuple[Any, ...]) -> tuple[Any, ...]:
    """Normalize Unicode fullwidth ASCII without using document-specific context."""
    full_text = "\n".join(str(line.text) for line in lines)
    if any(
        "\u3400" <= character <= "\u9fff" or "\u3040" <= character <= "\u30ff"
        for character in full_text
    ):
        return lines
    translation = {codepoint: codepoint - 0xFEE0 for codepoint in range(0xFF01, 0xFF5F)}
    translation[0x3000] = 0x20
    normalized: list[Any] = []
    for line in lines:
        text = str(line.text)
        normalized_text = text.translate(translation)
        if normalized_text == text:
            normalized.append(line)
            continue
        observation = replace(line.observation, text=normalized_text)
        normalized.append(replace(line, text=normalized_text, observation=observation))
    return tuple(normalized)


def normalize_latin_ligatures(lines: tuple[Any, ...]) -> tuple[Any, ...]:
    """Expand Unicode Latin presentation ligatures into their ordinary letters."""
    translation = str.maketrans(
        {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}
    )
    normalized: list[Any] = []
    for line in lines:
        text = str(line.text)
        normalized_text = text.translate(translation)
        if normalized_text == text:
            normalized.append(line)
            continue
        observation = replace(line.observation, text=normalized_text)
        normalized.append(replace(line, text=normalized_text, observation=observation))
    return tuple(normalized)


def extract_native_text(page: Any) -> tuple[str, tuple[Any, ...]]:
    profile = page_profile.get_page_profile(page)
    runs = native_text_runs_for_extraction(list(page.chars))
    runs = native_text_runs_inside_page_bounds(
        runs,
        page.media_box,
        rotate=page.rotation,
    )
    lines = render_page_observation_lines(
        runs,
        rotate=page.rotation,
        media_box=page.media_box,
        layout=True,
    )
    output_lines = normalize_latin_ligatures(
        normalize_fullwidth_ascii_text(resolved_text_lines_for_output(lines))
    )
    text = render_resolved_text_lines(output_lines)
    cache = getattr(page, "extraction_cache", None)
    if cache is not None:
        cache["native_text_profile"] = profile
        cache["native_output_lines"] = output_lines
    return text, output_lines
