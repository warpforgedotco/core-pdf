# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import contextlib
from bisect import bisect_right
from dataclasses import dataclass, replace
from statistics import median
from typing import Iterable, Literal, TypeAlias

from core_pdf.impl.engine.extraction.ocr.types import (
    OcrRow,
    OcrTextResult,
    ocr_float_value,
    ocr_int_value,
)

OcrLayoutWordRole: TypeAlias = Literal[
    "content",
    "decorative_punctuation",
    "geometry_separator",
    "inline_separator",
    "list_marker",
    "orphan_artifact",
]

OCR_DECORATIVE_PUNCTUATION_CHARS = frozenset(".·•°*")
OCR_GEOMETRY_SEPARATOR_CHARS = frozenset("|[]{}¥~_=¦¬^°•·")


@dataclass(frozen=True)
class OcrLayoutWord:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: int | None
    page_space: bool
    baseline_y: float | None
    row_index: int
    word_num: int

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def mid_y(self) -> float:
        return (self.y0 + self.y1) * 0.5

    @property
    def mid_x(self) -> float:
        return (self.x0 + self.x1) * 0.5

    @property
    def line_y(self) -> float:
        return self.baseline_y if self.baseline_y is not None else self.mid_y

    @property
    def visual_y_key(self) -> float:
        return -self.line_y if self.page_space else self.line_y


@dataclass(frozen=True)
class OcrRenderedLayout:
    lines: list[list[OcrLayoutWord]]
    mode: str
    roles: dict[int, OcrLayoutWordRole]


def geometry_rendered_ocr_result(result: OcrTextResult) -> OcrTextResult:
    text = render_ocr_text_from_geometry(result.word_rows)
    if not text:
        return result
    if not geometry_text_is_usable(result.text, text):
        return result
    confidence = geometry_text_confidence(result.word_rows, result.confidence)
    return replace(result, text=text, confidence=confidence)


def render_ocr_text_from_geometry(rows: Iterable[OcrRow]) -> str:
    words = [
        word
        for row_index, row in enumerate(rows)
        if (word := ocr_layout_word(row, row_index=row_index)) is not None
    ]
    if len(words) < 2:
        return ""
    layout = ocr_rendered_layout(words)
    if layout.mode == "table":
        rendered_lines = render_ocr_table_lines(layout.lines, layout.roles)
    else:
        rendered_lines = [render_ocr_word_line(line, layout.roles) for line in layout.lines]
    return "\n".join(line for line in rendered_lines if line)


def ocr_layout_word(
    row: OcrRow,
    *,
    row_index: int = 0,
) -> OcrLayoutWord | None:
    text = str(row.get("text", "")).strip()
    if not text:
        return None
    page_bbox = row.get("page_bbox")
    page_space = False
    if isinstance(page_bbox, (list, tuple)) and len(page_bbox) == 4:
        try:
            x0, y0, x1, y1 = (ocr_float_value(value) for value in page_bbox)
            page_space = True
        except (TypeError, ValueError):
            return None
    else:
        try:
            x0 = ocr_float_value(row["left"])
            y0 = ocr_float_value(row["top"])
            x1 = x0 + ocr_float_value(row["width"])
            y1 = y0 + ocr_float_value(row["height"])
        except (KeyError, TypeError, ValueError):
            return None
    if x1 <= x0 or y1 <= y0:
        return None
    confidence = row.get("conf")
    try:
        confidence_value = (
            int(round(ocr_float_value(confidence))) if confidence is not None else None
        )
    except (TypeError, ValueError):
        confidence_value = None
    word_num = ocr_row_word_num(row)
    baseline_y = ocr_row_baseline_y(row, page_space=page_space)
    return OcrLayoutWord(
        text,
        x0,
        y0,
        x1,
        y1,
        confidence_value,
        page_space,
        baseline_y,
        row_index,
        word_num,
    )


def ocr_row_word_num(row: OcrRow) -> int:
    try:
        return ocr_int_value(row.get("word_num", 0))
    except (TypeError, ValueError):
        return 0


def ocr_row_baseline_y(row: OcrRow, *, page_space: bool) -> float | None:
    baseline = row.get("page_baseline" if page_space else "baseline")
    if not isinstance(baseline, (list, tuple)) or len(baseline) != 4:
        return None
    try:
        return (ocr_float_value(baseline[1]) + ocr_float_value(baseline[3])) * 0.5
    except (TypeError, ValueError):
        return None


def ocr_words_to_lines(words: list[OcrLayoutWord]) -> list[list[OcrLayoutWord]]:
    return ocr_rendered_layout(words).lines


def ocr_rendered_layout(words: list[OcrLayoutWord]) -> OcrRenderedLayout:
    lines = ocr_geometry_word_lines(words)
    roles = ocr_layout_word_roles(lines)
    if ocr_lines_look_table_like(lines, roles):
        return OcrRenderedLayout(lines, "table", roles)
    split_points = ocr_column_split_points(lines, roles)
    if split_points:
        return OcrRenderedLayout(
            ocr_split_lines_by_columns(lines, split_points),
            "prose",
            roles,
        )
    return OcrRenderedLayout(lines, "prose", roles)


def ocr_geometry_word_lines(words: list[OcrLayoutWord]) -> list[list[OcrLayoutWord]]:
    sorted_words = sorted(words, key=lambda word: (word.visual_y_key, word.x0))
    lines: list[list[OcrLayoutWord]] = []
    for word in sorted_words:
        target = ocr_matching_line(lines, word)
        if target is None:
            lines.append([word])
        else:
            target.append(word)
    ordered_lines = [
        sorted(line, key=lambda word: (word.x0, word.word_num, word.row_index)) for line in lines
    ]
    return sorted(ordered_lines, key=ocr_line_visual_key)


def ocr_matching_line(
    lines: list[list[OcrLayoutWord]], word: OcrLayoutWord
) -> list[OcrLayoutWord] | None:
    for line in reversed(lines[-4:]):
        heights = [item.height for item in line if item.height > 0]
        line_height = median(heights) if heights else word.height
        max_delta = max(2.0, max(line_height, word.height) * 0.55)
        line_y_values = sorted(item.visual_y_key for item in line)
        line_y = line_y_values[len(line_y_values) // 2]
        if abs(word.visual_y_key - line_y) <= max_delta:
            return line
    return None


def ocr_line_visual_key(line: list[OcrLayoutWord]) -> tuple[float, float]:
    visual_y_values = sorted(word.visual_y_key for word in line)
    visual_y = visual_y_values[len(visual_y_values) // 2]
    x0 = min(word.x0 for word in line)
    return (visual_y, x0)


def ocr_column_split_points(
    lines: list[list[OcrLayoutWord]],
    roles: dict[int, OcrLayoutWordRole] | None = None,
) -> list[float]:
    candidate_centers: list[float] = []
    threshold = ocr_large_gap_threshold(lines)
    for line in lines:
        visible = ocr_visible_words(line, roles)
        if len(visible) < 2:
            continue
        for previous, current in zip(visible, visible[1:]):
            gap = current.x0 - previous.x1
            if gap >= threshold:
                candidate_centers.append((previous.x1 + current.x0) * 0.5)
    if not candidate_centers:
        return []
    tolerance = max(ocr_median_word_height(lines) * 2.0, 12.0)
    clusters = ocr_cluster_positions(sorted(candidate_centers), tolerance)
    visible_line_count = len([line for line in lines if ocr_visible_words(line, roles)])
    required = max(2, int(visible_line_count * 0.40))
    split_points = [sum(cluster) / len(cluster) for cluster in clusters if len(cluster) >= required]
    if not split_points:
        return []
    split_points = ocr_filter_usable_split_points(lines, split_points, roles)
    return sorted(split_points)


def ocr_split_lines_by_columns(
    lines: list[list[OcrLayoutWord]],
    split_points: list[float],
) -> list[list[OcrLayoutWord]]:
    columns: list[list[list[OcrLayoutWord]]] = [[] for _ in range(len(split_points) + 1)]
    for line in lines:
        buckets: list[list[OcrLayoutWord]] = [[] for _ in range(len(split_points) + 1)]
        for word in line:
            buckets[bisect_right(split_points, word.mid_x)].append(word)
        for column_index, bucket in enumerate(buckets):
            if bucket:
                columns[column_index].append(
                    sorted(
                        bucket,
                        key=lambda word: (word.x0, word.word_num, word.row_index),
                    )
                )
    ordered: list[list[OcrLayoutWord]] = []
    for column in columns:
        ordered.extend(sorted(column, key=ocr_line_visual_key))
    return ordered


def ocr_filter_usable_split_points(
    lines: list[list[OcrLayoutWord]],
    split_points: list[float],
    roles: dict[int, OcrLayoutWordRole] | None = None,
) -> list[float]:
    usable: list[float] = []
    all_words = [word for line in lines for word in ocr_visible_words(line, roles)]
    if not all_words:
        return []
    min_x = min(word.x0 for word in all_words)
    max_x = max(word.x1 for word in all_words)
    width = max_x - min_x
    if width <= 0:
        return []
    for split in split_points:
        if split <= min_x + width * 0.18 or split >= max_x - width * 0.18:
            continue
        left_lines = 0
        right_lines = 0
        crossing_lines = 0
        for line in lines:
            visible = ocr_visible_words(line, roles)
            if not visible:
                continue
            has_left = any(word.mid_x < split for word in visible)
            has_right = any(word.mid_x >= split for word in visible)
            if has_left:
                left_lines += 1
            if has_right:
                right_lines += 1
            if has_left and has_right:
                crossing_lines += 1
        if left_lines >= 2 and right_lines >= 2 and crossing_lines >= 2:
            usable.append(split)
    return usable


def ocr_lines_look_table_like(
    lines: list[list[OcrLayoutWord]],
    roles: dict[int, OcrLayoutWordRole] | None = None,
) -> bool:
    visible_lines = [ocr_visible_words(line, roles) for line in lines]
    visible_lines = [line for line in visible_lines if len(line) >= 3]
    if len(visible_lines) < 3:
        return False
    signal_lines = [line for line in visible_lines if ocr_line_has_table_signal(line)]
    if len(signal_lines) < 3:
        return False
    repeated_columns = ocr_repeated_column_count(visible_lines)
    return repeated_columns >= 3


def ocr_line_has_table_signal(words: list[OcrLayoutWord]) -> bool:
    tokens = [word.text for word in words if any(ch.isalnum() for ch in word.text)]
    if len(tokens) < 3:
        return False
    digit_tokens = sum(1 for token in tokens if any(ch.isdigit() for ch in token))
    if digit_tokens >= 2:
        return True
    return digit_tokens >= 1 and len(tokens) >= 4


def ocr_repeated_column_count(lines: list[list[OcrLayoutWord]]) -> int:
    positions = sorted(word.x0 for line in lines for word in line)
    if not positions:
        return 0
    tolerance = max(
        ocr_median_word_height(lines),
        ocr_median_char_width(lines) * 1.5,
        6.0,
    )
    clusters = ocr_cluster_positions(positions, tolerance)
    min_count = max(2, int(len(lines) * 0.5))
    return sum(1 for cluster in clusters if len(cluster) >= min_count)


def ocr_cluster_positions(values: list[float], tolerance: float) -> list[list[float]]:
    clusters: list[list[float]] = []
    for value in values:
        if not clusters:
            clusters.append([value])
            continue
        center = sum(clusters[-1]) / len(clusters[-1])
        if abs(value - center) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return clusters


def ocr_large_gap_threshold(lines: list[list[OcrLayoutWord]]) -> float:
    return max(
        ocr_median_word_height(lines) * 4.0,
        ocr_median_word_width(lines) * 2.25,
        36.0,
    )


def ocr_median_word_height(lines: list[list[OcrLayoutWord]]) -> float:
    heights = sorted(word.height for line in lines for word in line if word.height > 0)
    return heights[len(heights) // 2] if heights else 8.0


def ocr_median_word_width(lines: list[list[OcrLayoutWord]]) -> float:
    widths = sorted(word.width for line in lines for word in line if word.width > 0)
    return widths[len(widths) // 2] if widths else 12.0


def ocr_median_char_width(lines: list[list[OcrLayoutWord]]) -> float:
    widths = sorted(
        word.width / max(1, len(word.text.strip()))
        for line in lines
        for word in line
        if word.width > 0 and word.text.strip()
    )
    if widths:
        return widths[len(widths) // 2]
    return max(1.0, ocr_median_word_height(lines) * 0.45)


def ocr_visible_words(
    words: list[OcrLayoutWord],
    roles: dict[int, OcrLayoutWordRole] | None = None,
) -> list[OcrLayoutWord]:
    ordered = ocr_order_line_words(words)
    return [
        word
        for index, word in enumerate(ordered)
        if ocr_layout_word_role(ordered, index, roles) in {"content", "inline_separator"}
    ]


def render_ocr_word_line(
    words: list[OcrLayoutWord],
    roles: dict[int, OcrLayoutWordRole] | None = None,
) -> str:
    if not words:
        return ""
    ordered = ocr_order_line_words(words)
    parts: list[str] = []
    previous: OcrLayoutWord | None = None
    for index, word in enumerate(ordered):
        role = ocr_layout_word_role(ordered, index, roles)
        text = ocr_layout_word_render_text(ordered, index, role)
        if not text:
            continue
        if previous is not None and ocr_words_need_space(previous, word):
            parts.append(" ")
        parts.append(text)
        previous = word
    return "".join(parts).strip()


def render_ocr_table_lines(
    lines: list[list[OcrLayoutWord]],
    roles: dict[int, OcrLayoutWordRole] | None = None,
) -> list[str]:
    visible_lines = [ocr_visible_words(line, roles) for line in lines]
    visible_words = [word for line in visible_lines for word in line]
    if not visible_words:
        return []
    origin = min(word.x0 for word in visible_words)
    char_width = max(1.0, ocr_median_char_width(visible_lines))
    return [
        line
        for words in visible_lines
        if (
            line := render_ocr_table_line(
                words,
                origin=origin,
                char_width=char_width,
                roles=roles,
            )
        )
    ]


def render_ocr_table_line(
    words: list[OcrLayoutWord],
    *,
    origin: float,
    char_width: float,
    roles: dict[int, OcrLayoutWordRole] | None = None,
) -> str:
    if not words:
        return ""
    parts: list[str] = []
    cursor = 0
    ordered = ocr_order_line_words(words)
    for index, word in enumerate(ordered):
        role = ocr_layout_word_role(ordered, index, roles)
        text = ocr_layout_word_render_text(ordered, index, role)
        if not text:
            continue
        target = max(0, int(round((word.x0 - origin) / char_width)))
        if parts:
            gap = max(1, target - cursor)
            parts.append(" " * min(gap, 32))
            cursor += min(gap, 32)
        elif target > 0:
            parts.append(" " * min(target, 32))
            cursor += min(target, 32)
        parts.append(text)
        cursor += len(text)
    return "".join(parts).strip()


def ocr_words_need_space(previous: OcrLayoutWord, current: OcrLayoutWord) -> bool:
    if not previous.text or not current.text:
        return False
    if current.text[0] in ".,;:!?)]}%":
        return False
    if previous.text[-1] in "([{/$":
        return False
    gap = current.x0 - previous.x1
    return not gap < -max(previous.height, current.height) * 0.15


def ocr_order_line_words(words: list[OcrLayoutWord]) -> list[OcrLayoutWord]:
    return sorted(words, key=lambda word: (word.x0, word.word_num, word.row_index))


def ocr_layout_word_roles(
    lines: list[list[OcrLayoutWord]],
) -> dict[int, OcrLayoutWordRole]:
    ordered_lines = [ocr_order_line_words(line) for line in lines]
    roles = {
        word.row_index: ocr_classify_layout_word(line, index)
        for line in ordered_lines
        for index, word in enumerate(line)
    }
    for word in ocr_repeated_list_marker_words(ordered_lines):
        roles[word.row_index] = "list_marker"
    return roles


def ocr_layout_word_role(
    words: list[OcrLayoutWord],
    index: int,
    roles: dict[int, OcrLayoutWordRole] | None,
) -> OcrLayoutWordRole:
    word = words[index]
    if roles is not None and word.row_index in roles:
        return roles[word.row_index]
    return ocr_classify_layout_word(words, index)


def ocr_classify_layout_word(
    words: list[OcrLayoutWord],
    index: int,
) -> OcrLayoutWordRole:
    word = words[index]
    text = word.text.strip()
    if not text:
        return "geometry_separator"
    if ocr_word_is_recoverable_inline_separator(words, index):
        return "inline_separator"
    if ocr_word_is_orphan_artifact(words, index):
        return "orphan_artifact"
    if ocr_word_is_standalone_decorative_punctuation(words, index):
        return "decorative_punctuation"
    if ocr_word_is_geometry_separator(text, word):
        return "geometry_separator"
    return "content"


def ocr_layout_word_render_text(
    words: list[OcrLayoutWord],
    index: int,
    role: OcrLayoutWordRole,
) -> str:
    if role == "inline_separator":
        return "|"
    if role != "content":
        return ""
    return ocr_content_word_text(words, index)


def ocr_content_word_text(words: list[OcrLayoutWord], index: int) -> str:
    text = words[index].text.strip()
    tail_length = ocr_decorative_punctuation_tail_length(words, index)
    if tail_length:
        return text[:-tail_length].strip()
    return text


def ocr_repeated_list_marker_words(
    lines: list[list[OcrLayoutWord]],
) -> list[OcrLayoutWord]:
    candidates = [
        candidate for line in lines for candidate in ocr_line_list_marker_candidates(line)
    ]
    if len(candidates) < 2:
        return []
    tolerance = max(ocr_median_word_height(lines) * 0.8, 12.0)
    clusters: list[list[OcrLayoutWord]] = []
    for word in sorted(candidates, key=lambda candidate: candidate.mid_x):
        if not clusters:
            clusters.append([word])
            continue
        center = sum(candidate.mid_x for candidate in clusters[-1]) / len(clusters[-1])
        if abs(word.mid_x - center) <= tolerance:
            clusters[-1].append(word)
        else:
            clusters.append([word])
    markers: list[OcrLayoutWord] = []
    for cluster in clusters:
        if len(cluster) >= 2:
            markers.extend(cluster)
    return markers


def ocr_line_list_marker_candidates(
    words: list[OcrLayoutWord],
) -> list[OcrLayoutWord]:
    ordered = ocr_order_line_words(words)
    return [
        word
        for index, word in enumerate(ordered)
        if ocr_word_is_list_marker_candidate(ordered, index)
    ]


def ocr_word_is_list_marker_candidate(
    words: list[OcrLayoutWord],
    index: int,
) -> bool:
    word = words[index]
    text = word.text.strip()
    if len(text) != 1 or text.isdigit() or text in {"|", "¦"}:
        return False
    previous = ocr_neighbor_text_word(words, index, step=-1)
    next_word = ocr_neighbor_text_word(words, index, step=1)
    if next_word is None:
        return False
    following_words = [
        item
        for item in words[index + 1 :]
        if item.text.strip()
        and not ocr_word_is_geometry_separator(item.text, item)
        and any(ch.isalnum() for ch in item.text)
    ]
    if len(following_words) < 2:
        return False
    if not ocr_line_leading_marker_matches_next_word(word, next_word):
        return False
    if word.height > max(next_word.height * 0.75, 2.5):
        return False
    if previous is None:
        return True
    return ocr_word_gap(previous, word) >= max(
        next_word.height * 2.0,
        word.height * 3.0,
        12.0,
    )


def ocr_word_is_orphan_artifact(
    words: list[OcrLayoutWord],
    index: int,
) -> bool:
    word = words[index]
    text = word.text.strip()
    if len(text) != 1 or text.isdigit() or text in {"|", "¦"}:
        return False
    if word.confidence is None or word.confidence >= 60:
        return False
    line_heights = [
        item.height
        for item in words
        if item is not word and item.height > 0 and any(ch.isalnum() for ch in item.text)
    ]
    line_height = median(line_heights) if line_heights else word.height
    if not (
        word.height <= max(2.0, line_height * 0.20) or word.width <= max(2.0, line_height * 0.16)
    ):
        return False
    neighbors = (
        ocr_neighbor_text_word(words, index, step=-1),
        ocr_neighbor_text_word(words, index, step=1),
    )
    return not any(
        neighbor is not None and ocr_word_has_real_inline_neighbor(word, neighbor)
        for neighbor in neighbors
    )


def ocr_word_has_real_inline_neighbor(
    word: OcrLayoutWord,
    neighbor: OcrLayoutWord,
) -> bool:
    overlap = max(0.0, min(word.y1, neighbor.y1) - max(word.y0, neighbor.y0))
    if overlap < min(word.height, neighbor.height) * 0.35:
        return False
    if neighbor.x1 <= word.x0:
        gap = word.x0 - neighbor.x1
    elif word.x1 <= neighbor.x0:
        gap = neighbor.x0 - word.x1
    else:
        gap = 0.0
    return gap <= max(word.height, neighbor.height) * 3.0


def ocr_word_gap(left: OcrLayoutWord, right: OcrLayoutWord) -> float:
    if left.x1 <= right.x0:
        return right.x0 - left.x1
    if right.x1 <= left.x0:
        return left.x0 - right.x1
    return 0.0


def ocr_word_is_standalone_decorative_punctuation(
    words: list[OcrLayoutWord],
    index: int,
) -> bool:
    word = words[index]
    text = word.text.strip()
    if not text or any(ch.isalnum() for ch in text):
        return False
    if not all(ch in OCR_DECORATIVE_PUNCTUATION_CHARS for ch in text):
        return False
    previous = ocr_neighbor_text_word(words, index, step=-1)
    if previous is None:
        return False
    previous_index = words.index(previous)
    if not ocr_word_is_large_all_caps_title(words, previous_index):
        return False
    if word.height > previous.height * 0.55:
        return False
    gap = word.x0 - previous.x1
    if gap < -previous.height * 0.10 or gap > max(previous.height * 1.15, 32.0):
        return False
    return (
        previous.y0 - previous.height * 0.15 <= word.mid_y <= (previous.y1 + previous.height * 0.20)
    )


def ocr_decorative_punctuation_tail_length(
    words: list[OcrLayoutWord],
    index: int,
) -> int:
    word = words[index]
    text = word.text.strip()
    tail_length = 0
    cursor = len(text) - 1
    while cursor >= 0 and not text[cursor].isalnum():
        if text[cursor] not in OCR_DECORATIVE_PUNCTUATION_CHARS:
            return 0
        tail_length += 1
        cursor -= 1
    if tail_length < 2:
        return 0
    if cursor < 0:
        return 0
    if not ocr_word_is_large_all_caps_title(words, index):
        return 0
    return tail_length


def ocr_word_is_large_all_caps_title(
    words: list[OcrLayoutWord],
    index: int,
) -> bool:
    word = words[index]
    text = word.text.strip()
    while text and not text[-1].isalnum():
        text = text[:-1]
    if not ocr_text_is_all_caps_title(text):
        return False
    content_words = [
        item for item in words if item.text.strip() and any(ch.isalnum() for ch in item.text)
    ]
    if len(content_words) <= 2:
        return True
    heights = sorted(item.height for item in content_words if item.height > 0)
    line_height = heights[len(heights) // 2] if heights else word.height
    return word.height >= max(24.0, line_height * 1.25)


def ocr_text_is_all_caps_title(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 4:
        return False
    uppercase = sum(1 for ch in letters if ch.isupper())
    lowercase = sum(1 for ch in letters if ch.islower())
    return uppercase >= max(4, lowercase * 4)


def ocr_line_leading_marker_matches_next_word(
    marker: OcrLayoutWord,
    next_word: OcrLayoutWord,
) -> bool:
    overlap = max(0.0, min(marker.y1, next_word.y1) - max(marker.y0, next_word.y0))
    if overlap < min(marker.height, next_word.height) * 0.45:
        return False
    gap = next_word.x0 - marker.x1
    min_gap = max(4.0, marker.height * 0.20)
    max_gap = max(marker.height, next_word.height) * 2.6
    if not (min_gap <= gap <= max_gap):
        return False
    return marker.width <= max(next_word.height * 0.75, 18.0)


def ocr_word_is_recoverable_inline_separator(
    words: list[OcrLayoutWord],
    index: int,
) -> bool:
    word = words[index]
    if not ocr_word_is_inline_vertical_separator_glyph(word):
        return False
    previous = ocr_neighbor_text_word(words, index, step=-1)
    next_word = ocr_neighbor_text_word(words, index, step=1)
    if previous is None and next_word is None:
        return False
    return (
        previous is not None and ocr_inline_separator_matches_neighbor(word, previous, before=False)
    ) or (
        next_word is not None
        and ocr_inline_separator_matches_neighbor(word, next_word, before=True)
    )


def ocr_word_is_inline_vertical_separator_glyph(word: OcrLayoutWord) -> bool:
    if word.text.strip() not in {"|", "¦"}:
        return False
    if word.confidence is not None and word.confidence < 60:
        return False
    if word.height <= 0:
        return False
    return word.width <= max(4.0, min(16.0, word.height * 0.35))


def ocr_neighbor_text_word(
    words: list[OcrLayoutWord],
    index: int,
    *,
    step: int,
) -> OcrLayoutWord | None:
    cursor = index + step
    while 0 <= cursor < len(words):
        word = words[cursor]
        if word.text.strip() and not ocr_word_is_geometry_separator(word.text, word):
            return word
        cursor += step
    return None


def ocr_inline_separator_matches_neighbor(
    separator: OcrLayoutWord,
    neighbor: OcrLayoutWord,
    *,
    before: bool,
) -> bool:
    overlap = max(0.0, min(separator.y1, neighbor.y1) - max(separator.y0, neighbor.y0))
    if overlap < min(separator.height, neighbor.height) * 0.45:
        return False
    tolerance = max(separator.height, neighbor.height) * 0.20
    max_gap = max(separator.height, neighbor.height) * 1.35
    if before:
        gap = neighbor.x0 - separator.x1
        if separator.x1 > neighbor.x0 + tolerance:
            return False
    else:
        gap = separator.x0 - neighbor.x1
        if separator.x0 < neighbor.x1 - tolerance:
            return False
    return -tolerance <= gap <= max_gap


def ocr_word_is_geometry_separator(text: str, word: OcrLayoutWord) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if any(ch.isalnum() for ch in stripped):
        return False
    if not all(ch in OCR_GEOMETRY_SEPARATOR_CHARS for ch in stripped):
        return False
    if word.height <= 0:
        return True
    return word.width <= max(word.height * 1.4, 14.0) or len(stripped) >= 2


def geometry_text_is_usable(original: str, rendered: str) -> bool:
    rendered_tokens = count_text_tokens(rendered)
    if rendered_tokens < 2:
        return False
    original_tokens = count_text_tokens(original)
    if original_tokens == 0:
        return True
    if rendered_tokens < max(2, int(original_tokens * 0.65)):
        return False
    return not rendered_tokens > max(original_tokens + 300, int(original_tokens * 2.75))


def geometry_text_confidence(rows: Iterable[OcrRow], fallback: int | None) -> int | None:
    confidences: list[int] = []
    for row in rows:
        confidence = row.get("conf")
        if confidence is None:
            continue
        with contextlib.suppress(TypeError, ValueError):
            confidences.append(max(0, min(100, int(round(ocr_float_value(confidence))))))
    if not confidences:
        return fallback
    return int(round(sum(confidences) / len(confidences)))


def count_text_tokens(text: str) -> int:
    count = 0
    in_token = False
    for ch in text:
        if ch.isalnum() or ch == "_":
            if not in_token:
                count += 1
                in_token = True
        else:
            in_token = False
    return count
