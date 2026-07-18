# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import typing
from bisect import bisect_left, bisect_right
from dataclasses import replace
from statistics import median_low
from typing import Any

if typing.TYPE_CHECKING:
    from core_layout.impl.layout.geometry import RectBox

    from core_pdf.impl.engine.spec.s_07_content.capture import CapturedLine
    from core_pdf.impl.engine.spec.s_07_document import PdfPage
    from core_pdf.impl.models import AnnotationRecord

from core_layout.impl.layout.models import LayoutBox, LayoutLine, TextRun
from core_layout.impl.layout.text_lines import LayoutLineTextSegment, strip_private_use_chars
from core_layout.impl.layout.word_frequencies import word_rank
from core_ocr.impl import page_geometry
from core_ocr.impl.text_analysis import normalized_text_tokens

from core_pdf.impl.engine.extraction.common import observation_resolver
from core_pdf.impl.engine.extraction.common.ordering import (
    SKEW_ANGLE_TOLERANCE,
    LayoutAnalyzer,
)


def text_boxes_in_reading_order(boxes: list[LayoutBox], angle: int) -> list[LayoutBox]:
    """Order rotated text rows along their cross-line page axis."""
    if angle == 90:
        return sorted(boxes, key=lambda box: (box.x0, -box.y1))
    if angle == 270:
        return sorted(boxes, key=lambda box: (-box.x1, -box.y1))
    return boxes


class LayoutReconstructor:
    __slots__ = ("media_box",)

    def __init__(self, media_box: tuple[float, float, float, float]) -> None:
        self.media_box = media_box

    def render_page(self, runs: list[TextRun]) -> str:
        return render_resolved_text_lines(self.render_page_lines(runs))

    def render_page_lines(
        self,
        runs: list[TextRun],
    ) -> tuple[observation_resolver.ResolvedTextLine, ...]:
        if not runs:
            return ()

        if len(runs) >= 50:
            skew_angle = LayoutAnalyzer.detect_skew(runs)
            if abs(skew_angle) > SKEW_ANGLE_TOLERANCE:
                runs = LayoutAnalyzer.apply_skew_correction(runs, skew_angle)

        lines = LayoutAnalyzer.cluster_into_lines(runs)
        if not lines:
            return ()

        boxes = LayoutAnalyzer.order_boxes(LayoutAnalyzer.cluster_into_boxes(lines))
        if not boxes:
            return observation_resolver.resolve_text_lines(self.render_sorted_rows_lines(lines))
        if len(boxes) == 1:
            split_points = self.find_column_splits(lines)
            if split_points and (
                self.column_split_is_stable(lines, split_points)
                or self.multi_column_split_is_stable(lines, split_points)
                or (
                    len(split_points) == 1
                    and self.parallel_column_split_is_viable(lines, split_points[0])
                )
            ):
                return observation_resolver.resolve_text_lines(
                    self.render_line_segment_lines(lines)
                )

            if self.is_likely_single_column(lines):
                return observation_resolver.resolve_text_lines(self.render_sorted_rows_lines(lines))

            return observation_resolver.resolve_text_lines(self.render_box_lines(lines))

        boxes_by_angle: dict[int, list[LayoutBox]] = {}
        for box in boxes:
            angle = box.lines[0].rotation_angle if box.lines else 0
            boxes_by_angle.setdefault(angle, []).append(box)

        ordered_boxes = []
        for angle in sorted(boxes_by_angle):
            ordered_boxes.extend(text_boxes_in_reading_order(boxes_by_angle[angle], angle))

        output_lines: list[observation_resolver.ResolvedTextLine] = []
        for box in ordered_boxes:
            box_lines = self.render_box_lines(box.lines)
            if not box_lines:
                continue
            output_lines.extend(
                with_first_line_break(
                    box_lines,
                    break_before=2 if output_lines else 1,
                )
            )
        return observation_resolver.resolve_text_lines(output_lines)

    def render_box_lines(
        self,
        lines: list[LayoutLine],
    ) -> tuple[observation_resolver.ResolvedTextLine, ...]:
        split_points = self.find_column_splits(lines)
        if split_points and self.multi_column_split_is_stable(lines, split_points):
            return self.render_stable_multi_column_box_lines(lines, split_points)
        bands = partition_lines_into_vertical_bands(lines)
        if len(bands) == 1:
            return self.render_line_segment_lines(lines)
        output_lines: list[observation_resolver.ResolvedTextLine] = []
        for band in bands:
            band_lines = self.render_line_segment_lines(band)
            if not band_lines:
                continue
            output_lines.extend(
                with_first_line_break(
                    band_lines,
                    break_before=2 if output_lines else 1,
                )
            )
        return tuple(output_lines)

    def render_stable_multi_column_box_lines(
        self,
        lines: list[LayoutLine],
        split_points: list[float],
    ) -> tuple[observation_resolver.ResolvedTextLine, ...]:
        if not lines:
            return ()
        box_x0 = min(line.x0 for line in lines if line.runs)
        box_x1 = max(line.x1 for line in lines if line.runs)
        boundaries = [box_x0, *split_points, box_x1]
        column_width = max(
            boundaries[index + 1] - boundaries[index] for index in range(len(boundaries) - 1)
        )
        macro_column_anchors = self.macro_column_x0_anchors(
            lines,
            column_width=column_width,
        )
        sorted_rows = sorted(lines, key=lambda ln: (-ln.mid_y, ln.x0))
        output_lines: list[observation_resolver.ResolvedTextLine] = []
        column_buckets: list[list[LayoutLine]] = [[] for ignored in range(len(boundaries) - 1)]

        def append_part(
            part_lines: tuple[observation_resolver.ResolvedTextLine, ...],
        ) -> None:
            if not part_lines:
                return
            output_lines.extend(
                with_first_line_break(
                    part_lines,
                    break_before=2 if output_lines else 1,
                )
            )

        def flush_columns() -> None:
            nonlocal column_buckets
            if not any(column_buckets):
                return
            for bucket in column_buckets:
                if bucket:
                    append_part(self.render_line_segment_lines(bucket))
            column_buckets = [[] for ignored in range(len(boundaries) - 1)]

        for line in sorted_rows:
            if not line.runs:
                continue
            column_indexes = self.line_column_indexes(line, split_points)
            split_lines = self.split_stable_multi_column_line(line, split_points)
            if split_lines is not None:
                for split_line in split_lines:
                    split_index = self.dominant_line_column_index(
                        split_line,
                        split_points,
                    )
                    if split_index is None:
                        continue
                    column_buckets[split_index].append(split_line)
                continue
            dominant_index = self.dominant_line_column_index(line, split_points)
            if (
                dominant_index is not None
                and len(column_indexes) > 1
                and (
                    self.line_macro_column_index(
                        line,
                        macro_column_anchors,
                        column_width=column_width,
                    )
                    is not None
                    or self.line_can_use_dominant_column_assignment(
                        line,
                        split_points,
                        column_width=column_width,
                    )
                )
            ):
                macro_index = self.line_macro_column_index(
                    line,
                    macro_column_anchors,
                    column_width=column_width,
                )
                column_buckets[macro_index if macro_index is not None else dominant_index].append(
                    line
                )
                continue
            if len(column_indexes) != 1 or (line.x1 - line.x0) > column_width * 1.35:
                flush_columns()
                line_output = layout_line_to_resolved_text_line(line, break_before=1)
                if line_output is not None:
                    append_part((line_output,))
                continue
            column_buckets[column_indexes[0]].append(line)

        flush_columns()
        return tuple(output_lines)

    @staticmethod
    def split_stable_multi_column_line(
        line: LayoutLine,
        split_points: list[float],
    ) -> list[LayoutLine] | None:
        groups = [
            group for group in column_run_groups(line.runs) if run_group_bbox(group) is not None
        ]
        if len(groups) < 4:
            return None
        group_boxes = [
            typing.cast(tuple[float, float, float, float], run_group_bbox(group))
            for group in groups
        ]
        gaps = [
            group_boxes[index + 1][0] - group_boxes[index][2]
            for index in range(len(group_boxes) - 1)
        ]
        if not gaps:
            return None
        max_gap = max(gaps)
        split_index = gaps.index(max_gap) + 1
        if max_gap < 12.0:
            return None
        left_runs = [run for group in groups[:split_index] for run in group]
        right_runs = [run for group in groups[split_index:] for run in group]
        if not left_runs or not right_runs:
            return None
        split_lines = [LayoutLine(runs=left_runs), LayoutLine(runs=right_runs)]
        texts = [split_line.reconstructed_text().text.strip() for split_line in split_lines]
        if not stable_multi_column_split_is_safe(
            texts,
            split_lines=split_lines,
            split_points=split_points,
            max_gap=max_gap,
        ):
            return None
        return split_lines

    def render_line_segment_lines(
        self,
        lines: list[LayoutLine],
    ) -> tuple[observation_resolver.ResolvedTextLine, ...]:
        if len(lines) < 8:
            return self.render_rows_lines(lines)
        split_points = self.find_column_splits(lines)
        if not split_points:
            return self.render_rows_lines(lines)
        if not (
            self.column_split_is_stable(lines, split_points)
            or self.multi_column_split_is_stable(lines, split_points)
            or (
                len(split_points) == 1
                and self.parallel_column_split_is_viable(lines, split_points[0])
            )
        ):
            return self.render_rows_lines(lines)

        box_x0 = min(line.x0 for line in lines if line.runs)
        box_x1 = max(line.x1 for line in lines if line.runs)
        boundaries = [box_x0, *split_points, box_x1]
        column_width = max(
            boundaries[index + 1] - boundaries[index] for index in range(len(boundaries) - 1)
        )
        sorted_rows = sorted(lines, key=lambda ln: (-ln.mid_y, ln.x0))
        output_lines: list[observation_resolver.ResolvedTextLine] = []
        column_block: list[LayoutLine] = []

        def append_part(
            part_lines: tuple[observation_resolver.ResolvedTextLine, ...],
        ) -> None:
            if not part_lines:
                return
            output_lines.extend(
                with_first_line_break(
                    part_lines,
                    break_before=2 if output_lines else 1,
                )
            )

        def flush_columns() -> None:
            if not column_block:
                return
            column_lines: list[list[LayoutLine]] = [[] for ignored in range(len(split_points) + 1)]
            for line in column_block:
                split_runs: list[list[TextRun]] = [[] for ignored in range(len(split_points) + 1)]
                for group in column_run_groups(line.runs):
                    group_box = run_group_bbox(group)
                    if group_box is None:
                        continue
                    column_position = (
                        group_box[0]
                        if len(group) == 1 and narrow_boundary_punctuation_run(group[0])
                        else (group_box[0] + group_box[2]) * 0.5
                    )
                    column_index = 0
                    while (
                        column_index < len(split_points)
                        and column_position >= split_points[column_index]
                    ):
                        column_index += 1
                    split_runs[column_index].extend(group)
                for column_index, runs in enumerate(split_runs):
                    if runs:
                        column_lines[column_index].append(LayoutLine(runs=runs))

            for lines_for_column in column_lines:
                if lines_for_column:
                    append_part(self.render_line_segment_lines(lines_for_column))
            column_block.clear()

        for line in sorted_rows:
            if not line.runs:
                continue
            is_column_line = self.line_has_column_gutter(line, split_points, column_width)
            if not is_column_line and column_block:
                is_column_line = (line.x1 - line.x0) <= column_width * 1.15
            if is_column_line:
                column_block.append(line)
                continue

            flush_columns()
            line_output = layout_line_to_resolved_text_line(line, break_before=1)
            if line_output is not None:
                append_part((line_output,))

        flush_columns()
        return tuple(output_lines)

    @staticmethod
    def render_rows_lines(
        lines: list[LayoutLine],
    ) -> tuple[observation_resolver.ResolvedTextLine, ...]:
        sorted_rows = sorted(lines, key=lambda ln: (-ln.mid_y, ln.x0))
        output_lines: list[observation_resolver.ResolvedTextLine] = []
        for line in sorted_rows:
            if not line.runs:
                continue
            output_line = layout_line_to_resolved_text_line(
                line,
                break_before=1,
            )
            if output_line is not None:
                output_lines.append(output_line)
        return tuple(output_lines)

    @staticmethod
    def render_sorted_rows_lines(
        sorted_rows: list[LayoutLine],
    ) -> tuple[observation_resolver.ResolvedTextLine, ...]:
        output_lines: list[observation_resolver.ResolvedTextLine] = []
        for line in sorted_rows:
            if not line.runs:
                continue
            output_line = layout_line_to_resolved_text_line(
                line,
                break_before=1,
            )
            if output_line is not None:
                output_lines.append(output_line)
        return tuple(output_lines)

    def render_box(self, lines: list[LayoutLine]) -> str:
        return render_resolved_text_lines(self.render_box_lines(lines))

    def render_line_segment(self, lines: list[LayoutLine]) -> str:
        return render_resolved_text_lines(self.render_line_segment_lines(lines))

    @staticmethod
    def is_likely_single_column(lines: list[LayoutLine]) -> bool:
        if len(lines) < 24:
            return True

        x0s = [line.x0 for line in lines if line.runs]
        if len(x0s) < 24:
            return True

        x0s.sort()
        box_x0 = x0s[0]
        box_x1 = max(line.x1 for line in lines if line.runs)
        box_width = box_x1 - box_x0
        if box_width <= 0:
            return True

        best_gap = 0.0
        split_x = box_x0
        prev_x0 = x0s[0]
        for x0 in x0s[1:]:
            gap = x0 - prev_x0
            if gap > best_gap:
                best_gap = gap
                split_x = (prev_x0 + x0) * 0.5
            prev_x0 = x0

        if best_gap < max(120.0, box_width * 0.28):
            return True

        left_n = bisect_left(x0s, split_x)
        right_n = len(x0s) - left_n
        if left_n < 8 or right_n < 8:
            return True

        left_edge = max(line.x1 for line in lines if line.runs and line.x0 < split_x)
        right_edge = min(line.x0 for line in lines if line.runs and line.x0 >= split_x)
        return right_edge - left_edge < 18.0

    @staticmethod
    def column_split_is_stable(lines: list[LayoutLine], split_points: list[float]) -> bool:
        if len(lines) < 24 or len(split_points) != 1:
            return False
        split_x = split_points[0]
        left_only: list[LayoutLine] = []
        right_only: list[LayoutLine] = []
        spanning = 0
        gutter_rows = 0

        for line in lines:
            runs = [run for run in line.runs if run.has_text]
            if not runs:
                continue
            has_left = any(run.mid_x < split_x for run in runs)
            has_right = any(run.mid_x >= split_x for run in runs)
            if has_left and has_right:
                spanning += 1
                left_edge = max(
                    (run.x1 for run in runs if run.mid_x < split_x),
                    default=split_x,
                )
                right_edge = min(
                    (run.x0 for run in runs if run.mid_x >= split_x),
                    default=split_x,
                )
                if right_edge - left_edge >= 8.0:
                    gutter_rows += 1
            elif has_left:
                left_only.append(line)
            elif has_right:
                right_only.append(line)

        if len(left_only) < 8 or len(right_only) < 8:
            return False
        if spanning + gutter_rows < 5:
            return False

        left_top, left_bottom = LayoutReconstructor.line_y_extent(left_only)
        right_top, right_bottom = LayoutReconstructor.line_y_extent(right_only)
        overlap = min(left_top, right_top) - max(left_bottom, right_bottom)
        if overlap <= 0:
            return False
        left_height = left_top - left_bottom
        right_height = right_top - right_bottom
        if left_height <= 0 or right_height <= 0:
            return False
        return overlap >= min(left_height, right_height) * 0.45

    @staticmethod
    def multi_column_split_is_stable(lines: list[LayoutLine], split_points: list[float]) -> bool:
        if len(lines) < 24 or len(split_points) < 2:
            return False
        column_counts = [0] * (len(split_points) + 1)
        multi_column_rows = 0
        for line in lines:
            columns: set[int] = set()
            for group in column_run_groups(line.runs):
                group_box = run_group_bbox(group)
                if group_box is None:
                    continue
                group_mid_x = (group_box[0] + group_box[2]) * 0.5
                column_index = 0
                while (
                    column_index < len(split_points) and group_mid_x >= split_points[column_index]
                ):
                    column_index += 1
                column_counts[column_index] += 1
                columns.add(column_index)
            if len(columns) >= 2:
                multi_column_rows += 1

        active_columns = [count for count in column_counts if count >= 8]
        if len(active_columns) < 3:
            return False
        return multi_column_rows >= max(8, len(lines) // 4)

    @staticmethod
    def parallel_column_split_is_viable(
        lines: list[LayoutLine],
        split_x: float,
    ) -> bool:
        if len(lines) < 12:
            return False
        left_rows = 0
        right_rows = 0
        gutter_rows = 0
        for line in lines:
            runs = [run for run in line.runs if run.has_text]
            if not runs:
                continue
            has_left = any(run.mid_x < split_x for run in runs)
            has_right = any(run.mid_x >= split_x for run in runs)
            if has_left:
                left_rows += 1
            if has_right:
                right_rows += 1
            if has_left and has_right:
                left_edge = max(
                    (run.x1 for run in runs if run.mid_x < split_x),
                    default=split_x,
                )
                right_edge = min(
                    (run.x0 for run in runs if run.mid_x >= split_x),
                    default=split_x,
                )
                if right_edge - left_edge >= 8.0:
                    gutter_rows += 1
        if left_rows < 8 or right_rows < 8:
            return False
        return gutter_rows >= max(4, len(lines) // 10)

    @staticmethod
    def line_y_extent(lines: list[LayoutLine]) -> tuple[float, float]:
        return (
            max(line.y1 for line in lines if line.runs),
            min(line.y0 for line in lines if line.runs),
        )

    @staticmethod
    def line_has_column_gutter(
        line: LayoutLine,
        split_points: list[float],
        column_width: float | None = None,
    ) -> bool:
        runs = [r for r in line.runs if r.has_text]
        if len(runs) < 2:
            return False
        line_width = line.x1 - line.x0
        for split_x in split_points:
            has_left = False
            has_right = False
            left_edge = split_x
            right_edge = split_x
            for run in runs:
                if run.mid_x < split_x:
                    has_left = True
                    if run.x1 > left_edge:
                        left_edge = run.x1
                else:
                    has_right = True
                    if run.x0 < right_edge:
                        right_edge = run.x0
            if not has_left or not has_right:
                continue
            if right_edge - left_edge >= 8.0:
                return True
            if column_width is not None and line_width > column_width * 1.45:
                return True
        return False

    def find_column_split(self, lines: list[LayoutLine]) -> float | None:
        splits = self.find_column_splits(lines)
        return splits[0] if len(splits) == 1 else None

    def find_column_splits(self, lines: list[LayoutLine]) -> list[float]:
        if len(lines) < 8:
            return []
        all_runs: list[TextRun] = []
        gap_mids: list[float] = []
        box_x0: float = 1e9
        box_x1: float = -1e9

        for line in lines:
            current_line_runs = []
            for r in line.runs:
                if r.has_text:
                    rx0, rx1 = r.x0, r.x1
                    if rx0 < box_x0:
                        box_x0 = rx0
                    if rx1 > box_x1:
                        box_x1 = rx1
                    all_runs.append(r)
                    current_line_runs.append(r)
            if len(current_line_runs) < 2:
                continue
            current_line_runs.sort(key=lambda run: (run.x0, run.order))
            prev_x1 = current_line_runs[0].x1
            for run in current_line_runs[1:]:
                gap = run.x0 - prev_x1
                if gap >= 16.0:
                    gap_mids.append((prev_x1 + run.x0) * 0.5)
                if run.x1 > prev_x1:
                    prev_x1 = run.x1

        if len(all_runs) < 25:
            return []

        box_width = box_x1 - box_x0
        if box_width <= 0:
            return []

        if len(gap_mids) < 5:
            return []

        sorted_gap_mids = sorted(gap_mids)
        sorted_mid_x = sorted(r.mid_x for r in all_runs)

        clusters: list[list[float]] = []
        for mid in sorted_gap_mids:
            if not clusters or mid - clusters[-1][-1] > 35.0:
                clusters.append([mid])
            else:
                clusters[-1].append(mid)

        splits: list[float] = []
        for cluster in clusters:
            if len(cluster) < 4:
                continue
            split_x = median_low(cluster)
            split_rel = (split_x - box_x0) / box_width
            if not (0.12 <= split_rel <= 0.88):
                continue
            left_n = bisect_left(sorted_mid_x, split_x)
            right_n = len(all_runs) - left_n
            if left_n < 8 or right_n < 8:
                continue
            splits.append(split_x)

        if not splits:
            return []

        filtered: list[float] = []

        def split_density(split_x: float) -> int:
            return bisect_right(sorted_gap_mids, split_x + 17.5) - bisect_left(
                sorted_gap_mids, split_x - 17.5
            )

        for split_x in splits:
            if not filtered or split_x - filtered[-1] >= 45.0:
                filtered.append(split_x)
            elif split_density(split_x) > split_density(filtered[-1]):
                filtered[-1] = split_x
        return filtered

    @staticmethod
    def line_column_indexes(
        line: LayoutLine,
        split_points: list[float],
    ) -> tuple[int, ...]:
        indexes: set[int] = set()
        for group in column_run_groups(line.runs):
            group_box = run_group_bbox(group)
            if group_box is None:
                continue
            group_mid_x = (group_box[0] + group_box[2]) * 0.5
            column_index = 0
            while column_index < len(split_points) and group_mid_x >= split_points[column_index]:
                column_index += 1
            indexes.add(column_index)
        return tuple(sorted(indexes))

    @staticmethod
    def dominant_line_column_index(
        line: LayoutLine,
        split_points: list[float],
    ) -> int | None:
        weights: dict[int, float] = {}
        for group in column_run_groups(line.runs):
            group_box = run_group_bbox(group)
            if group_box is None:
                continue
            group_mid_x = (group_box[0] + group_box[2]) * 0.5
            column_index = 0
            while column_index < len(split_points) and group_mid_x >= split_points[column_index]:
                column_index += 1
            weights[column_index] = weights.get(column_index, 0.0) + max(
                1.0,
                group_box[2] - group_box[0],
            )
        if not weights:
            return None
        return max(weights.items(), key=lambda item: (item[1], -item[0]))[0]

    @staticmethod
    def line_can_use_dominant_column_assignment(
        line: LayoutLine,
        split_points: list[float],
        *,
        column_width: float,
    ) -> bool:
        if column_width <= 0:
            return False
        if (line.x1 - line.x0) > column_width * 1.5:
            return False
        weights: dict[int, float] = {}
        total_width = 0.0
        for group in column_run_groups(line.runs):
            group_box = run_group_bbox(group)
            if group_box is None:
                continue
            group_mid_x = (group_box[0] + group_box[2]) * 0.5
            column_index = 0
            while column_index < len(split_points) and group_mid_x >= split_points[column_index]:
                column_index += 1
            width = max(1.0, group_box[2] - group_box[0])
            weights[column_index] = weights.get(column_index, 0.0) + width
            total_width += width
        if total_width <= 0.0 or not weights:
            return False
        dominant_width = max(weights.values())
        return dominant_width / total_width >= 0.62

    @staticmethod
    def macro_column_x0_anchors(
        lines: list[LayoutLine],
        *,
        column_width: float,
    ) -> tuple[float, ...]:
        if column_width <= 0:
            return ()
        candidates = sorted(
            line.x0 for line in lines if line.runs and (line.x1 - line.x0) >= column_width * 0.55
        )
        if len(candidates) < 8:
            return ()
        tolerance = max(18.0, column_width * 0.14)
        clusters: list[list[float]] = []
        for x0 in candidates:
            if not clusters or abs(x0 - clusters[-1][-1]) > tolerance:
                clusters.append([x0])
            else:
                clusters[-1].append(x0)
        anchors = [sum(cluster) / len(cluster) for cluster in clusters if len(cluster) >= 4]
        return tuple(anchors)

    @staticmethod
    def line_macro_column_index(
        line: LayoutLine,
        anchors: tuple[float, ...],
        *,
        column_width: float,
    ) -> int | None:
        if not anchors or column_width <= 0:
            return None
        best_index = min(
            range(len(anchors)),
            key=lambda index: abs(line.x0 - anchors[index]),
        )
        if abs(line.x0 - anchors[best_index]) > max(28.0, column_width * 0.24):
            return None
        return best_index


def partition_lines_into_vertical_bands(
    lines: list[LayoutLine],
) -> list[list[LayoutLine]]:
    if len(lines) < 2:
        return [lines] if lines else []
    sorted_rows = sorted(lines, key=lambda ln: (-ln.mid_y, ln.x0))
    bands: list[list[LayoutLine]] = [[sorted_rows[0]]]
    for line in sorted_rows[1:]:
        current_band = bands[-1]
        previous = current_band[-1]
        if line_starts_new_vertical_band(current_band, previous, line):
            bands.append([line])
            continue
        current_band.append(line)
    return bands


def line_starts_new_vertical_band(
    current_band: list[LayoutLine],
    previous: LayoutLine,
    current: LayoutLine,
) -> bool:
    previous_height = max(1.0, previous.y1 - previous.y0)
    current_height = max(1.0, current.y1 - current.y0)
    line_height = max(1.0, min(previous_height, current_height))
    vertical_gap = previous.y0 - current.y1
    if vertical_gap > max(line_height * 1.6, 18.0):
        return True
    if len(current_band) < 6:
        return False
    current_width = max(1.0, current.x1 - current.x0)
    band_widths = sorted(max(1.0, line.x1 - line.x0) for line in current_band[-8:])
    median_width = band_widths[len(band_widths) // 2]
    band_x0 = min(line.x0 for line in current_band[-8:])
    band_x1 = max(line.x1 for line in current_band[-8:])
    current_extends_right = current.x1 > band_x1 + max(line_height * 2.5, 42.0)
    current_resets_left = current.x0 <= band_x0 + max(line_height * 1.5, 18.0)
    return bool(
        current_width > median_width * 1.45 and current_extends_right and current_resets_left
    )


def stable_multi_column_split_texts_look_safe(texts: list[str]) -> bool:
    if len(texts) != 2:
        return False
    token_lists = [normalized_text_tokens(text) for text in texts]
    if any(len(tokens) < 2 for tokens in token_lists):
        return False
    if any(
        sum(1 for token in tokens if any(ch.isalpha() for ch in token)) < 2
        for tokens in token_lists
    ):
        return False
    if any(
        len(tokens) <= 3 and sum(1 for token in tokens if any(ch.isdigit() for ch in token)) >= 1
        for tokens in token_lists
    ):
        return False
    return not any(len(text) < 18 for text in texts)


def stable_multi_column_split_is_safe(
    texts: list[str],
    *,
    split_lines: list[LayoutLine],
    split_points: list[float],
    max_gap: float,
) -> bool:
    if stable_multi_column_split_texts_look_safe(texts):
        return True
    marker_index = short_numeric_marker_fragment_index(texts)
    if marker_index is None or len(split_lines) != 2:
        return False
    prose_index = 1 - marker_index
    prose_text = texts[prose_index]
    if not stable_multi_column_prose_fragment_looks_safe(prose_text):
        return False
    marker_line = split_lines[marker_index]
    prose_line = split_lines[prose_index]
    marker_width = marker_line.x1 - marker_line.x0
    prose_width = prose_line.x1 - prose_line.x0
    if marker_width > 24.0 or prose_width < 120.0:
        return False
    if max_gap < max(32.0, marker_width * 6.0):
        return False
    marker_column = LayoutReconstructor.dominant_line_column_index(
        marker_line,
        split_points,
    )
    prose_column = LayoutReconstructor.dominant_line_column_index(
        prose_line,
        split_points,
    )
    return not (marker_column is None or prose_column is None or marker_column == prose_column)


def short_numeric_marker_fragment_index(texts: list[str]) -> int | None:
    if len(texts) != 2:
        return None
    for index, text in enumerate(texts):
        stripped = text.strip()
        if not stripped or len(stripped) > 4:
            continue
        tokens = normalized_text_tokens(stripped)
        if len(tokens) != 1:
            continue
        token = tokens[0]
        if token and all(ch.isdigit() for ch in token):
            return index
    return None


def stable_multi_column_prose_fragment_looks_safe(text: str) -> bool:
    tokens = normalized_text_tokens(text)
    if len(tokens) < 4 or len(text) < 18:
        return False
    alpha_tokens = sum(1 for token in tokens if any(ch.isalpha() for ch in token))
    return alpha_tokens >= 3


def render_resolved_text_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> str:
    return observation_resolver.text_from_resolved_lines(resolved_text_lines_for_output(lines))


def resolved_text_lines_for_output(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    resolved = resolve_native_paragraph_continuations(resolve_line_continuations(lines))
    return prune_standalone_separator_lines(resolved)


def with_first_line_break(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    break_before: int,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if not lines:
        return ()
    first, *rest = lines
    adjusted = observation_resolver.ResolvedTextLine(
        first.text,
        first.observation,
        break_before=break_before,
        contributing_observations=first.contributing_observations,
        resolution=first.resolution,
    )
    return (adjusted, *rest)


def layout_line_to_resolved_text_line(
    line: LayoutLine,
    *,
    break_before: int,
) -> observation_resolver.ResolvedTextLine | None:
    reconstructed = line.reconstructed_text()
    text = reconstructed.text.strip()
    if not text:
        return None
    contributing_observation_list: list[page_geometry.PageObservation] = []
    for index, segment in enumerate(reconstructed.segments):
        segment_observation = layout_segment_observation(segment, segment_index=index)
        if segment_observation is not None:
            contributing_observation_list.append(segment_observation)
    contributing_observations = tuple(contributing_observation_list)
    observation = page_geometry.PageObservation(
        kind="native_line",
        source="native_text",
        bbox=(line.x0, line.y0, line.x1, line.y1),
        advance_bbox=(line.x0, line.y0, line.x1, line.y1),
        ink_bbox=line_ink_bbox(contributing_observations) or (line.x0, line.y0, line.x1, line.y1),
        confidence=line_confidence(contributing_observations),
        text=text,
        provenance=page_geometry.provenance_tuple(
            object_type=type(line).__name__,
            min_order=line.min_order,
            max_order=line.max_order,
            max_depth=line.max_depth,
            rotation_angle=line.rotation_angle,
            is_vertical=line.is_vertical,
            run_count=len(line.runs),
        ),
    )
    return observation_resolver.ResolvedTextLine(
        text,
        observation,
        break_before=break_before,
        contributing_observations=contributing_observations,
    )


def layout_segment_observation(
    segment: LayoutLineTextSegment,
    *,
    segment_index: int,
) -> page_geometry.PageObservation | None:
    bbox = page_geometry.normalize_rect(segment.bbox)
    if not page_geometry.valid_rect(bbox):
        return None
    advance_bbox = page_geometry.normalize_rect(segment.advance_bbox)
    ink_bbox = page_geometry.normalize_rect(segment.ink_bbox)
    return page_geometry.PageObservation(
        kind="native_line_segment",
        source="native_text",
        bbox=bbox,
        advance_bbox=advance_bbox,
        ink_bbox=ink_bbox,
        confidence=segment.confidence,
        text=segment.text,
        baseline=segment.baseline,
        provenance=(
            *segment.provenance,
            ("object_type", "LayoutLineTextSegment"),
            ("segment_index", segment_index),
            ("spacing_decision", segment.spacing_decision),
            ("separator_before", segment.separator_before),
            ("writing_mode", segment.writing_mode),
            ("rotation_angle", segment.rotation_angle),
            ("visible", segment.visible),
        ),
    )


def line_ink_bbox(
    observations: tuple[page_geometry.PageObservation, ...],
) -> page_geometry.Rect | None:
    boxes: list[page_geometry.Rect] = []
    for observation in observations:
        box = observation.ink_bbox if observation.ink_bbox is not None else observation.bbox
        if box is not None:
            boxes.append(box)
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def line_confidence(
    observations: tuple[page_geometry.PageObservation, ...],
) -> float | None:
    values = [
        observation.confidence for observation in observations if observation.confidence is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def render_page_text(
    runs: list[TextRun],
    *,
    rotate: int = 0,
    media_box: tuple[float, float, float, float] | None = None,
    layout: bool = True,
) -> str:
    if not runs:
        return ""

    rotate = rotate % 360

    if rotate != 0 and media_box:
        runs = rotated_text_runs(runs, rotate=rotate, media_box=media_box)

    if not layout:
        return "".join(strip_private_use_chars(r.text) for r in runs if r.text)

    reconstructor = LayoutReconstructor(media_box or (0, 0, 612, 792))
    return reconstructor.render_page(runs)


def render_page_observation_lines(
    runs: list[TextRun],
    *,
    rotate: int = 0,
    media_box: tuple[float, float, float, float] | None = None,
    layout: bool = True,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if not runs:
        return ()
    rotate = rotate % 360
    if rotate != 0 and media_box:
        runs = rotated_text_runs(runs, rotate=rotate, media_box=media_box)
    if not layout:
        text = "".join(strip_private_use_chars(r.text) for r in runs if r.text).strip()
        if not text:
            return ()
        bbox = (
            min(run.x0 for run in runs),
            min(run.y0 for run in runs),
            max(run.x1 for run in runs),
            max(run.y1 for run in runs),
        )
        observation = page_geometry.PageObservation(
            kind="native_text",
            source="native_text",
            bbox=bbox,
            advance_bbox=bbox,
            ink_bbox=bbox,
            confidence=None,
            text=text,
            provenance=page_geometry.provenance_tuple(
                object_type="linear_text",
                run_count=len(runs),
            ),
        )
        return (
            observation_resolver.ResolvedTextLine(
                text,
                observation,
                contributing_observations=(observation,),
            ),
        )
    reconstructor = LayoutReconstructor(media_box or (0, 0, 612, 792))
    return reconstructor.render_page_lines(runs)


def rotated_text_runs(
    runs: list[TextRun],
    *,
    rotate: int,
    media_box: tuple[float, float, float, float],
) -> list[TextRun]:
    x0_mb, y0_mb, x1_mb, y1_mb = media_box
    page_width, page_height = x1_mb - x0_mb, y1_mb - y0_mb
    transformed = []
    for run in runs:
        if rotate == 90:
            new_x0, new_y0, new_x1, new_y1 = (
                run.y0,
                page_width - run.x1,
                run.y1,
                page_width - run.x0,
            )
        elif rotate == 180:
            new_x0, new_y0, new_x1, new_y1 = (
                page_width - run.x1,
                page_height - run.y1,
                page_width - run.x0,
                page_height - run.y0,
            )
        elif rotate == 270:
            new_x0, new_y0, new_x1, new_y1 = (
                page_height - run.y1,
                run.x0,
                page_height - run.y0,
                run.x1,
            )
        else:
            new_x0, new_y0, new_x1, new_y1 = run.x0, run.y0, run.x1, run.y1
        transformed.append(
            run.replace(
                x0=new_x0,
                y0=new_y0,
                x1=new_x1,
                y1=new_y1,
                rotation_angle=(run.rotation_angle - rotate) % 360,
            )
        )
    return transformed


SOFT_HYPHEN_CHARS = frozenset({"-", "–", "—"})
SEPARATOR_LINE_CHARS = frozenset({"-", "–", "—", "_", "="})


def resolve_line_continuations(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if len(lines) < 2:
        return lines

    output: list[observation_resolver.ResolvedTextLine] = []
    index = 0
    while index < len(lines):
        current = lines[index]
        next_index = index + 1
        while next_index < len(lines):
            if next_index + 1 < len(lines):
                merged = merge_standalone_soft_hyphen_continuation(
                    current,
                    lines[next_index],
                    lines[next_index + 1],
                )
                if merged is not None:
                    current = merged
                    next_index += 2
                    continue
            merged = merge_direct_soft_hyphen_continuation(current, lines[next_index])
            if merged is None:
                break
            current = merged
            next_index += 1
        output.append(current)
        index = next_index
    return tuple(output)


def resolve_native_paragraph_continuations(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if len(lines) < 2:
        return lines
    output: list[observation_resolver.ResolvedTextLine] = []
    index = 0
    while index < len(lines):
        current = lines[index]
        next_index = index + 1
        while next_index < len(lines):
            merged = merge_native_paragraph_hyphen_continuation(
                current,
                lines[next_index],
            )
            if merged is None:
                break
            current = merged
            next_index += 1
        output.append(current)
        index = next_index
    return tuple(output)


def prune_standalone_separator_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if not lines:
        return lines
    pruned: list[observation_resolver.ResolvedTextLine] = []
    changed = False
    for line in lines:
        if line_is_standalone_separator(line):
            changed = True
            continue
        pruned.append(line)
    return tuple(pruned) if changed else lines


def merge_direct_soft_hyphen_continuation(
    left: observation_resolver.ResolvedTextLine,
    right: observation_resolver.ResolvedTextLine,
) -> observation_resolver.ResolvedTextLine | None:
    if right.break_before > 1 or not lines_form_same_column_continuation(left, right):
        return None
    left_core = remove_trailing_soft_hyphen(left.text)
    if left_core == left.text:
        return None
    right_core = strip_leading_soft_hyphen_artifact(right.text)
    if not line_join_word_is_plausible(left_core, right_core):
        return None
    return merged_resolved_line(
        left,
        right,
        text=left_core.rstrip() + right_core.lstrip(),
        reason="soft_hyphen_continuation",
    )


def merge_native_paragraph_hyphen_continuation(
    left: observation_resolver.ResolvedTextLine,
    right: observation_resolver.ResolvedTextLine,
) -> observation_resolver.ResolvedTextLine | None:
    if not lines_form_same_native_paragraph(left, right):
        return None
    left_core = remove_trailing_soft_hyphen(left.text)
    if left_core == left.text:
        return None
    right_core = strip_leading_soft_hyphen_artifact(right.text)
    if not line_join_word_is_plausible(left_core, right_core):
        return None
    return merged_resolved_line(
        left,
        right,
        text=left_core.rstrip() + right_core.lstrip(),
        reason="native_paragraph_hyphen_continuation",
    )


def merge_standalone_soft_hyphen_continuation(
    left: observation_resolver.ResolvedTextLine,
    hyphen: observation_resolver.ResolvedTextLine,
    right: observation_resolver.ResolvedTextLine,
) -> observation_resolver.ResolvedTextLine | None:
    if hyphen.break_before > 1 or right.break_before > 1:
        return None
    if not line_is_standalone_soft_hyphen(hyphen):
        return None
    if not lines_form_same_column_continuation(left, right):
        return None
    if not standalone_hyphen_bridges_lines(left, hyphen, right):
        return None
    if not line_join_word_is_plausible(left.text, right.text):
        return None
    return merged_resolved_line(
        left,
        right,
        text=left.text.rstrip() + right.text.lstrip(),
        reason="standalone_soft_hyphen_continuation",
        skipped=(hyphen,),
    )


def lines_form_same_column_continuation(
    left: observation_resolver.ResolvedTextLine,
    right: observation_resolver.ResolvedTextLine,
) -> bool:
    left_box = line_effective_bbox(left)
    right_box = line_effective_bbox(right)
    if left_box is None or right_box is None:
        return False

    left_height = left_box[3] - left_box[1]
    right_height = right_box[3] - right_box[1]
    line_height = max(1.0, min(left_height, right_height))
    left_mid_y = (left_box[1] + left_box[3]) * 0.5
    right_mid_y = (right_box[1] + right_box[3]) * 0.5
    if left_mid_y <= right_mid_y:
        return False

    vertical_gap = left_box[1] - right_box[3]
    if vertical_gap < -line_height * 0.45:
        return False
    if vertical_gap > max(line_height * 1.7, 18.0):
        return False

    x_overlap = max(0.0, min(left_box[2], right_box[2]) - max(left_box[0], right_box[0]))
    narrow_width = max(1.0, min(left_box[2] - left_box[0], right_box[2] - right_box[0]))
    same_indent = abs(left_box[0] - right_box[0]) <= max(line_height * 1.5, 16.0)
    return same_indent or (x_overlap / narrow_width) >= 0.25


def lines_form_same_native_paragraph(
    left: observation_resolver.ResolvedTextLine,
    right: observation_resolver.ResolvedTextLine,
) -> bool:
    if left.observation.source != "native_text" or right.observation.source != "native_text":
        return False
    if right.break_before > 2:
        return False
    left_box = line_effective_bbox(left)
    right_box = line_effective_bbox(right)
    if left_box is None or right_box is None:
        return False
    left_height = max(1.0, left_box[3] - left_box[1])
    right_height = max(1.0, right_box[3] - right_box[1])
    line_height = max(1.0, min(left_height, right_height))
    if left_box[1] <= right_box[3]:
        return False
    vertical_gap = left_box[1] - right_box[3]
    if vertical_gap > max(line_height * 2.4, 26.0):
        return False
    if abs(left_box[0] - right_box[0]) > max(line_height * 1.8, 20.0):
        return False
    left_width = max(1.0, left_box[2] - left_box[0])
    right_width = max(1.0, right_box[2] - right_box[0])
    if max(left_width, right_width) / max(1.0, min(left_width, right_width)) > 3.2:
        return False
    return not (right.break_before == 2 and not native_paragraph_line_pair_looks_prose(left, right))


def native_paragraph_line_pair_looks_prose(
    left: observation_resolver.ResolvedTextLine,
    right: observation_resolver.ResolvedTextLine,
) -> bool:
    left_tokens = normalized_text_tokens(left.text)
    right_tokens = normalized_text_tokens(right.text)
    if len(left_tokens) < 3 or len(right_tokens) < 2:
        return False
    return not any(any(ch.isdigit() for ch in token) for token in (*left_tokens, *right_tokens))


def standalone_hyphen_bridges_lines(
    left: observation_resolver.ResolvedTextLine,
    hyphen: observation_resolver.ResolvedTextLine,
    right: observation_resolver.ResolvedTextLine,
) -> bool:
    left_box = line_effective_bbox(left)
    hyphen_box = line_effective_bbox(hyphen)
    right_box = line_effective_bbox(right)
    if left_box is None or hyphen_box is None or right_box is None:
        return False
    left_height = max(1.0, left_box[3] - left_box[1])
    hyphen_width = hyphen_box[2] - hyphen_box[0]
    if hyphen_width > max(10.0, left_height * 0.9):
        return False
    y_overlap = max(
        0.0,
        min(left_box[3], hyphen_box[3]) - max(left_box[1], hyphen_box[1]),
    )
    min_height = max(1.0, min(left_height, hyphen_box[3] - hyphen_box[1]))
    if y_overlap / min_height < 0.25:
        return False
    bridge_gap = hyphen_box[0] - left_box[2]
    if bridge_gap < -left_height * 0.35:
        return False
    if bridge_gap > max(14.0, left_height * 1.2):
        return False
    return right_box[0] <= left_box[0] + max(16.0, left_height * 1.5)


def line_effective_bbox(
    line: observation_resolver.ResolvedTextLine,
) -> page_geometry.Rect | None:
    return line.observation.ink_bbox or line.observation.bbox


def merged_resolved_line(
    left: observation_resolver.ResolvedTextLine,
    right: observation_resolver.ResolvedTextLine,
    *,
    text: str,
    reason: str,
    skipped: tuple[observation_resolver.ResolvedTextLine, ...] = (),
) -> observation_resolver.ResolvedTextLine:
    observations = (
        left.observation,
        *(line.observation for line in skipped),
        right.observation,
    )
    bbox = union_observation_bbox(observations, "bbox")
    advance_bbox = union_observation_bbox(observations, "advance_bbox")
    ink_bbox = union_observation_bbox(observations, "ink_bbox") or bbox
    confidence_values = [
        observation.confidence for observation in observations if observation.confidence is not None
    ]
    confidence = sum(confidence_values) / len(confidence_values) if confidence_values else None
    observation = replace(
        left.observation,
        text=text,
        bbox=bbox,
        advance_bbox=advance_bbox,
        ink_bbox=ink_bbox,
        confidence=confidence,
        provenance=(
            *left.observation.provenance,
            *page_geometry.provenance_tuple(
                line_resolution="continuation",
                continuation_reason=reason,
                merged_line_count=2 + len(skipped),
            ),
        ),
    )
    return observation_resolver.ResolvedTextLine(
        text,
        observation,
        break_before=left.break_before,
        contributing_observations=(
            *left.contributing_observations,
            *(item for line in skipped for item in line.contributing_observations),
            *right.contributing_observations,
        ),
        resolution=left.resolution,
    )


def union_observation_bbox(
    observations: tuple[page_geometry.PageObservation, ...],
    attr: str,
) -> page_geometry.Rect | None:
    boxes = [
        box
        for observation in observations
        if (box := page_geometry.normalize_rect(getattr(observation, attr))) is not None
    ]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def remove_trailing_soft_hyphen(text: str) -> str:
    end = len(text)
    while end > 0 and text[end - 1].isspace():
        end -= 1
    if end == 0 or text[end - 1] not in SOFT_HYPHEN_CHARS:
        return text
    return text[: end - 1] + text[end:]


def strip_leading_soft_hyphen_artifact(text: str) -> str:
    index = 0
    while index < len(text) and text[index].isspace():
        index += 1
    if index < len(text) and text[index] in SOFT_HYPHEN_CHARS:
        index += 1
        while index < len(text) and text[index].isspace():
            index += 1
    return text[index:]


def line_is_standalone_soft_hyphen(
    line: observation_resolver.ResolvedTextLine,
) -> bool:
    stripped = line.text.strip()
    return len(stripped) == 1 and stripped in SOFT_HYPHEN_CHARS


def line_is_standalone_separator(
    line: observation_resolver.ResolvedTextLine,
) -> bool:
    stripped = line.text.strip()
    if len(stripped) < 2:
        return False
    if any(ch.isalnum() for ch in stripped):
        return False
    if any(ch not in SEPARATOR_LINE_CHARS for ch in stripped):
        return False
    return len({ch for ch in stripped if not ch.isspace()}) == 1


def line_join_word_is_plausible(left_text: str, right_text: str) -> bool:
    prefix = trailing_alpha_fragment(left_text)
    suffix = leading_alpha_fragment(strip_leading_soft_hyphen_artifact(right_text))
    if len(prefix) < 2 or len(suffix) < 2:
        return False
    if not suffix[:1].islower():
        return False
    joined = f"{prefix}{suffix}".casefold()
    joined_rank = word_rank(joined)
    if joined_rank is None:
        return False
    prefix_rank = word_rank(prefix.casefold())
    suffix_rank = word_rank(suffix.casefold())
    if productive_prefix_join_is_plausible(
        prefix,
        suffix,
        joined_rank=joined_rank,
        prefix_rank=prefix_rank,
        suffix_rank=suffix_rank,
    ):
        return True
    if joined_rank > 150_000:
        return False
    if prefix_rank is None or suffix_rank is None:
        return True
    return joined_rank < max(prefix_rank, suffix_rank) or (
        len(suffix) <= 4 and joined_rank <= 75_000
    )


def productive_prefix_join_is_plausible(
    prefix: str,
    suffix: str,
    *,
    joined_rank: int,
    prefix_rank: int | None,
    suffix_rank: int | None,
) -> bool:
    if prefix.casefold() != "multi" or suffix_rank is None:
        return False
    if prefix_rank is not None and prefix_rank > 20_000:
        return False
    return joined_rank <= 350_000 and suffix_rank <= 150_000


def trailing_alpha_fragment(text: str) -> str:
    end = len(text)
    while end > 0 and text[end - 1].isspace():
        end -= 1
    if end > 0 and text[end - 1] in SOFT_HYPHEN_CHARS:
        end -= 1
        while end > 0 and text[end - 1].isspace():
            end -= 1
    start = end
    while start > 0 and text[start - 1].isalpha():
        start -= 1
    return text[start:end]


def leading_alpha_fragment(text: str) -> str:
    index = 0
    while index < len(text) and text[index].isspace():
        index += 1
    start = index
    while index < len(text) and text[index].isalpha():
        index += 1
    return text[start:index]


def column_run_groups(runs: list[TextRun]) -> list[list[TextRun]]:
    sorted_runs = sorted(runs, key=lambda run: (run.x0, run.order, run.stream_order))
    groups: list[list[TextRun]] = []
    current: list[TextRun] = []
    for run in sorted_runs:
        if not current:
            current.append(run)
            continue
        if column_group_breaks_between(current[-1], run):
            groups.append(current)
            current = [run]
            continue
        current.append(run)
    if current:
        groups.append(current)
    return groups


def column_group_breaks_between(left: TextRun, right: TextRun) -> bool:
    if left.text_is_space or right.text_is_space:
        return True
    if not left.has_text or not right.has_text:
        return True
    if left.rotation_angle != right.rotation_angle or left.is_vertical != right.is_vertical:
        return True
    if left.text and left.text[-1].isspace():
        return True
    if right.text and right.text[0].isspace():
        return True

    y_overlap = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
    min_height = max(1.0, min(left.height, right.height))
    if y_overlap / min_height < 0.45:
        return True

    gap = right.x0 - left.x1
    if gap < -max(1.5, min_height * 0.2):
        return False
    allowed_gap = max(1.8, min(left.space_width, right.space_width) * 0.45)
    return gap > allowed_gap


def run_group_bbox(group: list[TextRun]) -> page_geometry.Rect | None:
    boxes: list[page_geometry.Rect] = []
    for run in group:
        box = page_geometry.normalize_rect(run.ink_bbox or run.advance_bbox)
        if box is not None:
            boxes.append(box)
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def narrow_boundary_punctuation_run(run: TextRun) -> bool:
    text = run.text.strip()
    if text not in {"-", "–", "—"}:
        return False
    width = max(0.0, run.x1 - run.x0)
    height = max(0.0, run.y1 - run.y0)
    return width <= max(5.0, height * 0.5)


class MarkdownRenderer:
    __slots__ = (
        "page",
        "runs",
        "avg_fs",
        "vboxes_cache",
        "field_labels_cache",
        "separators_cache",
        "tables_cache",
    )

    page: PdfPage
    runs: list[TextRun]
    avg_fs: float
    vboxes_cache: list[RectBox] | None
    field_labels_cache: dict[str, str] | None
    separators_cache: list[CapturedLine] | None
    tables_cache: list[list[list[object]]] | None

    def __init__(self, page: PdfPage) -> None:
        self.page = page
        self.runs = page.chars
        self.vboxes_cache = None
        self.field_labels_cache = None
        self.separators_cache = None
        self.tables_cache = None

        all_fs = [r.font_size for r in self.runs if r.has_text]
        self.avg_fs = sum(all_fs) / len(all_fs) if all_fs else 12.0

    @property
    def vboxes(self) -> list[Any]:
        vbs = self.vboxes_cache
        if vbs is None:
            vbs = LayoutAnalyzer.detect_visual_boxes(self.page)
            self.vboxes_cache = vbs
        return vbs

    @property
    def separators(self) -> list[Any]:
        seps = self.separators_cache
        if seps is None:
            seps = LayoutAnalyzer.find_geometric_separators(self.page)
            self.separators_cache = seps
        return seps

    @property
    def field_labels(self) -> dict[str, str]:
        labels = self.field_labels_cache
        if labels is None:
            labels = LayoutAnalyzer.associate_field_labels(self.page)
            self.field_labels_cache = labels
        return labels

    @property
    def tables(self) -> list[list[list[object]]] | None:
        tbls = self.tables_cache
        if tbls is None:
            extract_tables = getattr(self.page, "extract_tables", None)
            tbls = extract_tables(flavor="lattice") if callable(extract_tables) else None
            self.tables_cache = tbls
        return tbls

    def render(self) -> str:
        if not self.runs:
            return ""

        lines = LayoutAnalyzer.cluster_into_lines(self.runs)
        if not lines:
            return ""
        boxes = LayoutAnalyzer.order_boxes(LayoutAnalyzer.cluster_into_boxes(lines))

        md_parts: list[str] = []

        current_y = 1e9
        seps = self.separators
        for box in boxes:
            remaining_seps = []
            for sep in seps:
                if abs(sep.y1 - sep.y0) < 1.5:
                    if current_y > sep.y0 > box.y1:
                        md_parts.append("---")
                        current_y = sep.y0
                    else:
                        remaining_seps.append(sep)
                else:
                    remaining_seps.append(sep)
            seps = remaining_seps

            md_parts.append(self.render_box(box))
            current_y = box.y0

        fields = self.page.get_fields()
        if fields:
            md_parts.append("### Form Fields")
            for field in fields:
                label = self.field_labels.get(field.name, field.name)
                val = str(field.value) if field.value is not None else "[Empty]"
                md_parts.append(f"- **{label}** ({field.type}): {val}")

        annots = self.page.get_annotations()
        if annots:
            self.render_annotations(md_parts, annots)

        for table in self.tables if isinstance(self.tables, list) else []:
            if not table:
                continue
            rendered = self.render_table(table)
            if rendered:
                md_parts.append(rendered)

        return "\n\n".join(md_parts)

    def render_table(self, table: list[list[object]]) -> str | None:

        if not table or not isinstance(table[0], list):
            raise ValueError("invalid table structure")
        n_cols = len(table[0])
        if n_cols == 0:
            return None
        empty_cols = set()
        for c in range(n_cols):
            for row in table:
                if not isinstance(row, list) or len(row) != n_cols:
                    raise ValueError("invalid table structure")
            if all(not str(row[c]).strip() for row in table):
                empty_cols.add(c)

        if len(empty_cols) == n_cols:
            return None

        md_table = []
        for row in table:
            cells = [str(row[c]).replace("\n", " ") for c in range(n_cols) if c not in empty_cols]
            md_table.append("| " + " | ".join(cells) + " |")

        if not md_table:
            return None

        header_sep = "|" + "|".join("---" for ignored in range(n_cols - len(empty_cols))) + "|"
        md_table.insert(1, header_sep)
        return "\n" + "\n".join(md_table)

    def render_box(self, box: LayoutBox) -> str:
        is_boxed = any(vb.contains_rect(box.bbox_rect) for vb in self.vboxes)

        box_text_lines = []
        for line in box.lines:
            line_text = line.text()
            if not line_text.strip():
                continue

            max_fs = line.max_font_size
            is_all_caps = line.is_all_caps_text

            if max_fs > self.avg_fs * 1.5:
                box_text_lines.append(f"# {line_text}")
            elif max_fs > self.avg_fs * 1.2 or (is_all_caps and max_fs > self.avg_fs):
                box_text_lines.append(f"## {line_text}")
            else:
                box_text_lines.append(line_text)

        text = "\n".join(box_text_lines)
        if is_boxed:
            text = "> " + text.replace("\n", "\n> ")
        return text

    def render_annotations(self, md_parts: list[str], annots: list[AnnotationRecord]) -> None:
        links = [a for a in annots if a.subtype == "Link"]
        if links:
            md_parts.append("### Links")
            for link in links:
                if link.contents:
                    md_parts.append(f"- {link.contents}")

        notes = [a for a in annots if a.subtype == "Text" and a.contents]
        if notes:
            md_parts.append("### Notes")
            for note in notes:
                md_parts.append(f"> {note.contents}")

    @classmethod
    def render_page(cls, page: PdfPage, *, rotate: int = 0) -> str:
        return cls(page).render()
