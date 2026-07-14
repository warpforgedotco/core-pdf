# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import re
import typing
from bisect import bisect_left, bisect_right
from statistics import median_low

if typing.TYPE_CHECKING:
    from core_pdf.impl.engine.spec.s_07_document.models import AnnotationRecord
    from core_pdf.impl.engine.spec.s_07_document.page import PdfPage

from core_pdf.impl.engine.spec.s_07_content.models import LayoutBox, LayoutLine, TextRun
from core_pdf.impl.engine.spec.s_07_content.ordering import (
    SKEW_ANGLE_TOLERANCE,
    LayoutAnalyzer,
)


def render_run_line(runs: list[TextRun]) -> str:
    if not runs:
        return ""
    angle = runs[0].rotation_angle

    if LayoutAnalyzer.is_right_to_left_line(runs):
        sorted_runs = sorted(runs, key=lambda r: (r.order, r.stream_order))
    else:
        if angle == 90:
            sorted_runs = sorted(runs, key=lambda r: (r.y0, r.order))
        elif angle == 270:
            sorted_runs = sorted(runs, key=lambda r: (-r.y1, r.order))
        else:
            sorted_runs = sorted(runs, key=lambda r: (r.x0, r.order))

    # Identify tables or all-caps lines for specialized spacing
    is_table_like_line = (
        len(sorted_runs) >= 4 and sum(1 for r in sorted_runs if r.text.strip().isdigit()) >= 2
    )
    is_all_caps_line = len(sorted_runs) >= 2 and all(
        r.text.strip().isupper() for r in sorted_runs if r.text.strip()
    )

    return render_sorted_runs(
        sorted_runs,
        is_table_like_line=is_table_like_line,
        is_all_caps_line=is_all_caps_line,
    )


def make_layout_line(runs: list[TextRun]) -> LayoutLine:
    line = LayoutLine()
    for run in runs:
        line.add(run)
    return line


def line_has_gap_at_split(runs: list[TextRun], split_x: float) -> bool:
    text_runs = sorted((r for r in runs if r.text.strip()), key=lambda r: (r.x0, r.order))
    prev_x1: float | None = None
    for run in text_runs:
        if run.x0 < split_x < run.x1:
            return False
        if prev_x1 is not None:
            gap = run.x0 - prev_x1
            if prev_x1 <= split_x <= run.x0 and gap >= 8.0:
                return True
        if prev_x1 is None or run.x1 > prev_x1:
            prev_x1 = run.x1
    return False


def run_has_text(run: TextRun) -> bool:
    return bool(run.text and not run.text.isspace())


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
    if not run_has_text(left) or not run_has_text(right):
        return True
    if left.rotation_angle != right.rotation_angle or left.is_vertical != right.is_vertical:
        return True
    if left.text[-1].isspace() or right.text[0].isspace():
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


def run_group_bbox(group: list[TextRun]) -> tuple[float, float, float, float] | None:
    text_runs = [run for run in group if run_has_text(run)]
    if not text_runs:
        return None
    return (
        min(run.x0 for run in text_runs),
        min(run.y0 for run in text_runs),
        max(run.x1 for run in text_runs),
        max(run.y1 for run in text_runs),
    )


class LayoutReconstructor:
    """Internal helper to manage recursive layout reconstruction state."""

    __slots__ = ("media_box",)

    def __init__(self, media_box: tuple[float, float, float, float]) -> None:
        self.media_box = media_box

    def render_page(self, runs: list[TextRun]) -> str:
        if not runs:
            return ""

        # 1. Skew correction
        if len(runs) >= 50:
            skew_angle = LayoutAnalyzer.detect_skew(runs)
            if abs(skew_angle) > SKEW_ANGLE_TOLERANCE:
                runs = LayoutAnalyzer.apply_skew_correction(runs)

        # 2. Re-cluster into lines
        lines = LayoutAnalyzer.cluster_into_lines(runs)
        if not lines:
            return ""

        # 3. Cluster lines into boxes and order them
        boxes = LayoutAnalyzer.order_boxes(LayoutAnalyzer.cluster_into_boxes(lines))

        # 4. Group boxes by rotation
        boxes_by_angle: dict[int, list[LayoutBox]] = {}
        for box in boxes:
            angle = box.lines[0].rotation_angle if box.lines else 0
            boxes_by_angle.setdefault(angle, []).append(box)

        ordered_boxes = []
        for angle in sorted(boxes_by_angle):
            ordered_boxes.extend(boxes_by_angle[angle])

        # 5. Final recursive rendering
        box_texts = [self.render_box(box.lines) for box in ordered_boxes]
        return "\n\n".join(text for text in box_texts if text)

    def render_box(self, lines: list[LayoutLine]) -> str:
        """Render a box to text, handling stable multi-column regions."""
        split_points = self.find_column_splits(lines)
        if split_points and (
            self.column_split_is_stable(lines, split_points)
            or self.multi_column_split_is_stable(lines, split_points)
            or (
                len(split_points) == 1
                and self.parallel_column_split_is_viable(lines, split_points[0])
            )
        ):
            return self.render_columnar_lines(lines, split_points)
        return self.render_rows(lines)

    def render_columnar_lines(self, lines: list[LayoutLine], split_points: list[float]) -> str:
        if not lines:
            return ""
        box_x0 = min(line.x0 for line in lines if line.runs)
        box_x1 = max(line.x1 for line in lines if line.runs)
        boundaries = [box_x0, *split_points, box_x1]
        column_count = len(boundaries) - 1
        column_width = max(
            boundaries[index + 1] - boundaries[index] for index in range(column_count)
        )
        sorted_rows = sorted(lines, key=lambda ln: (-ln.mid_y, ln.x0))
        output_parts: list[str] = []
        column_block: list[LayoutLine] = []

        def append_part(text: str) -> None:
            if text:
                output_parts.append(text)

        def flush_columns() -> None:
            if not column_block:
                return
            column_lines: list[list[LayoutLine]] = [[] for _ in range(column_count)]
            for line in column_block:
                split_runs: list[list[TextRun]] = [[] for _ in range(column_count)]
                for group in column_run_groups(line.runs):
                    group_box = run_group_bbox(group)
                    if group_box is None:
                        continue
                    column_position = (group_box[0] + group_box[2]) * 0.5
                    column_index = 0
                    while (
                        column_index < len(split_points)
                        and column_position >= split_points[column_index]
                    ):
                        column_index += 1
                    split_runs[column_index].extend(group)
                for column_index, runs in enumerate(split_runs):
                    if runs:
                        column_lines[column_index].append(make_layout_line(runs))

            for lines_for_column in column_lines:
                if lines_for_column:
                    append_part(self.render_box(lines_for_column))
            column_block.clear()

        for line in sorted_rows:
            if not line.runs:
                continue
            is_column_line = self.line_has_column_gutter(
                line, split_points, column_width
            )
            if not is_column_line and column_block:
                is_column_line = (line.x1 - line.x0) <= column_width * 1.15
            if is_column_line:
                column_block.append(line)
                continue

            flush_columns()
            append_part(self.render_line(line))

        flush_columns()
        return "\n\n".join(part for part in output_parts if part)

    @staticmethod
    def render_rows(lines: list[LayoutLine]) -> str:
        line_texts = []
        for line in sorted(lines, key=lambda ln: (-ln.mid_y, ln.x0)):
            line_text = LayoutReconstructor.render_line(line)
            if line_text:
                line_texts.append(line_text)
        return "\n".join(line_texts)

    @staticmethod
    def render_line(line: LayoutLine) -> str:
        if not line.runs:
            return ""
        line_text = render_run_line(line.runs).rstrip()
        return line_text if line_text.strip() else ""

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
                if run_has_text(r):
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
                if gap >= 8.0:
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
            if len(cluster) < 5:
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
    def column_split_is_stable(lines: list[LayoutLine], split_points: list[float]) -> bool:
        if len(lines) < 24 or len(split_points) != 1:
            return False
        split_x = split_points[0]
        left_only: list[LayoutLine] = []
        right_only: list[LayoutLine] = []
        spanning = 0
        gutter_rows = 0

        for line in lines:
            runs = [run for run in line.runs if run_has_text(run)]
            if not runs:
                continue
            has_left = any(run.mid_x < split_x for run in runs)
            has_right = any(run.mid_x >= split_x for run in runs)
            if has_left and has_right:
                spanning += 1
                left_edge = max((run.x1 for run in runs if run.mid_x < split_x), default=split_x)
                right_edge = min(
                    (run.x0 for run in runs if run.mid_x >= split_x), default=split_x
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
        box_x0 = min(line.x0 for line in lines if line.runs)
        box_x1 = max(line.x1 for line in lines if line.runs)
        boundaries = [box_x0, *split_points, box_x1]
        widths = [
            boundaries[index + 1] - boundaries[index]
            for index in range(len(boundaries) - 1)
        ]
        if min(widths) <= 0 or max(widths) / min(widths) > 1.8:
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
                    column_index < len(split_points)
                    and group_mid_x >= split_points[column_index]
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
    def parallel_column_split_is_viable(lines: list[LayoutLine], split_x: float) -> bool:
        if len(lines) < 12:
            return False
        left_rows = 0
        right_rows = 0
        gutter_rows = 0
        for line in lines:
            runs = [run for run in line.runs if run_has_text(run)]
            if not runs:
                continue
            has_left = any(run.mid_x < split_x for run in runs)
            has_right = any(run.mid_x >= split_x for run in runs)
            if has_left:
                left_rows += 1
            if has_right:
                right_rows += 1
            if has_left and has_right:
                left_edge = max((run.x1 for run in runs if run.mid_x < split_x), default=split_x)
                right_edge = min(
                    (run.x0 for run in runs if run.mid_x >= split_x), default=split_x
                )
                if right_edge - left_edge >= 8.0:
                    gutter_rows += 1
        if left_rows < 8 or right_rows < 8:
            return False
        return gutter_rows >= max(4, len(lines) // 6)

    @staticmethod
    def line_y_extent(lines: list[LayoutLine]) -> tuple[float, float]:
        return (
            max(line.y1 for line in lines if line.runs),
            min(line.y0 for line in lines if line.runs),
        )

    @staticmethod
    def line_has_column_gutter(
        line: LayoutLine, split_points: list[float], column_width: float | None = None
    ) -> bool:
        runs = [run for run in line.runs if run_has_text(run)]
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
                    rotation_angle=(run.rotation_angle + rotate) % 360,
                )
            )
        runs = transformed

    if not layout:
        return "".join(strip_private_use_chars(r.text) for r in runs if r.text)

    reconstructor = LayoutReconstructor(media_box or (0, 0, 612, 792))
    return reconstructor.render_page(runs)


PUA_RE = re.compile("[\ue000-\uf8ff\x00\xad]")
EXPAND_RE = re.compile(r"(?i)\bexpand\b")
SPACES_RE = re.compile(r" {2,}")


def strip_private_use_chars(text: str) -> str:
    # Strip PUA, nulls, and soft hyphens using fast C regex
    return PUA_RE.sub("", text)


def render_sorted_runs(
    sorted_runs: list[TextRun], *, is_table_like_line: bool, is_all_caps_line: bool
) -> str:
    parts: list[str] = []
    append_part = parts.append

    prev_x1: float | None = None
    prev_x0: float = 0.0
    prev_y0: float = 0.0
    prev_y1: float = 0.0
    has_prev_bbox = False
    prev_text: str = ""

    # Pre-bind for speed
    strip_func = strip_private_use_chars

    for run in sorted_runs:
        text = run.text
        if not text:
            continue

        # Micro-optimization: avoid full strip if no special chars
        # \xad is soft hyphen, \x00 is null
        if "\x00" in text or "\xad" in text:
            text = strip_func(text)

        # "expand" artifact removal (only if present)
        if "expand" in text.lower():
            text = EXPAND_RE.sub("", text)

        if not text:
            continue

        if text.isspace() and run.x1 - run.x0 <= 0.01:
            continue

        # Deduplication: if this run overlaps significantly with the previous one
        if has_prev_bbox and prev_x1 is not None:
            # Inline overlap calculation
            ox = run.x1 if run.x1 < prev_x1 else prev_x1
            ox -= run.x0 if run.x0 > prev_x0 else prev_x0
            oy = run.y1 if run.y1 < prev_y1 else prev_y1
            oy -= run.y0 if run.y0 > prev_y0 else prev_y0

            if ox > 0 and oy > 0:
                run_w = run.x1 - run.x0
                prev_w = prev_x1 - prev_x0
                run_h = run.y1 - run.y0
                prev_h = prev_y1 - prev_y0
                min_w = run_w if run_w < prev_w else prev_w
                min_h = run_h if run_h < prev_h else prev_h
                if (
                    text == prev_text
                    and abs(run.x0 - prev_x0) <= max(1.5, min_w * 0.75)
                    and abs(run.y0 - prev_y0) <= max(1.5, min_h * 0.25)
                ):
                    continue
                box_area = run_h * run_w
                prev_area = prev_h * prev_w
                overlap_area = ox * oy
                min_area = box_area if box_area < prev_area else prev_area
                if min_area > 0 and text == prev_text and overlap_area / min_area > 0.7:
                    continue
                if (
                    box_area > 0
                    and overlap_area / box_area > 0.8
                    and (text == prev_text or (len(text) == len(prev_text) and len(text) <= 2))
                ):
                    continue

        if prev_x1 is not None:
            angle = run.rotation_angle
            if angle == 90:
                axis_gap = run.y0 - prev_y1
                thickness = run.x1 - run.x0
                threshold = max(run.space_width * 0.25, thickness * 0.12, 1.0)
                if axis_gap > threshold:
                    append_part(" ")
            elif angle == 270:
                axis_gap = prev_y0 - run.y1
                thickness = run.x1 - run.x0
                threshold = max(run.space_width * 0.25, thickness * 0.12, 1.0)
                if axis_gap > threshold:
                    append_part(" ")
            else:
                x_gap = run.x0 - prev_x1
                # Use effective font size (height) for thresholding
                rh = run.y1 - run.y0
                # More balanced threshold for spacing
                threshold = max(run.space_width * 0.25, rh * 0.12, 1.0)
                if is_all_caps_line:
                    threshold = min(threshold, 0.5)

                if x_gap > threshold:
                    append_part(" ")
                elif (prev_x1 - run.x0) > max(threshold * 4.0, rh * 2.0, 24.0):
                    append_part("\n")

        append_part(text)
        prev_x1 = run.x1
        prev_x0, prev_y0, prev_y1 = run.x0, run.y0, run.y1
        has_prev_bbox = True
        prev_text = text

    combined = "".join(parts)
    if "  " in combined:
        combined = SPACES_RE.sub(" ", combined)
    return combined


class MarkdownRenderer:
    __slots__ = (
        "page",
        "runs",
        "avg_fs",
        "vboxes",
        "field_labels",
        "separators",
        "tables",
    )

    def __init__(self, page: PdfPage) -> None:
        self.page = page
        self.runs = page.chars
        self.vboxes = LayoutAnalyzer.detect_visual_boxes(page)
        self.field_labels = LayoutAnalyzer.associate_field_labels(page)
        self.separators = LayoutAnalyzer.find_geometric_separators(page)
        self.tables = page.extract_tables(flavor="lattice")

        # Calculate avg font size for header detection
        all_fs = [r.font_size for r in self.runs if r.text.strip()]
        self.avg_fs = sum(all_fs) / len(all_fs) if all_fs else 12.0

    def render(self) -> str:
        if not self.runs:
            return ""

        # 1. Clustering
        lines = LayoutAnalyzer.cluster_into_lines(self.runs)
        if not lines:
            return ""
        boxes = LayoutAnalyzer.order_boxes(LayoutAnalyzer.cluster_into_boxes(lines))

        md_parts: list[str] = []

        # Interleave separators and render boxes
        current_y = 1e9
        seps = self.separators
        for box in boxes:
            # Check for horizontal separators above this box
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

        # 2. Form Fields
        fields = self.page.get_fields()
        if fields:
            md_parts.append("### Form Fields")
            for field in fields:
                label = self.field_labels.get(field.name, field.name)
                val = str(field.value) if field.value is not None else "[Empty]"
                md_parts.append(f"- **{label}** ({field.type}): {val}")

        # 3. Annotations
        annots = self.page.get_annotations()
        if annots:
            self.render_annotations(md_parts, annots)

        # 4. Tables
        for table in self.tables if isinstance(self.tables, list) else []:
            if not table:
                continue
            rendered = self.render_table(table)
            if rendered:
                md_parts.append(rendered)

        return "\n\n".join(md_parts)

    def render_table(self, table: list[list[str]]) -> str | None:
        # Optimization: strip entirely empty columns
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

        # Add header separator
        header_sep = "|" + "|".join("---" for _ in range(n_cols - len(empty_cols))) + "|"
        md_table.insert(1, header_sep)
        return "\n" + "\n".join(md_table)

    def render_box(self, box: LayoutBox) -> str:
        is_boxed = any(vb.contains_rect(box.bbox_rect) for vb in self.vboxes)

        box_text_lines = []
        for line in box.lines:
            line_text = render_run_line(line.runs)
            if not line_text.strip():
                continue

            # Header detection
            max_fs = max(r.font_size for r in line.runs)
            is_all_caps = all(r.text.isupper() for r in line.runs if r.text.strip())

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
        """Convenience method to render a page to Markdown."""
        return cls(page).render()
