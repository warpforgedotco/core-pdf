# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import bisect
from collections.abc import Callable
from math import cos, radians, sin
from typing import TYPE_CHECKING, TypeAlias

from core_pdf.impl.engine.spec.s_07_content.models import LayoutBox, LayoutLine, TextRun

if TYPE_CHECKING:
    from core_pdf.impl.engine.spec.s_07_content.traces import CapturedLine
    from core_pdf.impl.engine.spec.s_07_document.page import PdfPage
    from core_pdf.impl.engine.spec.s_08_graphics.geometry import RectBox

SKEW_ANGLE_TOLERANCE: float = 0.5
SortableKey: TypeAlias = int | float | str | tuple[int | float | str, ...]


def cluster_runs_into_lines(
    runs: list[TextRun],
    *,
    lookback: int = 10,
    min_height: float = 8.0,
    threshold: float = 0.4,
    sort_key: Callable[[TextRun], SortableKey] | None = None,
) -> list[list[TextRun]]:
    if not runs:
        return []
    if sort_key is None:
        decorated = [(-r.mid_y, r.rotation_angle, r.x0, i, r) for i, r in enumerate(runs)]
        decorated.sort()
        sorted_runs = [item[4] for item in decorated]
    else:
        sorted_runs = sorted(runs, key=sort_key)

    rows: list[list[TextRun]] = []
    row_mid_sums: list[float] = []
    row_h_sums: list[float] = []
    row_counts: list[int] = []
    row_mid_avg: list[float] = []
    row_h_avg: list[float] = []

    for run in sorted_runs:
        ry_mid = run.mid_y
        rh = run.height
        placed = False

        start = max(0, len(rows) - lookback)
        for i in range(len(rows) - 1, start - 1, -1):
            if rows[i] and rows[i][0].rotation_angle != run.rotation_angle:
                continue
            ly_mid = row_mid_avg[i]
            lh = row_h_avg[i]
            dy = ry_mid - ly_mid
            max_h = rh if rh > lh else lh
            if max_h < min_height:
                max_h = min_height
            gap_threshold = max_h * threshold
            # Early exit: runs are sorted top-to-bottom; rows earlier in list
            # are higher on page. If run is too far below this row, earlier
            # (higher) rows are even further — no match possible.
            if dy < -gap_threshold * 1.8:
                break
            if abs(dy) < gap_threshold:
                rows[i].append(run)
                n = row_counts[i] + 1
                row_counts[i] = n
                row_mid_sums[i] += ry_mid
                row_h_sums[i] += rh
                row_mid_avg[i] = row_mid_sums[i] / n
                row_h_avg[i] = row_h_sums[i] / n
                placed = True
                break

        if not placed:
            rows.append([run])
            row_mid_sums.append(ry_mid)
            row_h_sums.append(rh)
            row_counts.append(1)
            row_mid_avg.append(ry_mid)
            row_h_avg.append(rh)

    return rows


class LayoutAnalyzer:
    """Namespace for higher-level layout analysis and clustering."""

    @staticmethod
    def cluster_into_lines(runs: list[TextRun]) -> list[LayoutLine]:
        rows = cluster_runs_into_lines(runs)
        all_lines: list[LayoutLine] = []
        line_add = LayoutLine.add
        for row in rows:
            line = LayoutLine()
            for run in row:
                line_add(line, run)
            all_lines.append(line)

        all_lines.sort(key=lambda ln: (-ln.y1, ln.x0))
        for line in all_lines:
            line.runs.sort(key=lambda r: r.x0)
        return all_lines

    @staticmethod
    def cluster_into_boxes(lines: list[LayoutLine]) -> list[LayoutBox]:
        boxes_by_angle: dict[int, list[LayoutBox]] = {}
        index_by_angle: dict[int, BoxIndex] = {}

        # Schwartzian transform: pre-compute sort keys to avoid lambda
        # function-call overhead during the sort.
        decorated = [(-ln.y1, ln.x0, ln.min_order, ln) for ln in lines]
        decorated.sort()
        for _, _, _, line in decorated:
            angle = line.rotation_angle
            candidates = boxes_by_angle.get(angle)
            if candidates is None:
                new_box = LayoutBox()
                new_box.add(line)
                boxes_by_angle[angle] = [new_box]
                idx = BoxIndex()
                idx.add(new_box)
                index_by_angle[angle] = idx
                continue

            idx = index_by_angle[angle]
            x_matching = idx.query_overlap_ids(line.x0, line.x1)

            placed = False
            lh = line.height
            line_x0, line_x1 = line.x0, line.x1
            for box in candidates:
                if id(box) not in x_matching:
                    continue

                # Inline overlap check to avoid function calls
                x_overlap = min(box.x1, line_x1) - max(box.x0, line_x0)
                if x_overlap <= 0:
                    continue
                box_width = box.x1 - box.x0
                line_width = line_x1 - line_x0
                min_width = min(box_width, line_width)
                overlap_percentage = x_overlap / min_width if min_width > 0 else 0.0

                # Early exit for non-overlapping boxes
                if overlap_percentage < 0.6 and x_overlap < 20.0:
                    continue

                gap = box.y0 - line.y1
                depth_factor = 4.0 if box.max_depth > 0 or line.max_depth > 0 else 1.0
                max_vertical_gap = max(lh, 12.0) * 2.2 * depth_factor
                if gap < max_vertical_gap:
                    old_x0 = box.x0
                    box.add(line)
                    idx.update(box, old_x0)
                    placed = True
                    break

            if not placed:
                new_box = LayoutBox()
                new_box.add(line)
                candidates.append(new_box)
                idx.add(new_box)

        return [box for group in boxes_by_angle.values() for box in group]

    @staticmethod
    def order_boxes(group: list[LayoutBox]) -> list[LayoutBox]:
        """Order boxes in a stable reading order, respecting columns."""
        if not group:
            return []

        # Sort primarily by center-Y (top-to-bottom) then X
        sorted_boxes = sorted(group, key=lambda b: (-b.mid_y, b.x0))

        # Group boxes into "visual rows" to detect tables or aligned cells
        visual_rows: list[list[LayoutBox]] = []
        visual_row_mid_sums: list[float] = []
        visual_row_height_sums: list[float] = []
        visual_row_ranges: list[list[tuple[float, float]]] = []
        for box in sorted_boxes:
            placed = False
            by_mid = box.mid_y
            bh = box.y1 - box.y0
            # Iterate backwards: boxes come Y-sorted, so the most recently created
            # row has the closest Y to the current box. Break when Y gap exceeds
            # the threshold — earlier rows are even further away.
            for idx in range(len(visual_rows) - 1, -1, -1):
                vrow = visual_rows[idx]
                row_len = len(vrow)
                ry_mid = visual_row_mid_sums[idx] / row_len
                rh = visual_row_height_sums[idx] / row_len
                max_gap = max(bh, rh, 10.0) * 0.3
                dy = by_mid - ry_mid
                if abs(dy) < max_gap:
                    if not any(
                        max(box.x0, x0) < min(box.x1, x1) for x0, x1 in visual_row_ranges[idx]
                    ):
                        vrow.append(box)
                        visual_row_mid_sums[idx] += by_mid
                        visual_row_height_sums[idx] += bh
                        visual_row_ranges[idx].append((box.x0, box.x1))
                        placed = True
                        break
                elif dy < -max_gap:
                    break
            if not placed:
                visual_rows.append([box])
                visual_row_mid_sums.append(by_mid)
                visual_row_height_sums.append(bh)
                visual_row_ranges.append([(box.x0, box.x1)])

        for vrow in visual_rows:
            vrow.sort(key=lambda b: b.x0)

        # Heuristic: if many rows have multiple boxes, it's likely a table/grid.
        multi_box_rows = sum(1 for r in visual_rows if len(r) > 1)
        if multi_box_rows >= 3 or (
            len(visual_rows) > 0 and multi_box_rows > len(visual_rows) * 0.4
        ):
            ordered: list[LayoutBox] = []
            for vrow in visual_rows:
                ordered.extend(vrow)
            return ordered

        # Otherwise, use standard column-major detection
        columns: list[tuple[float, float, float, list[LayoutBox]]] = []
        for box in sorted_boxes:
            placed = False
            bx0, bx1 = box.x0, box.x1
            box_w = bx1 - bx0
            for i, (cx0, cx1, x0_sum, col) in enumerate(columns):
                overlap = min(bx1, cx1) - max(bx0, cx0)
                if overlap > min(cx1 - cx0, box_w) * 0.5:
                    col.append(box)
                    columns[i] = (min(cx0, bx0), max(cx1, bx1), x0_sum + bx0, col)
                    placed = True
                    break
            if not placed:
                columns.append((bx0, bx1, bx0, [box]))

        columns.sort(key=lambda c: c[2] / len(c[3]))
        ordered_col: list[LayoutBox] = []
        for cx0, cx1, x0_sum, col in columns:
            col.sort(key=lambda b: -b.y1)
            ordered_col.extend(col)
        return ordered_col

    @staticmethod
    def detect_skew(runs: list[TextRun]) -> float:
        """
        Detects the skew angle of a page by projecting text runs into a histogram
        and finding the angle that maximizes the variance (Projective Profile).
        """
        if len(runs) < 30:
            return 0.0

        # Sample a good subset of runs for high signal
        sample_runs = runs[:: max(1, len(runs) // 1000)]

        best_angle = 0.0
        max_variance = -1.0
        variances = {}

        # Optimization: pre-fetch midpoints to local lists for faster access in the hot loop
        mid_xs = [run.mid_x for run in sample_runs]
        mid_ys = [run.mid_y for run in sample_runs]
        n_proj = len(mid_xs)

        # Helper to compute projections with local bindings
        def compute_projections(angles: list[float]) -> None:
            nonlocal max_variance, best_angle
            # Local bindings for speed
            local_mid_xs = mid_xs
            local_mid_ys = mid_ys

            for angle_deg in angles:
                angle_rad = radians(angle_deg)
                ca, sa = cos(angle_rad), sin(angle_rad)

                # Faster list comprehension
                projections = [int(-x * sa + y * ca) for x, y in zip(local_mid_xs, local_mid_ys)]

                if not projections:
                    continue

                min_p, max_p = min(projections), max(projections)
                num_bins = max_p - min_p + 1
                if num_bins > 5000:
                    continue

                bins = [0] * num_bins
                s2 = 0
                for p in projections:
                    idx = p - min_p
                    c = bins[idx]
                    s2 += 2 * c + 1
                    bins[idx] = c + 1

                variance = (s2 - (n_proj * n_proj / num_bins)) / num_bins
                variances[angle_deg] = variance

                if variance > max_variance:
                    max_variance = variance
                    best_angle = angle_deg

        # FAST PATH: Check 0.0, -1.0, and 1.0 first.
        # If 0.0 is a clear sharp peak, skip the rest of the search.
        compute_projections([0.0, -1.0, 1.0])
        v0 = variances.get(0.0, 0.0)
        v1 = variances.get(1.0, 0.0)
        vm1 = variances.get(-1.0, 0.0)

        if v0 > v1 * 1.8 and v0 > vm1 * 1.8:
            return 0.0

        # First pass: coarse search from -5 to +5 degrees in 1.0 degree increments
        coarse_angles = [float(d) for d in range(-5, 6) if d not in (0, 1, -1)]
        compute_projections(coarse_angles)

        if n_proj == 0:
            return 0.0

        # Optimization: if 0.0 is still the clear winner, skip fine pass
        if best_angle == 0.0 and max_variance > v1 * 1.5 and max_variance > vm1 * 1.5:
            return 0.0

        # Confidence check: if 0 is close to best, prefer 0
        if best_angle != 0.0 and max_variance < v0 * 1.05:
            return 0.0

        # Second pass: fine search around the best coarse angle (+/- 1.0 deg in 0.1 increments)
        fine_angles = [
            best_angle - 0.9 + i * 0.1
            for i in range(20)
            if abs(best_angle - 0.9 + i * 0.1 - best_angle) > 0.05
        ]
        compute_projections(fine_angles)

        return best_angle

    @staticmethod
    def apply_skew_correction(runs: list[TextRun]) -> list[TextRun]:
        skew_angle = LayoutAnalyzer.detect_skew(runs)
        if abs(skew_angle) <= SKEW_ANGLE_TOLERANCE:
            return runs

        x0s = [r.x0 for r in runs]
        y0s = [r.y0 for r in runs]
        x1s = [r.x1 for r in runs]
        y1s = [r.y1 for r in runs]
        cx = (min(x0s) + max(x1s)) * 0.5
        cy = (min(y0s) + max(y1s)) * 0.5

        theta = radians(-skew_angle)
        ca, sa = cos(theta), sin(theta)

        return [
            run.replace(
                x0=(run.x0 - cx) * ca - (run.y0 - cy) * sa + cx,
                y0=(run.x0 - cx) * sa + (run.y0 - cy) * ca + cy,
                x1=(run.x1 - cx) * ca - (run.y1 - cy) * sa + cx,
                y1=(run.x1 - cx) * sa + (run.y1 - cy) * ca + cy,
            )
            for run in runs
        ]

    @staticmethod
    def is_right_to_left_line(runs: list[TextRun]) -> bool:
        ordered = [run for run in runs if run.text.strip()]
        if len(ordered) < 2:
            return False
        # Check if stream order corresponds to right-to-left (x0 decreasing)
        stream_sorted = sorted(ordered, key=lambda r: (r.order, r.stream_order))
        decreases = 0
        increases = 0
        prev_x0 = stream_sorted[0].x0
        for run in stream_sorted[1:]:
            if run.x0 < prev_x0:
                decreases += 1
            elif run.x0 > prev_x0:
                increases += 1
            prev_x0 = run.x0
        return decreases > increases

    @staticmethod
    def find_geometric_separators(page: PdfPage) -> list[CapturedLine]:
        """Find long horizontal or vertical lines that likely act as logical dividers."""
        lines = page.get_grid_lines()
        if not lines:
            return []

        mb = page.media_box or (0, 0, 612, 792)
        page_width = mb[2] - mb[0]
        page_height = mb[3] - mb[1]

        separators: list[CapturedLine] = []

        # Thresholds: span > 25% of page dimension
        h_threshold = page_width * 0.25
        v_threshold = page_height * 0.25

        for line in lines:
            dx = abs(line.x1 - line.x0)
            dy = abs(line.y1 - line.y0)

            # Horizontal
            if dy < 1.5 and dx > h_threshold or dx < 1.5 and dy > v_threshold:
                separators.append(line)

        return separators

    @staticmethod
    def detect_visual_boxes(page: PdfPage) -> list[RectBox]:
        """Detect visually boxed regions (rectangles that likely contain text)."""
        state = page.get_graphics()
        drawings = state.drawings
        runs = state.runs

        visual_boxes: list[RectBox] = []
        for drawing in drawings:
            rect = drawing.rect
            if rect:
                # Normalize and check size (not too small, not too big)
                norm = rect.normalize()
                if norm.width > 30 and norm.height > 15:
                    # Check if it contains any text
                    has_text = any(
                        norm.contains_point(r.mid_x, r.mid_y) for r in runs if r.text.strip()
                    )
                    if has_text:
                        visual_boxes.append(norm)

        return visual_boxes

    @staticmethod
    def associate_field_labels(page: PdfPage) -> dict[str, str]:
        """Find text labels for AcroForm fields based on proximity."""
        fields = page.get_fields()
        field_to_label = {}
        for field in fields:
            if not field.widget:
                continue
            rect_data = field.widget.get("Rect")
            if not rect_data:
                continue
            rect = page.document.resolver.resolve_box(rect_data)
            if not rect:
                continue

            # Try left first (common for horizontal forms)
            nearby = page.find_text_near(rect, direction="left", distance=150.0)
            if nearby:
                field_to_label[field.name] = " ".join(r.text for r in nearby[:3])
            else:
                # Try above (common for vertical forms)
                nearby = page.find_text_near(rect, direction="above", distance=50.0)
                if nearby:
                    field_to_label[field.name] = " ".join(r.text for r in nearby[:3])
        return field_to_label


class BoxIndex:
    """Sorted-list interval index for O(log k + |result|) horizontal-overlap queries.

    Maintains boxes sorted by current x0. Augments with a suffix-max of x1 so that
    a prefix of boxes whose entire remaining x1 range falls below the query low bound
    can be pruned in one check.

    All structural operations are O(k) in box count. k is the number of boxes per
    rotation group — typically small (< 50) per page — so O(k) rebuilds are negligible.

    Correctness under mutation: LayoutBox.add() can only decrease x0 and increase x1.
    Callers must call update(box, old_x0) after every box.add() so the sorted order
    and suffix max stay valid.
    """

    __slots__ = ("x0s", "boxes", "suf")

    def __init__(self) -> None:
        self.x0s: list[float] = []
        self.boxes: list[LayoutBox] = []
        self.suf: list[float] = []  # suf[i] = max(box.x1 for box in boxes[i:])

    def rebuild_suf(self) -> None:
        n = len(self.boxes)
        self.suf = [0.0] * n
        if n:
            self.suf[-1] = self.boxes[-1].x1
            for i in range(n - 2, -1, -1):
                self.suf[i] = max(self.boxes[i].x1, self.suf[i + 1])

    def add(self, box: LayoutBox) -> None:
        pos = bisect.bisect_left(self.x0s, box.x0)
        self.x0s.insert(pos, box.x0)
        self.boxes.insert(pos, box)
        self.suf.insert(pos, 0.0)
        self.rebuild_suf()

    def update(self, box: LayoutBox, old_x0: float) -> None:
        try:
            pos = self.boxes.index(box)
        except ValueError:
            return
        self.x0s.pop(pos)
        self.boxes.pop(pos)
        self.suf.pop(pos)
        new_pos = bisect.bisect_left(self.x0s, box.x0)
        self.x0s.insert(new_pos, box.x0)
        self.boxes.insert(new_pos, box)
        self.suf.insert(new_pos, 0.0)
        self.rebuild_suf()

    def query_overlap_ids(self, lx0: float, lx1: float) -> set[int]:
        """Return id()s of all boxes whose x-interval overlaps [lx0, lx1].

        Uses binary search to skip boxes with x0 > lx1 and suffix-max pruning
        to stop early when no remaining box can have x1 >= lx0.
        """
        right = bisect.bisect_right(self.x0s, lx1)
        result: set[int] = set()
        for i in range(right):
            if self.suf[i] < lx0:
                # max x1 of boxes[i:] is below lx0 — none can overlap
                break
            if self.boxes[i].x1 >= lx0:
                result.add(id(self.boxes[i]))
        return result
