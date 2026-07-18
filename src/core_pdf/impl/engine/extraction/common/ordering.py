# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import bisect
from collections.abc import Sequence
from math import cos, radians, sin
from typing import TYPE_CHECKING, Protocol

from core_layout.impl.layout.models import LayoutBox, LayoutLine, TextRun

if TYPE_CHECKING:
    from core_layout.impl.layout.geometry import RectBox

    from core_pdf.impl.engine.spec.s_07_content.capture import CapturedLine


class VisualGraphicsState(Protocol):
    @property
    def drawings(self) -> Sequence[DrawingLike]: ...

    @property
    def runs(self) -> Sequence[TextRun]: ...


class DrawingLike(Protocol):
    @property
    def rect(self) -> RectBox | None: ...


class GeometricPage(Protocol):
    @property
    def media_box(self) -> tuple[float, float, float, float] | None: ...

    def get_grid_lines(self) -> list[CapturedLine]: ...


class VisualPage(Protocol):
    def get_graphics(self) -> VisualGraphicsState: ...


class FieldLike(Protocol):
    name: str
    rect: tuple[float, float, float, float] | None


class FieldLabelPage(Protocol):
    def get_fields(self) -> Sequence[FieldLike]: ...

    def find_text_near(
        self,
        target_box: tuple[float, float, float, float],
        direction: str = "left",
        distance: float = 100.0,
    ) -> list[TextRun]: ...


SKEW_ANGLE_TOLERANCE: float = 0.5


def cluster_runs_into_lines(
    runs: list[TextRun],
    *,
    lookback: int = 1,
    min_height: float = 8.0,
    threshold: float = 0.4,
    sort_by_horizontal_position: bool = False,
) -> list[list[TextRun]]:
    if not runs:
        return []
    decorated: list[tuple[float, int | float, float, int, float, float, TextRun]] = []
    if sort_by_horizontal_position:
        for i, run in enumerate(runs):
            mid_y = run.mid_y_value
            x0 = run.coords[TextRun.X0]
            decorated.append((-mid_y, x0, 0.0, i, mid_y, run.height_value, run))
    else:
        for i, run in enumerate(runs):
            if run.rotation_angle in (90, 270):
                x0 = run.coords[TextRun.X0]
                x1 = run.coords[TextRun.X1]
                mid_x = (x0 + x1) * 0.5
                decorated.append(
                    (
                        x0,
                        run.rotation_angle,
                        -run.mid_y_value,
                        i,
                        mid_x,
                        x1 - x0,
                        run,
                    )
                )
                continue
            mid_y = run.mid_y_value
            decorated.append(
                (
                    -mid_y,
                    run.rotation_angle,
                    run.coords[TextRun.X0],
                    i,
                    mid_y,
                    run.height_value,
                    run,
                )
            )
    decorated.sort()

    rows: list[list[TextRun]] = []
    row_mid_sums: list[float] = []
    row_h_sums: list[float] = []
    row_counts: list[int] = []
    row_mid_avg: list[float] = []
    row_h_avg: list[float] = []
    row_angles: list[int] = []

    if lookback <= 1:
        for item in decorated:
            ry_mid = item[4]
            rh = item[5]
            run = item[6]
            if rows:
                i = len(rows) - 1
                if row_angles[i] == run.rotation_angle:
                    ly_mid = row_mid_avg[i]
                    lh = row_h_avg[i]
                    dy = ry_mid - ly_mid
                    max_h = rh if rh > lh else lh
                    if max_h < min_height:
                        max_h = min_height
                    gap_threshold = max_h * threshold
                    if dy >= -gap_threshold * 1.8 and abs(dy) < gap_threshold:
                        rows[i].append(run)
                        n = row_counts[i] + 1
                        row_counts[i] = n
                        row_mid_sums[i] += ry_mid
                        row_h_sums[i] += rh
                        row_mid_avg[i] = row_mid_sums[i] / n
                        row_h_avg[i] = row_h_sums[i] / n
                        continue
            rows.append([run])
            row_mid_sums.append(ry_mid)
            row_h_sums.append(rh)
            row_counts.append(1)
            row_mid_avg.append(ry_mid)
            row_h_avg.append(rh)
            row_angles.append(run.rotation_angle)
        return rows

    for item in decorated:
        ry_mid = item[4]
        rh = item[5]
        run = item[6]
        placed = False

        start = max(0, len(rows) - lookback)
        for i in range(len(rows) - 1, start - 1, -1):
            if row_angles[i] != run.rotation_angle:
                continue
            ly_mid = row_mid_avg[i]
            lh = row_h_avg[i]
            dy = ry_mid - ly_mid
            max_h = rh if rh > lh else lh
            if max_h < min_height:
                max_h = min_height
            gap_threshold = max_h * threshold

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
            row_angles.append(run.rotation_angle)

    return rows


class LayoutAnalyzer:
    @staticmethod
    def cluster_into_lines(runs: list[TextRun]) -> list[LayoutLine]:
        rows = cluster_runs_into_lines(runs)
        all_lines: list[LayoutLine] = []
        for row in rows:
            all_lines.append(LayoutLine(runs=row))
        return all_lines

    @staticmethod
    def cluster_into_boxes(lines: list[LayoutLine]) -> list[LayoutBox]:
        boxes_by_angle: dict[int, list[LayoutBox]] = {}
        index_by_angle: dict[int, BoxIndex] = {}

        decorated = [(-ln.y1, ln.x0, ln.min_order, ln) for ln in lines]
        decorated.sort()
        for ignored, ignored, ignored, line in decorated:
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

                x_overlap = min(box.x1, line_x1) - max(box.x0, line_x0)
                if x_overlap <= 0:
                    continue
                box_width = box.x1 - box.x0
                line_width = line_x1 - line_x0
                min_width = min(box_width, line_width)
                overlap_percentage = x_overlap / min_width if min_width > 0 else 0.0

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
        if not group:
            return []

        sorted_boxes = sorted(group, key=lambda b: (-b.mid_y, b.x0))

        visual_rows: list[list[LayoutBox]] = []
        visual_row_mid_sums: list[float] = []
        visual_row_height_sums: list[float] = []
        visual_row_ranges: list[list[tuple[float, float]]] = []
        for box in sorted_boxes:
            placed = False
            by_mid = box.mid_y
            bh = box.y1 - box.y0

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

        multi_box_rows = sum(1 for r in visual_rows if len(r) > 1)
        if multi_box_rows >= 3 or (
            len(visual_rows) > 0 and multi_box_rows > len(visual_rows) * 0.4
        ):
            ordered: list[LayoutBox] = []
            for vrow in visual_rows:
                ordered.extend(vrow)
            return ordered

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
        if len(runs) < 30:
            return 0.0

        sample_runs = runs[:: max(1, len(runs) // 1000)]

        best_angle = 0.0
        max_variance = -1.0
        variances = {}

        mid_xs = [run.mid_x for run in sample_runs]
        mid_ys = [run.mid_y for run in sample_runs]
        n_proj = len(mid_xs)
        mid_points = list(zip(mid_xs, mid_ys))

        def compute_projections(angles: list[float]) -> None:
            nonlocal max_variance, best_angle

            for angle_deg in angles:
                angle_rad = radians(angle_deg)
                ca, sa = cos(angle_rad), sin(angle_rad)

                projections = [int(-x * sa + y * ca) for x, y in mid_points]

                if not projections:
                    continue

                min_p, max_p = min(projections), max(projections)

                num_bins = max_p - min_p + 1
                if num_bins > 5000:
                    continue

                counts = [0] * num_bins
                for projection in projections:
                    counts[projection - min_p] += 1
                s2 = 0
                for count in counts:
                    s2 += count * count

                variance = (s2 - (n_proj * n_proj / num_bins)) / num_bins
                variances[angle_deg] = variance

                if variance > max_variance:
                    max_variance = variance
                    best_angle = angle_deg

        compute_projections([0.0, -1.0, 1.0])
        v0 = variances.get(0.0, 0.0)
        v1 = variances.get(1.0, 0.0)
        vm1 = variances.get(-1.0, 0.0)

        if v0 > v1 * 1.8 and v0 > vm1 * 1.8:
            return 0.0

        if v1 > 5.0 and v1 > v0 * 4.0 and v1 > vm1 * 4.0:
            best_angle = 1.0
            max_variance = v1
            fine_angles = [
                best_angle - 0.9 + i * 0.1
                for i in range(20)
                if abs(best_angle - 0.9 + i * 0.1 - best_angle) > 0.05
            ]
            compute_projections(fine_angles)
            return best_angle
        if vm1 > 5.0 and vm1 > v0 * 4.0 and vm1 > v1 * 4.0:
            best_angle = -1.0
            max_variance = vm1
            fine_angles = [
                best_angle - 0.9 + i * 0.1
                for i in range(20)
                if abs(best_angle - 0.9 + i * 0.1 - best_angle) > 0.05
            ]
            compute_projections(fine_angles)
            return best_angle

        coarse_angles = [float(d) for d in range(-5, 6) if d not in (0, 1, -1)]
        compute_projections(coarse_angles)

        if n_proj == 0:
            return 0.0

        if best_angle == 0.0 and max_variance > v1 * 1.5 and max_variance > vm1 * 1.5:
            return 0.0

        if best_angle != 0.0 and max_variance < v0 * 1.05:
            return 0.0

        fine_angles = [
            best_angle - 0.9 + i * 0.1
            for i in range(20)
            if abs(best_angle - 0.9 + i * 0.1 - best_angle) > 0.05
        ]
        compute_projections(fine_angles)

        return best_angle

    @staticmethod
    def apply_skew_correction(
        runs: list[TextRun], skew_angle: float | None = None
    ) -> list[TextRun]:
        if skew_angle is None:
            skew_angle = LayoutAnalyzer.detect_skew(runs)
        if abs(skew_angle) <= SKEW_ANGLE_TOLERANCE:
            return runs

        x0_index = TextRun.X0
        y0_index = TextRun.Y0
        x1_index = TextRun.X1
        y1_index = TextRun.Y1
        first_coords = runs[0].coords
        min_x0 = first_coords[x0_index]
        max_x1 = first_coords[x1_index]
        min_y0 = first_coords[y0_index]
        max_y1 = first_coords[y1_index]
        for run_index in range(1, len(runs)):
            run = runs[run_index]
            coords = run.coords
            x0 = coords[x0_index]
            x1 = coords[x1_index]
            y0 = coords[y0_index]
            y1 = coords[y1_index]
            if x0 < min_x0:
                min_x0 = x0
            if x1 > max_x1:
                max_x1 = x1
            if y0 < min_y0:
                min_y0 = y0
            if y1 > max_y1:
                max_y1 = y1
        cx = (min_x0 + max_x1) * 0.5
        cy = (min_y0 + max_y1) * 0.5

        theta = radians(-skew_angle)
        ca, sa = cos(theta), sin(theta)

        corrected: list[TextRun] = []
        append = corrected.append
        for run in runs:
            coords = run.coords
            x0 = coords[x0_index]
            y0 = coords[y0_index]
            x1 = coords[x1_index]
            y1 = coords[y1_index]
            dx0 = x0 - cx
            dy0 = y0 - cy
            dx1 = x1 - cx
            dy1 = y1 - cy
            append(
                run.with_coords(
                    dx0 * ca - dy0 * sa + cx,
                    dx0 * sa + dy0 * ca + cy,
                    dx1 * ca - dy1 * sa + cx,
                    dx1 * sa + dy1 * ca + cy,
                )
            )
        return corrected

    @staticmethod
    def is_right_to_left_line(runs: list[TextRun]) -> bool:
        if len(runs) < 2:
            return False
        ordered = runs
        for run in runs:
            if not run.has_text:
                ordered = [text_run for text_run in runs if text_run.has_text]
                if len(ordered) < 2:
                    return False
                break

        stream_sorted = sorted(ordered, key=lambda r: (r.order, r.stream_order))
        decreases = 0
        increases = 0
        text_run_x0 = TextRun.X0
        prev_x0 = stream_sorted[0].coords[text_run_x0]
        for idx in range(1, len(stream_sorted)):
            x0 = stream_sorted[idx].coords[text_run_x0]
            if x0 < prev_x0:
                decreases += 1
            elif x0 > prev_x0:
                increases += 1
            prev_x0 = x0
        return decreases > increases

    @staticmethod
    def find_geometric_separators(page: GeometricPage) -> list[CapturedLine]:
        lines = page.get_grid_lines()
        if not lines:
            return []

        mb = page.media_box or (0, 0, 612, 792)
        page_width = mb[2] - mb[0]
        page_height = mb[3] - mb[1]

        separators: list[CapturedLine] = []

        h_threshold = page_width * 0.25
        v_threshold = page_height * 0.25

        for line in lines:
            dx = abs(line.x1 - line.x0)
            dy = abs(line.y1 - line.y0)

            if dy < 1.5 and dx > h_threshold or dx < 1.5 and dy > v_threshold:
                separators.append(line)

        return separators

    @staticmethod
    def detect_visual_boxes(page: VisualPage) -> list[RectBox]:
        state = page.get_graphics()
        drawings = state.drawings
        runs = state.runs

        visual_boxes: list[RectBox] = []
        for drawing in drawings:
            rect = drawing.rect
            if rect:
                norm = rect.normalize()
                if norm.width > 30 and norm.height > 15:
                    has_text = any(
                        norm.contains_point(r.mid_x, r.mid_y) for r in runs if r.text.strip()
                    )
                    if has_text:
                        visual_boxes.append(norm)

        return visual_boxes

    @staticmethod
    def associate_field_labels(page: FieldLabelPage) -> dict[str, str]:
        fields = page.get_fields()
        field_to_label = {}
        for field in fields:
            if not field.rect:
                continue

            nearby = page.find_text_near(field.rect, direction="left", distance=150.0)
            if nearby:
                field_to_label[field.name] = " ".join(r.text for r in nearby[:3])
            else:
                nearby = page.find_text_near(field.rect, direction="above", distance=50.0)
                if nearby:
                    field_to_label[field.name] = " ".join(r.text for r in nearby[:3])
        return field_to_label


class BoxIndex:
    __slots__ = ("x0s", "boxes", "suf")

    def __init__(self) -> None:
        self.x0s: list[float] = []
        self.boxes: list[LayoutBox] = []
        self.suf: list[float] = []

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
        right = bisect.bisect_right(self.x0s, lx1)
        result: set[int] = set()
        for i in range(right):
            if self.suf[i] < lx0:
                break
            if self.boxes[i].x1 >= lx0:
                result.add(id(self.boxes[i]))
        return result
