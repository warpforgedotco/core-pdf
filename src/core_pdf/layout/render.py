# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import re
import typing
from statistics import median_low
from typing import Any

if typing.TYPE_CHECKING:
    from core_pdf.document.models import AnnotationRecord
    from core_pdf.document.page import PdfPage

from core_pdf.layout.models import LayoutBox, LayoutLine, TextRun
from core_pdf.layout.ordering import (
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
        """Render a box to text, handling columns recursively."""
        segments: list[list[LayoutLine]] = [lines]
        i = 0
        while i < len(segments):
            seg = segments[i]
            if not seg:
                i += 1
                continue
            split_x = self.find_column_split(seg)
            if split_x is None:
                i += 1
                continue

            left_lines: list[LayoutLine] = []
            right_lines: list[LayoutLine] = []
            for line in seg:
                left_runs = [r for r in line.runs if r.mid_x < split_x]
                right_runs = [r for r in line.runs if r.mid_x >= split_x]
                if left_runs:
                    left_lines.append(LayoutLine(runs=left_runs))
                if right_runs:
                    right_lines.append(LayoutLine(runs=right_runs))
            segments[i : i + 1] = [left_lines, right_lines]

        parts: list[str] = []
        for seg in segments:
            if not seg:
                continue
            sorted_rows = sorted(seg, key=lambda ln: (-ln.mid_y, ln.x0))
            parts.append("\n".join(render_run_line(ln.runs) for ln in sorted_rows if ln.runs))

        return "\n\n".join(p for p in parts if p)

    def find_column_split(self, lines: list[LayoutLine]) -> float | None:
        all_runs: list[TextRun] = []
        filtered_lines: list[list[TextRun]] = []
        box_x0: float = 1e9
        box_x1: float = -1e9

        for line in lines:
            current_line_runs: list[TextRun] = []
            for r in line.runs:
                text = r.text
                if text and not text.isspace():
                    rx0, rx1 = r.x0, r.x1
                    if rx0 < box_x0:
                        box_x0 = rx0
                    if rx1 > box_x1:
                        box_x1 = rx1
                    all_runs.append(r)
                    current_line_runs.append(r)
            if current_line_runs:
                filtered_lines.append(current_line_runs)

        if len(all_runs) < 100:
            return None

        box_width = box_x1 - box_x0
        if box_width <= 0:
            return None

        gap_mids: list[float] = []
        for line_runs in filtered_lines:
            if len(line_runs) < 2:
                continue
            best_gap = 0.0
            best_mid = 0.0
            prev_x1 = line_runs[0].x1
            for run in line_runs[1:]:
                gap = run.x0 - prev_x1
                if gap > best_gap:
                    best_gap = gap
                    best_mid = (prev_x1 + run.x0) * 0.5
                if run.x1 > prev_x1:
                    prev_x1 = run.x1
            if best_gap >= 20.0:
                gap_mids.append(best_mid)

        if len(gap_mids) < 10:
            return None

        split_x = median_low(gap_mids)
        consistent_count = sum(1 for mid in gap_mids if abs(mid - split_x) <= 30.0)
        if consistent_count < len(gap_mids) * 0.6:
            return None

        split_rel = (split_x - box_x0) / box_width
        if not (0.20 <= split_rel <= 0.80):
            return None

        left_n = sum(1 for r in all_runs if r.mid_x < split_x)
        if left_n < 30 or (len(all_runs) - left_n) < 30:
            return None

        return split_x


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

        # Deduplication: if this run overlaps significantly with the previous one
        if has_prev_bbox and prev_x1 is not None:
            # Inline overlap calculation
            ox = run.x1 if run.x1 < prev_x1 else prev_x1
            ox -= run.x0 if run.x0 > prev_x0 else prev_x0
            oy = run.y1 if run.y1 < prev_y1 else prev_y1
            oy -= run.y0 if run.y0 > prev_y0 else prev_y0

            if ox > 0 and oy > 0:
                box_area = (run.y1 - run.y0) * (run.x1 - run.x0)
                if (
                    box_area > 0
                    and (ox * oy) / box_area > 0.8
                    and (text == prev_text or (len(text) == len(prev_text) and len(text) <= 2))
                ):
                    continue

        if prev_x1 is not None:
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

    def render_table(self, table: list[list[Any]]) -> str | None:
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
