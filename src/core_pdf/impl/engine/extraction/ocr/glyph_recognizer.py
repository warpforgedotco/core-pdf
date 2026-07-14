# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import replace
from typing import Any

from core_pdf.impl.engine.layout.glyphs import GlyphCluster
from core_pdf.impl.engine.layout.models import TextRun


BITMAP_REPAIR_LABELS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,-+/()[]{}<>|_~"
)
SUSPICIOUS_GLYPH_TEXT = {"\ufffd", "\ufffc"}
CONTEXTUAL_PUNCTUATION_GLYPH_TEXT = frozenset("()[]{}<>|_~")
LEGITIMATE_MULTI_CHAR_GLYPHS = {"ff", "fi", "fl", "ffi", "ffl", "st"}
MAX_BITMAP_REPAIR_DISTANCE = 0.18


def repair_text_runs_with_glyph_bitmaps(
    runs: list[Any],
    rendered_page: Any,
    *,
    repair_contextual_punctuation: bool = False,
) -> list[Any]:
    display_list = getattr(rendered_page, "display_list", None)
    display_items = getattr(display_list, "items", ())
    glyph_items: list[dict[str, Any]] = []
    for display_index, item in enumerate(display_items):
        if item.kind != "glyph" or not item.data.get("bitmap"):
            continue
        data = dict(item.data)
        data["seqno"] = item.seqno
        data["glyph_index"] = display_index
        glyph_items.append(data)
    if not glyph_items:
        return runs
    repairs = glyph_bitmap_item_repairs(
        glyph_items,
        repair_contextual_punctuation=repair_contextual_punctuation,
    )
    if not repairs:
        return runs
    changed = False
    repaired_runs: list[Any] = []
    glyphs_by_seqno: dict[int, list[dict[str, Any]]] = {}
    for item in glyph_items:
        seqno = item.get("seqno")
        if isinstance(seqno, int):
            glyphs_by_seqno.setdefault(seqno, []).append(item)
    for run in runs:
        run_glyphs = glyphs_by_seqno.get(getattr(run, "seqno", -1), [])
        repaired_run = repair_text_run_from_glyph_items(run, run_glyphs, repairs)
        if repaired_run is run:
            repaired_runs.append(run)
            continue
        repaired_runs.append(repaired_run)
        changed = True
    return repaired_runs if changed else runs


def text_runs_from_rendered_glyphs(rendered_page: Any) -> list[TextRun]:
    display_list = getattr(rendered_page, "display_list", None)
    display_items = getattr(display_list, "items", ())
    glyph_items: list[dict[str, Any]] = []
    for display_index, item in enumerate(display_items):
        if item.kind != "glyph" or not item.data.get("bitmap"):
            continue
        data = dict(item.data)
        data["seqno"] = item.seqno
        data["glyph_index"] = display_index
        glyph_items.append(data)
    if not glyph_items:
        return []
    repairs = glyph_bitmap_item_repairs(glyph_items)
    run_items: list[
        tuple[
            float,
            float,
            float,
            float,
            tuple[float, float, float, float],
            str,
            int,
            dict[str, Any],
        ]
    ] = []
    for item in glyph_items:
        if item.get("visible") is False:
            continue
        ink_bbox = numeric_bbox(item.get("bbox"))
        if ink_bbox is None:
            continue
        advance_bbox = numeric_bbox(item.get("advance_bbox")) or ink_bbox
        x0, y0, x1, y1 = advance_bbox
        ix0, iy0, ix1, iy1 = ink_bbox
        if x1 <= x0 or y1 <= y0 or ix1 <= ix0 or iy1 <= iy0:
            continue
        seqno = item.get("seqno")
        item_glyph_index = glyph_item_index(item)
        text = repairs.get(item_glyph_index) if item_glyph_index is not None else None
        if text is None:
            text = item.get("text")
        if not isinstance(text, str) or not text:
            continue
        seqno_int = seqno if isinstance(seqno, int) else -1
        run_items.append((x0, y0, x1, y1, ink_bbox, text, seqno_int, item))
    run_items.sort(
        key=lambda entry: (-((entry[1] + entry[3]) * 0.5), entry[0], entry[6])
    )

    runs: list[TextRun] = []
    for order, (x0, y0, x1, y1, ink_bbox, text, seqno, item) in enumerate(run_items):
        font_size = max(1.0, y1 - y0)
        item_glyph_index = glyph_item_index(item)
        repaired = item_glyph_index is not None and item_glyph_index in repairs
        confidence = 0.88 if repaired else 0.75
        provenance: tuple[tuple[str, object], ...] = (
            ("source", "rendered_glyph"),
            ("glyph_index", item.get("glyph_index")),
            ("seqno", seqno),
        )
        if repaired:
            provenance = (*provenance, ("repair", "glyph_bitmap"))
        runs.append(
            TextRun(
                text=text,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                tx=x0,
                ty=y0,
                font_size=font_size,
                space_width=max(1.0, font_size * 0.35),
                order=order,
                stream_order=seqno,
                xobject_depth=0,
                font_name=item.get("font_name")
                if isinstance(item.get("font_name"), str)
                else None,
                visible=True,
                seqno=seqno,
                fill_color=item.get("fill_color")
                if isinstance(item.get("fill_color"), tuple)
                else None,
                advance_bbox=(x0, y0, x1, y1),
                ink_bbox=ink_bbox,
                provenance=provenance,
                confidence=confidence,
                glyph_clusters=(
                    GlyphCluster(
                        cluster_id=order,
                        text=text,
                        glyphs=(),
                        kind="rendered_glyph",
                        advance_bbox=(x0, y0, x1, y1),
                        ink_bbox=ink_bbox,
                        baseline=None,
                        writing_mode="horizontal",
                        rotation_angle=0,
                        font_name=item.get("font_name")
                        if isinstance(item.get("font_name"), str)
                        else None,
                        seqno=seqno,
                        confidence=confidence,
                        provenance=provenance,
                    ),
                ),
            )
        )
    return runs


def numeric_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (
            float(value[0]),
            float(value[1]),
            float(value[2]),
            float(value[3]),
        )
    except TypeError, ValueError:
        return None
    return (x0, y0, x1, y1)


def glyph_bitmap_item_repairs(
    glyph_items: list[dict[str, Any]],
    *,
    repair_contextual_punctuation: bool = False,
) -> dict[int, str]:
    code_repairs = glyph_code_repairs(glyph_items)
    examples: dict[
        tuple[str | None, int, int], dict[tuple[tuple[int, ...], str], None]
    ] = {}
    targets: list[dict[str, Any]] = []
    target_bitmaps: dict[int, tuple[int, ...]] = {}
    contextual_targets = (
        contextual_punctuation_target_indexes(glyph_items)
        if repair_contextual_punctuation
        else set()
    )
    for item in glyph_items:
        glyph_index = glyph_item_index(item)
        if glyph_index is not None and glyph_index in code_repairs:
            continue
        text = item.get("text")
        bitmap = normalized_bitmap(item)
        if bitmap is None:
            continue
        key = glyph_repair_key(item)
        if glyph_index is not None and glyph_index in contextual_targets:
            targets.append(item)
            target_bitmaps[id(item)] = bitmap
        elif is_trustworthy_bitmap_label(text):
            assert isinstance(text, str)
            examples.setdefault(key, {})[(bitmap, text)] = None
        elif is_suspicious_bitmap_label(text):
            targets.append(item)
            target_bitmaps[id(item)] = bitmap
    if not examples or not targets:
        return code_repairs

    repairs: dict[int, str] = dict(code_repairs)
    best_label_cache: dict[
        tuple[tuple[str | None, int, int], tuple[int, ...]], str | None
    ] = {}
    for item in targets:
        bitmap = target_bitmaps.get(id(item))
        if bitmap is None:
            continue
        key = glyph_repair_key(item)
        cache_key = (key, bitmap)
        if cache_key in best_label_cache:
            best_label = best_label_cache[cache_key]
        else:
            candidates = examples.get(key)
            best_label = None
            if candidates:
                best_label = best_repair_label(bitmap, list(candidates.keys()))
            best_label_cache[cache_key] = best_label
        if best_label is not None:
            glyph_index = glyph_item_index(item)
            if glyph_index is not None:
                repairs[glyph_index] = best_label
    return repairs


def repaired_run_text_from_glyph_items(
    text: str,
    glyph_items: list[dict[str, Any]],
    repairs: dict[int, str],
) -> str | None:
    if not text or not glyph_items or not repairs:
        return None
    ordered: list[tuple[int, str]] = []
    for item in sorted(glyph_items, key=lambda entry: glyph_item_index(entry) or -1):
        glyph_index = glyph_item_index(item)
        glyph_text = item.get("text")
        if (
            glyph_index is None
            or not isinstance(glyph_text, str)
            or len(glyph_text) != 1
        ):
            return None
        ordered.append((glyph_index, glyph_text))
    if not any(glyph_index in repairs for glyph_index, ignored_text in ordered):
        return None

    captured_text = "".join(glyph_text for ignored_index, glyph_text in ordered)
    nonspace_text = "".join(ch for ch in text if not ch.isspace())
    if captured_text != nonspace_text:
        return None

    output: list[str] = []
    glyph_pos = 0
    for ch in text:
        if ch.isspace():
            output.append(ch)
            continue
        glyph_index, glyph_text = ordered[glyph_pos]
        output.append(repairs.get(glyph_index, glyph_text))
        glyph_pos += 1
    return "".join(output)


def repair_text_run_from_glyph_items(
    run: Any,
    glyph_items: list[dict[str, Any]],
    repairs: dict[int, str],
) -> Any:
    text = getattr(run, "text", "")
    replacement = repaired_run_text_from_glyph_items(text, glyph_items, repairs)
    if replacement is None or replacement == text:
        return run
    clusters = repaired_glyph_clusters_for_text(run, replacement)
    replace_method = getattr(run, "replace", None)
    if not callable(replace_method):
        return run
    try:
        return replace_method(text=replacement, glyph_clusters=clusters)
    except TypeError:
        return replace_method(text=replacement)


def repaired_glyph_clusters_for_text(
    run: Any,
    repaired_text: str,
) -> tuple[GlyphCluster, ...]:
    clusters = tuple(getattr(run, "glyph_clusters", ()) or ())
    if not clusters:
        return ()
    original_text = getattr(run, "text", "")
    if "".join(cluster.text for cluster in clusters) != original_text:
        return ()
    repaired_clusters: list[GlyphCluster] = []
    cursor = 0
    changed = False
    for cluster in clusters:
        length = len(cluster.text)
        cluster_text = repaired_text[cursor : cursor + length]
        if len(cluster_text) != length:
            return ()
        cursor += length
        if cluster_text == cluster.text:
            repaired_clusters.append(cluster)
            continue
        changed = True
        confidence = cluster.confidence
        repaired_confidence = 0.88 if confidence is None else max(confidence, 0.88)
        repaired_clusters.append(
            replace(
                cluster,
                text=cluster_text,
                confidence=repaired_confidence,
                provenance=(*cluster.provenance, ("repair", "glyph_bitmap")),
            )
        )
    if cursor != len(repaired_text):
        return ()
    return tuple(repaired_clusters) if changed else clusters


def contextual_punctuation_target_indexes(
    glyph_items: list[dict[str, Any]],
) -> set[int]:
    rows: list[tuple[float, float, float, float, str, int]] = []
    for item in glyph_items:
        glyph_index = glyph_item_index(item)
        text = item.get("text")
        bbox = item.get("bbox")
        if glyph_index is None or not isinstance(text, str) or len(text) != 1:
            continue
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x0, y0, x1, y1 = (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            )
        except TypeError, ValueError:
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        rows.append((x0, y0, x1, y1, text, glyph_index))
    if len(rows) < 3:
        return set()

    lines: list[list[tuple[float, float, float, float, str, int]]] = []
    for row in sorted(
        rows, key=lambda entry: (-((entry[1] + entry[3]) * 0.5), entry[0])
    ):
        mid_y = (row[1] + row[3]) * 0.5
        height = row[3] - row[1]
        for line in lines:
            line_mid = sum((item[1] + item[3]) * 0.5 for item in line) / len(line)
            line_height = max(item[3] - item[1] for item in line)
            if abs(mid_y - line_mid) <= max(height, line_height) * 0.55:
                line.append(row)
                break
        else:
            lines.append([row])

    targets: set[int] = set()
    for line in lines:
        ordered = sorted(line, key=lambda entry: (entry[0], entry[5]))
        for index, row_item in enumerate(ordered):
            text = row_item[4]
            seqno = row_item[5]
            if text not in CONTEXTUAL_PUNCTUATION_GLYPH_TEXT:
                continue
            left = ordered[index - 1] if index > 0 else None
            right = ordered[index + 1] if index + 1 < len(ordered) else None
            if punctuation_inside_word_context(row_item, left, right):
                targets.add(seqno)
            elif left is not None and index > 1:
                left2 = ordered[index - 2]
                if punctuation_after_word_context(row_item, left, left2):
                    targets.add(seqno)
            elif right is not None and index + 2 < len(ordered):
                right2 = ordered[index + 2]
                if punctuation_before_word_context(row_item, right, right2):
                    targets.add(seqno)
    return targets


def punctuation_inside_word_context(
    item: tuple[float, float, float, float, str, int],
    left: tuple[float, float, float, float, str, int] | None,
    right: tuple[float, float, float, float, str, int] | None,
) -> bool:
    if left is None or right is None:
        return False
    return (
        is_alpha_glyph_text(left[4])
        and is_alpha_glyph_text(right[4])
        and glyph_gap(left, item) <= glyph_context_gap(left, item)
        and glyph_gap(item, right) <= glyph_context_gap(item, right)
    )


def punctuation_after_word_context(
    item: tuple[float, float, float, float, str, int],
    left: tuple[float, float, float, float, str, int],
    left2: tuple[float, float, float, float, str, int],
) -> bool:
    return (
        is_alpha_glyph_text(left[4])
        and is_alpha_glyph_text(left2[4])
        and glyph_gap(left2, left) <= glyph_context_gap(left2, left)
        and glyph_gap(left, item) <= glyph_context_gap(left, item)
    )


def punctuation_before_word_context(
    item: tuple[float, float, float, float, str, int],
    right: tuple[float, float, float, float, str, int],
    right2: tuple[float, float, float, float, str, int],
) -> bool:
    return (
        is_alpha_glyph_text(right[4])
        and is_alpha_glyph_text(right2[4])
        and glyph_gap(item, right) <= glyph_context_gap(item, right)
        and glyph_gap(right, right2) <= glyph_context_gap(right, right2)
    )


def is_alpha_glyph_text(text: str) -> bool:
    return len(text) == 1 and text.isalpha()


def glyph_gap(
    left: tuple[float, float, float, float, str, int],
    right: tuple[float, float, float, float, str, int],
) -> float:
    return right[0] - left[2]


def glyph_context_gap(
    left: tuple[float, float, float, float, str, int],
    right: tuple[float, float, float, float, str, int],
) -> float:
    left_width = max(1.0, left[2] - left[0])
    right_width = max(1.0, right[2] - right[0])
    return max(2.0, min(left_width, right_width) * 0.8)


def glyph_code_repairs(glyph_items: list[dict[str, Any]]) -> dict[int, str]:
    examples: dict[tuple[str | None, str, int], dict[str, int]] = {}
    targets: list[dict[str, Any]] = []
    for item in glyph_items:
        keys = glyph_identity_repair_keys(item)
        text = item.get("text")
        if is_trustworthy_bitmap_label(text):
            assert isinstance(text, str)
            for key in keys:
                labels = examples.setdefault(key, {})
                labels[text] = labels.get(text, 0) + 1
        elif is_suspicious_bitmap_label(text):
            targets.append(item)
    if not examples or not targets:
        return {}

    repairs: dict[int, str] = {}
    for item in targets:
        label = None
        for key in glyph_identity_repair_keys(item):
            label = dominant_code_label(examples.get(key, {}))
            if label is not None:
                break
        if label is None:
            continue
        glyph_index = glyph_item_index(item)
        if glyph_index is not None:
            repairs[glyph_index] = label
    return repairs


def glyph_item_index(item: dict[str, Any]) -> int | None:
    glyph_index = item.get("glyph_index")
    if isinstance(glyph_index, int):
        return glyph_index
    seqno = item.get("seqno")
    return seqno if isinstance(seqno, int) else None


def glyph_identity_repair_keys(
    item: dict[str, Any],
) -> list[tuple[str | None, str, int]]:
    font_name = (
        item.get("font_name") if isinstance(item.get("font_name"), str) else None
    )
    keys: list[tuple[str | None, str, int]] = []
    gid = item.get("gid")
    if isinstance(gid, int):
        keys.append((font_name, "gid", gid))
    code = item.get("code")
    if isinstance(code, int):
        keys.append((font_name, "code", code))
    return keys


def dominant_code_label(labels: dict[str, int]) -> str | None:
    if not labels:
        return None
    ordered = sorted(labels.items(), key=lambda item: item[1], reverse=True)
    label, count = ordered[0]
    if count < 1:
        return None
    if len(ordered) > 1 and ordered[1][1] == count:
        return None
    return label


def best_repair_label(
    bitmap: tuple[int, ...], candidates: list[tuple[tuple[int, ...], str]]
) -> str | None:
    best_distance = 1.0
    best_label: str | None = None
    second_distance = 1.0
    for candidate_bitmap, label in candidates:
        distance = bitmap_row_distance(bitmap, candidate_bitmap)
        if distance < best_distance:
            second_distance = best_distance
            best_distance = distance
            best_label = label
        elif distance < second_distance:
            second_distance = distance
    if (
        best_label is not None
        and best_distance <= MAX_BITMAP_REPAIR_DISTANCE
        and best_distance + 0.04 <= second_distance
    ):
        return best_label
    return None


def glyph_repair_key(item: dict[str, Any]) -> tuple[str | None, int, int]:
    bitmap = item.get("bitmap")
    return (
        item.get("font_name") if isinstance(item.get("font_name"), str) else None,
        int(item.get("bitmap_width") or 0),
        len(bitmap) if isinstance(bitmap, (list, tuple)) else 0,
    )


def normalized_bitmap(item: dict[str, Any]) -> tuple[int, ...] | None:
    bitmap = item.get("bitmap")
    width = int(item.get("bitmap_width") or 0)
    if width <= 0 or not isinstance(bitmap, (list, tuple)):
        return None
    rows = tuple(int(row) for row in bitmap if type(row) is int)
    if not rows:
        return None
    mask = (1 << width) - 1
    return tuple(row & mask for row in rows)


def is_trustworthy_bitmap_label(text: Any) -> bool:
    return isinstance(text, str) and len(text) == 1 and text in BITMAP_REPAIR_LABELS


def is_suspicious_bitmap_label(text: Any) -> bool:
    if not isinstance(text, str) or not text:
        return False
    if len(text) != 1:
        if text in LEGITIMATE_MULTI_CHAR_GLYPHS:
            return False
        if len(text) > 3:
            return True
        return any(not (ch.isalnum() or ch.isspace()) for ch in text)
    if text in SUSPICIOUS_GLYPH_TEXT:
        return True
    category = ord(text)
    return 0xE000 <= category <= 0xF8FF or category < 32


def bitmap_row_distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or not right or len(left) != len(right):
        return 1.0
    intersection = 0
    union = 0
    for left_row, right_row in zip(left, right, strict=True):
        intersection += (left_row & right_row).bit_count()
        union += (left_row | right_row).bit_count()
    if union == 0:
        return 1.0
    return 1.0 - intersection / union
