# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Any, TypeGuard, cast

from core_pdf.impl.engine.layout.geometry import RectBox, rect_tuple
from core_pdf.impl.engine.spec.s_07_content.capture import CapturedPath
from core_pdf.impl.engine.spec.s_07_filters.pipeline import decode_stream_data
from core_pdf.impl.engine.spec.s_07_objects.coercion import (
    normalize_pdf_name,
    parse_float,
    parse_int,
)
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_08_graphics.color import (
    ImageColorManager,
    evaluate_sampled_tint_function,
)
from core_pdf.impl.exceptions import PdfRasterTooLargeError
from core_pdf.impl.objects import PdfStream

BIT_IMAGE_MASK_ALPHA = tuple(
    bytes(255 if byte & (0x80 >> bit) else 0 for bit in range(8)) for byte in range(256)
)


BITMAP_GLYPHS_5X7: dict[str, tuple[str, ...]] = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ",": ("00000", "00000", "00000", "00000", "01100", "00100", "01000"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    ";": ("00000", "01100", "01100", "00000", "01100", "00100", "01000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    "\\": ("10000", "01000", "01000", "00100", "00010", "00010", "00001"),
    "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
    ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    "[": ("01110", "01000", "01000", "01000", "01000", "01000", "01110"),
    "]": ("01110", "00010", "00010", "00010", "00010", "00010", "01110"),
    "%": ("11001", "11010", "00010", "00100", "01000", "01011", "10011"),
    "$": ("00100", "01111", "10100", "01110", "00101", "11110", "00100"),
    "#": ("01010", "01010", "11111", "01010", "11111", "01010", "01010"),
    "&": ("01100", "10010", "10100", "01000", "10101", "10010", "01101"),
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
    "!": ("00100", "00100", "00100", "00100", "00100", "00000", "00100"),
    "'": ("01100", "00100", "01000", "00000", "00000", "00000", "00000"),
    '"': ("01010", "01010", "01010", "00000", "00000", "00000", "00000"),
}


def pdf_int(value: Any, default: int) -> int:
    if type(value) is bool:
        return default
    parsed = parse_int(value, None)
    return default if parsed is None else parsed


def pdf_positive_int(value: Any, default: int = 0) -> int:
    parsed = pdf_int(value, default)
    return parsed if parsed > 0 else default


def pdf_number(value: Any) -> TypeGuard[int | float]:
    return type(value) is int or type(value) is float


def image_filter_names(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [name for item in value if (name := normalize_pdf_name(item))]
    name = normalize_pdf_name(value)
    return [name] if name else []


def image_color_space_name(value: Any) -> str | None:
    prefixes: list[str] = []
    seen: set[int] = set()
    current = value
    while True:
        name = normalize_pdf_name(current)
        if name is not None:
            return ":".join((*prefixes, name))
        if not isinstance(current, (list, tuple)) or not current:
            return ":".join(prefixes) if prefixes else None
        marker = id(current)
        if marker in seen:
            return ":".join(prefixes) if prefixes else None
        seen.add(marker)
        kind = normalize_pdf_name(current[0])
        if kind == "Indexed":
            prefixes.append("Indexed")
            if len(current) <= 1:
                return ":".join(prefixes)
            current = current[1]
            continue
        if kind == "ICCBased":
            prefixes.append("ICCBased")
            if len(current) <= 1 or not isinstance(current[1], dict):
                return ":".join(prefixes)
            alternate = lookup_dict_key(current[1], "Alternate")
            if alternate is None:
                return ":".join(prefixes)
            current = alternate
            continue
        if kind is None:
            return ":".join(prefixes) if prefixes else None
        return ":".join((*prefixes, kind))


def image_display_metadata(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    dictionary = data.get("dictionary")
    if not isinstance(dictionary, dict):
        return {}

    width = pdf_positive_int(lookup_dict_key(dictionary, "Width"))
    height = pdf_positive_int(lookup_dict_key(dictionary, "Height"))
    image_mask = lookup_dict_key(dictionary, "ImageMask") is True
    bpc = pdf_positive_int(
        lookup_dict_key(dictionary, "BitsPerComponent"),
        1 if image_mask else 0,
    )
    bbox = data.get("bbox")
    metadata: dict[str, Any] = {
        "kind": kind,
        "width": width,
        "height": height,
        "pixels": width * height if width > 0 and height > 0 else 0,
        "bits_per_component": bpc if bpc > 0 else None,
        "color_space": image_color_space_name(lookup_dict_key(dictionary, "ColorSpace")),
        "filters": image_filter_names(lookup_dict_key(dictionary, "Filter")),
        "image_mask": image_mask,
        "has_mask": lookup_dict_key(dictionary, "Mask") is not None,
        "has_soft_mask": lookup_dict_key(dictionary, "SMask") is not None,
    }

    raw_data = data.get("raw_data", data.get("data"))
    if isinstance(raw_data, (bytes, bytearray, memoryview)):
        metadata["raw_bytes"] = len(raw_data)

    bbox_tuple = rect_tuple(bbox)
    if bbox_tuple is not None:
        x0, y0, x1, y1 = bbox_tuple
        display_width = abs(x1 - x0)
        display_height = abs(y1 - y0)
        metadata["display_width"] = display_width
        metadata["display_height"] = display_height
        metadata["display_area"] = display_width * display_height

    return metadata


@dataclass(slots=True)
class RenderOptions:
    page_number: int | None = None
    rotate: int = 0
    crop: tuple[float, float, float, float] | None = None
    include_annotations: bool = True
    include_layers: bool = True
    include_text: bool = True


@dataclass(slots=True)
class DisplayListItem:
    kind: str
    seqno: int
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DisplayList:
    width: float
    height: float
    items: list[DisplayListItem] = field(default_factory=list)

    def append(self, kind: str, seqno: int, **data: Any) -> None:
        if kind in {"image", "inline-image"}:
            metadata = image_display_metadata(kind, data)
            if metadata:
                explicit = data.get("image_metadata")
                if isinstance(explicit, dict):
                    metadata.update(explicit)
                data["image_metadata"] = metadata
        self.items.append(DisplayListItem(kind=kind, seqno=seqno, data=data))


@dataclass(slots=True)
class RenderedPage:
    page_number: int
    width: float
    height: float
    rotate: int
    display_list: DisplayList
    metadata: dict[str, Any] = field(default_factory=dict)
    raster_cache: dict[tuple[Any, ...], bytes] = field(default_factory=dict, repr=False)
    image_conversion_cache: dict[tuple[int, int, bytes], bytes] = field(
        default_factory=dict, repr=False
    )
    ppm_cache: bytes | None = field(default=None, repr=False)

    def unrotated_raster_size(self, scale: float = 1.0) -> tuple[int, int]:
        """Return the raster size before applying the page rotation."""
        scale = max(0.01, float(scale))
        crop = self.metadata.get("crop")
        if isinstance(crop, (list, tuple)) and len(crop) == 4:
            width = max(1, int(round((float(crop[2]) - float(crop[0])) * scale)))
            height = max(1, int(round((float(crop[3]) - float(crop[1])) * scale)))
            return width, height
        return (
            max(1, int(round(self.width * scale))),
            max(1, int(round(self.height * scale))),
        )

    def raster_size(self, scale: float = 1.0) -> tuple[int, int]:
        """Return the width and height of the bytes produced by ``rasterize``."""
        width, height = self.unrotated_raster_size(scale)
        return (height, width) if self.rotate % 180 else (width, height)

    def validate_raster_size(self, scale: float = 1.0, max_pixels: int | None = None) -> None:
        """Reject a raster request before allocating an oversized RGBA canvas."""
        if max_pixels is None or max_pixels <= 0:
            return
        width, height = self.unrotated_raster_size(scale)
        pixels = width * height
        if pixels > max_pixels:
            raise PdfRasterTooLargeError(
                "PDF page would render to too many pixels for safe processing: "
                f"page={self.page_number}, pixels={pixels}, maximum={max_pixels}. "
                "Try splitting the PDF, reducing the page dimensions, or using a lower render DPI."
            )

    def rasterize(
        self,
        *,
        background: tuple[int, int, int, int] = (255, 255, 255, 0),
        scale: float = 1.0,
        max_pixels: int | None = None,
    ) -> bytes:
        scale = max(0.01, float(scale))
        self.validate_raster_size(scale, max_pixels)
        crop = self.metadata.get("crop")
        cache_key = (
            "raster",
            tuple(background),
            scale,
            tuple(crop) if isinstance(crop, (list, tuple)) else None,
            self.rotate,
        )
        cached = self.raster_cache.get(cache_key)
        if cached is not None:
            return cached
        if isinstance(crop, (list, tuple)) and len(crop) == 4:
            crop_x0, crop_y0, _crop_x1, crop_y1 = (
                float(crop[0]),
                float(crop[1]),
                float(crop[2]),
                float(crop[3]),
            )
        else:
            crop_x0 = 0.0
            crop_y0 = 0.0
            crop_y1 = self.height
        width, height = self.unrotated_raster_size(scale)
        background_bytes = bytes(background)
        pixels = bytearray(background_bytes * (width * height))
        page_group_alpha = self.metadata.get("group_alpha")
        if not pdf_number(page_group_alpha):
            page_group_alpha = None
        buffer_stack: list[tuple[bytearray, float | None, str | None]] = [
            (pixels, page_group_alpha, None)
        ]
        rotate = self.rotate % 360
        clip_path_stack: list[tuple[CapturedPath, str]] = []
        clip_state_stack: list[int] = []
        clip_edge_cache: dict[int, list[tuple[float, float, float, float]]] = {}
        clip_metadata_dirty = True
        clip_stack_generation = 0
        cached_clip_box: tuple[float, float, float, float] | None = None
        cached_clip_is_rectangular = True
        raw_bytes_cache: dict[int, tuple[object, bytes]] = {}
        glyph_seqnos = {
            item.seqno
            for item in self.display_list.items
            if item.kind == "glyph" and item.data.get("bitmap")
        }

        def image_raw_bytes(raw: bytes | bytearray | memoryview) -> bytes:
            if type(raw) is bytes:
                return raw
            key = id(raw)
            cached = raw_bytes_cache.get(key)
            if cached is not None and cached[0] is raw:
                return cached[1]
            data = bytes(raw)
            raw_bytes_cache[key] = (raw, data)
            return data

        def nearest_sample_map(output_count: int, source_count: int) -> list[int]:
            if output_count <= 0 or source_count <= 0:
                return []
            return [
                min(source_count - 1, (index * source_count) // output_count)
                for index in range(output_count)
            ]

        def unit_sample_map(output_count: int) -> list[float]:
            if output_count <= 0:
                return []
            return [index / output_count for index in range(output_count)]

        def intersect_box(
            a: tuple[float, float, float, float],
            b: tuple[float, float, float, float],
        ) -> tuple[float, float, float, float] | None:
            x0 = max(a[0], b[0])
            y0 = max(a[1], b[1])
            x1 = min(a[2], b[2])
            y1 = min(a[3], b[3])
            if x1 <= x0 or y1 <= y0:
                return None
            return x0, y0, x1, y1

        crop_box = (crop_x0, crop_y0, crop_x0 + width / scale, crop_y1)
        path_bbox_cache: dict[int, tuple[float, float, float, float] | None] = {}
        path_edge_cache: dict[int, list[tuple[float, float, float, float]]] = {}
        path_rect_cache: dict[int, tuple[float, float, float, float] | None] = {}
        clip_row_span_cache: dict[
            tuple[int, int, str],
            tuple[tuple[int, int], ...],
        ] = {}
        clip_visible_row_cache: dict[
            tuple[int, int],
            tuple[tuple[int, int], ...],
        ] = {}

        def item_paint_box(
            item: DisplayListItem,
        ) -> tuple[float, float, float, float] | None:
            data = item.data
            value = None
            if item.kind in {"text", "glyph", "image", "inline-image"}:
                value = data.get("bbox")
            elif item.kind in {"annotation", "widget"}:
                value = data.get("rect")
            elif item.kind == "shading":
                value = data.get("bbox") or data.get("rect")
            elif item.kind in {"fill", "fillstroke", "stroke"}:
                path = data.get("path")
                if type(path) is CapturedPath:
                    value = path_bbox(path)
                else:
                    value = data.get("bbox")
            box = rect_tuple(value)
            if box is None:
                return None
            if item.kind in {"stroke", "fillstroke"}:
                line_width = data.get("line_width")
                if pdf_number(line_width):
                    pad = max(0.0, float(line_width) * 0.5)
                    if pad:
                        box = (
                            box[0] - pad,
                            box[1] - pad,
                            box[2] + pad,
                            box[3] + pad,
                        )
            return box

        def rect_tuple(value: Any) -> tuple[float, float, float, float] | None:
            if type(value) is RectBox:
                return value.x0, value.y0, value.x1, value.y1
            value_type = type(value)
            if (value_type is list or value_type is tuple) and len(value) == 4:
                try:
                    x0 = float(value[0])
                    y0 = float(value[1])
                    x1 = float(value[2])
                    y1 = float(value[3])
                except (TypeError, ValueError):
                    return None
                return x0, y0, x1, y1
            return None

        def item_is_outside_crop(item: DisplayListItem) -> bool:
            if crop_x0 <= 0.0 and crop_y0 <= 0.0 and crop_y1 >= self.height:
                return False
            box = item_paint_box(item)
            return box is not None and intersect_box(box, crop_box) is None

        def refresh_clip_metadata() -> None:
            nonlocal clip_metadata_dirty, cached_clip_box, cached_clip_is_rectangular
            if not clip_metadata_dirty:
                return
            clip: tuple[float, float, float, float] | None = None
            rectangular = True
            for path, _rule in clip_path_stack:
                if axis_aligned_rect_box(path) is None:
                    rectangular = False
                box = path_bbox(path)
                if box is None:
                    continue
                clip = box if clip is None else intersect_box(clip, box)
                if clip is None:
                    break
            cached_clip_box = clip
            cached_clip_is_rectangular = rectangular
            clip_metadata_dirty = False

        def mark_clip_metadata_dirty() -> None:
            nonlocal clip_metadata_dirty, clip_stack_generation
            clip_metadata_dirty = True
            clip_stack_generation += 1

        def current_clip() -> tuple[float, float, float, float] | None:
            refresh_clip_metadata()
            return cached_clip_box

        def clip_paths_are_axis_aligned_rects() -> bool:
            refresh_clip_metadata()
            return cached_clip_is_rectangular

        def path_bbox(path: Any) -> tuple[float, float, float, float] | None:
            if type(path) is not CapturedPath:
                return None
            cache_key = id(path)
            if cache_key in path_bbox_cache:
                return path_bbox_cache[cache_key]
            box = path.bbox()
            path_bbox_cache[cache_key] = box
            return box

        def axis_aligned_rect_box(
            path: CapturedPath,
        ) -> tuple[float, float, float, float] | None:
            cache_key = id(path)
            if cache_key in path_rect_cache:
                return path_rect_cache[cache_key]
            if len(path.subpaths) != 1:
                path_rect_cache[cache_key] = None
                return None
            subpath = path.subpaths[0]
            points = list(subpath.points)
            if len(points) >= 2 and points[0] == points[-1]:
                points.pop()
            if len(points) != 4:
                path_rect_cache[cache_key] = None
                return None
            if not subpath.closed and subpath.points[0] != subpath.points[-1]:
                path_rect_cache[cache_key] = None
                return None
            xs = {point[0] for point in points}
            ys = {point[1] for point in points}
            if len(xs) != 2 or len(ys) != 2:
                path_rect_cache[cache_key] = None
                return None
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            if x1 <= x0 or y1 <= y0:
                path_rect_cache[cache_key] = None
                return None
            corners = {(x0, y0), (x0, y1), (x1, y0), (x1, y1)}
            if set(points) != corners:
                path_rect_cache[cache_key] = None
                return None
            for (px0, py0), (px1, py1) in zip(points, points[1:] + points[:1], strict=False):
                if px0 != px1 and py0 != py1:
                    path_rect_cache[cache_key] = None
                    return None
            rect = (x0, y0, x1, y1)
            path_rect_cache[cache_key] = rect
            return rect

        def path_edges(
            path: CapturedPath,
        ) -> list[tuple[float, float, float, float]]:
            cache_key = id(path)
            cached = path_edge_cache.get(cache_key)
            if cached is not None:
                return cached
            edges = path.fill_edges()
            path_edge_cache[cache_key] = edges
            return edges

        def translate_rect(rect: Any, tx: float, ty: float) -> Any:
            if type(rect) is RectBox:
                return RectBox(
                    rect.x0 + tx,
                    rect.y0 + ty,
                    rect.x1 + tx,
                    rect.y1 + ty,
                    seqno=rect.seqno,
                    fill=rect.fill,
                    fill_opacity=rect.fill_opacity,
                )
            rect_type = type(rect)
            if (rect_type is list or rect_type is tuple) and len(rect) == 4:
                return (
                    float(rect[0]) + tx,
                    float(rect[1]) + ty,
                    float(rect[2]) + tx,
                    float(rect[3]) + ty,
                )
            return rect

        def page_x_to_pixel_span(start_x: float, end_x: float) -> tuple[int, int] | None:
            if end_x <= start_x:
                return None
            start = math.ceil((start_x - crop_x0) * scale - 0.5)
            end = math.ceil((end_x - crop_x0) * scale - 0.5)
            start = max(0, min(width, start))
            end = max(0, min(width, end))
            if end <= start:
                return None
            return start, end

        def clip_path_row_spans(
            path: CapturedPath,
            py: int,
            fill_rule: str,
        ) -> tuple[tuple[int, int], ...]:
            cache_key = (id(path), py, fill_rule)
            cached = clip_row_span_cache.get(cache_key)
            if cached is not None:
                return cached
            edges = clip_edge_cache.get(id(path))
            if edges is None:
                edges = path_edges(path)
                clip_edge_cache[id(path)] = edges
            if not edges:
                clip_row_span_cache[cache_key] = ()
                return ()
            page_y = crop_y1 - (py + 0.5) / scale
            crossings: list[tuple[float, int]] = []
            for x0, y0, x1, y1 in edges:
                if y0 == y1:
                    continue
                low = y0 if y0 < y1 else y1
                high = y1 if y1 > y0 else y0
                if not (low <= page_y < high):
                    continue
                t = (page_y - y0) / (y1 - y0)
                crossings.append((x0 + t * (x1 - x0), 1 if y1 > y0 else -1))
            if not crossings:
                clip_row_span_cache[cache_key] = ()
                return ()
            spans: list[tuple[int, int]] = []
            if fill_rule == "evenodd":
                xs = sorted(x for x, _delta in crossings)
                for start_x, end_x in zip(xs[0::2], xs[1::2], strict=False):
                    span = page_x_to_pixel_span(start_x, end_x)
                    if span is not None:
                        spans.append(span)
            else:
                crossings.sort(key=lambda item: item[0])
                winding = 0
                previous_x: float | None = None
                index = 0
                while index < len(crossings):
                    x = crossings[index][0]
                    if previous_x is not None and winding != 0 and x > previous_x:
                        span = page_x_to_pixel_span(previous_x, x)
                        if span is not None:
                            spans.append(span)
                    delta = 0
                    while index < len(crossings) and crossings[index][0] == x:
                        delta += crossings[index][1]
                        index += 1
                    winding += delta
                    previous_x = x
            cached_spans = tuple(spans)
            clip_row_span_cache[cache_key] = cached_spans
            return cached_spans

        def pixel_in_clip(px: int, py: int) -> bool:
            spans = clip_row_visible_spans(py)
            if not spans:
                return False
            index = bisect_left(spans, (px + 1, -1))
            if index <= 0:
                return False
            start, end = spans[index - 1]
            return start <= px < end

        def clip_row_visible_spans(py: int) -> tuple[tuple[int, int], ...]:
            cache_key = (clip_stack_generation, py)
            cached = clip_visible_row_cache.get(cache_key)
            if cached is not None:
                return cached
            if not clip_path_stack:
                clip_visible_row_cache[cache_key] = ((0, width),)
                return clip_visible_row_cache[cache_key]
            spans: tuple[tuple[int, int], ...] | None = None
            for path, fill_rule in clip_path_stack:
                path_spans = clip_path_row_spans(path, py, fill_rule)
                if not path_spans:
                    clip_visible_row_cache[cache_key] = ()
                    return ()
                if spans is None:
                    spans = path_spans
                    continue
                left_index = 0
                right_index = 0
                merged: list[tuple[int, int]] = []
                while left_index < len(spans) and right_index < len(path_spans):
                    left_start, left_end = spans[left_index]
                    right_start, right_end = path_spans[right_index]
                    start = max(left_start, right_start)
                    end = min(left_end, right_end)
                    if end > start:
                        merged.append((start, end))
                    if left_end < right_end:
                        left_index += 1
                    else:
                        right_index += 1
                spans = tuple(merged)
                if not spans:
                    clip_visible_row_cache[cache_key] = ()
                    return ()
            result = spans or ()
            clip_visible_row_cache[cache_key] = result
            return result

        def page_box_to_pixels(
            x0: float, y0: float, x1: float, y1: float
        ) -> tuple[int, int, int, int] | None:
            ix0 = max(0, min(width, math.floor((x0 - crop_x0) * scale)))
            ix1 = max(0, min(width, math.ceil((x1 - crop_x0) * scale)))
            iy0 = max(0, min(height, math.floor((crop_y1 - y1) * scale)))
            iy1 = max(0, min(height, math.ceil((crop_y1 - y0) * scale)))
            if ix1 <= ix0 or iy1 <= iy0:
                return None
            return ix0, iy0, ix1, iy1

        def blend_px(
            idx: int, rgba: tuple[int, int, int, int], blend_mode: str | None = None
        ) -> None:
            nonlocal pixels
            sr, sg, sb, sa = rgba
            if sa <= 0:
                return
            target_alpha = buffer_stack[-1][1] if buffer_stack else None
            if sa >= 255 and target_alpha is None and blend_mode is None:
                pixels[idx] = sr
                pixels[idx + 1] = sg
                pixels[idx + 2] = sb
                pixels[idx + 3] = 255
                return
            if pdf_number(target_alpha):
                sa = max(0, min(255, int(round(sa * float(target_alpha)))))
                if sa <= 0:
                    return
            dr = pixels[idx]
            dg = pixels[idx + 1]
            db = pixels[idx + 2]
            da = pixels[idx + 3]
            src_a = sa / 255.0
            dst_a = da / 255.0
            src_r = sr / 255.0
            src_g = sg / 255.0
            src_b = sb / 255.0
            dst_r = dr / 255.0
            dst_g = dg / 255.0
            dst_b = db / 255.0
            mode = blend_mode.lower() if isinstance(blend_mode, str) else None
            if mode == "multiply":
                src_r *= dst_r
                src_g *= dst_g
                src_b *= dst_b
            elif mode == "screen":
                src_r = 1.0 - (1.0 - src_r) * (1.0 - dst_r)
                src_g = 1.0 - (1.0 - src_g) * (1.0 - dst_g)
                src_b = 1.0 - (1.0 - src_b) * (1.0 - dst_b)
            out_a = src_a + dst_a * (1.0 - src_a)
            if out_a <= 0:
                pixels[idx] = 0
                pixels[idx + 1] = 0
                pixels[idx + 2] = 0
                pixels[idx + 3] = 0
                return
            out_r = int(round(((src_r * 255.0) * src_a + dr * dst_a * (1.0 - src_a)) / out_a))
            out_g = int(round(((src_g * 255.0) * src_a + dg * dst_a * (1.0 - src_a)) / out_a))
            out_b = int(round(((src_b * 255.0) * src_a + db * dst_a * (1.0 - src_a)) / out_a))
            out_a_i = int(round(out_a * 255.0))
            pixels[idx] = max(0, min(255, out_r))
            pixels[idx + 1] = max(0, min(255, out_g))
            pixels[idx + 2] = max(0, min(255, out_b))
            pixels[idx + 3] = max(0, min(255, out_a_i))

        def can_blend_normal_fast(blend_mode: str | None) -> bool:
            return blend_mode is None and buffer_stack[-1][1] is None

        def blend_normal_pixel(idx: int, sr: int, sg: int, sb: int, sa: int) -> None:
            if sa <= 0:
                return
            if sa >= 255:
                pixels[idx] = sr
                pixels[idx + 1] = sg
                pixels[idx + 2] = sb
                pixels[idx + 3] = 255
                return
            dr = pixels[idx]
            dg = pixels[idx + 1]
            db = pixels[idx + 2]
            da = pixels[idx + 3]
            src_a = sa / 255.0
            dst_a = da / 255.0
            out_a = src_a + dst_a * (1.0 - src_a)
            if out_a <= 0:
                pixels[idx] = 0
                pixels[idx + 1] = 0
                pixels[idx + 2] = 0
                pixels[idx + 3] = 0
                return
            out_r = int(round((sr * src_a + dr * dst_a * (1.0 - src_a)) / out_a))
            out_g = int(round((sg * src_a + dg * dst_a * (1.0 - src_a)) / out_a))
            out_b = int(round((sb * src_a + db * dst_a * (1.0 - src_a)) / out_a))
            out_a_i = int(round(out_a * 255.0))
            pixels[idx] = max(0, min(255, out_r))
            pixels[idx + 1] = max(0, min(255, out_g))
            pixels[idx + 2] = max(0, min(255, out_b))
            pixels[idx + 3] = max(0, min(255, out_a_i))

        def blend_normal_solid_span(
            row: int, start: int, end: int, rgba: tuple[int, int, int, int]
        ) -> None:
            sr, sg, sb, sa = rgba
            if sa <= 0 or end <= start:
                return
            start_offset = row + start * 4
            stop_offset = row + end * 4
            if sa >= 255:
                pixels[start_offset:stop_offset] = bytes((sr, sg, sb, 255)) * (end - start)
                return
            src_a = sa / 255.0
            one_minus_src_a = 1.0 - src_a
            for idx in range(start_offset, stop_offset, 4):
                dr = pixels[idx]
                dg = pixels[idx + 1]
                db = pixels[idx + 2]
                da = pixels[idx + 3]
                dst_a = da / 255.0
                out_a = src_a + dst_a * one_minus_src_a
                if out_a <= 0:
                    pixels[idx] = 0
                    pixels[idx + 1] = 0
                    pixels[idx + 2] = 0
                    pixels[idx + 3] = 0
                    continue
                out_r = int(round((sr * src_a + dr * dst_a * one_minus_src_a) / out_a))
                out_g = int(round((sg * src_a + dg * dst_a * one_minus_src_a) / out_a))
                out_b = int(round((sb * src_a + db * dst_a * one_minus_src_a) / out_a))
                out_a_i = int(round(out_a * 255.0))
                pixels[idx] = max(0, min(255, out_r))
                pixels[idx + 1] = max(0, min(255, out_g))
                pixels[idx + 2] = max(0, min(255, out_b))
                pixels[idx + 3] = max(0, min(255, out_a_i))

        def composite_group(
            child: bytearray, group_alpha: float | None, group_blend_mode: str | None
        ) -> None:
            nonlocal pixels
            for idx in range(0, len(child), 4):
                sa = child[idx + 3]
                if sa <= 0:
                    continue
                if pdf_number(group_alpha):
                    sa = max(0, min(255, int(round(sa * float(group_alpha)))))
                    if sa <= 0:
                        continue
                blend_px(
                    idx,
                    (child[idx], child[idx + 1], child[idx + 2], sa),
                    group_blend_mode,
                )

        def fill_rect(
            box: tuple[float, float, float, float] | None,
            rgba: tuple[int, int, int, int],
            blend_mode: str | None = None,
        ) -> None:
            if box is None:
                return
            x0, y0, x1, y1 = box
            clip_box = current_clip()
            if clip_box is not None:
                cx0, cy0, cx1, cy1 = clip_box
                x0 = max(x0, cx0)
                y0 = max(y0, cy0)
                x1 = min(x1, cx1)
                y1 = min(y1, cy1)
                if x1 <= x0 or y1 <= y0:
                    return
            pixel_box = page_box_to_pixels(x0, y0, x1, y1)
            if pixel_box is None:
                return
            ix0, iy0, ix1, iy1 = pixel_box
            rectangular_clip = clip_paths_are_axis_aligned_rects()
            if (
                rgba[3] == 255
                and blend_mode is None
                and buffer_stack[-1][1] is None
                and rectangular_clip
            ):
                span = ix1 - ix0
                if span <= 0:
                    return
                fill = bytes(rgba) * span
                start = ix0 * 4
                stop = ix1 * 4
                for y in range(iy0, iy1):
                    row = y * width * 4
                    pixels[row + start : row + stop] = fill
                return
            normal_fast = can_blend_normal_fast(blend_mode)
            for y in range(iy0, iy1):
                row = y * width * 4
                visible_spans = clip_row_visible_spans(y)
                if not visible_spans:
                    continue
                if rectangular_clip and normal_fast:
                    for start, end in visible_spans:
                        start = max(ix0, start)
                        end = min(ix1, end)
                        if end > start:
                            blend_normal_solid_span(row, start, end, rgba)
                    continue
                for start, end in visible_spans:
                    start = max(ix0, start)
                    end = min(ix1, end)
                    if end <= start:
                        continue
                    if normal_fast:
                        for x in range(start, end):
                            blend_normal_pixel(row + x * 4, *rgba)
                    else:
                        for x in range(start, end):
                            blend_px(row + x * 4, rgba, blend_mode)

        def fill_path_scanlines(
            edge_segments: list[tuple[float, float, float, float, float, float]],
            pixel_box: tuple[int, int, int, int],
            rgba: tuple[int, int, int, int],
            blend_mode: str | None,
            fill_rule: str,
        ) -> None:
            ix0, iy0, ix1, iy1 = pixel_box
            rectangular_clip = clip_paths_are_axis_aligned_rects()
            simple_opaque = (
                rgba[3] == 255
                and blend_mode is None
                and buffer_stack[-1][1] is None
                and rectangular_clip
            )
            normal_fast = can_blend_normal_fast(blend_mode)

            def span_pixels(start_x: float, end_x: float) -> tuple[int, int] | None:
                if end_x <= start_x:
                    return None
                start = math.ceil((start_x - crop_x0) * scale - 0.5)
                end = math.ceil((end_x - crop_x0) * scale - 0.5)
                start = max(ix0, min(ix1, start))
                end = max(ix0, min(ix1, end))
                if end <= start:
                    return None
                return start, end

            for py in range(iy0, iy1):
                visible_spans = clip_row_visible_spans(py)
                if not visible_spans:
                    continue
                page_y = crop_y1 - (py + 0.5) / scale
                crossings: list[tuple[float, int]] = []
                for ex0, ey0, ex1, ey1, low, high in edge_segments:
                    if not (low <= page_y < high):
                        continue
                    t = (page_y - ey0) / (ey1 - ey0)
                    x_intersection = ex0 + t * (ex1 - ex0)
                    crossings.append(
                        (
                            x_intersection,
                            1 if ey1 > ey0 else -1,
                        )
                    )
                if not crossings:
                    continue
                row = py * width * 4
                if fill_rule == "evenodd":
                    xs = sorted(x for x, _delta in crossings)
                    scan_spans = list(zip(xs[0::2], xs[1::2], strict=False))
                else:
                    crossings.sort(key=lambda item: item[0])
                    spans_list: list[tuple[float, float]] = []
                    winding = 0
                    previous_x: float | None = None
                    index = 0
                    while index < len(crossings):
                        x = crossings[index][0]
                        if previous_x is not None and winding != 0 and x > previous_x:
                            spans_list.append((previous_x, x))
                        delta = 0
                        while index < len(crossings) and crossings[index][0] == x:
                            delta += crossings[index][1]
                            index += 1
                        winding += delta
                        previous_x = x
                    scan_spans = spans_list
                for start_x, end_x in scan_spans:
                    span = span_pixels(start_x, end_x)
                    if span is None:
                        continue
                    start, end = span
                    for clip_start, clip_end in visible_spans:
                        visible_start = max(start, clip_start)
                        visible_end = min(end, clip_end)
                        if visible_end <= visible_start:
                            continue
                        if simple_opaque:
                            pixels[row + visible_start * 4 : row + visible_end * 4] = bytes(
                                rgba
                            ) * (visible_end - visible_start)
                            continue
                        if rectangular_clip and normal_fast:
                            blend_normal_solid_span(row, visible_start, visible_end, rgba)
                            continue
                        for px in range(visible_start, visible_end):
                            if normal_fast:
                                blend_normal_pixel(row + px * 4, *rgba)
                            else:
                                blend_px(row + px * 4, rgba, blend_mode)

        def fill_path_sample_crossings(
            edge_segments: list[tuple[float, float, float, float, float, float]],
            page_y: float,
        ) -> list[tuple[float, int]]:
            crossings: list[tuple[float, int]] = []
            for ex0, ey0, ex1, ey1, low, high in edge_segments:
                if not (low <= page_y < high):
                    continue
                t = (page_y - ey0) / (ey1 - ey0)
                x_intersection = ex0 + t * (ex1 - ex0)
                crossings.append((x_intersection, 1 if ey1 > ey0 else -1))
            return crossings

        def fill_path_crossings_contain_point(
            crossings: list[tuple[float, int]],
            page_x: float,
            fill_rule: str,
        ) -> bool:
            if fill_rule == "evenodd":
                odd = False
                for x_intersection, _delta in crossings:
                    if x_intersection > page_x:
                        odd = not odd
                return odd
            winding = 0
            for x_intersection, delta in crossings:
                if x_intersection > page_x:
                    winding += delta
            return winding != 0

        def fill_path_crossing_spans(
            crossings: list[tuple[float, int]],
            fill_rule: str,
        ) -> list[tuple[float, float]]:
            if not crossings:
                return []
            if fill_rule == "evenodd":
                xs = sorted(x for x, _delta in crossings)
                return [
                    (start, end)
                    for start, end in zip(xs[0::2], xs[1::2], strict=False)
                    if end > start
                ]
            crossings.sort(key=lambda item: item[0])
            spans: list[tuple[float, float]] = []
            winding = 0
            previous_x: float | None = None
            index = 0
            while index < len(crossings):
                x = crossings[index][0]
                if previous_x is not None and winding != 0 and x > previous_x:
                    spans.append((previous_x, x))
                delta = 0
                while index < len(crossings) and crossings[index][0] == x:
                    delta += crossings[index][1]
                    index += 1
                winding += delta
                previous_x = x
            return spans

        def draw_bitmap_text(
            box: tuple[float, float, float, float] | None,
            text: Any,
            rgba: tuple[int, int, int, int],
            blend_mode: str | None = None,
        ) -> None:
            if box is None or not isinstance(text, str) or not text:
                return
            x0, y0, x1, y1 = box
            if x1 <= x0 or y1 <= y0:
                return
            drawable = [char for char in text if char == " " or char.upper() in BITMAP_GLYPHS_5X7]
            if not drawable:
                return
            glyph_units = max(1, sum(3 if char == " " else 6 for char in drawable) - 1)
            unit_w = (x1 - x0) / glyph_units
            unit_h = (y1 - y0) / 7.0
            if unit_w <= 0 or unit_h <= 0:
                return
            cursor = x0
            for char in drawable:
                if char == " ":
                    cursor += unit_w * 3.0
                    continue
                pattern = BITMAP_GLYPHS_5X7.get(char.upper())
                if pattern is None:
                    pattern = BITMAP_GLYPHS_5X7["?"]
                for row_index, row in enumerate(pattern):
                    cell_y1 = y1 - row_index * unit_h
                    cell_y0 = y1 - (row_index + 1) * unit_h
                    for col_index, value in enumerate(row):
                        if value != "1":
                            continue
                        cell_x0 = cursor + col_index * unit_w
                        cell_x1 = cursor + (col_index + 1) * unit_w
                        fill_rect((cell_x0, cell_y0, cell_x1, cell_y1), rgba, blend_mode)
                cursor += unit_w * 6.0

        def draw_glyph_bitmap(
            box: tuple[float, float, float, float] | None,
            bitmap: Any,
            rgba: tuple[int, int, int, int],
            blend_mode: str | None = None,
            bitmap_width: Any = None,
            bitmap_height: Any = None,
        ) -> None:
            bitmap_type = type(bitmap)
            if box is None or (bitmap_type is not list and bitmap_type is not tuple) or not bitmap:
                return
            x0, y0, x1, y1 = box
            if x1 <= x0 or y1 <= y0:
                return
            rows = [int(row) for row in bitmap if type(row) is int]
            if not rows:
                return
            bitmap_h = pdf_int(bitmap_height, 0) or len(rows)
            bitmap_w = pdf_int(bitmap_width, 0) or max(
                (row.bit_length() for row in rows), default=0
            )
            if bitmap_w <= 0 or bitmap_h <= 0:
                return
            cell_w = (x1 - x0) / bitmap_w
            cell_h = (y1 - y0) / bitmap_h
            if cell_w <= 0 or cell_h <= 0:
                return
            for row_index, row in enumerate(rows):
                cell_y1 = y1 - row_index * cell_h
                cell_y0 = y1 - (row_index + 1) * cell_h
                for col_index in range(bitmap_w):
                    if not (row & (1 << col_index)):
                        continue
                    cell_x0 = x0 + col_index * cell_w
                    cell_x1 = x0 + (col_index + 1) * cell_w
                    fill_rect((cell_x0, cell_y0, cell_x1, cell_y1), rgba, blend_mode)

        def fill_line(
            x0: float,
            y0: float,
            x1: float,
            y1: float,
            line_width: float,
            rgba: tuple[int, int, int, int],
            dash_pattern: tuple[list[float], float] | None = None,
            blend_mode: str | None = None,
            line_cap: int = 0,
        ) -> None:
            if dash_pattern and dash_pattern[0]:
                dash_array, phase = dash_pattern
                total = sum((max(0.0, float(v)) for v in dash_array), 0.0)
                if total > 0:
                    seg_len = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                    if seg_len > 0:
                        pos = float(phase) % total
                        on = True
                        remaining = seg_len
                        while remaining > 0:
                            dash_idx = 0
                            acc = 0.0
                            for i, val in enumerate(dash_array):
                                acc += max(0.0, float(val))
                                if pos < acc:
                                    dash_idx = i
                                    break
                            on = (dash_idx % 2) == 0
                            dash_end = acc
                            step = min(
                                remaining,
                                dash_end - pos if dash_end > pos else total - pos,
                            )
                            if on and step > 0:
                                t0 = (seg_len - remaining) / seg_len
                                t1 = (seg_len - remaining + step) / seg_len
                                sx0 = x0 + (x1 - x0) * t0
                                sy0 = y0 + (y1 - y0) * t0
                                sx1 = x0 + (x1 - x0) * t1
                                sy1 = y0 + (y1 - y0) * t1
                                fill_line(
                                    sx0,
                                    sy0,
                                    sx1,
                                    sy1,
                                    line_width,
                                    rgba,
                                    None,
                                    blend_mode,
                                    line_cap,
                                )
                            remaining -= step
                            pos = (pos + step) % total
                            if step <= 0:
                                break
                        return
            dx = x1 - x0
            dy = y1 - y0
            seg_len2 = dx * dx + dy * dy
            half = max(0.5 / scale, float(line_width) * 0.5)
            if seg_len2 <= 1e-12:
                if line_cap == 1:
                    fill_circle(x0, y0, half, rgba, blend_mode)
                else:
                    fill_rect(
                        (x0 - half, y0 - half, x0 + half, y0 + half),
                        rgba,
                        blend_mode,
                    )
                return

            seg_len = seg_len2**0.5
            ux = dx / seg_len
            uy = dy / seg_len
            cap_extension = half if line_cap == 2 else 0.0
            box = (
                min(x0, x1) - half - abs(ux) * cap_extension,
                min(y0, y1) - half - abs(uy) * cap_extension,
                max(x0, x1) + half + abs(ux) * cap_extension,
                max(y0, y1) + half + abs(uy) * cap_extension,
            )
            clip_box = current_clip()
            if clip_box is not None:
                clipped = intersect_box(box, clip_box)
                if clipped is None:
                    return
                box = clipped
            pixel_box = page_box_to_pixels(*box)
            if pixel_box is None:
                return

            ix0, iy0, ix1, iy1 = pixel_box
            samples = 4
            sample_total = samples * samples
            half2 = half * half
            inv_seg_len2 = 1.0 / seg_len2
            extension_t = cap_extension / seg_len
            normal_fast = can_blend_normal_fast(blend_mode)
            for py in range(iy0, iy1):
                row = py * width * 4
                for px in range(ix0, ix1):
                    if not pixel_in_clip(px, py):
                        continue
                    covered = 0
                    for sy in range(samples):
                        page_y = crop_y1 - (py + (sy + 0.5) / samples) / scale
                        for sx in range(samples):
                            page_x = crop_x0 + (px + (sx + 0.5) / samples) / scale
                            t = ((page_x - x0) * dx + (page_y - y0) * dy) * inv_seg_len2
                            if line_cap == 0:
                                if t < 0.0 or t > 1.0:
                                    continue
                                closest_t = t
                            elif line_cap == 2:
                                if t < -extension_t or t > 1.0 + extension_t:
                                    continue
                                closest_t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
                            else:
                                closest_t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
                            qx = x0 + dx * closest_t
                            qy = y0 + dy * closest_t
                            dist_x = page_x - qx
                            dist_y = page_y - qy
                            if dist_x * dist_x + dist_y * dist_y <= half2:
                                covered += 1
                    if covered:
                        alpha = max(1, min(255, round(rgba[3] * covered / sample_total)))
                        if normal_fast:
                            blend_normal_pixel(row + px * 4, rgba[0], rgba[1], rgba[2], alpha)
                        else:
                            blend_px(
                                row + px * 4,
                                (rgba[0], rgba[1], rgba[2], alpha),
                                blend_mode,
                            )

        def fill_path(
            path: CapturedPath,
            rgba: tuple[int, int, int, int],
            blend_mode: str | None = None,
            fill_rule: str = "nonzero",
        ) -> None:
            rect = axis_aligned_rect_box(path)
            if rect is not None:
                fill_rect(rect, rgba, blend_mode)
                return
            edges = path_edges(path)
            if not edges:
                return
            bbox = path_bbox(path)
            if bbox is None:
                return
            x0, y0, x1, y1 = bbox
            clip_box = current_clip()
            if clip_box is not None:
                clipped = intersect_box((x0, y0, x1, y1), clip_box)
                if clipped is None:
                    return
                x0, y0, x1, y1 = clipped
            pixel_box = page_box_to_pixels(x0, y0, x1, y1)
            if pixel_box is None:
                return
            ix0, iy0, ix1, iy1 = pixel_box
            edge_segments = [
                (
                    ex0,
                    ey0,
                    ex1,
                    ey1,
                    ey0 if ey0 < ey1 else ey1,
                    ey1 if ey1 > ey0 else ey0,
                )
                for ex0, ey0, ex1, ey1 in edges
                if ey0 != ey1
            ]
            if not edge_segments:
                return
            pixel_area = (ix1 - ix0) * (iy1 - iy0)
            if pixel_area >= 50_000:
                fill_path_scanlines(edge_segments, pixel_box, rgba, blend_mode, fill_rule)
                return
            samples = 4
            rectangular_clip = clip_paths_are_axis_aligned_rects()
            normal_fast = can_blend_normal_fast(blend_mode)
            for py in range(iy0, iy1):
                row = py * width * 4
                sample_spans = []
                for sy in range(samples):
                    crossings = fill_path_sample_crossings(
                        edge_segments,
                        crop_y1 - (py + (sy + 0.5) / samples) / scale,
                    )
                    sample_spans.append(fill_path_crossing_spans(crossings, fill_rule))
                for px in range(ix0, ix1):
                    covered = 0
                    sample_x0 = crop_x0 + (px + 0.5 / samples) / scale
                    sample_step = 1.0 / (samples * scale)
                    for spans in sample_spans:
                        if not spans:
                            continue
                        for sx in range(samples):
                            page_x = sample_x0 + sx * sample_step
                            for start_x, end_x in spans:
                                if start_x <= page_x < end_x:
                                    covered += 1
                                    break
                    if covered:
                        if not rectangular_clip and not pixel_in_clip(px, py):
                            continue
                        alpha = max(
                            1,
                            min(255, round(rgba[3] * covered / (samples * samples))),
                        )
                        if normal_fast:
                            blend_normal_pixel(row + px * 4, rgba[0], rgba[1], rgba[2], alpha)
                        else:
                            blend_px(
                                row + px * 4,
                                (rgba[0], rgba[1], rgba[2], alpha),
                                blend_mode,
                            )

        def fill_circle(
            cx: float,
            cy: float,
            radius: float,
            rgba: tuple[int, int, int, int],
            blend_mode: str | None = None,
        ) -> None:
            pixel_box = page_box_to_pixels(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
            )
            if pixel_box is None:
                return
            ix0, iy0, ix1, iy1 = pixel_box
            radius2 = radius * radius
            normal_fast = can_blend_normal_fast(blend_mode)
            for py in range(iy0, iy1):
                page_y = crop_y1 - (py + 0.5) / scale
                row = py * width * 4
                visible_spans = clip_row_visible_spans(py)
                if not visible_spans:
                    continue
                for clip_start, clip_end in visible_spans:
                    start = max(ix0, clip_start)
                    end = min(ix1, clip_end)
                    if end <= start:
                        continue
                    for px in range(start, end):
                        page_x = crop_x0 + (px + 0.5) / scale
                        dx = page_x - cx
                        dy = page_y - cy
                        if dx * dx + dy * dy > radius2:
                            continue
                        if normal_fast:
                            blend_normal_pixel(row + px * 4, *rgba)
                        else:
                            blend_px(row + px * 4, rgba, blend_mode)

        def fill_join(
            px: float,
            py: float,
            line_width: float,
            rgba: tuple[int, int, int, int],
            line_join: int = 0,
            blend_mode: str | None = None,
        ) -> None:
            radius = max(0.5 / scale, float(line_width) * 0.5)
            if line_join == 1:
                fill_circle(px, py, radius, rgba, blend_mode)
            elif line_join == 2:
                fill_rect(
                    (px - radius, py - radius, px + radius, py + radius),
                    rgba,
                    blend_mode,
                )
            else:
                fill_rect(
                    (px - radius, py - radius, px + radius, py + radius),
                    rgba,
                    blend_mode,
                )

        def fill_cap(
            px: float,
            py: float,
            line_width: float,
            rgba: tuple[int, int, int, int],
            line_cap: int,
            blend_mode: str | None = None,
        ) -> None:
            if line_cap == 0:
                return
            radius = max(0.5 / scale, float(line_width) * 0.5)
            if line_cap == 1:
                fill_circle(px, py, radius, rgba, blend_mode)
            else:
                fill_rect(
                    (px - radius, py - radius, px + radius, py + radius),
                    rgba,
                    blend_mode,
                )

        def stroke_path(
            path: CapturedPath,
            line_width: float,
            rgba: tuple[int, int, int, int],
            dash_pattern: tuple[list[float], float] | None = None,
            blend_mode: str | None = None,
            line_cap: int = 0,
            line_join: int = 0,
        ) -> None:
            for subpath in path.subpaths:
                segments = subpath.edges(close_open=False)
                if not segments:
                    continue
                if dash_pattern and dash_pattern[0]:
                    for x0, y0, x1, y1 in segments:
                        fill_line(
                            x0,
                            y0,
                            x1,
                            y1,
                            line_width,
                            rgba,
                            dash_pattern,
                            blend_mode,
                            line_cap,
                        )
                    continue
                for x0, y0, x1, y1 in segments:
                    fill_line(
                        x0,
                        y0,
                        x1,
                        y1,
                        line_width,
                        rgba,
                        None,
                        blend_mode,
                        0,
                    )
                points = subpath.points
                if len(points) < 2:
                    continue
                for x, y in points[1:-1]:
                    fill_join(x, y, line_width, rgba, line_join, blend_mode)
                if subpath.closed:
                    x, y = points[0]
                    fill_join(x, y, line_width, rgba, line_join, blend_mode)
                else:
                    fill_cap(
                        points[0][0],
                        points[0][1],
                        line_width,
                        rgba,
                        line_cap,
                        blend_mode,
                    )
                    fill_cap(
                        points[-1][0],
                        points[-1][1],
                        line_width,
                        rgba,
                        line_cap,
                        blend_mode,
                    )

        def color_component(value: Any, default: int = 0) -> int:
            if type(value) is bool:
                return default
            try:
                return max(0, min(255, int(round(float(value) * 255.0))))
            except (TypeError, ValueError):
                return default

        def color_rgba(color: Any, opacity: Any) -> tuple[int, int, int, int]:
            alpha = 255
            if pdf_number(opacity):
                alpha = color_component(opacity, 255)
            if isinstance(color, (list, tuple)) and color:
                if len(color) == 1:
                    gray = color_component(color[0])
                    return gray, gray, gray, alpha
                rgb = [color_component(c) for c in color[:3]]
                while len(rgb) < 3:
                    rgb.append(rgb[-1] if rgb else 0)
                return rgb[0], rgb[1], rgb[2], alpha
            return 0, 0, 0, alpha

        def pdf_float(value: Any, default: float) -> float:
            if type(value) is bool:
                return default
            parsed = parse_float(value, None)
            return default if parsed is None else parsed

        def number_array(value: Any) -> list[float]:
            if not isinstance(value, (list, tuple)):
                return []
            out: list[float] = []
            for item in value:
                parsed = parse_float(item, None)
                if parsed is None:
                    return []
                out.append(parsed)
            return out

        def clamp01(value: float) -> float:
            return max(0.0, min(1.0, value))

        def evaluate_pdf_function(function: Any, value: float) -> list[float]:
            if isinstance(function, PdfStream):
                function_type = pdf_int(lookup_dict_key(function.dictionary, "FunctionType"), -1)
                if function_type == 0:
                    try:
                        return evaluate_sampled_tint_function(function, value)
                    except Exception:
                        return [value]
                dictionary = function.dictionary
            elif isinstance(function, dict):
                function_type = pdf_int(lookup_dict_key(function, "FunctionType"), -1)
                dictionary = function
            else:
                return [value]

            if function_type == 2:
                exponent = pdf_float(lookup_dict_key(dictionary, "N"), 1.0)
                c0 = number_array(lookup_dict_key(dictionary, "C0")) or [0.0]
                c1 = number_array(lookup_dict_key(dictionary, "C1")) or [1.0]
                count = max(len(c0), len(c1))
                if len(c0) < count:
                    c0.extend([c0[-1] if c0 else 0.0] * (count - len(c0)))
                if len(c1) < count:
                    c1.extend([c1[-1] if c1 else 1.0] * (count - len(c1)))
                factor = value**exponent
                return [c0[i] + factor * (c1[i] - c0[i]) for i in range(count)]

            if function_type == 3:
                functions = lookup_dict_key(dictionary, "Functions")
                if not isinstance(functions, (list, tuple)) or not functions:
                    return [value]
                bounds = number_array(lookup_dict_key(dictionary, "Bounds"))
                encode = number_array(lookup_dict_key(dictionary, "Encode"))
                index = 0
                while index < len(bounds) and value >= bounds[index]:
                    index += 1
                low = bounds[index - 1] if index > 0 else 0.0
                high = bounds[index] if index < len(bounds) else 1.0
                enc0 = encode[index * 2] if index * 2 < len(encode) else 0.0
                enc1 = encode[index * 2 + 1] if index * 2 + 1 < len(encode) else 1.0
                if high == low:
                    encoded = enc0
                else:
                    encoded = enc0 + (value - low) * (enc1 - enc0) / (high - low)
                return evaluate_pdf_function(functions[min(index, len(functions) - 1)], encoded)

            return [value]

        def shading_color_rgba(
            color_space: Any, components: list[float], opacity: Any
        ) -> tuple[int, int, int, int]:
            alpha = color_component(opacity, 255) if pdf_number(opacity) else 255
            name = image_color_space_name(color_space) or "DeviceRGB"
            if name.endswith("DeviceGray") or len(components) == 1:
                gray = color_component(components[0] if components else 0.0)
                return gray, gray, gray, alpha
            if name.endswith("DeviceCMYK") and len(components) >= 4:
                c, m, y, k = (clamp01(v) for v in components[:4])
                return (
                    max(0, min(255, int(round(255.0 * (1.0 - c) * (1.0 - k))))),
                    max(0, min(255, int(round(255.0 * (1.0 - m) * (1.0 - k))))),
                    max(0, min(255, int(round(255.0 * (1.0 - y) * (1.0 - k))))),
                    alpha,
                )
            rgb = [color_component(c) for c in components[:3]]
            while len(rgb) < 3:
                rgb.append(rgb[-1] if rgb else 0)
            return rgb[0], rgb[1], rgb[2], alpha

        def shading_box(data: dict[str, Any]) -> tuple[float, float, float, float]:
            dictionary = data.get("dictionary")
            box = None
            if isinstance(dictionary, dict):
                bbox_values = number_array(lookup_dict_key(dictionary, "BBox"))
                if len(bbox_values) >= 4:
                    box = tuple(bbox_values[:4])
            if box is None:
                raw_box = data.get("bbox")
                if isinstance(raw_box, RectBox):
                    box = (raw_box.x0, raw_box.y0, raw_box.x1, raw_box.y1)
                elif isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
                    try:
                        box = tuple(float(value) for value in raw_box[:4])
                    except (TypeError, ValueError):
                        box = None
            if box is None:
                box = (crop_x0, crop_y0, crop_x0 + width / scale, crop_y1)
            x0, y0, x1, y1 = box
            return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

        def paint_shading(data: dict[str, Any], blend_mode: str | None) -> None:
            dictionary = data.get("dictionary")
            if not isinstance(dictionary, dict):
                return
            shading_type = pdf_int(lookup_dict_key(dictionary, "ShadingType"), 0)
            if shading_type not in {2, 3}:
                return
            coords = number_array(lookup_dict_key(dictionary, "Coords"))
            if (shading_type == 2 and len(coords) < 4) or (shading_type == 3 and len(coords) < 6):
                return
            domain = number_array(lookup_dict_key(dictionary, "Domain"))
            if len(domain) < 2:
                domain = [0.0, 1.0]
            extend = lookup_dict_key(dictionary, "Extend")
            extend0 = isinstance(extend, (list, tuple)) and len(extend) > 0 and extend[0] is True
            extend1 = isinstance(extend, (list, tuple)) and len(extend) > 1 and extend[1] is True
            function = lookup_dict_key(dictionary, "Function")
            color_space = lookup_dict_key(dictionary, "ColorSpace")
            x0, y0, x1, y1 = shading_box(data)
            clip_box = current_clip()
            if clip_box is not None:
                clipped = intersect_box((x0, y0, x1, y1), clip_box)
                if clipped is None:
                    return
                x0, y0, x1, y1 = clipped
            pixel_box = page_box_to_pixels(x0, y0, x1, y1)
            if pixel_box is None:
                return
            ix0, iy0, ix1, iy1 = pixel_box
            soft_mask_alpha = data.get("soft_mask_alpha")
            normal_fast = can_blend_normal_fast(blend_mode)
            for py in range(iy0, iy1):
                page_y = crop_y1 - (py + 0.5) / scale
                row = py * width * 4
                visible_spans = clip_row_visible_spans(py)
                if not visible_spans:
                    continue
                for px in range(ix0, ix1):
                    index = bisect_left(visible_spans, (px + 1, -1))
                    if index <= 0:
                        continue
                    start, end = visible_spans[index - 1]
                    if not (start <= px < end):
                        continue
                    page_x = crop_x0 + (px + 0.5) / scale
                    unit_t = (
                        axial_shading_t(coords, page_x, page_y)
                        if shading_type == 2
                        else radial_shading_t(coords, page_x, page_y)
                    )
                    if unit_t is None:
                        continue
                    if unit_t < 0.0:
                        if not extend0:
                            continue
                        unit_t = 0.0
                    elif unit_t > 1.0:
                        if not extend1:
                            continue
                        unit_t = 1.0
                    value = domain[0] + unit_t * (domain[1] - domain[0])
                    rgba = shading_color_rgba(
                        color_space,
                        evaluate_pdf_function(function, value),
                        data.get("fill_opacity"),
                    )
                    if pdf_number(soft_mask_alpha):
                        rgba = (
                            rgba[0],
                            rgba[1],
                            rgba[2],
                            max(
                                0,
                                min(255, int(round(rgba[3] * float(soft_mask_alpha)))),
                            ),
                        )
                    if normal_fast:
                        blend_normal_pixel(row + px * 4, *rgba)
                    else:
                        blend_px(row + px * 4, rgba, blend_mode)

        def paint_fill_pattern(data: dict[str, Any], blend_mode: str | None) -> bool:
            pattern = data.get("fill_pattern")
            if not isinstance(pattern, dict):
                return False
            path = data.get("path")
            pushed_clip = False
            if type(path) is CapturedPath and path.has_segments():
                clip_path_stack.append((path, data.get("fill_rule") or "nonzero"))
                mark_clip_metadata_dirty()
                pushed_clip = True
            try:
                if pattern.get("kind") == "shading":
                    dictionary = pattern.get("dictionary")
                    if not isinstance(dictionary, dict):
                        return False
                    shading_data = {
                        "dictionary": dictionary,
                        "bbox": data.get("bbox") or path_bbox(path),
                        "fill_opacity": data.get("fill_opacity"),
                        "soft_mask_alpha": data.get("soft_mask_alpha"),
                    }
                    paint_shading(shading_data, blend_mode)
                    return True
                if pattern.get("kind") == "tiling":
                    return paint_tiling_pattern(pattern, data, blend_mode)
            finally:
                if pushed_clip:
                    clip_path_stack.pop()
                    mark_clip_metadata_dirty()
            return False

        def paint_tiling_pattern(
            pattern: dict[str, Any],
            target_data: dict[str, Any],
            blend_mode: str | None,
        ) -> bool:
            raw_bbox = pattern.get("bbox")
            raw_bbox_type = type(raw_bbox)
            if raw_bbox_type is not list and raw_bbox_type is not tuple:
                return False
            raw_bbox = cast(list[Any] | tuple[Any, ...], raw_bbox)
            if len(raw_bbox) < 4:
                return False
            try:
                cell_x0, cell_y0, cell_x1, cell_y1 = (
                    float(raw_bbox[0]),
                    float(raw_bbox[1]),
                    float(raw_bbox[2]),
                    float(raw_bbox[3]),
                )
                x_step = abs(float(pattern.get("x_step", 0.0)))
                y_step = abs(float(pattern.get("y_step", 0.0)))
            except (TypeError, ValueError):
                return False
            if x_step <= 0.0 or y_step <= 0.0:
                return False
            drawings = pattern.get("drawings")
            runs = pattern.get("runs")
            glyphs = pattern.get("glyphs")
            if type(drawings) is not list:
                drawings = []
            if type(runs) is not list:
                runs = []
            if type(glyphs) is not list:
                glyphs = []
            if not drawings and not runs and not glyphs:
                return False
            target_box = target_data.get("bbox") or path_bbox(target_data.get("path"))
            target_box_type = type(target_box)
            if target_box_type is RectBox:
                target_rect = cast(RectBox, target_box)
                x0, y0, x1, y1 = (
                    target_rect.x0,
                    target_rect.y0,
                    target_rect.x1,
                    target_rect.y1,
                )
            elif target_box_type is list or target_box_type is tuple:
                target_box = cast(list[Any] | tuple[Any, ...], target_box)
                if len(target_box) == 4:
                    try:
                        x0, y0, x1, y1 = (float(value) for value in target_box)
                    except (TypeError, ValueError):
                        return False
                else:
                    x0, y0, x1, y1 = (
                        crop_x0,
                        crop_y0,
                        crop_x0 + width / scale,
                        crop_y1,
                    )
            else:
                x0, y0, x1, y1 = crop_x0, crop_y0, crop_x0 + width / scale, crop_y1
            clip_box = current_clip()
            if clip_box is not None:
                clipped = intersect_box((x0, y0, x1, y1), clip_box)
                if clipped is None:
                    return True
                x0, y0, x1, y1 = clipped
            start_x = cell_x0 + math.floor((x0 - cell_x0) / x_step) * x_step
            start_y = cell_y0 + math.floor((y0 - cell_y0) / y_step) * y_step
            cells = 0
            y = start_y
            while y < y1 + y_step and cells < 10000:
                x = start_x
                while x < x1 + x_step and cells < 10000:
                    tx = x - cell_x0
                    ty = y - cell_y0
                    if x + (cell_x1 - cell_x0) >= x0 and y + (cell_y1 - cell_y0) >= y0:
                        for drawing in drawings:
                            if type(drawing) is not dict:
                                continue
                            paint_tiling_drawing(drawing, tx, ty, blend_mode)
                        paint_tiling_text(runs, tx, ty, blend_mode)
                        paint_tiling_glyphs(glyphs, tx, ty, blend_mode)
                    cells += 1
                    x += x_step
                y += y_step
            return True

        def paint_tiling_text(
            runs: Any,
            tx: float,
            ty: float,
            blend_mode: str | None,
        ) -> None:
            if type(runs) is not list:
                return
            for run in runs:
                if type(run) is not dict or run.get("visible") is False:
                    continue
                bbox = translate_rect(run.get("bbox"), tx, ty)
                rgba = color_rgba(run.get("fill_color"), None)
                draw_bitmap_text(bbox, run.get("text"), rgba, blend_mode)

        def paint_tiling_glyphs(
            glyphs: Any,
            tx: float,
            ty: float,
            blend_mode: str | None,
        ) -> None:
            if type(glyphs) is not list:
                return
            for glyph in glyphs:
                if type(glyph) is not dict or glyph.get("visible") is False:
                    continue
                bbox = translate_rect(glyph.get("bbox"), tx, ty)
                rgba = color_rgba(glyph.get("fill_color"), None)
                draw_glyph_bitmap(
                    bbox,
                    glyph.get("bitmap"),
                    rgba,
                    blend_mode,
                    glyph.get("bitmap_width"),
                    glyph.get("bitmap_height"),
                )

        def paint_tiling_drawing(
            drawing: dict[str, Any],
            tx: float,
            ty: float,
            parent_blend_mode: str | None,
        ) -> None:
            kind = drawing.get("kind")
            blend = drawing.get("blend_mode") or parent_blend_mode
            raw_path = drawing.get("path")
            path = raw_path.translated(tx, ty) if type(raw_path) is CapturedPath else None
            if kind == "shading" and isinstance(drawing.get("dictionary"), dict):
                paint_shading(
                    {
                        "dictionary": drawing.get("dictionary"),
                        "bbox": translate_rect(drawing.get("rect"), tx, ty),
                        "fill_opacity": drawing.get("fill_opacity"),
                        "soft_mask_alpha": drawing.get("soft_mask_alpha"),
                    },
                    blend,
                )
                return
            if kind not in {"fill", "fillstroke", "stroke"}:
                return
            if path is None:
                return
            fill_rgba = color_rgba(drawing.get("fill"), drawing.get("fill_opacity"))
            if kind in {"fill", "fillstroke"}:
                fill_path(
                    path,
                    fill_rgba,
                    blend,
                    drawing.get("fill_rule") or "nonzero",
                )
            if kind in {"stroke", "fillstroke"}:
                stroke_rgba = color_rgba(drawing.get("stroke_color"), drawing.get("stroke_opacity"))
                stroke_path(
                    path,
                    float(drawing.get("line_width") or 1.0),
                    stroke_rgba,
                    drawing.get("dash_pattern"),
                    blend,
                    int(drawing.get("line_cap") or 0),
                    int(drawing.get("line_join") or 0),
                )

        def axial_shading_t(coords: list[float], px: float, py: float) -> float | None:
            x0, y0, x1, y1 = coords[:4]
            dx = x1 - x0
            dy = y1 - y0
            denom = dx * dx + dy * dy
            if denom <= 1e-12:
                return None
            return ((px - x0) * dx + (py - y0) * dy) / denom

        def radial_shading_t(coords: list[float], px: float, py: float) -> float | None:
            x0, y0, r0, x1, y1, r1 = coords[:6]
            dx = x1 - x0
            dy = y1 - y0
            dr = r1 - r0
            qx = px - x0
            qy = py - y0
            a = dx * dx + dy * dy - dr * dr
            b = -2.0 * (qx * dx + qy * dy + r0 * dr)
            c = qx * qx + qy * qy - r0 * r0
            if abs(a) <= 1e-12:
                if abs(b) <= 1e-12:
                    return None
                return -c / b
            disc = b * b - 4.0 * a * c
            if disc < 0.0:
                return None
            root = disc**0.5
            t0 = (-b - root) / (2.0 * a)
            t1 = (-b + root) / (2.0 * a)
            valid = [t for t in (t0, t1) if math.isfinite(t)]
            if not valid:
                return None
            in_range = [t for t in valid if 0.0 <= t <= 1.0]
            return max(in_range) if in_range else min(valid, key=lambda t: abs(t - 0.5))

        def image_rgba(data: dict[str, Any]) -> tuple[int, int, int, int]:
            raw = data.get("raw_data")
            dictionary = data.get("dictionary")
            if isinstance(raw, (bytes, bytearray, memoryview)) and isinstance(dictionary, dict):
                if lookup_dict_key(dictionary, "ImageMask") is True:
                    width_px = pdf_int(lookup_dict_key(dictionary, "Width"), 0)
                    height_px = pdf_int(lookup_dict_key(dictionary, "Height"), 0)
                    mask = image_mask_samples(image_raw_bytes(raw), dictionary, width_px, height_px)
                    if mask:
                        level = mask[0]
                        if len(mask) > 1:
                            level = max(mask)
                        decode = lookup_dict_key(dictionary, "Decode")
                        if image_mask_decode_inverts(decode):
                            level = 255 - level
                        return 0, 0, 0, level
                try:
                    converted = ImageColorManager.convert_image_data(
                        image_raw_bytes(raw), dictionary
                    )
                except Exception:
                    converted = None
                if converted:
                    if len(converted) >= 3:
                        return converted[0], converted[1], converted[2], 255
                    if len(converted) == 1:
                        return converted[0], converted[0], converted[0], 255
            return color_rgba(data.get("fill") or data.get("fill_color"), data.get("fill_opacity"))

        def image_samples(
            raw: bytes, dictionary: dict[Any, Any]
        ) -> tuple[bytes, dict[Any, Any]] | None:
            width_px = pdf_int(lookup_dict_key(dictionary, "Width"), 0)
            height_px = pdf_int(lookup_dict_key(dictionary, "Height"), 0)
            if width_px <= 0 or height_px <= 0:
                return None
            bpc = pdf_int(lookup_dict_key(dictionary, "BitsPerComponent"), 8)
            expected_gray = width_px * height_px
            expected_rgb = expected_gray * 3
            if len(raw) in {expected_gray, expected_rgb}:
                return raw, dictionary

            try:
                decoded = decode_stream_data(raw, dictionary)
            except Exception:
                return None
            if len(decoded) in {expected_gray, expected_rgb}:
                return decoded, dictionary
            if bpc in {1, 2, 4}:
                row_bytes = (width_px * bpc + 7) // 8
                if len(decoded) >= row_bytes * height_px:
                    return decoded, dictionary
            return None

        def image_filter_names(dictionary: dict[Any, Any]) -> list[str | None]:
            filters = lookup_dict_key(dictionary, "Filter")
            return (
                [normalize_pdf_name(item) for item in filters]
                if isinstance(filters, (list, tuple))
                else [normalize_pdf_name(filters)]
            )

        def image_mask_samples(
            raw: bytes, dictionary: dict[Any, Any], width_px: int, height_px: int
        ) -> bytes:
            if width_px <= 0 or height_px <= 0:
                return b""
            try:
                decoded = decode_stream_data(raw, dictionary)
            except Exception:
                decoded = raw
            row_bytes = (width_px + 7) >> 3
            if len(decoded) < row_bytes * height_px:
                return b""
            mask = BIT_IMAGE_MASK_ALPHA
            out = bytearray(width_px * height_px)
            dst = 0
            for row in range(height_px):
                row_start = row * row_bytes
                row_end = row_start + row_bytes
                expanded = b"".join(mask[byte] for byte in decoded[row_start:row_end])
                out[dst : dst + width_px] = expanded[:width_px]
                dst += width_px
            return bytes(out)

        def image_mask_decode_inverts(value: Any) -> bool:
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                return False
            try:
                return float(value[0]) > float(value[1])
            except (TypeError, ValueError):
                return False

        def soft_mask_samples(data: dict[str, Any]) -> tuple[bytes, int, int] | None:
            dictionary = data.get("dictionary")
            if not isinstance(dictionary, dict):
                return None
            raw = dictionary.get("__soft_mask_raw_data__")
            mask_dict = dictionary.get("__soft_mask_dictionary__")
            if not isinstance(raw, (bytes, bytearray, memoryview)) or not isinstance(
                mask_dict, dict
            ):
                return None
            mask_width = pdf_int(lookup_dict_key(mask_dict, "Width"), 0)
            mask_height = pdf_int(lookup_dict_key(mask_dict, "Height"), 0)
            if mask_width <= 0 or mask_height <= 0:
                return None
            sample_dict = dict(mask_dict)
            sample_dict.setdefault("ColorSpace", "DeviceGray")
            sample_dict.setdefault("BitsPerComponent", 8)
            raw_bytes = image_raw_bytes(raw)
            sample_result = image_samples(raw_bytes, sample_dict)
            if sample_result is None:
                samples = raw_bytes
            else:
                samples, sample_dict = sample_result
            try:
                converted = ImageColorManager.convert_image_data(samples, sample_dict)
            except Exception:
                converted = samples
            if converted is None:
                converted = samples
            pixel_count = mask_width * mask_height
            if len(converted) >= pixel_count * 3:
                converted = converted[0 : pixel_count * 3 : 3]
            if len(converted) < pixel_count:
                return None
            return bytes(converted[:pixel_count]), mask_width, mask_height

        def soft_mask_alpha_at(mask: tuple[bytes, int, int] | None, u: float, v: float) -> int:
            if mask is None:
                return 255
            samples, mask_width, mask_height = mask
            src_x = min(mask_width - 1, max(0, int(u * mask_width)))
            src_y = min(mask_height - 1, max(0, int((1.0 - v) * mask_height)))
            idx = src_y * mask_width + src_x
            return samples[idx] if idx < len(samples) else 255

        def image_quad(data: dict[str, Any]) -> tuple[tuple[float, float], ...] | None:
            quad = data.get("quad")
            if isinstance(quad, (list, tuple)) and len(quad) >= 3:
                try:
                    return tuple((float(point[0]), float(point[1])) for point in quad)
                except (TypeError, ValueError, IndexError):
                    return None
            items = data.get("items")
            if not isinstance(items, list):
                return None
            for kind, value in items:
                if kind != "quad":
                    continue
                if not isinstance(value, (list, tuple)) or len(value) < 3:
                    return None
                try:
                    return tuple((float(point[0]), float(point[1])) for point in value)
                except (TypeError, ValueError, IndexError):
                    return None
            return None

        def blit_affine_image(
            quad: tuple[tuple[float, float], ...],
            converted: bytes,
            width_px: int,
            height_px: int,
            comps: int,
            data: dict[str, Any],
            blend_mode: str | None,
        ) -> bool:
            if len(quad) < 3:
                return False
            converted_len = len(converted)
            p00 = quad[0]
            p10 = quad[1]
            p01 = quad[2]
            x0 = min(point[0] for point in quad)
            y0 = min(point[1] for point in quad)
            x1 = max(point[0] for point in quad)
            y1 = max(point[1] for point in quad)
            clip_box = current_clip()
            rectangular_clip = clip_box is not None and clip_paths_are_axis_aligned_rects()
            if clip_box is not None:
                clipped = intersect_box((x0, y0, x1, y1), clip_box)
                if clipped is None:
                    return True
                x0, y0, x1, y1 = clipped
            pixel_box = page_box_to_pixels(x0, y0, x1, y1)
            if pixel_box is None:
                return True
            ix0, iy0, ix1, iy1 = pixel_box
            ux = p10[0] - p00[0]
            uy = p10[1] - p00[1]
            vx = p01[0] - p00[0]
            vy = p01[1] - p00[1]
            det = ux * vy - uy * vx
            if abs(det) < 1e-9:
                return False
            inv_det = 1.0 / det
            soft_mask_alpha = data.get("soft_mask_alpha")
            alpha = 255
            if pdf_number(soft_mask_alpha):
                alpha = max(0, min(255, int(round(alpha * float(soft_mask_alpha)))))
            soft_mask = soft_mask_samples(data)
            if soft_mask is None:
                soft_mask_data = None
                soft_mask_width = 0
                soft_mask_height = 0
                soft_mask_len = 0
            else:
                soft_mask_data, soft_mask_width, soft_mask_height = soft_mask
                soft_mask_len = len(soft_mask_data)
            can_write_opaque = (
                alpha == 255
                and blend_mode is None
                and not buffer_stack[-1][1]
                and soft_mask is None
            )
            normal_fast = can_blend_normal_fast(blend_mode)
            rect_tolerance = max(abs(ux), abs(vy), 1.0) * 1e-6
            if (
                abs(uy) <= rect_tolerance
                and abs(vx) <= rect_tolerance
                and ux > 0
                and vy > 0
                and alpha == 255
                and blend_mode is None
                and can_write_opaque
                and (not clip_path_stack or rectangular_clip)
            ):
                inv_ux = 1.0 / ux
                inv_vy = 1.0 / vy
                x_span = max(1, ix1 - ix0)
                src_x_map: list[int] = []
                for px in range(ix0, ix1):
                    page_x = crop_x0 + (px + 0.5) / scale
                    u = (page_x - p00[0]) * inv_ux
                    if u < 0.0 or u > 1.0:
                        src_x_map.append(-1)
                    else:
                        src_x = int(u * width_px)
                        if src_x < 0:
                            src_x = 0
                        elif src_x >= width_px:
                            src_x = width_px - 1
                        src_x_map.append(src_x)
                affine_row_cache: dict[int, bytes] = {}
                for py in range(iy0, iy1):
                    page_y = crop_y1 - (py + 0.5) / scale
                    v = (page_y - p00[1]) * inv_vy
                    if v < 0.0 or v > 1.0:
                        continue
                    src_y = int((1.0 - v) * height_px)
                    if src_y < 0:
                        src_y = 0
                    elif src_y >= height_px:
                        src_y = height_px - 1
                    row_bytes = affine_row_cache.get(src_y)
                    if row_bytes is None:
                        src_row = src_y * width_px * comps
                        row_out = bytearray(x_span * 4)
                        out = 0
                        if comps == 1:
                            for src_x in src_x_map:
                                if src_x < 0:
                                    out += 4
                                    continue
                                src_idx = src_row + src_x
                                if src_idx >= converted_len:
                                    out += 4
                                    continue
                                gray = converted[src_idx]
                                row_out[out] = gray
                                row_out[out + 1] = gray
                                row_out[out + 2] = gray
                                row_out[out + 3] = 255
                                out += 4
                        else:
                            for src_x in src_x_map:
                                if src_x < 0:
                                    out += 4
                                    continue
                                src_idx = src_row + src_x * comps
                                if src_idx + 2 >= converted_len:
                                    out += 4
                                    continue
                                row_out[out] = converted[src_idx]
                                row_out[out + 1] = converted[src_idx + 1]
                                row_out[out + 2] = converted[src_idx + 2]
                                row_out[out + 3] = 255
                                out += 4
                        row_bytes = bytes(row_out)
                        affine_row_cache[src_y] = row_bytes
                    row = py * width * 4 + ix0 * 4
                    pixels[row : row + x_span * 4] = row_bytes
                return True
            u_from_x = abs(uy) <= rect_tolerance and abs(ux) > rect_tolerance
            u_from_y = abs(ux) <= rect_tolerance and abs(uy) > rect_tolerance
            v_from_x = abs(vy) <= rect_tolerance and abs(vx) > rect_tolerance
            v_from_y = abs(vx) <= rect_tolerance and abs(vy) > rect_tolerance
            if (
                alpha == 255
                and blend_mode is None
                and can_write_opaque
                and (not clip_path_stack or rectangular_clip)
                and ((u_from_x and v_from_y) or (u_from_y and v_from_x))
            ):
                if u_from_x:
                    inv_ux = 1.0 / ux
                    inv_vy = 1.0 / vy
                    src_x_map = [
                        (
                            max(0, min(width_px - 1, int(u * width_px)))
                            if 0.0 <= (u := (crop_x0 + (px + 0.5) / scale - p00[0]) * inv_ux) <= 1.0
                            else -1
                        )
                        for px in range(ix0, ix1)
                    ]
                    src_y_map = [
                        (
                            max(
                                0,
                                min(height_px - 1, int((1.0 - v) * height_px)),
                            )
                            if 0.0 <= (v := (crop_y1 - (py + 0.5) / scale - p00[1]) * inv_vy) <= 1.0
                            else -1
                        )
                        for py in range(iy0, iy1)
                    ]
                    x_span = max(1, ix1 - ix0)
                    orthogonal_row_cache: dict[int, bytes] = {}
                    for dy, py in enumerate(range(iy0, iy1)):
                        src_y = src_y_map[dy]
                        if src_y < 0:
                            continue
                        row_bytes = orthogonal_row_cache.get(src_y)
                        if row_bytes is None:
                            src_row = src_y * width_px * comps
                            row_out = bytearray(x_span * 4)
                            out = 0
                            if comps == 1:
                                for src_x in src_x_map:
                                    if src_x < 0:
                                        out += 4
                                        continue
                                    src_idx = src_row + src_x
                                    if src_idx >= converted_len:
                                        out += 4
                                        continue
                                    gray = converted[src_idx]
                                    row_out[out] = gray
                                    row_out[out + 1] = gray
                                    row_out[out + 2] = gray
                                    row_out[out + 3] = 255
                                    out += 4
                            else:
                                for src_x in src_x_map:
                                    if src_x < 0:
                                        out += 4
                                        continue
                                    src_idx = src_row + src_x * comps
                                    if src_idx + 2 >= converted_len:
                                        out += 4
                                        continue
                                    row_out[out] = converted[src_idx]
                                    row_out[out + 1] = converted[src_idx + 1]
                                    row_out[out + 2] = converted[src_idx + 2]
                                    row_out[out + 3] = 255
                                    out += 4
                            row_bytes = bytes(row_out)
                            orthogonal_row_cache[src_y] = row_bytes
                        row = py * width * 4 + ix0 * 4
                        pixels[row : row + x_span * 4] = row_bytes
                    return True
                inv_uy = 1.0 / uy
                inv_vx = 1.0 / vx
                src_y_map = [
                    (
                        max(0, min(height_px - 1, int((1.0 - v) * height_px)))
                        if 0.0 <= (v := (crop_x0 + (px + 0.5) / scale - p00[0]) * inv_vx) <= 1.0
                        else -1
                    )
                    for px in range(ix0, ix1)
                ]
                x_span = max(1, ix1 - ix0)
                vertical_row_cache: dict[int, bytes] = {}
                for py in range(iy0, iy1):
                    u = (crop_y1 - (py + 0.5) / scale - p00[1]) * inv_uy
                    if u < 0.0 or u > 1.0:
                        continue
                    src_x = max(0, min(width_px - 1, int(u * width_px)))
                    row_bytes = vertical_row_cache.get(src_x)
                    if row_bytes is None:
                        row_out = bytearray(x_span * 4)
                        out = 0
                        if comps == 1:
                            for src_y in src_y_map:
                                if src_y < 0:
                                    out += 4
                                    continue
                                src_idx = src_y * width_px + src_x
                                if src_idx >= converted_len:
                                    out += 4
                                    continue
                                gray = converted[src_idx]
                                row_out[out] = gray
                                row_out[out + 1] = gray
                                row_out[out + 2] = gray
                                row_out[out + 3] = 255
                                out += 4
                        else:
                            for src_y in src_y_map:
                                if src_y < 0:
                                    out += 4
                                    continue
                                src_idx = (src_y * width_px + src_x) * comps
                                if src_idx + 2 >= converted_len:
                                    out += 4
                                    continue
                                row_out[out] = converted[src_idx]
                                row_out[out + 1] = converted[src_idx + 1]
                                row_out[out + 2] = converted[src_idx + 2]
                                row_out[out + 3] = 255
                                out += 4
                        row_bytes = bytes(row_out)
                        vertical_row_cache[src_x] = row_bytes
                    row = py * width * 4 + ix0 * 4
                    pixels[row : row + x_span * 4] = row_bytes
                return True
            for py in range(iy0, iy1):
                page_y = crop_y1 - (py + 0.5) / scale
                row = py * width * 4
                visible_spans = clip_row_visible_spans(py)
                if not visible_spans:
                    continue
                for px in range(ix0, ix1):
                    if not rectangular_clip:
                        index = bisect_left(visible_spans, (px + 1, -1))
                        if index <= 0:
                            continue
                        start, end = visible_spans[index - 1]
                        if not (start <= px < end):
                            continue
                    page_x = crop_x0 + (px + 0.5) / scale
                    rel_x = page_x - p00[0]
                    rel_y = page_y - p00[1]
                    u = (rel_x * vy - rel_y * vx) * inv_det
                    v = (ux * rel_y - uy * rel_x) * inv_det
                    if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
                        continue
                    src_x = int(u * width_px)
                    if src_x < 0:
                        src_x = 0
                    elif src_x >= width_px:
                        src_x = width_px - 1
                    src_y = int((1.0 - v) * height_px)
                    if src_y < 0:
                        src_y = 0
                    elif src_y >= height_px:
                        src_y = height_px - 1
                    src_idx = (src_y * width_px + src_x) * comps
                    if src_idx >= converted_len:
                        continue
                    if comps == 1:
                        gray = converted[src_idx]
                        if can_write_opaque:
                            pixels[row + px * 4 : row + px * 4 + 4] = (
                                gray,
                                gray,
                                gray,
                                255,
                            )
                            continue
                        rgba = (gray, gray, gray, 255)
                    else:
                        if src_idx + 3 > converted_len:
                            continue
                        if can_write_opaque:
                            pixels[row + px * 4 : row + px * 4 + 4] = (
                                converted[src_idx],
                                converted[src_idx + 1],
                                converted[src_idx + 2],
                                255,
                            )
                            continue
                        rgba = (
                            converted[src_idx],
                            converted[src_idx + 1],
                            converted[src_idx + 2],
                            255,
                        )
                    if soft_mask_data is None:
                        mask_alpha = 255
                    else:
                        mask_x = int(u * soft_mask_width)
                        if mask_x < 0:
                            mask_x = 0
                        elif mask_x >= soft_mask_width:
                            mask_x = soft_mask_width - 1
                        mask_y = int((1.0 - v) * soft_mask_height)
                        if mask_y < 0:
                            mask_y = 0
                        elif mask_y >= soft_mask_height:
                            mask_y = soft_mask_height - 1
                        mask_idx = mask_y * soft_mask_width + mask_x
                        mask_alpha = soft_mask_data[mask_idx] if mask_idx < soft_mask_len else 255
                    if mask_alpha <= 0:
                        continue
                    if alpha != 255:
                        rgba = (
                            rgba[0],
                            rgba[1],
                            rgba[2],
                            alpha,
                        )
                    if mask_alpha != 255:
                        rgba = (
                            rgba[0],
                            rgba[1],
                            rgba[2],
                            max(0, min(255, int(round(rgba[3] * mask_alpha / 255)))),
                        )
                    if normal_fast:
                        blend_normal_pixel(row + px * 4, rgba[0], rgba[1], rgba[2], rgba[3])
                    else:
                        blend_px(row + px * 4, rgba, blend_mode)
            return True

        def blit_image(
            box: tuple[float, float, float, float] | None,
            data: dict[str, Any],
            blend_mode: str | None,
        ) -> None:
            if box is None:
                return
            dictionary = data.get("dictionary")
            raw = data.get("raw_data")
            if not isinstance(dictionary, dict) or not isinstance(
                raw, (bytes, bytearray, memoryview)
            ):
                return
            if lookup_dict_key(dictionary, "ImageMask") is True:
                width_px = pdf_int(lookup_dict_key(dictionary, "Width"), 0)
                height_px = pdf_int(lookup_dict_key(dictionary, "Height"), 0)
                if width_px <= 0 or height_px <= 0:
                    return
                mask = image_mask_samples(image_raw_bytes(raw), dictionary, width_px, height_px)
                if not mask:
                    return
                x0, y0, x1, y1 = box
                clip_box = current_clip()
                if clip_box is not None:
                    cx0, cy0, cx1, cy1 = clip_box
                    x0 = max(x0, cx0)
                    y0 = max(y0, cy0)
                    x1 = min(x1, cx1)
                    y1 = min(y1, cy1)
                    if x1 <= x0 or y1 <= y0:
                        return
                pixel_box = page_box_to_pixels(x0, y0, x1, y1)
                if pixel_box is None:
                    return
                ix0, iy0, ix1, iy1 = pixel_box
                x_span = max(1, ix1 - ix0)
                y_span = max(1, iy1 - iy0)
                src_x_map = nearest_sample_map(x_span, width_px)
                src_y_map = nearest_sample_map(y_span, height_px)
                decode = lookup_dict_key(dictionary, "Decode")
                invert = image_mask_decode_inverts(decode)
                target_alpha = buffer_stack[-1][1] if buffer_stack else None
                if not clip_path_stack and blend_mode is None and not pdf_number(target_alpha):
                    for dy, py in enumerate(range(iy0, iy1)):
                        src_y = src_y_map[dy]
                        row = py * width * 4
                        source_row = src_y * width_px
                        for dx, px in enumerate(range(ix0, ix1)):
                            src_idx = source_row + src_x_map[dx]
                            if src_idx >= len(mask):
                                continue
                            alpha = 255 - mask[src_idx] if invert else mask[src_idx]
                            if alpha:
                                idx = row + px * 4
                                pixels[idx] = 0
                                pixels[idx + 1] = 0
                                pixels[idx + 2] = 0
                                pixels[idx + 3] = alpha
                    return
                normal_fast = can_blend_normal_fast(blend_mode)
                for dy, py in enumerate(range(iy0, iy1)):
                    src_y = src_y_map[dy]
                    row = py * width * 4
                    visible_spans = clip_row_visible_spans(py)
                    if not visible_spans:
                        continue
                    for dx, px in enumerate(range(ix0, ix1)):
                        index = bisect_left(visible_spans, (px + 1, -1))
                        if index <= 0:
                            continue
                        start, end = visible_spans[index - 1]
                        if not (start <= px < end):
                            continue
                        src_x = src_x_map[dx]
                        src_idx = src_y * width_px + src_x
                        if src_idx >= len(mask):
                            continue
                        alpha = mask[src_idx]
                        if invert:
                            alpha = 255 - alpha
                        if normal_fast:
                            blend_normal_pixel(row + px * 4, 0, 0, 0, alpha)
                        else:
                            blend_px(row + px * 4, (0, 0, 0, alpha), blend_mode)
                return
            width_px = pdf_int(lookup_dict_key(dictionary, "Width"), 0)
            height_px = pdf_int(lookup_dict_key(dictionary, "Height"), 0)
            if width_px <= 0 or height_px <= 0:
                return
            raw_bytes = image_raw_bytes(raw)
            page_cache_key = (width_px, height_px, raw_bytes)
            converted = self.image_conversion_cache.get(page_cache_key)
            converted_cache_key = "__core_pdf_render_converted_image_data__"
            converted_cache = dictionary.get(converted_cache_key)
            if (
                converted is None
                and isinstance(converted_cache, tuple)
                and len(converted_cache) == 4
                and converted_cache[0] == len(raw_bytes)
                and converted_cache[1] == width_px
                and converted_cache[2] == height_px
                and isinstance(converted_cache[3], bytes)
            ):
                converted = converted_cache[3]
                self.image_conversion_cache[page_cache_key] = converted
            try:
                if converted is None:
                    sample_result = image_samples(raw_bytes, dictionary)
                    if sample_result is None:
                        samples = raw_bytes
                        sample_dictionary = dictionary
                    else:
                        samples, sample_dictionary = sample_result
                    converted = ImageColorManager.convert_image_data(samples, sample_dictionary)
                    if converted is None:
                        return
                    dictionary[converted_cache_key] = (
                        len(raw_bytes),
                        width_px,
                        height_px,
                        converted,
                    )
                    self.image_conversion_cache[page_cache_key] = converted
            except Exception:
                converted = None
            if not converted:
                return
            quad = image_quad(data)
            comps = 3 if len(converted) >= width_px * height_px * 3 else 1
            if quad is not None and blit_affine_image(
                quad, converted, width_px, height_px, comps, data, blend_mode
            ):
                return
            x0, y0, x1, y1 = box
            clip_box = current_clip()
            if clip_box is not None:
                cx0, cy0, cx1, cy1 = clip_box
                x0 = max(x0, cx0)
                y0 = max(y0, cy0)
                x1 = min(x1, cx1)
                y1 = min(y1, cy1)
                if x1 <= x0 or y1 <= y0:
                    return
            pixel_box = page_box_to_pixels(x0, y0, x1, y1)
            if pixel_box is None:
                return
            ix0, iy0, ix1, iy1 = pixel_box
            x_span = max(1, ix1 - ix0)
            y_span = max(1, iy1 - iy0)
            src_x_map = nearest_sample_map(x_span, width_px)
            src_y_map = nearest_sample_map(y_span, height_px)
            soft_mask = soft_mask_samples(data)
            x_unit_map = unit_sample_map(x_span) if soft_mask is not None else []
            y_unit_map = unit_sample_map(y_span) if soft_mask is not None else []
            soft_mask_alpha = data.get("soft_mask_alpha")
            if pdf_number(soft_mask_alpha):
                has_constant_alpha = True
                constant_alpha = float(soft_mask_alpha)
            else:
                has_constant_alpha = False
                constant_alpha = 1.0
            target_alpha = buffer_stack[-1][1] if buffer_stack else None
            can_write_opaque_rows = (
                not clip_path_stack
                and blend_mode is None
                and soft_mask is None
                and (not has_constant_alpha or constant_alpha >= 1.0)
                and not pdf_number(target_alpha)
            )
            normal_fast = can_blend_normal_fast(blend_mode)
            if can_write_opaque_rows:
                rect_row_cache: dict[int, bytes] = {}
                for dy, py in enumerate(range(iy0, iy1)):
                    src_y = src_y_map[dy]
                    row_bytes = rect_row_cache.get(src_y)
                    if row_bytes is None:
                        row_out = bytearray(x_span * 4)
                        out = 0
                        if comps == 1:
                            row_base = src_y * width_px
                            for src_x in src_x_map:
                                src_idx = row_base + src_x
                                if src_idx >= len(converted):
                                    break
                                gray = converted[src_idx]
                                row_out[out] = gray
                                row_out[out + 1] = gray
                                row_out[out + 2] = gray
                                row_out[out + 3] = 255
                                out += 4
                        else:
                            row_base = src_y * width_px * comps
                            for src_x in src_x_map:
                                src_idx = row_base + src_x * comps
                                if src_idx + 2 >= len(converted):
                                    break
                                row_out[out] = converted[src_idx]
                                row_out[out + 1] = converted[src_idx + 1]
                                row_out[out + 2] = converted[src_idx + 2]
                                row_out[out + 3] = 255
                                out += 4
                        row_bytes = bytes(row_out)
                        rect_row_cache[src_y] = row_bytes
                    row = py * width * 4 + ix0 * 4
                    pixels[row : row + x_span * 4] = row_bytes
                return
            for dy, py in enumerate(range(iy0, iy1)):
                src_y = src_y_map[dy]
                mask_v = 1.0 - y_unit_map[dy] if soft_mask is not None else 1.0
                row = py * width * 4
                visible_spans = clip_row_visible_spans(py)
                if not visible_spans:
                    continue
                for clip_start, clip_end in visible_spans:
                    start = max(ix0, clip_start)
                    end = min(ix1, clip_end)
                    if end <= start:
                        continue
                    for px in range(start, end):
                        dx = px - ix0
                        src_x = src_x_map[dx]
                        src_idx = (src_y * width_px + src_x) * comps
                        if src_idx >= len(converted):
                            continue
                        if comps == 1:
                            gray = converted[src_idx]
                            rgba = (gray, gray, gray, 255)
                        else:
                            rgba = (
                                converted[src_idx],
                                converted[src_idx + 1],
                                converted[src_idx + 2],
                                255,
                            )
                        if soft_mask is not None:
                            mask_alpha = soft_mask_alpha_at(
                                soft_mask,
                                x_unit_map[dx],
                                mask_v,
                            )
                            if mask_alpha <= 0:
                                continue
                            if mask_alpha != 255:
                                rgba = (
                                    rgba[0],
                                    rgba[1],
                                    rgba[2],
                                    max(
                                        0,
                                        min(255, int(round(rgba[3] * mask_alpha / 255))),
                                    ),
                                )
                        if has_constant_alpha:
                            rgba = (
                                rgba[0],
                                rgba[1],
                                rgba[2],
                                max(0, min(255, int(round(rgba[3] * constant_alpha)))),
                            )
                        if normal_fast:
                            blend_normal_pixel(row + px * 4, rgba[0], rgba[1], rgba[2], rgba[3])
                        else:
                            blend_px(row + px * 4, rgba, blend_mode)

        for item in self.display_list.items:
            data = item.data
            blend_mode = data.get("blend_mode")
            if item.kind == "state-push":
                clip_state_stack.append(len(clip_path_stack))
                continue
            if item.kind == "state-pop":
                if clip_state_stack:
                    clip_path_stack = clip_path_stack[: clip_state_stack.pop()]
                else:
                    clip_path_stack.clear()
                mark_clip_metadata_dirty()
                continue
            if item.kind == "clip":
                path = data.get("path")
                if type(path) is CapturedPath and path.has_segments():
                    clip_path_stack.append((path, data.get("fill_rule") or "nonzero"))
                    mark_clip_metadata_dirty()
                continue
            if item.kind == "group-begin":
                buffer_stack.append(
                    (
                        bytearray(background_bytes * (width * height)),
                        data.get("fill_opacity"),
                        data.get("blend_mode"),
                    )
                )
                pixels, _parent_alpha, _parent_blend_mode = buffer_stack[-1]
                continue
            if item.kind == "group-end":
                if len(buffer_stack) > 1:
                    child, group_alpha, group_blend_mode = buffer_stack.pop()
                    pixels, _parent_alpha, _parent_blend_mode = buffer_stack[-1]
                    composite_group(
                        child,
                        group_alpha if pdf_number(group_alpha) else data.get("fill_opacity"),
                        group_blend_mode
                        if type(group_blend_mode) is str
                        else data.get("blend_mode"),
                    )
                continue
            if item_is_outside_crop(item):
                continue
            if item.kind == "text":
                if item.seqno in glyph_seqnos:
                    continue
                rgba = color_rgba(data.get("fill_color"), None)
                draw_bitmap_text(data.get("bbox"), data.get("text"), rgba, blend_mode)
            elif item.kind == "glyph":
                if data.get("visible") is False:
                    continue
                rgba = color_rgba(data.get("fill_color"), None)
                draw_glyph_bitmap(
                    data.get("bbox"),
                    data.get("bitmap"),
                    rgba,
                    blend_mode,
                    data.get("bitmap_width"),
                    data.get("bitmap_height"),
                )
            elif item.kind == "shading":
                paint_shading(data, blend_mode)
            elif item.kind in {"fill", "fillstroke", "stroke", "image", "inline-image"}:
                rgba = (
                    image_rgba(data)
                    if item.kind in {"image", "inline-image"}
                    else color_rgba(
                        data.get("fill") or data.get("fill_color"),
                        data.get("fill_opacity"),
                    )
                )
                soft_mask_alpha = data.get("soft_mask_alpha")
                if pdf_number(soft_mask_alpha):
                    rgba = (
                        rgba[0],
                        rgba[1],
                        rgba[2],
                        max(0, min(255, int(round(rgba[3] * float(soft_mask_alpha))))),
                    )
                if item.kind in {"image", "inline-image"}:
                    blit_image(data.get("bbox"), data, blend_mode)
                    continue
                path = data.get("path")
                if type(path) is not CapturedPath:
                    continue
                pattern_painted = item.kind in {
                    "fill",
                    "fillstroke",
                } and paint_fill_pattern(data, blend_mode)
                if item.kind in {"fill", "fillstroke"} and not pattern_painted:
                    fill_path(
                        path,
                        rgba,
                        blend_mode,
                        data.get("fill_rule") or "nonzero",
                    )
                if item.kind in {"stroke", "fillstroke"}:
                    stroke_rgba = color_rgba(data.get("stroke_color"), data.get("stroke_opacity"))
                    if pdf_number(soft_mask_alpha):
                        stroke_rgba = (
                            stroke_rgba[0],
                            stroke_rgba[1],
                            stroke_rgba[2],
                            max(
                                0,
                                min(
                                    255,
                                    int(round(stroke_rgba[3] * float(soft_mask_alpha))),
                                ),
                            ),
                        )
                    stroke_path(
                        path,
                        float(data.get("line_width") or 1.0),
                        stroke_rgba,
                        data.get("dash_pattern"),
                        blend_mode,
                        int(data.get("line_cap") or 0),
                        int(data.get("line_join") or 0),
                    )
            elif item.kind == "annotation":
                if not data.get("appearance_rendered"):
                    fill_rect(data.get("rect"), (255, 215, 0, 96))
            elif item.kind == "widget":
                if not data.get("appearance_rendered"):
                    fill_rect(data.get("rect"), (80, 160, 255, 72))
        if rotate == 0:
            result = bytes(pixels)
            self.raster_cache[cache_key] = result
            return result

        rotated = bytearray(background_bytes * (width * height))
        for y in range(height):
            for x in range(width):
                src_idx = (y * width + x) * 4
                if rotate == 90:
                    dst_x, dst_y = height - 1 - y, x
                    dst_w, dst_h = height, width
                elif rotate == 180:
                    dst_x, dst_y = width - 1 - x, height - 1 - y
                    dst_w, dst_h = width, height
                elif rotate == 270:
                    dst_x, dst_y = y, width - 1 - x
                    dst_w, dst_h = height, width
                else:
                    dst_x, dst_y = x, y
                    dst_w, dst_h = width, height
                if 0 <= dst_x < dst_w and 0 <= dst_y < dst_h:
                    dst_idx = (dst_y * dst_w + dst_x) * 4
                    rotated[dst_idx : dst_idx + 4] = pixels[src_idx : src_idx + 4]
        result = bytes(rotated)
        self.raster_cache[cache_key] = result
        return result

    def to_ppm(self) -> bytes:
        if self.ppm_cache is not None:
            return self.ppm_cache
        rgba = self.rasterize()
        width, height = self.raster_size()
        header = f"P6\n{width} {height}\n255\n".encode("ascii")
        pixel_count = len(rgba) // 4
        out = bytearray(len(header) + pixel_count * 3)
        out[: len(header)] = header
        src = 0
        dst = len(header)
        for ignored in range(pixel_count):
            out[dst] = rgba[src]
            out[dst + 1] = rgba[src + 1]
            out[dst + 2] = rgba[src + 2]
            src += 4
            dst += 3
        result = bytes(out)
        self.ppm_cache = result
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "rotate": self.rotate,
            "display_list": [
                {"kind": item.kind, "seqno": item.seqno, "data": dict(item.data)}
                for item in self.display_list.items
            ],
            "metadata": dict(self.metadata),
        }
