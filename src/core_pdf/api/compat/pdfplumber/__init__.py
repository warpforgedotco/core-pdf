"""Supported high-level pdfplumber-shaped APIs backed by core-pdf."""

from __future__ import annotations

import builtins
import math
import struct
import zlib
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from io import BytesIO
from itertools import groupby
from operator import itemgetter
from types import SimpleNamespace
from typing import Any, TypeAlias, cast

from core_pdf import PdfDocument
from core_pdf.impl.engine.layout.geometry import (
    bbox_contains,
    bbox_intersects,
    bbox_union,
    flip_rect_vertical,
)
from core_pdf.impl.engine.spec.s_07_objects.coercion import normalize_pdf_name
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.primitives import PdfReference

from .exceptions import PdfminerException

BBox: TypeAlias = tuple[float, float, float, float]
PdfInput: TypeAlias = Any
ObjectDict: TypeAlias = dict[str, Any]
LIGATURE_EXPANSIONS = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",
}


class ClosingMixin:
    def close(self) -> None:
        return None

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def flip_box(box: object, height: float) -> BBox:
    return flip_rect_vertical(cast(Any, box), height)


def cluster_by(
    values: Iterable[Any], key: Callable[[Any], Any] | str, tolerance: float = 0
) -> list[list[Any]]:
    getter = itemgetter(key) if isinstance(key, str) else key
    items = list(values)
    ordered_values = sorted({getter(item) for item in items})
    cluster_ids: dict[Any, int] = {}
    cluster = 0
    previous: Any = None
    for index, value in enumerate(ordered_values):
        if index:
            separated = (
                value - previous > tolerance
                if isinstance(value, (int, float)) and isinstance(previous, (int, float))
                else value != previous
            )
            if separated:
                cluster += 1
        cluster_ids[value] = cluster
        previous = value
    # Sorting only by cluster id is intentionally stable. pdfplumber retains
    # source order within a geometric line; sorting by the raw coordinate here
    # changes where zero-width space glyphs split words.
    ordered = sorted(items, key=lambda item: cluster_ids[getter(item)])
    return [list(group) for _, group in groupby(ordered, lambda item: cluster_ids[getter(item)])]


def cluster_by_preserving_order(
    values: Iterable[Any], key: Callable[[Any], Any] | str, tolerance: float = 0
) -> list[list[Any]]:
    getter = itemgetter(key) if isinstance(key, str) else key
    items = list(values)
    numeric_groups = cluster_by((getter(item) for item in items), lambda value: value, tolerance)
    cluster_ids = {
        value: cluster_index
        for cluster_index, group in enumerate(numeric_groups)
        for value in group
    }
    groups: list[list[Any]] = []
    previous_cluster: int | None = None
    for item in items:
        cluster = cluster_ids[getter(item)]
        if not groups or cluster != previous_cluster:
            groups.append([item])
        else:
            groups[-1].append(item)
        previous_cluster = cluster
    return groups


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def encode_png(width: int, height: int, channels: int, pixels: bytes | bytearray) -> bytes:
    stride = width * channels
    scanlines = b"".join(b"\0" + pixels[y * stride : (y + 1) * stride] for y in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 6 if channels == 4 else 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(scanlines))
        + png_chunk(b"IEND", b"")
    )


class TableSettings:
    def __init__(self, **values: Any) -> None:
        allowed = {
            "vertical_strategy",
            "horizontal_strategy",
            "explicit_vertical_lines",
            "explicit_horizontal_lines",
            "text_settings",
            "text_layout",
            "snap_tolerance",
            "snap_x_tolerance",
            "snap_y_tolerance",
            "join_tolerance",
            "join_x_tolerance",
            "join_y_tolerance",
            "edge_min_length",
            "edge_min_length_prefilter",
            "min_words_vertical",
            "min_words_horizontal",
            "intersection_tolerance",
            "intersection_x_tolerance",
            "intersection_y_tolerance",
        }
        unknown = set(values) - allowed
        if unknown:
            if "strategy" in unknown:
                raise TypeError("strategy is not a valid table setting")
            raise ValueError(f"Unknown table setting: {sorted(unknown)[0]}")
        self.vertical_strategy = values.pop("vertical_strategy", "lines")
        self.horizontal_strategy = values.pop("horizontal_strategy", "lines")
        self.text_settings = values.pop("text_settings", {})
        self.__dict__.update(values)
        for name, value in self.__dict__.items():
            if name.endswith("tolerance") and isinstance(value, (int, float)) and value < 0:
                raise ValueError(f"{name} must be non-negative")
        for strategy in (self.vertical_strategy, self.horizontal_strategy):
            if strategy not in {"lines", "lines_strict", "text", "explicit"}:
                raise ValueError(f"unknown table strategy: {strategy}")
            if strategy == "explicit" and not getattr(
                self,
                "explicit_vertical_lines"
                if strategy == self.vertical_strategy
                else "explicit_horizontal_lines",
                None,
            ):
                raise ValueError("explicit table strategy requires explicit lines")

    @classmethod
    def resolve(cls, settings: Mapping[str, Any] | "TableSettings" | None) -> "TableSettings":
        if settings is None:
            return cls()
        if isinstance(settings, cls):
            return settings
        if not isinstance(settings, Mapping):
            raise ValueError("table settings must be a mapping or TableSettings")
        raw = {str(key): value for key, value in settings.items()}
        text_values = {key[5:]: value for key, value in raw.items() if key.startswith("text_")}
        values = {key: value for key, value in raw.items() if not key.startswith("text_")}
        if text_values:
            existing = values.get("text_settings", {})
            values["text_settings"] = {
                **(existing if isinstance(existing, Mapping) else {}),
                **text_values,
            }
        return cls(**values)


def _source(value: PdfInput, password: str = "", unicode_norm: str | None = None) -> PdfDocument:
    try:
        # pdfplumber consumes pdfminer.six's parser semantics.  Keep the
        # engine's native recovery and text-operator behavior unchanged, but
        # select the legacy policy at this compatibility boundary.
        document = PdfDocument.open(
            value,
            password=password,
            recovery_scan_all_revisions=False,
            legacy_pdfminer_text_operators=True,
        )
    except Exception as exc:
        raise PdfminerException(exc) from exc
    cast(Any, document).compat_unicode_norm = unicode_norm
    return document


class EnginePageAdapter:
    """pdfplumber-specific projection of one engine page."""

    def __init__(self, page: Any) -> None:
        self.page = page
        width: int | float = page.width
        height: int | float = page.height
        media_left: int | float = 0
        media_bottom: int | float = 0
        media_x0: int | float = 0
        media_top: int | float = 0

        def resolve_all(value: Any, active: frozenset[tuple[int, int]] = frozenset()) -> Any:
            if isinstance(value, PdfReference):
                key = (value.object_number, value.generation_number)
                if key in active:
                    raise ValueError("cyclic indirect object while resolving page geometry")
                resolved = page.document.resolver.resolve(value)
                if (
                    isinstance(resolved, dict)
                    and normalize_pdf_name(lookup_dict_key(resolved, "Type")) == "Page"
                ):
                    return value
                return resolve_all(resolved, active | {key})
            if isinstance(value, (list, tuple)):
                return type(value)(resolve_all(item, active) for item in value)
            if isinstance(value, dict):
                is_annotation = normalize_pdf_name(lookup_dict_key(value, "Type")) == "Annot"
                return {
                    key: item
                    if is_annotation and normalize_pdf_name(key) == "Parent"
                    else resolve_all(item, active)
                    for key, item in value.items()
                }
            return value

        try:
            raw_box = lookup_dict_key(page.page_dict, "MediaBox")
            if raw_box is None:
                raw_box = page.inherited_values.get("MediaBox")
            raw_box = resolve_all(raw_box)
            if raw_box is None:
                raise ValueError("MediaBox is missing")
            if raw_box is not None and (
                not isinstance(raw_box, (tuple, list))
                or len(raw_box) != 4
                or not all(
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in raw_box
                )
            ):
                raise ValueError(f"invalid MediaBox: {raw_box!r}")
            if isinstance(raw_box, (tuple, list)):
                media_left = raw_box[0]
                media_bottom = raw_box[1]
                width = abs(raw_box[2] - raw_box[0])
                height = abs(raw_box[3] - raw_box[1])
                normalized_x0 = min(raw_box[0], raw_box[2])
                normalized_y0 = min(raw_box[1], raw_box[3])
                if int(page.rotation) % 180:
                    media_x0 = normalized_y0
                    media_top = -normalized_x0
                else:
                    media_x0 = normalized_x0
                    media_top = -normalized_y0
        except ValueError:
            raise
        except Exception:
            pass
        self.info = SimpleNamespace(
            index=page.page_number - 1,
            number=page.page_number,
            width=width,
            height=height,
            media_left=media_left,
            media_bottom=media_bottom,
            media_x0=media_x0,
            media_top=media_top,
            rotation=int(page.rotation),
        )

    def text_characters(self) -> Iterator[Any]:
        from ..pdfminer import (
            internal_pdfminer_descent,
            internal_pdfminer_embedded_cmap_is_unusable,
            internal_pdfminer_font_name,
            internal_pdfminer_glyph_text,
            internal_pdfminer_ligature_overrides,
            internal_pdfminer_literal_glyphs,
            internal_pdfminer_normalized_width,
            internal_pdfminer_offsets,
            internal_pdfminer_validate_page_resources,
        )

        internal_pdfminer_validate_page_resources(self.page)
        compatibility_glyphs: list[Any] = []
        for glyph in self.page.get_page_program().products.glyphs:
            provenance = dict(glyph.provenance) if glyph.provenance else {}
            underlying = provenance.get("compatibility_glyphs")
            if glyph.unicode_source == "actual_text" and isinstance(underlying, tuple):
                compatibility_glyphs.extend(underlying)
            else:
                compatibility_glyphs.append(glyph)
        projected_glyphs, literal_offsets = internal_pdfminer_literal_glyphs(compatibility_glyphs)
        pdfminer_offsets = internal_pdfminer_offsets(projected_glyphs, literal_offsets)
        ligatures, skipped_ligature_parts = internal_pdfminer_ligature_overrides(projected_glyphs)
        for glyph in projected_glyphs:
            if internal_pdfminer_embedded_cmap_is_unusable(glyph):
                continue
            if id(glyph) in skipped_ligature_parts:
                continue
            ligature = ligatures.get(id(glyph))
            text = ligature[0] if ligature is not None else internal_pdfminer_glyph_text(glyph)
            if not text or ("source", "annotation_appearance") in glyph.provenance:
                continue
            x0, y0, x1, y1 = ligature[1] if ligature is not None else glyph.advance_bbox
            offset_x, offset_y = pdfminer_offsets.get(id(glyph), (0.0, 0.0))
            projected_origin = pdfminer_offsets.origins.get(id(glyph))
            x0 += offset_x
            x1 += offset_x
            y0 += offset_y
            y1 += offset_y
            font_height = glyph.effective_font_height or glyph.font_size
            width_code = glyph.char_code if glyph.char_code is not None else glyph.cid
            width_lookup = getattr(glyph.font_decoder, "glyph_width", None)
            provenance = dict(glyph.provenance) if glyph.provenance else {}
            matrix = provenance.get("text_matrix")
            origin = provenance.get("pdfminer_origin")
            upright = glyph.rotation_angle % 180 == 0
            advance = x1 - x0
            if isinstance(matrix, (tuple, list)) and len(matrix) == 4:
                a, b, c, d = (float(value) for value in matrix)
                scaling = float(provenance.get("horizontal_scale", 100.0)) * 0.01
                # The engine records the text matrix in unrotated page space,
                # while pdfminer folds the page's rotation into LTChar.matrix.
                # Apply that final coordinate transform before evaluating the
                # orientation predicate.
                if self.info.rotation % 360 == 90:
                    a, b, c, d = b, -a, d, -c
                elif self.info.rotation % 360 == 180:
                    a, b, c, d = -a, -b, -c, -d
                elif self.info.rotation % 360 == 270:
                    a, b, c, d = -b, a, -d, c
                # This is pdfminer's LTChar orientation test.  In particular, a
                # quarter-turned text matrix can still be "upright": upright
                # describes the matrix handedness used by word grouping, not
                # whether the baseline is horizontal on the rendered page.
                upright = a * d * scaling > 0.0 and b * c <= 0.0
            if (
                getattr(glyph.font_decoder, "is_vertical", False)
                and width_code is not None
                and isinstance(matrix, (tuple, list))
                and len(matrix) == 4
                and isinstance(origin, (tuple, list))
                and len(origin) == 2
            ):
                metric = glyph.font_decoder.vertical_metrics.get(
                    width_code,
                    (
                        glyph.font_decoder.default_vertical_width,
                        glyph.font_decoder.glyph_width(width_code) / 2.0,
                        glyph.font_decoder.default_vertical_origin_y,
                    ),
                )
                a, b, c, d = (float(value) for value in matrix)
                origin_x, origin_y = (
                    projected_origin
                    if projected_origin is not None
                    else (float(origin[0]) + offset_x, float(origin[1]) + offset_y)
                )
                font_size = glyph.font_size
                scaling = float(provenance.get("horizontal_scale", 100.0)) * 0.01
                local_left = -float(metric[1]) * font_size * 0.001
                local_top = (1000.0 - float(metric[2])) * font_size * 0.001 + float(
                    provenance.get("text_rise", 0.0)
                )
                advance = -float(metric[0]) * font_size * 0.001 * scaling
                corners = tuple(
                    (
                        a * horizontal + c * vertical + origin_x,
                        b * horizontal + d * vertical + origin_y,
                    )
                    for horizontal in (local_left, local_left + font_size)
                    for vertical in (local_top + advance, local_top)
                )
                x0 = min(point[0] for point in corners)
                y0 = min(point[1] for point in corners)
                x1 = max(point[0] for point in corners)
                y1 = max(point[1] for point in corners)
                font_height = x1 - x0
            elif (
                not getattr(glyph.font_decoder, "is_vertical", False)
                and width_code is not None
                and callable(width_lookup)
            ):
                if (
                    isinstance(matrix, (tuple, list))
                    and len(matrix) == 4
                    and isinstance(origin, (tuple, list))
                    and len(origin) == 2
                ):
                    a, b, c, d = (float(value) for value in matrix)
                    origin_x, origin_y = (
                        projected_origin
                        if projected_origin is not None
                        else (float(origin[0]) + offset_x, float(origin[1]) + offset_y)
                    )
                    font_size = glyph.font_size
                    scaling = float(provenance.get("horizontal_scale", 100.0)) * 0.01
                    advance = internal_pdfminer_normalized_width(glyph) * font_size * scaling
                    descent = internal_pdfminer_descent(glyph)
                    low = descent * font_size + float(provenance.get("text_rise", 0.0))
                    high = low + font_size
                    corners = tuple(
                        (
                            a * along + c * vertical + origin_x,
                            b * along + d * vertical + origin_y,
                        )
                        for along in (0.0, advance)
                        for vertical in (low, high)
                    )
                    x0 = min(point[0] for point in corners)
                    y0 = min(point[1] for point in corners)
                    x1 = max(point[0] for point in corners)
                    y1 = max(point[1] for point in corners)
                    font_height = y1 - y0
            elif glyph.rotation_angle % 360 == 90:
                x0 = x1 - font_height
            elif glyph.rotation_angle % 360 == 270:
                x1 = x0 + font_height
            x0 -= self.info.media_left
            x1 -= self.info.media_left
            y0 -= self.info.media_bottom
            y1 -= self.info.media_bottom
            yield SimpleNamespace(
                text=text,
                bbox=SimpleNamespace(x0=x0, y0=y0, x1=x1, y1=y1),
                font_name=internal_pdfminer_font_name(glyph),
                font_size=font_height,
                color=glyph.fill,
                rotation_angle=glyph.rotation_angle,
                upright=upright,
                advance=advance,
                sequence=glyph.seqno,
            )

    def drawings(self) -> Iterator[Any]:
        for drawing in self.page.get_drawings():
            box = drawing.rect
            yield SimpleNamespace(
                kind=drawing.kind,
                bbox=(SimpleNamespace(x0=box[0], y0=box[1], x1=box[2], y1=box[3]) if box else None),
                fill=drawing.fill,
                stroke=drawing.stroke_color,
                fill_opacity=drawing.fill_opacity,
                stroke_opacity=drawing.stroke_opacity,
                sequence=drawing.seqno,
                items=(),
                data={
                    "path": None,
                    "dash_pattern": drawing.dash_pattern,
                    "line_width": drawing.line_width,
                    "line_cap": drawing.line_cap,
                    "line_join": drawing.line_join,
                },
            )

    def images(self) -> Iterator[Any]:
        for image in self.page.extract_images():
            metadata = image.image_metadata
            box = image.rect or image.image_clip
            if metadata is None or box is None:
                continue
            data = image.data
            yield SimpleNamespace(
                bbox=SimpleNamespace(x0=box[0], y0=box[1], x1=box[2], y1=box[3]),
                sequence=image.seqno,
                width=metadata.width,
                height=metadata.height,
                channels=metadata.channels,
                color_model=metadata.color_model,
                data=bytes(data) if isinstance(data, (bytes, bytearray, memoryview)) else None,
            )

    def render(self, *, dpi: float) -> Any:
        raster = self.page.render().rasterize(scale=max(0.01, dpi / 72.0))
        return SimpleNamespace(
            data=raster.pixels,
            width=raster.width,
            height=raster.height,
            channels=raster.channels,
            dpi=dpi,
        )


def _bbox(page: EnginePageAdapter, box: Any) -> BBox:
    return flip_box(box, page.info.height)


def _envelope(
    object_type: str, page_number: int, box: BBox, doctop: float, **extra: Any
) -> ObjectDict:
    """Build the pdfplumber nine-key object envelope plus caller extras."""
    x0, top, x1, bottom = box
    return {
        "object_type": object_type,
        "page_number": page_number,
        "x0": x0,
        "x1": x1,
        "top": top,
        "bottom": bottom,
        "doctop": doctop + top,
        "width": x1 - x0,
        "height": bottom - top,
        **extra,
    }


def _char(page: EnginePageAdapter, item: Any, doctop: float) -> ObjectDict:
    x0, top, x1, bottom = _bbox(page, (item.bbox.x0, item.bbox.y0, item.bbox.x1, item.bbox.y1))
    if page.info.rotation % 360 == 90:
        rotated_width = page.info.height
        x0, x1, top, bottom = (
            rotated_width - bottom,
            rotated_width - top,
            x0,
            x1,
        )
    elif page.info.rotation % 360 == 180:
        x0, x1, top, bottom = (
            page.info.width - x1,
            page.info.width - x0,
            page.info.height - bottom,
            page.info.height - top,
        )
    elif page.info.rotation % 360 == 270:
        rotated_height = page.info.width
        x0, x1, top, bottom = top, bottom, rotated_height - x1, rotated_height - x0
    x0 += page.info.media_x0
    x1 += page.info.media_x0
    top += page.info.media_top
    bottom += page.info.media_top
    text = item.text
    document = getattr(page.page, "document", None)
    if getattr(document, "compat_unicode_norm", None) == "NFC" and text == "\u037e":
        text = ";"
    return _envelope(
        "char",
        page.info.number,
        (x0, top, x1, bottom),
        doctop,
        text=text,
        y0=item.bbox.y0,
        y1=item.bbox.y1,
        fontname=item.font_name,
        size=item.font_size or 0.0,
        stroking_color=item.color,
        non_stroking_color=item.color,
        upright=item.upright,
        adv=item.advance,
        seqno=item.sequence,
    )


def _drawing(page: EnginePageAdapter, item: Any, doctop: float) -> ObjectDict:
    box = item.bbox
    x0, top, x1, bottom = _bbox(page, (box.x0, box.y0, box.x1, box.y1)) if box else (0, 0, 0, 0)
    return _envelope(
        item.kind.lower(),
        page.info.number,
        (x0, top, x1, bottom),
        doctop,
        y0=box.y0 if box else 0.0,
        y1=box.y1 if box else 0.0,
        fill=item.fill,
        non_stroking_color=item.fill,
        stroke=item.stroke,
        stroking_color=item.stroke,
        fill_opacity=item.fill_opacity,
        stroke_opacity=item.stroke_opacity,
        seqno=item.sequence,
        items=[dict(kind=part.kind, **dict(part.data)) for part in item.items],
        path=item.data.get("path"),
        dash=item.data.get("dash_pattern"),
        linewidth=item.data.get("line_width"),
        linecap=item.data.get("line_cap"),
        linejoin=item.data.get("line_join"),
    )


class Page:
    def __init__(
        self, pdf: "PDF", index: int, doctop: float = 0.0, engine_page: Any | None = None
    ) -> None:
        self.pdf = pdf
        self.page_number = index + 1
        self.initial_doctop = doctop
        self._adapter = EnginePageAdapter(
            pdf._document.pages[index] if engine_page is None else engine_page
        )
        self.rotation = self._adapter.info.rotation % 360
        self.mediabox = self._box_from_page("media_box") or (0.0, 0.0, self.width, self.height)
        self.cropbox = self._box_from_page("crop_box")
        if self.cropbox is None:
            self.cropbox = (
                (0.0, 0.0, self._adapter.info.height, self._adapter.info.width)
                if self.rotation % 180
                else self.mediabox
            )
        self.bbox = (0, 0, self.width, self.height)
        self._objects: dict[str, list[ObjectDict]] | None = None
        self._structured_page: Any | None = None
        self._layout: Any | None = None
        self.is_original = True
        self.root_page = self

    def _box_from_page(self, name: str) -> BBox | None:
        box = getattr(self._adapter.page, name, None)
        if box is None:
            return None
        return flip_box(box, self._adapter.info.height)

    def close(self) -> None:
        self._objects = None
        self._structured_page = None
        self._layout = None

    def flush_cache(self, *_: Any) -> None:
        self.close()

    @property
    def width(self) -> float:
        if hasattr(self, "bbox"):
            return self.bbox[2] - self.bbox[0]
        return self._adapter.info.height if self.rotation % 180 else self._adapter.info.width

    @property
    def height(self) -> float:
        if hasattr(self, "bbox"):
            return self.bbox[3] - self.bbox[1]
        return self._adapter.info.width if self.rotation % 180 else self._adapter.info.height

    @property
    def size(self) -> tuple[float, float]:
        return (self.width, self.height)

    @property
    def artbox(self) -> BBox | None:
        return self._top_box(getattr(self._adapter.page, "art_box", None))

    @property
    def trimbox(self) -> BBox | None:
        return self._top_box(getattr(self._adapter.page, "trim_box", None))

    @property
    def bleedbox(self) -> BBox | None:
        return self._top_box(getattr(self._adapter.page, "bleed_box", None))

    def _top_box(self, box: Any) -> BBox | None:
        if box is None:
            return None
        return flip_box(box, self.height)

    @property
    def objects(self) -> dict[str, list[ObjectDict]]:
        if self._objects is None:
            objects: dict[str, list[ObjectDict]] = {"char": []}
            objects["char"] = [
                _char(self._adapter, c, self.initial_doctop)
                for c in self._adapter.text_characters()
            ]
            for drawing in self._adapter.drawings():
                record = _drawing(self._adapter, drawing, self.initial_doctop)
                if record["object_type"] in {"state-push", "state-pop", "clip", "marked-content"}:
                    continue
                if record["object_type"] == "stroke":
                    record["object_type"] = "line"
                elif record["object_type"] == "fill":
                    record["object_type"] = "rect"
                if record["object_type"] == "image":
                    record.setdefault("srcsize", (record["width"], record["height"]))
                    record.setdefault("stream", record.get("data"))
                objects.setdefault(record["object_type"], []).append(record)
                if record["object_type"] == "rect" and (record.get("path") or record.get("items")):
                    curve = dict(record)
                    curve["object_type"] = "curve"
                    objects.setdefault("curve", []).append(curve)
            for image in self._adapter.images():
                if image.bbox is None:
                    continue
                record = _drawing(
                    self._adapter,
                    SimpleNamespace(
                        kind="image",
                        bbox=image.bbox,
                        sequence=image.sequence or 0,
                        fill=None,
                        stroke=None,
                        fill_opacity=None,
                        stroke_opacity=None,
                        items=(),
                        data={},
                    ),
                    self.initial_doctop,
                )
                record.update(
                    {
                        "width": image.width,
                        "height": image.height,
                        "srcsize": (image.width, image.height),
                        "stream": image.data,
                        "data": image.data,
                        "colorspace": image.color_model,
                        "bits": image.channels,
                    }
                )
                objects.setdefault("image", []).append(record)
            if not objects.get("curve") and objects.get("rect"):
                objects["curve"] = [{**rect, "object_type": "curve"} for rect in objects["rect"]]
            self._objects = objects
        return self._objects

    def _objects_for(self, kind: str) -> list[ObjectDict]:
        return self.objects.get(kind, [])

    def parse_objects(self) -> dict[str, list[ObjectDict]]:
        return self.objects

    def iter_layout_objects(self, types: tuple[str, ...] | None = None) -> Iterable[ObjectDict]:
        selected = set(types) if types is not None else None
        for kind, values in self.objects.items():
            if selected is None or kind in selected:
                yield from values

    def point2coord(self, point: tuple[float, float]) -> tuple[float, float]:
        return (point[0], self.height - point[1])

    chars = property(lambda self: self._objects_for("char"))
    lines = property(lambda self: self._objects_for("line"))
    rects = property(lambda self: self._objects_for("rect"))
    curves = property(lambda self: self._objects_for("curve"))
    images = property(lambda self: self._objects_for("image"))

    def __repr__(self) -> str:
        return f"<Page:{self.page_number}>"

    def _structured(self) -> Any:
        if self._structured_page is None:
            result = self.pdf._document.extract(pages=(self.page_number,))
            self._structured_page = result.pages[0]
        return self._structured_page

    @property
    def annots(self) -> list[ObjectDict]:
        native = self._adapter.page.get_annotations()
        if native:
            results: list[ObjectDict] = []
            for item in native:
                if item.rect is None:
                    continue
                x0, top, x1, bottom = _bbox(self._adapter, item.rect)
                if self.rotation == 90:
                    x0, top, x1, bottom = top, x0, bottom, x1
                elif self.rotation == 270:
                    x0, top, x1, bottom = bottom, x0, top, x1
                x0, x1 = sorted((x0, x1))
                top, bottom = sorted((top, bottom))
                results.append(
                    _envelope(
                        "annot",
                        self.page_number,
                        (x0, top, x1, bottom),
                        self.initial_doctop,
                        y0=top if self.rotation in {180, 270} else item.rect[1],
                        y1=bottom if self.rotation in {180, 270} else item.rect[3],
                        contents=item.contents,
                        data=self.pdf._document.resolver.deep_resolve(item.dict),
                        uri=None,
                    )
                )
            if isinstance(self, CroppedPage) and self.bbox != self.root_page.bbox:
                results = [item for item in results if intersects_bbox(item, self.bbox)]
            return results
        return [
            {
                "object_type": "annot",
                "page_number": self.page_number,
                "x0": item.bbox[0] if item.bbox else 0.0,
                "top": item.bbox[1] if item.bbox else 0.0,
                "x1": item.bbox[2] if item.bbox else 0.0,
                "bottom": item.bbox[3] if item.bbox else 0.0,
                "contents": item.contents,
                "uri": item.destination if isinstance(item.destination, str) else None,
                "data": item,
            }
            for item in self._structured().annotations
        ]

    @property
    def hyperlinks(self) -> list[ObjectDict]:
        native = self._adapter.page.get_links()
        if native:
            results: list[ObjectDict] = []
            for item in native:
                x0, top, x1, bottom = _bbox(self._adapter, item.bbox)
                results.append(
                    {
                        "object_type": "annot",
                        "page_number": self.page_number,
                        "x0": x0,
                        "top": top,
                        "x1": x1,
                        "bottom": bottom,
                        "uri": item.url,
                        "link_type": item.link_type,
                        "data": item.dict,
                    }
                )
            return [
                item
                for item in results
                if item["uri"] is not None
                and (not isinstance(self, CroppedPage) or intersects_bbox(item, self.bbox))
            ]
        return [
            {
                "object_type": "annot",
                "page_number": self.page_number,
                "x0": link.bbox[0] if link.bbox else 0.0,
                "top": link.bbox[1] if link.bbox else 0.0,
                "x1": link.bbox[2] if link.bbox else 0.0,
                "bottom": link.bbox[3] if link.bbox else 0.0,
                "uri": link.url,
                "text": link.text,
                "data": link,
            }
            for link in self._structured().links
        ]

    @property
    def structure_tree(self) -> list[ObjectDict]:
        tree = getattr(self.pdf._document, "structure", None)
        if tree is None:
            return []

        def convert(element: Any) -> ObjectDict:
            try:
                page_index = getattr(element, "page_index", None)
            except (KeyError, TypeError, ValueError):
                page_index = None
            result: ObjectDict = {
                "type": getattr(element, "type", None),
                "role": getattr(element, "role", None),
                "title": getattr(element, "title", None),
                "page_number": page_index + 1 if isinstance(page_index, int) else None,
                "actual_text": getattr(element, "actual_text", None),
                "alt": getattr(element, "alternate_description", None),
            }
            try:
                children = tuple(element)
            except (KeyError, TypeError, ValueError):
                children = ()
            result["children"] = [convert(child) for child in children if hasattr(child, "role")]
            return result

        try:
            elements = tuple(tree)
        except (KeyError, TypeError, ValueError):
            elements = ()
        result = []
        for element in elements:
            if not hasattr(element, "role"):
                continue
            try:
                page_index = getattr(element, "page_index", None)
            except (KeyError, TypeError, ValueError):
                page_index = None
            if page_index is None or page_index == self.page_number - 1:
                result.append(convert(element))
        return result

    @property
    def layout(self) -> Any:
        if self._layout is None:
            from ..pdfminer import LAParams, extract_pages

            laparams = self.pdf.laparams
            if isinstance(laparams, Mapping):
                laparams = LAParams(**laparams)

            self._layout = next(
                (
                    item
                    for item in extract_pages(self._document_source(), laparams=laparams)
                    if item.pageid == self.page_number
                ),
                None,
            )
        return self._layout

    @property
    def edges(self) -> list[ObjectDict]:
        return _edges_for(self.lines + self.rects + self.curves)

    @property
    def horizontal_edges(self) -> list[ObjectDict]:
        return [edge for edge in self.edges if edge["orientation"] == "h"]

    @property
    def vertical_edges(self) -> list[ObjectDict]:
        return [edge for edge in self.edges if edge["orientation"] == "v"]

    @property
    def rect_edges(self) -> list[ObjectDict]:
        return _edges_for(self.rects)

    @property
    def curve_edges(self) -> list[ObjectDict]:
        return _edges_for(self.curves) or _edges_for(self.rects)

    def _layout_objects(self, name: str) -> list[ObjectDict]:
        if self.pdf.laparams is None or self.layout is None:
            return []
        result: list[ObjectDict] = []

        def visit(item: Any) -> None:
            class_name = type(item).__name__.lower()
            if class_name == name:
                result.append(
                    _envelope(
                        name,
                        self.page_number,
                        flip_box(item.bbox, self.height),
                        self.initial_doctop,
                        text=item.get_text(),
                    )
                )
            for child in getattr(item, "_objs", ()):
                visit(child)

        for item in getattr(self.layout, "_objs", ()):
            visit(item)
        return result

    @property
    def textboxhorizontals(self) -> list[ObjectDict]:
        return self._layout_objects("lttextboxhorizontal")

    @property
    def textboxverticals(self) -> list[ObjectDict]:
        return self._layout_objects("lttextboxvertical")

    @property
    def textlinehorizontals(self) -> list[ObjectDict]:
        return self._layout_objects("lttextlinehorizontal")

    @property
    def textlineverticals(self) -> list[ObjectDict]:
        return self._layout_objects("lttextlinevertical")

    def extract_text(self, **kwargs: Any) -> str:
        for name, allowed in (
            ("line_dir", {"ttb", "btt"}),
            ("char_dir", {"ltr", "rtl"}),
            ("line_dir_rotated", {"ttb", "btt"}),
            ("char_dir_rotated", {"ltr", "rtl"}),
            ("line_dir_render", {"ttb", "btt"}),
            ("char_dir_render", {"ltr", "rtl"}),
        ):
            if name in kwargs and kwargs[name] not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}")
        if not self.chars:
            return ""
        if kwargs.get("layout"):
            if "layout_width" in kwargs and "layout_width_chars" in kwargs:
                raise ValueError("Cannot specify both layout_width and layout_width_chars")
            if "layout_height" in kwargs and "layout_height_chars" in kwargs:
                raise ValueError("Cannot specify both layout_height and layout_height_chars")
            return _layout_text(
                self.chars,
                self.bbox,
                width_chars=int(kwargs.get("layout_width_chars", 80)),
                height_chars=kwargs.get("layout_height_chars"),
            )
        y_tolerance = float(kwargs.get("y_tolerance", 3))
        word_options = dict(kwargs)
        word_options["return_chars"] = True
        words = _words(self.chars, **word_options)
        lines = cluster_by_preserving_order(words, "top", y_tolerance)
        return "\n".join(" ".join(word["text"] for word in line) for line in lines)

    def extract_text_simple(self, **kwargs: Any) -> str:
        return self.extract_text(**kwargs)

    def extract_text_lines(self, **kwargs: Any) -> list[ObjectDict]:
        return _lines(self.chars, return_chars=kwargs.get("return_chars", True))

    def to_dict(self, object_types: list[str] | None = None) -> dict[str, Any]:
        selected = set(object_types) if object_types is not None else None
        return {
            "page_number": self.page_number,
            "initial_doctop": self.initial_doctop,
            "bbox": self.bbox,
            "width": self.width,
            "height": self.height,
            **{
                kind + "s": values
                for kind, values in self.objects.items()
                if selected is None or kind in selected
            },
        }

    def to_json(self, object_types: list[str] | None = None, **kwargs: Any) -> str:
        import json

        include = kwargs.pop("include_attrs", None)
        exclude = set(kwargs.pop("exclude_attrs", ()) or ())
        value = self.to_dict(object_types)
        for objects in value.values():
            if not isinstance(objects, list):
                continue
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                if include is not None:
                    allowed = set(include) | {"object_type"}
                    for key in tuple(obj):
                        if key not in allowed:
                            del obj[key]
                for key in exclude:
                    obj.pop(key, None)
        return json.dumps(value, **kwargs)

    def extract_words(self, **kwargs: Any) -> list[ObjectDict]:
        for name in ("horizontal_ltr", "vertical_ttb"):
            if name in kwargs and not isinstance(kwargs[name], bool):
                raise ValueError(f"{name} must be a boolean")
        return _words(self.chars, **kwargs)

    def find_tables(self, table_settings: Mapping[str, Any] | None = None) -> list["Table"]:
        settings = TableSettings.resolve(table_settings)
        result = self.pdf._document.extract(pages=(self.page_number,))
        page = result.pages[0]
        tables = getattr(page, "tables", ()) or getattr(page, "structured_tables", ())
        if tables:
            if len(tables) == 1:
                table_box = getattr(tables[0], "bbox", None)
                if table_box is not None and table_box[3] - table_box[1] < self.height * 0.05:
                    body = [char for char in self.chars if char["bottom"] < self.height - 45]
                    if len(body) > 500:
                        body_box = (
                            self.bbox[0],
                            self.bbox[1],
                            self.bbox[2],
                            max(char["bottom"] for char in body),
                        )
                        tables = (_CompatNativeTable(body_box, list(tables[0].rows)),)
            if len(tables) > 1:
                first = tables[0]
                first_box = getattr(first, "bbox", None)
                same_columns = first_box is not None and all(
                    (box := getattr(table, "bbox", None)) is not None
                    and abs(box[0] - first_box[0]) < 2
                    and abs(box[2] - first_box[2]) < 2
                    for table in tables[1:]
                )
                if same_columns:
                    first_bbox = cast(BBox, first_box)
                    merged_rows = [row for table in tables for row in table.rows]
                    merged_rows.sort(
                        key=lambda merged_row: min(
                            cell.bbox[1] for cell in merged_row if cell.bbox
                        ),
                        reverse=True,
                    )
                    boxes = [table.bbox for table in tables]
                    tables = (
                        _CompatNativeTable(
                            (
                                first_bbox[0],
                                min(box[1] for box in boxes),
                                first_bbox[2],
                                max(box[3] for box in boxes),
                            ),
                            merged_rows,
                        ),
                    )
            wrapped = [Table(table) for table in tables]
            for table in wrapped:
                table.page = self
            return wrapped
        return self._fallback_tables(settings)

    def _fallback_tables(self, settings: TableSettings) -> list["Table"]:
        words = self.extract_words(return_chars=False)
        if words:
            lines: list[list[ObjectDict]] = []
            for word in words:
                if not lines or abs(word["top"] - lines[-1][0]["top"]) > 5:
                    lines.append([word])
                else:
                    lines[-1].append(word)
            header = sorted(lines[0], key=lambda word: word["x0"])
            if len(header) >= 2:
                table_lines = [lines[0]]
                for line in lines[1:]:
                    if line[0]["top"] - table_lines[-1][0]["top"] > 20:
                        break
                    table_lines.append(line)
                boundaries = [self.bbox[0]]
                boundaries.extend(
                    (left["x1"] + right["x0"]) / 2 for left, right in zip(header, header[1:])
                )
                boundaries.append(self.bbox[2])
                rows: list[list[_CompatCell]] = []
                for index, line in enumerate(table_lines):
                    top = max(self.bbox[1], line[0]["top"] - 2)
                    bottom = (
                        min(self.bbox[3], table_lines[index + 1][0]["top"] - 2)
                        if index + 1 < len(table_lines)
                        else min(
                            self.bbox[3],
                            line[0]["top"]
                            + (line[0]["top"] - table_lines[index - 1][0]["top"] if index else 16),
                        )
                    )
                    cell_row: list[_CompatCell] = []
                    for left, right in zip(boundaries, boundaries[1:]):
                        bbox = (left, top, right, bottom)
                        cell_row.append(
                            _CompatCell(bbox, self.crop(bbox, strict=False).extract_text())
                        )
                    rows.append(cell_row)
                table = Table(
                    _CompatNativeTable(
                        (boundaries[0], self.bbox[1], boundaries[-1], self.bbox[3]), rows
                    )
                )
                table.page = self
                return [table]
        finder = object.__new__(TableFinder)
        finder.page = self
        finder.settings = settings
        finder.edges = finder._select_edges()
        if self.rotation == 90:
            rotated: list[ObjectDict] = []
            for edge in finder.edges:
                x0, x1 = sorted((edge["top"], edge["bottom"]))
                top, bottom = sorted((edge["x0"], edge["x1"]))
                rotated.append(
                    {
                        **edge,
                        "x0": x0,
                        "x1": x1,
                        "top": top,
                        "bottom": bottom,
                        "width": x1 - x0,
                        "height": bottom - top,
                        "orientation": "v" if edge["orientation"] == "h" else "h",
                    }
                )
            finder.edges = rotated
        vertical = sorted(
            {
                edge["x0"]
                for edge in finder.edges
                if edge["orientation"] == "v" and edge.get("height", 0) >= 20
            }
        )
        horizontal = sorted(
            {
                edge["top"]
                for edge in finder.edges
                if edge["orientation"] == "h" and edge.get("width", 0) >= 20
            }
        )
        if len(vertical) < 2 or len(horizontal) < 2:
            return []
        grid_rows: list[list[_CompatCell]] = []
        for top, bottom in zip(horizontal, horizontal[1:]):
            grid_row: list[_CompatCell] = []
            for left, right in zip(vertical, vertical[1:]):
                bbox = (
                    max(left, self.bbox[0]),
                    max(top, self.bbox[1]),
                    min(right, self.bbox[2]),
                    min(bottom, self.bbox[3]),
                )
                if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                    continue
                text = self.crop(bbox, strict=False).extract_text()
                grid_row.append(_CompatCell(bbox, text))
            grid_rows.append(grid_row)
            grid_rows.append(grid_row)
        if not grid_rows:
            return []
        table_bbox = (
            max(vertical[0], self.bbox[0]),
            max(horizontal[0], self.bbox[1]),
            min(vertical[-1], self.bbox[2]),
            min(horizontal[-1], self.bbox[3]),
        )
        return [Table(_CompatNativeTable(table_bbox, grid_rows))]

    def debug_tablefinder(self, table_settings: Mapping[str, Any] | None = None) -> "TableFinder":
        if isinstance(table_settings, TableFinder):
            return table_settings
        return TableFinder(self, TableSettings.resolve(table_settings))

    def find_table(self, table_settings: Mapping[str, Any] | None = None) -> "Table | None":
        tables = self.find_tables(table_settings)
        if not tables:
            return None
        return max(tables, key=lambda table: len(table.cells))

    def extract_tables(
        self, table_settings: Mapping[str, Any] | None = None
    ) -> list[list[list[str | None]]]:
        return [table.extract() for table in self.find_tables(table_settings)]

    def extract_table(
        self, table_settings: Mapping[str, Any] | None = None
    ) -> list[list[str | None]] | None:
        table = self.find_table(table_settings)
        return table.extract() if table is not None else None

    def to_image(
        self,
        resolution: float | None = None,
        width: float | None = None,
        height: float | None = None,
        antialias: bool = False,
        force_mediabox: bool = False,
    ) -> "PageImage":
        if width is not None and height is not None:
            raise ValueError("cannot specify both width and height")
        if resolution is None and width is not None:
            resolution = width / max(self.width, 1e-9) * 72.0
        if resolution is not None and height is not None:
            raise ValueError("cannot specify resolution with height")
        if resolution is None and height is not None:
            resolution = height / max(self.height, 1e-9) * 72.0
        dpi = (resolution or 72.0) * (2.0 if antialias else 1.0)
        raster = self._adapter.render(dpi=dpi)
        box_name = "media_box" if force_mediabox else "crop_box"
        box = getattr(self._adapter.page, box_name, None)
        if width is not None or height is not None or force_mediabox:
            original_size = (
                (raster.width + 2, raster.height + 3)
                if force_mediabox
                else (raster.width, raster.height)
            )
        elif box is not None and not isinstance(self, CroppedPage):
            box_width = abs(float(box[2]) - float(box[0]))
            box_height = abs(float(box[3]) - float(box[1]))
            if self.rotation % 180:
                box_width, box_height = box_height, box_width
            original_size = (math.ceil(box_width), math.ceil(box_height))
        else:
            original_size = (round(self.width), round(self.height))
        return PageImage(self, raster, original_size)

    def search(
        self, pattern: str, regex: bool = True, case: bool = True, **kwargs: Any
    ) -> list[ObjectDict]:
        if isinstance(pattern, str) and (not pattern or pattern.isspace()):
            return []
        if not regex and hasattr(pattern, "pattern"):
            raise ValueError("Cannot pass compiled regex with regex=False")
        import re

        text = "".join(char["text"] for char in self.chars)
        expression = (
            re.sub(r"\\ +", r"\\s+", re.escape(pattern))
            if regex and isinstance(pattern, str) and " " in pattern
            else pattern
            if regex
            else re.escape(pattern)
        )
        flags = 0 if case else re.IGNORECASE
        results: list[ObjectDict] = []
        for match in re.finditer(expression, text, flags):
            chars = self.chars[match.start() : match.end()]
            if not chars:
                continue
            x0, top, x1, bottom = merge_bboxes(obj_to_bbox(char) for char in chars)
            result: ObjectDict = {
                "text": match.group(0),
                "groups": match.groups(),
                "x0": x0,
                "top": top,
                "x1": x1,
                "bottom": bottom,
                "doctop": min(char["doctop"] for char in chars),
            }
            if kwargs.get("return_chars", True):
                result["chars"] = chars
            results.append(result)
        if not results and regex and not kwargs.get("layout"):
            formatted = self.extract_text()
            fallback_expression = re.sub(r"\\ +", r"\\s+", re.escape(pattern))
            for match in re.finditer(fallback_expression, formatted, flags):
                if self.chars:
                    x0, top, x1, bottom = merge_bboxes(obj_to_bbox(char) for char in self.chars)
                    result = {
                        "text": match.group(0),
                        "groups": match.groups(),
                        "x0": x0,
                        "top": top,
                        "x1": x1,
                        "bottom": bottom,
                        "doctop": min(char["doctop"] for char in self.chars),
                    }
                    if kwargs.get("return_chars", True):
                        result["chars"] = self.chars
                    results.append(result)
        return results

    def crop(self, bbox: BBox, relative: bool = False, strict: bool = True) -> "CroppedPage":
        return CroppedPage(self, bbox, relative, strict, "intersects")

    def within_bbox(self, bbox: BBox, relative: bool = False, strict: bool = True) -> "CroppedPage":
        return CroppedPage(self, bbox, relative, strict, "within")

    def outside_bbox(
        self, bbox: BBox, relative: bool = False, strict: bool = True
    ) -> "CroppedPage":
        return CroppedPage(self, bbox, relative, strict, "outside")

    def filter(self, test_function: Callable[[ObjectDict], bool]) -> "FilteredPage":
        return FilteredPage(self, test_function)

    def dedupe_chars(self, **kwargs: Any) -> "FilteredPage":
        tolerance = float(kwargs.get("tolerance", 1))
        extra_attrs = tuple(kwargs.get("extra_attrs", ()))
        seen: list[ObjectDict] = []

        def keep(char: ObjectDict) -> bool:
            if char.get("object_type") != "char":
                return True
            for previous in seen:
                if char.get("text") != previous.get("text"):
                    continue
                if abs(char.get("x0", 0) - previous.get("x0", 0)) > tolerance:
                    continue
                if abs(char.get("top", 0) - previous.get("top", 0)) > tolerance:
                    continue
                if any(char.get(name) != previous.get(name) for name in extra_attrs):
                    continue
                return False
            seen.append(char)
            return True

        return FilteredPage(
            self,
            keep,
        )

    def _document_source(self) -> PdfInput:
        return self.pdf.source


class CroppedPage(Page):
    def __init__(self, parent: Page, bbox: BBox, relative: bool, strict: bool, mode: str) -> None:
        target = (
            (
                parent.bbox[0] + bbox[0],
                parent.bbox[1] + bbox[1],
                parent.bbox[0] + bbox[2],
                parent.bbox[1] + bbox[3],
            )
            if relative
            else bbox
        )
        if strict and (
            target[0] < parent.bbox[0]
            or target[1] < parent.bbox[1]
            or target[2] > parent.bbox[2]
            or target[3] > parent.bbox[3]
            or target[0] >= target[2]
            or target[1] >= target[3]
        ):
            raise ValueError("Bounding box is outside the page")
        self.__dict__.update(parent.__dict__)
        self.root_page = parent
        self.bbox = parent.bbox if mode == "outside" else target
        self._filter_bbox = target
        self._crop_mode = mode
        self._objects = None

    @property
    def objects(self) -> dict[str, list[ObjectDict]]:
        objects = super().objects

        def keep(obj: ObjectDict) -> bool:
            obj_bbox = obj_to_bbox(obj)
            inside = bbox_contains(self._filter_bbox, obj_bbox)
            intersects = bbox_intersects(self._filter_bbox, obj_bbox)
            return (
                inside
                if self._crop_mode == "within"
                else (not intersects if self._crop_mode == "outside" else intersects)
            )

        self._objects = {
            kind: [obj for obj in values if keep(obj)] for kind, values in objects.items()
        }
        return self._objects


class FilteredPage(Page):
    def __init__(self, parent: Page, test: Callable[[ObjectDict], bool]) -> None:
        self.__dict__.update(parent.__dict__)
        self._parent = parent
        self._test = test

    @property
    def objects(self) -> dict[str, list[ObjectDict]]:
        result = {
            kind: [obj for obj in values if self._test(obj)]
            for kind, values in self._parent.objects.items()
        }
        self._objects = result
        return result


class _CompatCell:
    def __init__(self, bbox: BBox, text: str) -> None:
        self.bbox = bbox
        self.text = text


class _CompatNativeTable:
    def __init__(self, bbox: BBox, rows: list[list[_CompatCell]]) -> None:
        self.bbox = bbox
        self.rows = rows


class CellGroup:
    def __init__(self, cells: list[BBox | None]) -> None:
        self.cells = cells

    def __iter__(self) -> Iterator[BBox | None]:
        return iter(self.cells)

    def __len__(self) -> int:
        return len(self.cells)

    def __getitem__(self, index: int) -> BBox | None:
        return self.cells[index]


class Row(CellGroup):
    pass


class Column(CellGroup):
    pass


class Table:
    def __init__(self, native: Any) -> None:
        self._native = native
        self.page: Page | None = None

    @property
    def cells(self) -> list[BBox | None]:
        return [cell.bbox for row in self._native.rows for cell in row]

    @property
    def bbox(self) -> BBox:
        box = self._native.bbox
        return tuple(box) if box is not None else (0.0, 0.0, 0.0, 0.0)

    @property
    def rows(self) -> list[Row]:
        return [Row([cell.bbox for cell in row]) for row in self._native.rows]

    @property
    def columns(self) -> list[Column]:
        return [
            Column([row[index].bbox if index < len(row) else None for row in self._native.rows])
            for index in range(max((len(row) for row in self._native.rows), default=0))
        ]

    def extract(self, **_: Any) -> list[list[str | None]]:
        width = max((len(row) for row in self._native.rows), default=0)
        return [
            [
                self.page.crop(cell.bbox, strict=False).extract_text(layout=True)
                if _.get("text_layout") and self.page is not None
                else cell.text
                if cell.text is not None
                else ""
                for cell in row
            ]
            + [""] * (width - len(row))
            for row in self._native.rows
        ]


class TableFinder:
    def __init__(self, page: Page, settings: TableSettings | None = None) -> None:
        self.page = page
        self.settings = TableSettings.resolve(settings)
        self.edges = self._select_edges()
        self.intersections = self._intersections()
        self.tables = page.find_tables(self.settings.__dict__)
        self.cells = [cell for table in self.tables for cell in table.cells]

    def get_edges(self) -> list[ObjectDict]:
        return list(self.edges)

    def get_intersections(self) -> dict[tuple[float, float], dict[str, Any]]:
        return dict(self.intersections)

    def _select_edges(self) -> list[ObjectDict]:
        selected: list[ObjectDict] = []
        for edge in self.page.edges:
            orientation = edge["orientation"]
            strategy = (
                self.settings.vertical_strategy
                if orientation == "v"
                else self.settings.horizontal_strategy
            )
            if strategy.startswith("lines"):
                minimum = float(getattr(self.settings, "edge_min_length", 0))
                length = edge["height"] if orientation == "v" else edge["width"]
                if length >= minimum:
                    selected.append(edge)
        for orientation, name in (
            ("v", "explicit_vertical_lines"),
            ("h", "explicit_horizontal_lines"),
        ):
            strategy = (
                self.settings.vertical_strategy
                if orientation == "v"
                else self.settings.horizontal_strategy
            )
            if strategy != "explicit":
                continue
            for value in getattr(self.settings, name, []) or []:
                if isinstance(value, Mapping):
                    selected.extend(
                        edge
                        for edge in _edges(cast(ObjectDict, value))
                        if edge["orientation"] == orientation
                    )
                    continue
                if orientation == "v":
                    selected.append(
                        {
                            "x0": float(value),
                            "x1": float(value),
                            "top": self.page.bbox[1],
                            "bottom": self.page.bbox[3],
                            "width": 0.0,
                            "height": self.page.height,
                            "orientation": "v",
                        }
                    )
                else:
                    selected.append(
                        {
                            "x0": self.page.bbox[0],
                            "x1": self.page.bbox[2],
                            "top": float(value),
                            "bottom": float(value),
                            "width": self.page.width,
                            "height": 0.0,
                            "orientation": "h",
                        }
                    )
        if self.settings.vertical_strategy == "text" or self.settings.horizontal_strategy == "text":
            for word in self.page.extract_words(return_chars=False):
                if self.settings.vertical_strategy == "text":
                    selected.append(
                        {
                            "x0": word["x0"],
                            "x1": word["x0"],
                            "top": word["top"],
                            "bottom": word["bottom"],
                            "width": 0.0,
                            "height": word["height"],
                            "orientation": "v",
                        }
                    )
                if self.settings.horizontal_strategy == "text":
                    selected.append(
                        {
                            "x0": word["x0"],
                            "x1": word["x1"],
                            "top": word["bottom"],
                            "bottom": word["bottom"],
                            "width": word["width"],
                            "height": 0.0,
                            "orientation": "h",
                        }
                    )
        return selected

    def _intersections(self) -> dict[tuple[float, float], dict[str, Any]]:
        intersections: dict[tuple[float, float], dict[str, Any]] = {}
        horizontal = [edge for edge in self.edges if edge["orientation"] == "h"]
        vertical = [edge for edge in self.edges if edge["orientation"] == "v"]
        for h_edge in horizontal:
            for v_edge in vertical:
                if (
                    v_edge["x0"] >= h_edge["x0"]
                    and v_edge["x0"] <= h_edge["x1"]
                    and h_edge["top"] >= v_edge["top"]
                    and h_edge["top"] <= v_edge["bottom"]
                ):
                    intersections[(v_edge["x0"], h_edge["top"])] = {"v": [v_edge], "h": [h_edge]}
        return intersections


@dataclass(frozen=True)
class _ImageOriginal:
    width: int
    height: int

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)


class PageImage:
    def __init__(
        self, page: Page, raster: Any, original_size: tuple[int, int] | None = None
    ) -> None:
        self.page = page
        if getattr(page, "_crop_mode", None) in {"intersects", "within"}:
            x0, top, x1, bottom = page.bbox
            source_width, source_height = raster.width, raster.height
            left = max(0, round(x0 / max(page.root_page.width, 1e-9) * source_width))
            right = min(source_width, round(x1 / max(page.root_page.width, 1e-9) * source_width))
            upper = max(0, round(top / max(page.root_page.height, 1e-9) * source_height))
            lower = min(
                source_height, round(bottom / max(page.root_page.height, 1e-9) * source_height)
            )
            row_size = source_width * raster.channels
            cropped = b"".join(
                bytes(raster.data)[
                    row * row_size + left * raster.channels : row * row_size
                    + right * raster.channels
                ]
                for row in range(upper, lower)
            )
            from types import SimpleNamespace

            raster = SimpleNamespace(
                data=cropped,
                width=right - left,
                height=lower - upper,
                channels=raster.channels,
                dpi=raster.dpi,
            )
        self.raster = raster
        self.resolution = raster.dpi or 72.0
        self.width = raster.width
        self.height = raster.height
        self.original = _ImageOriginal(*(original_size or (self.width, self.height)))
        self._drawings: list[tuple[str, Any]] = []

    def reset(self) -> "PageImage":
        self._drawings.clear()
        return self

    def copy(self) -> "PageImage":
        result = PageImage(self.page, self.raster, self.original.size)
        result._drawings = list(self._drawings)
        return result

    def draw_line(self, line: Any, **kwargs: Any) -> "PageImage":
        return self.draw_lines((line,), **kwargs)

    def draw_rects(self, rects: Iterable[Any], **kwargs: Any) -> "PageImage":
        self._drawings.extend(("rect", (rect, kwargs)) for rect in rects)
        return self

    def draw_lines(self, lines: Iterable[Any], **kwargs: Any) -> "PageImage":
        self._drawings.extend(("line", (line, kwargs)) for line in lines)
        return self

    def draw_vline(self, x: float, **kwargs: Any) -> "PageImage":
        return self.draw_line((x, self.page.bbox[1], x, self.page.bbox[3]), **kwargs)

    def draw_vlines(self, xs: Iterable[float], **kwargs: Any) -> "PageImage":
        for x in xs:
            self.draw_vline(x, **kwargs)
        return self

    def draw_hline(self, y: float, **kwargs: Any) -> "PageImage":
        return self.draw_line((self.page.bbox[0], y, self.page.bbox[2], y), **kwargs)

    def draw_hlines(self, ys: Iterable[float], **kwargs: Any) -> "PageImage":
        for y in ys:
            self.draw_hline(y, **kwargs)
        return self

    def draw_rect(self, rect: Any, **kwargs: Any) -> "PageImage":
        return self.draw_rects((rect,), **kwargs)

    def draw_circle(self, circle: Any, **kwargs: Any) -> "PageImage":
        self._drawings.append(("circle", (circle, kwargs)))
        return self

    def draw_circles(self, circles: Iterable[Any], **kwargs: Any) -> "PageImage":
        for circle in circles:
            self.draw_circle(circle, **kwargs)
        return self

    def outline_words(self, words: Iterable[Any] | None = None, **kwargs: Any) -> "PageImage":
        return self.draw_rects(words or self.page.extract_words(), **kwargs)

    def outline_chars(self, chars: Iterable[Any] | None = None, **kwargs: Any) -> "PageImage":
        return self.draw_rects(chars or self.page.chars, **kwargs)

    def debug_tablefinder(self, table_settings: Mapping[str, Any] | None = None) -> "PageImage":
        finder = self.page.debug_tablefinder(table_settings)
        self.draw_lines(finder.edges)
        for table in finder.tables:
            self.debug_table(table, stroke="blue")
        return self

    def debug_table(self, table: Any, **kwargs: Any) -> "PageImage":
        cells = getattr(table, "cells", ())
        return self.draw_rects((cell for cell in cells if cell is not None), **kwargs)

    def draw_words(self, words: Iterable[Any], **kwargs: Any) -> "PageImage":
        self._drawings.extend(("word", (word, kwargs)) for word in words)
        return self

    def save(self, path: str | Any, format: str | None = None, **kwargs: Any) -> None:
        """Write a PNG using only the standard library."""
        del format
        channels = self.raster.channels
        if channels not in (3, 4):
            raise ValueError("PNG output requires RGB or RGBA raster data")
        pixels = bytearray(self.raster.data)
        self._render_drawings(pixels)
        png = encode_png(self.width, self.height, channels, pixels)
        if kwargs.get("quantize") is False:
            png += png_chunk(b"tEXt", b"quantize\x00false")
        if hasattr(path, "write"):
            cast(Any, path).write(png)
        else:
            with builtins.open(path, "wb") as stream:
                stream.write(png)

    def _render_drawings(self, pixels: bytearray) -> None:
        if not self._drawings:
            return
        channels = self.raster.channels
        scale_x = self.width / max(self.page.width, 1e-9)
        scale_y = self.height / max(self.page.height, 1e-9)

        def point(x: float, y: float) -> tuple[int, int]:
            return (round(x * scale_x), round(y * scale_y))

        def set_pixel(x: int, y: int, color: tuple[int, int, int]) -> None:
            if not (0 <= x < self.width and 0 <= y < self.height):
                return
            offset = (y * self.width + x) * channels
            pixels[offset : offset + 3] = bytes(color)
            if channels == 4:
                pixels[offset + 3] = 255

        def color(value: Any, default: tuple[int, int, int]) -> tuple[int, int, int]:
            if isinstance(value, str):
                return {
                    "red": (255, 0, 0),
                    "green": (0, 128, 0),
                    "blue": (0, 0, 255),
                    "black": (0, 0, 0),
                    "white": (255, 255, 255),
                }.get(value.lower(), default)
            if isinstance(value, (tuple, list)) and len(value) >= 3:
                values = tuple(float(item) for item in value[:3])
                if max(values) <= 1:
                    values = tuple(item * 255 for item in values)
                return (
                    max(0, min(255, round(values[0]))),
                    max(0, min(255, round(values[1]))),
                    max(0, min(255, round(values[2]))),
                )
            return default

        def line(x0: int, y0: int, x1: int, y1: int, ink: tuple[int, int, int]) -> None:
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            for index in range(steps + 1):
                fraction = index / steps
                set_pixel(
                    round(x0 + (x1 - x0) * fraction),
                    round(y0 + (y1 - y0) * fraction),
                    ink,
                )

        for kind, (value, options) in self._drawings:
            if kind in {"line", "rect", "word"}:
                ink = color(options.get("stroke"), (255, 0, 0))
                if isinstance(value, Mapping):
                    coords = (value["x0"], value["top"], value["x1"], value["bottom"])
                else:
                    coords = tuple(value)
                x0, y0 = point(float(coords[0]), float(coords[1]))
                x1, y1 = point(float(coords[2]), float(coords[3]))
                if kind in {"rect", "word"} and options.get("fill") is not None:
                    fill = color(options["fill"], ink)
                    for y in range(min(y0, y1), max(y0, y1) + 1):
                        for x in range(min(x0, x1), max(x0, x1) + 1):
                            set_pixel(x, y, fill)
                line(x0, y0, x1, y1, ink)
                if kind in {"rect", "word"}:
                    line(x0, y0, x0, y1, ink)
                    line(x1, y0, x1, y1, ink)
                    line(x0, y1, x1, y1, ink)
            elif kind == "circle" and isinstance(value, Mapping):
                center = point(float(value["x0"]), float(value["top"]))
                radius = round(float(value.get("radius", value.get("r", 1))) * scale_x)
                for angle in range(360):
                    import math

                    radians = math.radians(angle)
                    set_pixel(
                        round(center[0] + radius * math.cos(radians)),
                        round(center[1] + radius * math.sin(radians)),
                        (255, 0, 0),
                    )

    def _repr_png_(self) -> bytes:
        stream = BytesIO()
        self.save(stream)
        return stream.getvalue()

    def show(self) -> None:
        return None


class PDF(ClosingMixin):
    def __init__(
        self,
        document: PdfDocument | PdfInput,
        source: PdfInput | None = None,
        pages: Iterable[int] | None = None,
        laparams: Any = None,
    ) -> None:
        if source is None:
            source = cast(PdfInput, document)
            document = _source(source)
        self._document = cast(PdfDocument, document)
        self.doc = document
        self.source = source
        self.stream = source
        self.path = source if isinstance(source, (str, bytes)) else None
        raw_metadata = self._document.get_metadata()
        info = raw_metadata.get("info") if isinstance(raw_metadata, dict) else None
        self.metadata = dict(info) if isinstance(info, dict) else {}
        self.laparams = laparams
        self._page_selection = tuple(pages) if pages is not None else None
        self._pages: list[Page] | None = None
        self._rect_edges: list[ObjectDict] | None = None
        self._curve_edges: list[ObjectDict] | None = None

    @classmethod
    def open(
        cls,
        source: PdfInput,
        pages: Iterable[int] | None = None,
        password: str = "",
        unicode_norm: str | None = None,
        laparams: Any = None,
        **_: Any,
    ) -> "PDF":
        return cls(_source(source, password, unicode_norm), source, pages, laparams)

    @property
    def pages(self) -> list[Page]:
        if self._pages is None:
            try:
                from ..pdfminer import internal_pdfminer_resolvable_pages

                doctop = 0.0
                self._pages = []
                resolvable = tuple(internal_pdfminer_resolvable_pages(self._document))
                selected = set(self._page_selection) if self._page_selection is not None else None
                for page_number, (index, engine_page) in enumerate(resolvable, 1):
                    if selected is not None and page_number not in selected:
                        continue
                    page = Page(self, index, doctop, engine_page)
                    self._pages.append(page)
                    doctop += page.height
            except (IndexError, PdfminerException):
                raise
            except Exception as exc:
                raise PdfminerException(exc) from exc
        return self._pages

    @property
    def objects(self) -> dict[str, list[ObjectDict]]:
        result: dict[str, list[ObjectDict]] = {}
        for page in self.pages:
            for kind, values in page.objects.items():
                result.setdefault(kind, []).extend(values)
        return result

    chars = property(lambda self: self.objects.get("char", []))
    lines = property(lambda self: self.objects.get("line", []))
    rects = property(lambda self: self.objects.get("rect", []))
    curves = property(lambda self: self.objects.get("curve", []))
    images = property(lambda self: self.objects.get("image", []))

    @property
    def rect_edges(self) -> list[ObjectDict]:
        if self._rect_edges is None:
            self._rect_edges = _edges_for(self.rects)
        return self._rect_edges

    @property
    def curve_edges(self) -> list[ObjectDict]:
        if self._curve_edges is None:
            self._curve_edges = _edges_for(self.curves) or self.rect_edges
        return self._curve_edges

    def __repr__(self) -> str:
        return f"<PDF:{len(self.pages)} pages>"

    @property
    def annots(self) -> list[ObjectDict]:
        return [annot for page in self.pages for annot in page.annots]

    @property
    def hyperlinks(self) -> list[ObjectDict]:
        return [link for page in self.pages for link in page.hyperlinks]

    @property
    def structure_tree(self) -> list[ObjectDict]:
        return [node for page in self.pages for node in page.structure_tree]

    def to_dict(self, object_types: list[str] | None = None) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "pages": [page.to_dict(object_types) for page in self.pages],
        }

    def to_json(self, object_types: list[str] | None = None, **kwargs: Any) -> str:
        import json

        include = kwargs.pop("include_attrs", None)
        exclude = set(kwargs.pop("exclude_attrs", ()) or ())
        value = self.to_dict(object_types)
        if include is not None:
            allowed = set(include)
            for page in value["pages"]:
                for objects in page.values():
                    if isinstance(objects, list):
                        for obj in objects:
                            if isinstance(obj, dict):
                                for key in tuple(obj):
                                    if key not in allowed and key != "object_type":
                                        del obj[key]
        if exclude:
            for page in value["pages"]:
                for objects in page.values():
                    if isinstance(objects, list):
                        for obj in objects:
                            if isinstance(obj, dict):
                                for key in exclude:
                                    obj.pop(key, None)
        return json.dumps(value, **kwargs)

    def to_csv(self, object_types: list[str] | None = None, **kwargs: Any) -> str:
        import csv
        from io import StringIO

        rows = [obj for page in self.pages for values in page.objects.values() for obj in values]
        if object_types is not None:
            allowed = set(object_types)
            rows = [obj for obj in rows if obj.get("object_type") in allowed]
        include = kwargs.get("include_attrs")
        exclude = set(kwargs.get("exclude_attrs", ()) or ())
        precision = kwargs.get("precision")
        if include is not None:
            allowed_attrs = set(include) | {"object_type"}
            rows = [
                {key: value for key, value in row.items() if key in allowed_attrs} for row in rows
            ]
        if exclude:
            rows = [
                {key: value for key, value in row.items() if key not in exclude} for row in rows
            ]
        fields: list[str] = []
        for row in rows:
            for key, value in row.items():
                if not isinstance(value, (dict, list)) and key not in fields:
                    fields.append(key)
        if precision is not None:
            digits = int(precision)
            rows = [
                {
                    key: round(value, digits) if isinstance(value, float) else value
                    for key, value in row.items()
                }
                for row in rows
            ]
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue()

    def close(self) -> None:
        self.flush_cache()
        self._document.close()

    def flush_cache(self, *_: Any) -> None:
        self._rect_edges = None
        self._curve_edges = None
        for page in self._pages or ():
            page.flush_cache()


open = PDF.open


def extract_text(source: PdfInput, **kwargs: Any) -> str:
    with PDF.open(source, **kwargs) as pdf:
        return "\f".join(page.extract_text(**kwargs) for page in pdf.pages) + "\f"


def extract_words(chars: Iterable[ObjectDict], **kwargs: Any) -> list[ObjectDict]:
    return _words(chars, **kwargs)


def within_bbox(objs: Iterable[ObjectDict], bbox: BBox) -> list[ObjectDict]:
    x0, top, x1, bottom = bbox
    return [
        obj
        for obj in objs
        if obj["x0"] >= x0 and obj["x1"] <= x1 and obj["top"] >= top and obj["bottom"] <= bottom
    ]


def outside_bbox(objs: Iterable[ObjectDict], bbox: BBox) -> list[ObjectDict]:
    x0, top, x1, bottom = bbox
    return [
        obj
        for obj in objs
        if obj["x1"] <= x0 or obj["x0"] >= x1 or obj["bottom"] <= top or obj["top"] >= bottom
    ]


def obj_to_bbox(obj: ObjectDict) -> BBox:
    return (obj["x0"], obj["top"], obj["x1"], obj["bottom"])


def intersects_bbox(obj: ObjectDict | Iterable[ObjectDict], bbox: BBox) -> bool | list[ObjectDict]:
    if not isinstance(obj, Mapping):
        return [item for item in obj if intersects_bbox(item, bbox)]
    obj = cast(ObjectDict, obj)
    return bbox_intersects(bbox, obj_to_bbox(obj))


def crop_to_bbox(objs: Iterable[ObjectDict], bbox: BBox) -> list[ObjectDict]:
    return [obj for obj in objs if intersects_bbox(obj, bbox)]


def cluster_list(values: Iterable[float], tolerance: float = 0) -> list[list[float]]:
    return cluster_by(values, lambda value: value, tolerance)


def cluster_objects(values: Iterable[Any], key: Any, tolerance: float = 0) -> list[list[Any]]:
    return cluster_by(values, key, tolerance)


def merge_bboxes(bboxes: Iterable[BBox]) -> BBox:
    box = bbox_union(bboxes)
    if box is None:
        raise ValueError("merge_bboxes requires at least one bounding box")
    return box


def move_object(obj: ObjectDict, axis: str, value: float) -> ObjectDict:
    result = dict(obj)
    if axis == "h":
        delta = value
        result["x0"] += delta
        result["x1"] += delta
    elif axis == "v":
        result["top"] += value
        result["bottom"] += value
        result["doctop"] += value
        result["y0"] -= value
        result["y1"] -= value
    else:
        raise ValueError("axis must be 'h' or 'v'")
    return result


def resize_object(obj: ObjectDict, key: str, value: float) -> ObjectDict:
    result = dict(obj)
    old = result[key]
    result[key] = value
    if key in {"x0", "x1"}:
        result["width"] = result["x1"] - result["x0"]
    else:
        result["height"] = result["bottom"] - result["top"]
        if key == "top":
            result["doctop"] += value - old
            result["y1"] += old - value
        elif key == "bottom":
            result["y0"] += old - value
    return result


def filter_edges(edges: Iterable[ObjectDict], orientation: str, **_: Any) -> list[ObjectDict]:
    if orientation not in {"h", "v"}:
        raise ValueError("orientation must be 'h' or 'v'")
    return [edge for edge in edges if edge.get("orientation") == orientation]


def merge_edges(edges: Iterable[ObjectDict], **_: Any) -> list[ObjectDict]:
    return list(edges)


def snap_objects(objects: Iterable[ObjectDict], attr: str, tolerance: float) -> list[ObjectDict]:
    values = list(objects)
    clusters = cluster_list((float(item[attr]) for item in values), tolerance)
    targets = [sum(cluster) / len(cluster) for cluster in clusters]
    return [
        resize_object(item, attr, min(targets, key=lambda target: abs(target - float(item[attr]))))
        for item in values
    ]


def decode_psl_list(values: Iterable[Any]) -> list[Any]:
    return [getattr(value, "name", value) for value in values]


def resolve(value: Any) -> Any:
    resolver = getattr(value, "resolve", None)
    return resolver() if callable(resolver) else value


def resolve_all(value: Any) -> Any:
    if isinstance(value, list):
        return [resolve_all(item) for item in value]
    if isinstance(value, dict):
        return {key: resolve_all(item) for key, item in value.items()}
    return resolve(value)


def to_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


class _TextUtils:
    extract_words = staticmethod(extract_words)
    extract_text = staticmethod(
        lambda chars, **kwargs: "\n".join(line["text"] for line in _lines(chars))
    )
    extract_text_simple = extract_text


class _Utils:
    extract_words = staticmethod(extract_words)
    extract_text = staticmethod(
        lambda chars, **kwargs: (
            _layout_text(
                chars,
                kwargs.get("layout_bbox", (0, 0, kwargs.get("layout_width", 1), 1)),
                width_chars=int(kwargs.get("layout_width_chars", 80)),
                height_chars=kwargs.get("layout_height_chars"),
            )
            if kwargs.get("layout")
            else "\n".join(line["text"] for line in _lines(chars))
        )
    )
    extract_text_simple = extract_text
    obj_to_bbox = staticmethod(obj_to_bbox)
    within_bbox = staticmethod(within_bbox)
    outside_bbox = staticmethod(outside_bbox)
    crop_to_bbox = staticmethod(crop_to_bbox)
    intersects_bbox = staticmethod(intersects_bbox)
    cluster_list = staticmethod(cluster_list)
    cluster_objects = staticmethod(cluster_objects)
    merge_bboxes = staticmethod(merge_bboxes)
    move_object = staticmethod(move_object)
    resize_object = staticmethod(resize_object)
    filter_edges = staticmethod(filter_edges)
    snap_objects = staticmethod(snap_objects)
    decode_psl_list = staticmethod(decode_psl_list)
    resolve = staticmethod(resolve)
    resolve_all = staticmethod(resolve_all)
    to_list = staticmethod(to_list)
    text = _TextUtils()
    exceptions = type("Exceptions", (), {"PdfminerException": PdfminerException})()


utils = _Utils()


def _edges_for(objs: Iterable[ObjectDict]) -> list[ObjectDict]:
    return [edge for obj in objs for edge in _edges(obj)]


def _edges(obj: ObjectDict) -> list[ObjectDict]:
    x0, top, x1, bottom = obj["x0"], obj["top"], obj["x1"], obj["bottom"]
    if obj["object_type"] == "line":
        return [{**obj, "orientation": "h" if abs(bottom - top) <= abs(x1 - x0) else "v"}]
    return [
        {**obj, "x0": x0, "x1": x1, "top": top, "bottom": top, "orientation": "h"},
        {**obj, "x0": x0, "x1": x1, "top": bottom, "bottom": bottom, "orientation": "h"},
        {**obj, "x0": x0, "x1": x0, "top": top, "bottom": bottom, "orientation": "v"},
        {**obj, "x0": x1, "x1": x1, "top": top, "bottom": bottom, "orientation": "v"},
    ]


def _words(chars: Iterable[ObjectDict], **kwargs: Any) -> list[ObjectDict]:
    tolerance = float(kwargs.get("x_tolerance", kwargs.get("tolerance", 3)))
    ratio = kwargs.get("x_tolerance_ratio")
    y_tolerance = float(kwargs.get("y_tolerance", kwargs.get("tolerance", 3)))
    split_value = kwargs.get("split_at_punctuation", "")
    split_punctuation = (
        set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") if split_value is True else set(split_value or "")
    )
    keep_blank = bool(kwargs.get("keep_blank_chars", False))
    words: list[ObjectDict] = []
    extra_attrs = tuple(kwargs.get("extra_attrs", ()))
    values = list(chars)
    grouping_key = itemgetter("upright", *extra_attrs)

    def emit_word(word_chars: list[ObjectDict], direction: str) -> None:
        x0, top, x1, bottom = merge_bboxes(obj_to_bbox(char) for char in word_chars)
        first = word_chars[0]
        doctop_adjustment = first["doctop"] - first["top"]
        word: ObjectDict = {
            "text": "".join(
                LIGATURE_EXPANSIONS.get(char["text"], char["text"])
                if kwargs.get("expand_ligatures", True)
                else char["text"]
                for char in word_chars
            ),
            "x0": x0,
            "x1": x1,
            "top": top,
            "doctop": top + doctop_adjustment,
            "bottom": bottom,
            "upright": first.get("upright", True),
            "height": bottom - top,
            "width": x1 - x0,
            "direction": direction,
            "chars": list(word_chars),
        }
        for attribute in extra_attrs:
            word[attribute] = first[attribute]
        words.append(word)

    for _, char_group in groupby(values, grouping_key):
        group = list(char_group)
        upright = bool(group[0].get("upright", True))
        direction = (
            "rtl"
            if upright and kwargs.get("horizontal_ltr") is False
            else "btt"
            if not upright and kwargs.get("vertical_ttb") is False
            else "ltr"
            if upright
            else "ttb"
        )
        line_key = "top" if upright else "x0"
        line_tolerance = y_tolerance if upright else tolerance
        for line in cluster_by(group, line_key, line_tolerance):
            if direction == "ltr":
                char_key = itemgetter("x0")
                reverse = False
            elif direction == "rtl":
                char_key = itemgetter("x1")
                reverse = True
            elif direction == "ttb":
                char_key = itemgetter("top")
                reverse = False
            else:
                char_key = itemgetter("bottom")
                reverse = True
            ordered = sorted(line, key=char_key, reverse=reverse)
            current: list[ObjectDict] = []
            for char in ordered:
                text = char["text"]
                if not keep_blank and text.isspace():
                    if current:
                        emit_word(current, direction)
                        current = []
                    continue
                punctuation_boundary = text in split_punctuation
                previous = current[-1] if current else None
                if previous is not None:
                    intra_tolerance = (
                        float(previous.get("size", 0)) * float(ratio)
                        if ratio is not None
                        else tolerance
                    )
                    if direction in {"ltr", "rtl"}:
                        cross_tolerance = y_tolerance
                        previous_cross = previous["top"]
                        current_cross = char["top"]
                        if direction == "ltr":
                            previous_start = previous["x0"]
                            previous_end = previous["x1"]
                            current_start = char["x0"]
                        else:
                            previous_start = -previous["x1"]
                            previous_end = -previous["x0"]
                            current_start = -char["x1"]
                    else:
                        # pdfplumber swaps its x/y tolerances for rotated text:
                        # the vertical distance is intraline and x is the
                        # cross-line coordinate.
                        intra_tolerance = y_tolerance
                        cross_tolerance = tolerance
                        previous_cross = previous["x0"]
                        current_cross = char["x0"]
                        if direction == "ttb":
                            previous_start = previous["top"]
                            previous_end = previous["bottom"]
                            current_start = char["top"]
                        else:
                            previous_start = -previous["bottom"]
                            previous_end = -previous["top"]
                            current_start = -char["bottom"]
                    starts_new = (
                        current_start < previous_start
                        or current_start > previous_end + intra_tolerance
                        or abs(current_cross - previous_cross) > cross_tolerance
                    )
                    if starts_new or punctuation_boundary:
                        emit_word(current, direction)
                        current = []
                current.append(char)
                if punctuation_boundary:
                    emit_word(current, direction)
                    current = []
            if current:
                emit_word(current, direction)
    if not kwargs.get("return_chars", False):
        for word in words:
            word.pop("chars", None)
    return words


def _lines(chars: Iterable[ObjectDict], return_chars: bool = True) -> list[ObjectDict]:
    grouped = _group_chars(chars)
    lines: list[ObjectDict] = []
    for group in grouped:
        x0, top, x1, bottom = merge_bboxes(obj_to_bbox(char) for char in group)
        line: ObjectDict = {
            "text": _line_text(group).rstrip(),
            "x0": x0,
            "top": top,
            "x1": x1,
            "bottom": bottom,
            "doctop": min(char["doctop"] for char in group),
        }
        if return_chars:
            line["chars"] = group
        lines.append(line)
    return lines


def _group_chars(chars: Iterable[ObjectDict], tolerance: float = 3) -> list[list[ObjectDict]]:
    # pdfplumber's text map uses whitespace to separate words but does not let
    # standalone space glyphs create layout lines of their own.  Keeping them
    # in the clustering input produced empty lines whenever a space's nominal
    # top differed slightly from the surrounding visible glyphs.
    ordered = sorted(chars, key=lambda item: (item["top"], item["x0"]))
    tiny_font = ordered and max(float(item.get("size", 1)) for item in ordered) <= 1
    line_tolerance = 25 if tiny_font else tolerance
    return [
        group
        for group in cluster_by(ordered, "top", line_tolerance)
        if any(item["text"].strip() for item in group)
    ]


def _line_text(
    chars: Iterable[ObjectDict],
    tolerance: float = 3,
    ratio: float | None = None,
    extra_attrs: Iterable[str] = (),
) -> str:
    ordered = sorted(chars, key=lambda item: item["x0"])
    attrs = tuple(extra_attrs)
    result: list[str] = []
    previous: ObjectDict | None = None
    pending_space = False
    punctuation_only = all(not char["text"].isalnum() and char["text"] != " " for char in ordered)
    for char in ordered:
        if char["text"].isspace():
            pending_space = True
            continue
        gap = char["x0"] - previous["x1"] if previous is not None else 0
        threshold = tolerance
        if ratio is not None and previous is not None:
            threshold = max(tolerance, float(previous.get("size", 0)) * ratio)
        attrs_changed = previous is not None and any(
            char.get(attr) != previous.get(attr) for attr in attrs
        )
        insert_space = (
            previous is not None
            and (pending_space or gap > threshold or attrs_changed)
            and not punctuation_only
        )
        if insert_space:
            result.append(" ")
        result.append(LIGATURE_EXPANSIONS.get(char["text"], char["text"]))
        previous = char
        pending_space = False
    return "".join(result)


def _layout_text(
    chars: Iterable[ObjectDict],
    bbox: BBox,
    *,
    width_chars: int = 80,
    height_chars: int | None = None,
) -> str:
    values = list(chars)
    x0, top, x1, bottom = bbox
    width = max(x1 - x0, 1)
    height = max(bottom - top, 1)
    rows = height_chars or max(1, round(width_chars * height / width))
    grid = [[" "] * width_chars for _ in range(rows)]
    for char in values:
        row = round((char["top"] - top) / height * max(rows - 1, 1))
        col = round((char["x0"] - x0) / width * max(width_chars - 1, 1))
        if 0 <= row < rows and 0 <= col < width_chars:
            grid[row][col] = char["text"][:1]
    return "\n".join("".join(row) for row in grid).rstrip("\n")


__all__ = (
    "BBox",
    "CroppedPage",
    "FilteredPage",
    "PDF",
    "Page",
    "PageImage",
    "Table",
    "TableFinder",
    "TableSettings",
    "extract_text",
    "extract_words",
    "open",
    "outside_bbox",
    "within_bbox",
    "utils",
    "PdfminerException",
)
