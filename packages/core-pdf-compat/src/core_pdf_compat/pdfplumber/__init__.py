from __future__ import annotations

import builtins
import math
import re
import struct
import warnings
import zlib
from bisect import bisect_right
from collections.abc import Callable, Iterable, Iterator, Mapping
from copy import copy
from io import BytesIO
from itertools import accumulate, groupby, pairwise
from operator import itemgetter
from types import SimpleNamespace
from typing import Any, ClassVar, TypeAlias, cast

from core_pdf import PdfDocument
from core_pdf.impl.capture_program import CapturedProgram, CaptureOptions
from core_pdf.impl.exceptions import PdfError
from core_pdf.impl.fonts_helpers import LIGATURE_TEXT_OVERRIDES
from core_pdf.impl.geometry import (
    bbox_contains,
    bbox_union,
    flip_rect_vertical,
)
from core_pdf.impl.graphics_stream_decoding import PNG_SIGNATURE, png_chunk
from core_pdf.impl.pdf_values import recover_pdf_name
from core_pdf.impl.raw_media import ImageRecord
from core_pdf.impl.types import (
    FrozenFields,
    PdfName,
    PdfReference,
    PdfString,
    ReplaceFields,
    ReprFields,
    frozen_setattr,
)
from core_pdf_spec.s_07_content.model import PATH_CLOSE, PATH_CURVE, PATH_OPERAND_COUNTS, PATH_RECT

from .._shared import BBox, ClosingMixin, PdfInput
from ..pdfminer.pdftypes import DocumentView
from .convert import Container
from .exceptions import PdfminerException
from .repair import RepairSetting, repair, repair_stream

ObjectDict: TypeAlias = dict[str, Any]
LIGATURE_EXPANSIONS = {**LIGATURE_TEXT_OVERRIDES, "ﬅ": "ft", "ﬆ": "st"}
flip_box = flip_rect_vertical


def cluster_by(
    values: Iterable[Any], key: Callable[[Any], Any] | str, tolerance: float = 0
) -> list[list[Any]]:
    getter = itemgetter(key) if isinstance(key, str) else key
    items = list(values)
    ordered_values = sorted({getter(item) for item in items})

    def separated(previous: Any, value: Any) -> int:
        if isinstance(value, (int, float)) and isinstance(previous, (int, float)):
            return 1 if value - previous > tolerance else 0
        return 1 if value != previous else 0

    cluster_ids: dict[Any, int] = {}
    if ordered_values:
        steps = (separated(previous, value) for previous, value in pairwise(ordered_values))
        cluster_ids = dict(zip(ordered_values, accumulate(steps, initial=0), strict=True))
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


def _source(value: PdfInput, password: str = "") -> PdfDocument:
    try:
        return PdfDocument.open(
            value,
            password=password,
            recovery_scan_all_revisions=False,
        )
    except Exception as exc:
        raise PdfminerException(exc) from exc


DRAWINGS_CAPTURE = CaptureOptions(
    ink_bounds=False, text_runs=False, render_details=False, glyphs=False
)


class LatestCapture:
    # A document keeps the pdfminer capture of only its most recently walked page, so
    # chars and layout of one page share a walk without every page retaining its own.
    __slots__ = ("adapter",)

    def __init__(self) -> None:
        self.adapter: EnginePageAdapter | None = None

    def hold(self, adapter: EnginePageAdapter) -> None:
        previous, self.adapter = self.adapter, adapter
        if previous is not None and previous is not adapter:
            previous._pdfminer_program = None


class EnginePageAdapter:
    def __init__(
        self,
        page: Any,
        unicode_norm: str | None = None,
        latest_capture: LatestCapture | None = None,
    ) -> None:
        self.page = page
        self.unicode_norm = unicode_norm
        self.latest_capture = latest_capture
        self._program: Any | None = None
        self._pdfminer_program: CapturedProgram | None = None
        self._painted_paths: tuple[Any, ...] = ()
        self._text_marks: tuple[Any, ...] = ()
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
                if isinstance(resolved, dict) and recover_pdf_name(resolved.get("Type")) == "Page":
                    return value
                return resolve_all(resolved, active | {key})
            if isinstance(value, (list, tuple)):
                return type(value)(resolve_all(item, active) for item in value)
            if isinstance(value, dict):
                is_annotation = recover_pdf_name(value.get("Type")) == "Annot"
                return {
                    key: item
                    if is_annotation and recover_pdf_name(key) == "Parent"
                    else resolve_all(item, active)
                    for key, item in value.items()
                }
            return value

        try:
            raw_box = page.page_dict.get("MediaBox")
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
        from ..pdfminer._capture import pdfminer_validate_page_resources
        from ..pdfminer._fonts import (
            pdfminer_descent,
            pdfminer_embedded_cmap_is_unusable,
            pdfminer_font_name,
            pdfminer_glyph_text,
            pdfminer_ligature_overrides,
            pdfminer_normalized_width,
        )

        pdfminer_validate_page_resources(self.page)
        projected_glyphs: tuple[Any, ...] = self.pdfminer_program().glyphs
        ligatures, skipped_ligature_parts = pdfminer_ligature_overrides(projected_glyphs)
        provenances: dict[int, tuple[object, bool, dict[str, Any]]] = {}
        # A page repeats a few hundred glyphs thousands of times; these depend only on
        # the font decoder and the glyph fields in the key.
        glyph_texts: dict[tuple[Any, ...], str] = {}
        widths: dict[tuple[Any, ...], float] = {}
        descents: dict[int, tuple[Any, float]] = {}

        def glyph_key(glyph: Any) -> tuple[Any, ...]:
            return (
                glyph.font_decoder,
                glyph.char_code,
                glyph.cid,
                glyph.code_bytes,
                glyph.unicode_source,
                glyph.text,
                glyph.alternates,
            )

        marks = self._text_marks
        mark_starts = [mark.glyph_index for mark in marks]
        for glyph_index, glyph in enumerate(projected_glyphs):
            if pdfminer_embedded_cmap_is_unusable(glyph):
                continue
            if id(glyph) in skipped_ligature_parts:
                continue
            ligature = ligatures.get(id(glyph))
            key = glyph_key(glyph)
            if ligature is not None:
                text = ligature[0]
            else:
                text = glyph_texts.get(key)  # type: ignore[assignment]
                if text is None:
                    text = glyph_texts[key] = pdfminer_glyph_text(glyph)
            if not text:
                continue
            source_provenance = glyph.provenance
            known = provenances.get(id(source_provenance))
            if known is None or known[0] is not source_provenance:
                known = provenances[id(source_provenance)] = (
                    source_provenance,
                    ("source", "annotation_appearance") in source_provenance,
                    dict(source_provenance) if source_provenance else {},
                )
            if known[1]:
                continue
            x0, y0, x1, y1 = ligature[1] if ligature is not None else glyph.advance_bbox
            font_height = glyph.effective_font_height or glyph.font_size
            provenance = known[2]
            matrix = provenance.get("text_matrix")
            text_matrix: tuple[float, ...] | None = None
            upright = glyph.rotation_angle % 180 == 0
            advance = x1 - x0
            if isinstance(matrix, (tuple, list)) and len(matrix) == 4:
                a, b, c, d = (float(value) for value in matrix)
                scaling = float(provenance.get("horizontal_scale", 100.0)) * 0.01
                baseline = ligature[2] if ligature is not None else glyph.baseline
                if baseline is not None:
                    if glyph.font_decoder.is_vertical:
                        metric = glyph.font_decoder.vertical_glyph_metric(glyph.cid)
                        left = -float(metric[1]) * glyph.font_size * 0.001
                        advance = float(metric[0]) * glyph.font_size * 0.001 * scaling
                        horizontal = (left, left + glyph.font_size)
                        vertical = (glyph.font_size + advance, glyph.font_size)
                    else:
                        decoder = glyph.font_decoder
                        known_descent = descents.get(id(decoder))
                        if known_descent is None or known_descent[0] is not decoder:
                            known_descent = descents[id(decoder)] = (
                                decoder,
                                pdfminer_descent(glyph),
                            )
                        descent = known_descent[1] * glyph.font_size
                        projected = None if ligature else text
                        width_key = (*key, projected)
                        width = widths.get(width_key)
                        if width is None:
                            width = widths[width_key] = pdfminer_normalized_width(glyph, projected)
                        advance = width * glyph.font_size * scaling
                        horizontal = (0.0, advance)
                        vertical = (descent, descent + glyph.font_size)
                    near, far = horizontal
                    low, high = vertical
                    origin_x, origin_y = baseline[0], baseline[1]
                    text_matrix = (a, b, c, d, origin_x, origin_y)
                    # The corners in the order (near, low), (near, high), (far, low), (far, high).
                    ax = near * a + low * c + origin_x
                    ay = near * b + low * d + origin_y
                    bx = near * a + high * c + origin_x
                    by = near * b + high * d + origin_y
                    cx = far * a + low * c + origin_x
                    cy = far * b + low * d + origin_y
                    dx = far * a + high * c + origin_x
                    dy = far * b + high * d + origin_y
                    x0 = min(ax, bx, cx, dx)
                    y0 = min(ay, by, cy, dy)
                    x1 = max(ax, bx, cx, dx)
                    y1 = max(ay, by, cy, dy)
                    font_height = (
                        x1 - x0
                        if glyph.font_decoder.is_vertical or glyph.rotation_angle % 180
                        else y1 - y0
                    )
                if self.info.rotation % 360 == 90:
                    a, b, c, d = b, -a, d, -c
                elif self.info.rotation % 360 == 180:
                    a, b, c, d = -a, -b, -c, -d
                elif self.info.rotation % 360 == 270:
                    a, b, c, d = -b, a, -d, c
                upright = a * d * scaling > 0.0 and b * c <= 0.0
            if glyph.rotation_angle % 360 == 90:
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
                font_name=pdfminer_font_name(glyph),
                font_size=font_height,
                color=glyph.fill,
                rotation_angle=glyph.rotation_angle,
                upright=upright,
                advance=advance,
                sequence=glyph.seqno,
                matrix=text_matrix,
                mark=marks[bisect_right(mark_starts, glyph_index) - 1] if marks else None,
            )

    def page_program(self) -> Any:
        # Only drawings and images are read from this program (characters come from
        # the pdfminer capture), so it captures no glyphs or text runs.
        if self._program is None:
            self._program = self.page.get_page_program(options=DRAWINGS_CAPTURE)
        return self._program

    def pdfminer_program(self) -> CapturedProgram:
        # Characters and the pdfminer layout both read pdfminer's own text walk of the page.
        program = self._pdfminer_program
        if program is None:
            from ..pdfminer._capture import pdfminer_page_capture

            program, self._painted_paths, self._text_marks = pdfminer_page_capture(self.page)
            self._pdfminer_program = program
            if self.latest_capture is not None:
                self.latest_capture.hold(self)
        return program

    def painted_paths(self) -> tuple[Any, ...]:
        self.pdfminer_program()
        return self._painted_paths

    def flush_program(self) -> None:
        self._program = None
        self._pdfminer_program = None

    def images(self, program: Any | None = None) -> tuple[ImageRecord, ...]:
        if program is None:
            program = self.page_program()
        return self.page.extract_program_images(program)

    def render(self, *, dpi: float, antialias: bool = False) -> Any:
        rendered = self.page.render()
        rendered.width = rendered.display_list.width
        rendered.height = rendered.display_list.height
        rendered.user_unit = 1.0
        raster = rendered.rasterize(scale=max(0.01, dpi / 72.0) * (2 if antialias else 1))
        if antialias:
            import numpy

            samples = raster.array()
            height, width = (raster.height + 1) // 2, (raster.width + 1) // 2
            samples = numpy.pad(
                samples, ((0, raster.height % 2), (0, raster.width % 2), (0, 0)), mode="edge"
            )
            reduced = samples.reshape(height, 2, width, 2, raster.channels).mean(axis=(1, 3))
            return SimpleNamespace(
                data=numpy.rint(reduced).astype(numpy.uint8).tobytes(),
                width=width,
                height=height,
                channels=raster.channels,
                dpi=dpi,
            )
        return SimpleNamespace(
            data=raster.pixels,
            width=raster.width,
            height=raster.height,
            channels=raster.channels,
            dpi=dpi,
        )


def _envelope(
    object_type: str, page_number: int, box: BBox, doctop: float, **extra: Any
) -> ObjectDict:
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


def _annotation_text(value: Any, name: str, raise_errors: bool) -> Any:
    if isinstance(value, PdfString):
        value = value.data
    if not isinstance(value, (bytes, bytearray)):
        return value
    try:
        return bytes(value).decode("utf-8")
    except UnicodeDecodeError:
        try:
            return bytes(value).decode("utf-16")
        except UnicodeDecodeError:
            if raise_errors:
                raise
            warnings.warn(
                f"Could not decode {name} of annotation. {name} will be missing.", stacklevel=2
            )
            return value


def _raw_color(color: tuple[float, ...] | None) -> tuple[float, ...] | None:
    # pdfminer keeps operands as written, and a bare gray operand is an integer.
    if color is not None and len(color) == 1 and color[0] == int(color[0]):
        return (int(color[0]),)
    return color


def _char(page: EnginePageAdapter, item: Any, doctop: float) -> ObjectDict:
    x0, top, x1, bottom = flip_box(
        (item.bbox.x0, item.bbox.y0, item.bbox.x1, item.bbox.y1), page.info.height
    )
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
    if page.unicode_norm == "NFC" and text == "\u037e":
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
        stroking_color=item.color if item.mark is None else _raw_color(item.mark.stroke_color),
        non_stroking_color=item.color if item.mark is None else _raw_color(item.mark.fill_color),
        upright=item.upright,
        adv=item.advance,
        seqno=item.sequence,
        matrix=item.matrix,
        ncs=None if item.mark is None else item.mark.color_space,
        mcid=None if item.mark is None else item.mark.mcid,
        tag=None if item.mark is None else item.mark.tag,
    )


def _page_point(page: EnginePageAdapter, point: tuple[float, float]) -> tuple[float, float]:
    info = page.info
    x, top = point[0], info.height - point[1]
    rotation = info.rotation % 360
    if rotation == 90:
        x, top = info.height - top, x
    elif rotation == 180:
        x, top = info.width - x, info.height - top
    elif rotation == 270:
        x, top = top, info.width - x
    return x + info.media_x0, top + info.media_top


def _path_segments(path: Any, media_left: float, media_bottom: float) -> list[tuple[Any, ...]]:
    matrix = path.ctm

    def device(x: float, y: float) -> tuple[float, float]:
        return (
            matrix.a * x + matrix.c * y + matrix.e - media_left,
            matrix.b * x + matrix.d * y + matrix.f - media_bottom,
        )

    coords = path.coords
    position = 0
    segments: list[tuple[Any, ...]] = []
    for op in path.ops:
        if op == PATH_CLOSE:
            segments.append(("h",))
        elif op == PATH_RECT:
            x, y, width, height = coords[position : position + 4]
            segments += [
                ("m", device(x, y)),
                ("l", device(x + width, y)),
                ("l", device(x + width, y + height)),
                ("l", device(x, y + height)),
                ("h",),
            ]
        elif op == PATH_CURVE:
            _, _, x1, y1, x2, y2, x3, y3 = coords[position : position + 8]
            segments.append(("c", device(x1, y1), device(x2, y2), device(x3, y3)))
        else:
            segments.append((chr(op), device(coords[position], coords[position + 1])))
        position += PATH_OPERAND_COUNTS[op]
    return segments


def _split_subpaths(segments: list[tuple[Any, ...]]) -> list[list[tuple[Any, ...]]]:
    shape = "".join(segment[0] for segment in segments)
    if not shape.startswith("m"):
        return []
    if shape.count("m") == 1:
        return [segments]
    return [segments[match.start() : match.end()] for match in re.finditer(r"m[^m]+", shape)]


def _classify_subpath(segments: list[tuple[Any, ...]]) -> tuple[str, list[tuple[float, float]]]:
    # pdfminer.six's PDFLayoutAnalyzer.paint_path, Copyright (c) 2004-2016 Yusuke Shinyama,
    # under the MIT License.
    shape = "".join(segment[0] for segment in segments)
    points = [segments[0][-1] if segment[0] == "h" else segment[-1] for segment in segments]
    if len(shape) > 3 and shape.endswith("lh") and points[-2] == points[0]:
        shape = f"{shape[:-2]}h"
        points.pop()
    if shape in {"mlh", "ml"}:
        return "line", points[:2]
    if shape in {"mlllh", "mllll"}:
        (x0, y0), (x1, y1), (x2, y2), (x3, y3), last = points
        axis_aligned = (x0 == x1 and y1 == y2 and x2 == x3 and y3 == y0) or (
            y0 == y1 and x1 == x2 and y2 == y3 and x3 == x0
        )
        if points[0] == last and axis_aligned:
            return "rect", [(x0, y0), (x2, y0), (x2, y2), (x0, y2)]
    return "curve", points


def _path_objects(page: EnginePageAdapter, doctop: float) -> list[ObjectDict]:
    objects: list[ObjectDict] = []
    for path in page.painted_paths():
        segments_by_subpath = _split_subpaths(
            _path_segments(path, page.info.media_left, page.info.media_bottom)
        )
        for segments in segments_by_subpath:
            object_type, points = _classify_subpath(segments)
            ys = [point[1] for point in points]
            placed = [_page_point(page, point) for point in points]
            objects.append(
                _envelope(
                    object_type,
                    page.info.number,
                    (
                        min(point[0] for point in placed),
                        min(point[1] for point in placed),
                        max(point[0] for point in placed),
                        max(point[1] for point in placed),
                    ),
                    doctop,
                    y0=min(ys),
                    y1=max(ys),
                    pts=placed,
                    path=[
                        (segment[0], *(_page_point(page, point) for point in segment[1:]))
                        for segment in segments
                    ],
                    fill=path.kind != "stroke",
                    stroke=path.kind != "fill",
                    evenodd=path.fill_rule == "evenodd",
                    linewidth=path.line_width,
                    stroking_color=path.stroke_color,
                    non_stroking_color=path.fill_color,
                    dash=path.dash,
                    mcid=path.mcid,
                    tag=path.tag,
                    seqno=path.sequence,
                )
            )
    return objects


def _image(page: EnginePageAdapter, image: ImageRecord, doctop: float) -> ObjectDict | None:
    metadata = image.image_metadata
    box = image.rect or image.image_clip
    if metadata is None or box is None:
        return None
    data = image.data
    payload = bytes(data) if isinstance(data, (bytes, bytearray, memoryview)) else None
    x0, top, x1, bottom = flip_box(box, page.info.height)
    dictionary = image.dictionary or {}
    return _envelope(
        "image",
        page.info.number,
        (x0, top, x1, bottom),
        doctop,
        y0=box[1],
        y1=box[3],
        seqno=image.seqno or 0,
        name=None,
        srcsize=(metadata.width, metadata.height),
        stream=payload,
        imagemask=dictionary.get(PdfName(b"ImageMask")),
        colorspace=metadata.color_model,
        bits=dictionary.get(PdfName(b"BitsPerComponent")),
        mcid=None,
        tag=None,
    )


class Page(Container):
    def __init__(
        self, pdf: PDF, index: int, doctop: float = 0.0, engine_page: Any | None = None
    ) -> None:
        self.pdf = pdf
        self.page_number = index + 1
        self.initial_doctop = doctop
        self._adapter = EnginePageAdapter(
            pdf._document.pages[index] if engine_page is None else engine_page,
            pdf._unicode_norm,
            pdf._latest_capture,
        )
        self.rotation = self._adapter.info.rotation % 360
        self.mediabox = self._box_from_page("media_box") or (0.0, 0.0, self.width, self.height)
        self.cropbox = self._box_from_page("crop_box")
        if self.cropbox is None:
            self.cropbox = self.mediabox
        self.bbox: BBox = self.mediabox
        self._objects: dict[str, list[ObjectDict]] | None = None
        self._edges: list[ObjectDict] | None = None
        self._structured_page: Any | None = None
        self._layout_page: Any | None = None
        self.is_original = True
        self.root_page = self

    def _box_from_page(self, name: str) -> BBox | None:
        box = getattr(self._adapter.page, name, None)
        if box is None:
            return None

        def normalized(raw: Any) -> BBox:
            x0, x1 = sorted((raw[0], raw[2]))
            y0, y1 = sorted((raw[1], raw[3]))
            return (y0, x0, y1, x1) if self.rotation % 180 else (x0, y0, x1, y1)

        media = normalized(self._adapter.page.media_box)
        height = media[3] - media[1]
        x0, y0, x1, y1 = normalized(box)
        return (x0, height - y1, x1, height - y0)

    def close(self) -> None:
        self._objects = None
        self._edges = None
        self._structured_page = None
        self._layout_page = None
        self._adapter.flush_program()
        self.pdf.flush_merged_objects()

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
            try:
                self._objects = self._parse_objects()
            except PdfError as error:
                raise PdfminerException(error) from error
        return self._objects

    def _parse_objects(self) -> dict[str, list[ObjectDict]]:
        objects: dict[str, list[ObjectDict]] = {
            "char": [
                _char(self._adapter, c, self.initial_doctop)
                for c in self._adapter.text_characters()
            ]
        }
        program = self._adapter.page_program()
        for path_object in _path_objects(self._adapter, self.initial_doctop):
            objects.setdefault(path_object["object_type"], []).append(path_object)
        for image in self._adapter.images(program):
            image_record = _image(self._adapter, image, self.initial_doctop)
            if image_record is not None:
                objects.setdefault("image", []).append(image_record)
        if self.pdf.laparams is not None and objects.get("char"):
            objects["char"] = self._chars_in_layout_order(objects["char"])
        # pdfplumber keeps each type where it first appears in the content stream.
        ordered = dict(
            sorted(
                ((kind, values) for kind, values in objects.items() if values),
                key=lambda item: min(obj["seqno"] for obj in item[1]),
            )
        )
        for values in ordered.values():
            for value in values:
                del value["seqno"]
        return ordered

    def _chars_in_layout_order(self, chars: list[ObjectDict]) -> list[ObjectDict]:
        # With laparams, pdfminer's analysis moves characters into text boxes.
        def key(text: str, x0: float, y0: float) -> tuple[str, float, float]:
            return text, round(x0, 2), round(y0, 2)

        waiting: dict[tuple[str, float, float], list[ObjectDict]] = {}
        for char in reversed(chars):
            waiting.setdefault(key(char["text"], char["x0"], char["y0"]), []).append(char)
        ordered: list[ObjectDict] = []

        def visit(item: Any) -> None:
            if hasattr(item, "fontname") and hasattr(item, "bbox"):
                matches = waiting.get(key(item.get_text(), item.bbox[0], item.bbox[1]))
                if matches:
                    ordered.append(matches.pop())
            for child in getattr(item, "_objs", ()):
                visit(child)

        visit(self.layout)
        seen = {id(char) for char in ordered}
        return ordered + [char for char in chars if id(char) not in seen]

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
            try:
                result = self.pdf._document.extract(pages=(self.page_number,))
            except PdfError as error:
                raise PdfminerException(error) from error
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
                x0, top, x1, bottom = flip_box(item.rect, self._adapter.info.height)
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
                        contents=_annotation_text(
                            item.dict.get(PdfName(b"Contents")),
                            "contents",
                            self.pdf.raise_unicode_errors,
                        ),
                        data=self.pdf.doc.bridge(item.dict),
                        uri=None,
                        title=_annotation_text(
                            item.dict.get(PdfName(b"T")), "title", self.pdf.raise_unicode_errors
                        ),
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
                x0, top, x1, bottom = flip_box(item.bbox, self._adapter.info.height)
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
        try:
            return [element.to_dict() for element in structure.PDFStructTree(self.pdf, self)]
        except structure.StructTreeMissing:
            return []

    @property
    def page_obj(self) -> SimpleNamespace:
        document = self.pdf.doc
        return SimpleNamespace(
            pageid=document.page_ids()[self.page_number - 1],
            attrs=document.bridge(self._adapter.page.page_dict),
        )

    @property
    def layout(self) -> Any:
        if self._layout_page is None:
            from ..pdfminer import LAParams, extract_pages
            from ..pdfminer._projection import project_page

            laparams = self.pdf.laparams
            if isinstance(laparams, Mapping):
                laparams = LAParams(**laparams)

            if self.pdf._opened_as_pdfminer:
                engine_page = self._adapter.page
                self._layout_page = (
                    project_page(
                        engine_page,
                        laparams or LAParams(),
                        products=self._adapter.pdfminer_program(),
                    )
                    if engine_page.page_number == self.page_number
                    else None
                )
            else:
                self._layout_page = next(
                    (
                        item
                        for item in extract_pages(
                            self._document_source(),
                            page_numbers=(self.page_number - 1,),
                            laparams=laparams,
                        )
                        if item.pageid == self.page_number
                    ),
                    None,
                )
        return self._layout_page

    @property
    def edges(self) -> list[ObjectDict]:
        if self._edges is None:
            self._edges = _edges_for(self.lines + self.rects + self.curves)
        return self._edges

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

    @property
    def _layout(self) -> Any:
        return self.layout

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

    def get_textmap(self, **kwargs: Any) -> Any:
        options: dict[str, Any] = {"layout_bbox": self.bbox}
        if "layout_width_chars" not in kwargs:
            options["layout_width"] = self.width
        if "layout_height_chars" not in kwargs:
            options["layout_height"] = self.height
        return utils.chars_to_textmap(self.chars, **{**options, **kwargs})

    def extract_text(self, **kwargs: Any) -> str:
        return self.get_textmap(**kwargs).as_string

    def extract_text_simple(self, **kwargs: Any) -> str:
        return utils.extract_text_simple(self.chars, **kwargs)

    def extract_text_lines(
        self, strip: bool = True, return_chars: bool = True, **kwargs: Any
    ) -> list[ObjectDict]:
        return self.get_textmap(**kwargs).extract_text_lines(strip=strip, return_chars=return_chars)

    def to_dict(self, object_types: list[str] | None = None) -> dict[str, Any]:
        names = [*self.objects, "annot"] if object_types is None else object_types
        return {
            "page_number": self.page_number,
            "initial_doctop": self.initial_doctop,
            "rotation": self.rotation,
            "cropbox": self.cropbox,
            "mediabox": self.mediabox,
            "bbox": self.bbox,
            "width": self.width,
            "height": self.height,
            **{f"{name}s": getattr(self, f"{name}s") for name in names},
        }

    def extract_words(self, **kwargs: Any) -> list[ObjectDict]:
        return utils.extract_words(self.chars, **kwargs)

    def find_tables(
        self, table_settings: Mapping[str, Any] | TableSettings | None = None
    ) -> list[Table]:
        return TableFinder(self, TableSettings.resolve(table_settings)).tables

    def debug_tablefinder(
        self, table_settings: Mapping[str, Any] | TableFinder | None = None
    ) -> TableFinder:
        if isinstance(table_settings, TableFinder):
            return table_settings
        return TableFinder(self, TableSettings.resolve(table_settings))

    def find_table(
        self, table_settings: Mapping[str, Any] | TableSettings | None = None
    ) -> Table | None:
        tables = self.find_tables(table_settings)
        if not tables:
            return None
        return min(tables, key=lambda table: (-len(table.cells), table.bbox[1], table.bbox[0]))

    def extract_tables(
        self, table_settings: Mapping[str, Any] | TableSettings | None = None
    ) -> list[list[list[str | None]]]:
        settings = TableSettings.resolve(table_settings)
        return [table.extract(**settings.text_settings) for table in self.find_tables(settings)]

    def extract_table(
        self, table_settings: Mapping[str, Any] | TableSettings | None = None
    ) -> list[list[str | None]] | None:
        settings = TableSettings.resolve(table_settings)
        table = self.find_table(settings)
        return None if table is None else table.extract(**settings.text_settings)

    def to_image(
        self,
        resolution: float | None = None,
        width: float | None = None,
        height: float | None = None,
        antialias: bool = False,
        force_mediabox: bool = False,
    ) -> PageImage:
        if sum(value is not None for value in (resolution, width, height)) > 1:
            raise ValueError("specify only one of resolution, width, or height")
        if width is not None:
            resolution = width / max(self.width, 1e-9) * 72.0
        elif height is not None:
            resolution = height / max(self.height, 1e-9) * 72.0
        try:
            raster = self._adapter.render(dpi=resolution or 72.0, antialias=antialias)
        except (OverflowError, MemoryError, PdfError) as error:
            raise PdfminerException(error) from error
        bbox = (
            self.bbox
            if self.bbox != self.mediabox
            else (self.mediabox if force_mediabox else self.cropbox)
        )
        return PageImage(self, raster, bbox=bbox)

    def search(
        self,
        pattern: str | re.Pattern[str],
        regex: bool = True,
        case: bool = True,
        main_group: int = 0,
        return_chars: bool = True,
        return_groups: bool = True,
        **kwargs: Any,
    ) -> list[ObjectDict]:
        return self.get_textmap(**kwargs).search(
            pattern,
            regex=regex,
            case=case,
            main_group=main_group,
            return_chars=return_chars,
            return_groups=return_groups,
        )

    def crop(self, bbox: BBox, relative: bool = False, strict: bool = True) -> CroppedPage:
        return CroppedPage(self, bbox, relative, strict, "intersects")

    def within_bbox(self, bbox: BBox, relative: bool = False, strict: bool = True) -> CroppedPage:
        return CroppedPage(self, bbox, relative, strict, "within")

    def outside_bbox(self, bbox: BBox, relative: bool = False, strict: bool = True) -> CroppedPage:
        return CroppedPage(self, bbox, relative, strict, "outside")

    def filter(self, test_function: Callable[[ObjectDict], bool]) -> FilteredPage:
        return FilteredPage(self, test_function)

    def dedupe_chars(self, **kwargs: Any) -> FilteredPage:
        result = FilteredPage(self, lambda _obj: True)
        result._objects = {**self.objects, "char": utils.dedupe_chars(self.chars, **kwargs)}
        return result

    def _document_source(self) -> PdfInput:
        return self.pdf.source


class CroppedPage(Page):
    def __init__(self, parent: Page, bbox: BBox, relative: bool, strict: bool, mode: str) -> None:
        target: BBox = (
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
        self.root_page = parent.root_page
        self.parent_page = parent
        self.is_original = False
        self.bbox = parent.bbox if mode == "outside" else target
        self._filter_bbox = target
        self._crop_mode = mode
        self._objects = None
        self._edges = None

    @property
    def _selector(self) -> Callable[[Iterable[ObjectDict], BBox], list[ObjectDict]]:
        return {
            "within": within_bbox,
            "outside": outside_bbox,
            "intersects": crop_to_bbox,
        }[self._crop_mode]

    def _crop_fn(self, objects: list[ObjectDict]) -> list[ObjectDict]:
        return self._selector(objects, self._filter_bbox)

    @property
    def objects(self) -> dict[str, list[ObjectDict]]:
        if self._objects is not None:
            return self._objects
        select = self._selector
        self._objects = {
            kind: select(values, self._filter_bbox)
            for kind, values in self.parent_page.objects.items()
        }
        return self._objects


class FilteredPage(Page):
    def __init__(self, parent: Page, test: Callable[[ObjectDict], bool]) -> None:
        self.__dict__.update(parent.__dict__)
        self._parent = parent
        self.is_original = False
        self._test = test
        self._objects = None
        self._edges = None

    @property
    def objects(self) -> dict[str, list[ObjectDict]]:
        if self._objects is not None:
            return self._objects
        result = {
            kind: [obj for obj in values if self._test(obj)]
            for kind, values in self._parent.objects.items()
        }
        self._objects = result
        return result


class _ImageOriginal(FrozenFields, ReprFields, ReplaceFields):
    width: int
    height: int

    __fields__: ClassVar[tuple[str, ...]] = ("width", "height")
    __match_args__ = ("width", "height")

    def __init__(self, width: int, height: int) -> None:
        frozen_setattr(self, "width", width)
        frozen_setattr(self, "height", height)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.width == other.width and self.height == other.height

    def __hash__(self) -> int:
        return hash((self.width, self.height))

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)


class PageImage:
    def __init__(
        self,
        page: Page,
        raster: Any,
        original_size: tuple[int, int] | None = None,
        *,
        bbox: BBox | None = None,
    ) -> None:
        self.page = page
        self.bbox = bbox if bbox is not None else page.bbox
        if self.bbox != page.root_page.bbox:
            x0, top, x1, bottom = self.bbox
            source_width, source_height = raster.width, raster.height
            scale_x = source_width / max(page.root_page.width, 1e-9)
            scale_y = source_height / max(page.root_page.height, 1e-9)
            left = max(0, round(x0 * scale_x))
            right = min(source_width, left + math.ceil((x1 - x0) * scale_x))
            upper = max(0, round(top * scale_y))
            lower = min(source_height, upper + math.ceil((bottom - top) * scale_y))
            row_size = source_width * raster.channels
            cropped = b"".join(
                bytes(raster.data)[
                    row * row_size + left * raster.channels : row * row_size
                    + right * raster.channels
                ]
                for row in range(upper, lower)
            )
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

    def reset(self) -> PageImage:
        self._drawings.clear()
        return self

    def copy(self) -> PageImage:
        result = copy(self)
        result._drawings = list(self._drawings)
        return result

    def draw_line(self, line: Any, **kwargs: Any) -> PageImage:
        return self.draw_lines((line,), **kwargs)

    def draw_rects(self, rects: Iterable[Any], **kwargs: Any) -> PageImage:
        self._drawings.extend(("rect", (rect, kwargs)) for rect in rects)
        return self

    def draw_lines(self, lines: Iterable[Any], **kwargs: Any) -> PageImage:
        self._drawings.extend(("line", (line, kwargs)) for line in lines)
        return self

    def draw_vline(self, x: float, **kwargs: Any) -> PageImage:
        return self.draw_line((x, self.page.bbox[1], x, self.page.bbox[3]), **kwargs)

    def draw_vlines(self, xs: Iterable[float], **kwargs: Any) -> PageImage:
        for x in xs:
            self.draw_vline(x, **kwargs)
        return self

    def draw_hline(self, y: float, **kwargs: Any) -> PageImage:
        return self.draw_line((self.page.bbox[0], y, self.page.bbox[2], y), **kwargs)

    def draw_hlines(self, ys: Iterable[float], **kwargs: Any) -> PageImage:
        for y in ys:
            self.draw_hline(y, **kwargs)
        return self

    def draw_rect(self, rect: Any, **kwargs: Any) -> PageImage:
        return self.draw_rects((rect,), **kwargs)

    def draw_circle(self, circle: Any, **kwargs: Any) -> PageImage:
        self._drawings.append(("circle", (circle, kwargs)))
        return self

    def draw_circles(self, circles: Iterable[Any], **kwargs: Any) -> PageImage:
        for circle in circles:
            self.draw_circle(circle, **kwargs)
        return self

    def outline_words(self, words: Iterable[Any] | None = None, **kwargs: Any) -> PageImage:
        return self.draw_rects(words or self.page.extract_words(), **kwargs)

    def outline_chars(self, chars: Iterable[Any] | None = None, **kwargs: Any) -> PageImage:
        return self.draw_rects(chars or self.page.chars, **kwargs)

    def debug_tablefinder(self, table_settings: Mapping[str, Any] | None = None) -> PageImage:
        finder = self.page.debug_tablefinder(table_settings)
        self.draw_lines(finder.edges)
        for table in finder.tables:
            self.debug_table(table, stroke="blue")
        return self

    def debug_table(self, table: Any, **kwargs: Any) -> PageImage:
        cells = getattr(table, "cells", ())
        return self.draw_rects((cell for cell in cells if cell is not None), **kwargs)

    def draw_words(self, words: Iterable[Any], **kwargs: Any) -> PageImage:
        self._drawings.extend(("word", (word, kwargs)) for word in words)
        return self

    def save(self, path: str | Any, format: str | None = None, **kwargs: Any) -> None:
        del format
        channels = self.raster.channels
        if channels not in (3, 4):
            raise ValueError("PNG output requires RGB or RGBA raster data")
        pixels = bytearray(self.raster.data)
        self._render_drawings(pixels)
        png = _encode_png(self.width, self.height, channels, pixels)
        if kwargs.get("quantize", True):
            png = _quantized_png(png)
        if hasattr(path, "write"):
            path.write(png)  # ty: ignore[call-non-callable]
        else:
            with builtins.open(path, "wb") as stream:
                stream.write(png)

    def _render_drawings(self, pixels: bytearray) -> None:
        if not self._drawings:
            return
        channels = self.raster.channels
        scale_x = self.width / max(self.bbox[2] - self.bbox[0], 1e-9)
        scale_y = self.height / max(self.bbox[3] - self.bbox[1], 1e-9)

        def point(x: float, y: float) -> tuple[int, int]:
            return (
                round((x - self.bbox[0]) * scale_x),
                round((y - self.bbox[1]) * scale_y),
            )

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
                if kind == "line" and isinstance(value, Mapping) and "pts" in value:
                    coords = tuple(value["pts"])
                elif isinstance(value, Mapping):
                    coords = (value["x0"], value["top"], value["x1"], value["bottom"])
                else:
                    coords = tuple(value)
                if kind == "line" and coords and isinstance(coords[0], (tuple, list)):
                    points = [point(float(x), float(y)) for x, y in coords]
                    for first, last in zip(points, points[1:], strict=False):
                        line(*first, *last, ink)
                    continue
                x0, y0 = point(float(coords[0]), float(coords[1]))
                x1, y1 = point(float(coords[2]), float(coords[3]))
                if kind in {"rect", "word"} and options.get("fill") is not None:
                    fill = color(options["fill"], ink)
                    for y in range(min(y0, y1), max(y0, y1) + 1):
                        for x in range(min(x0, x1), max(x0, x1) + 1):
                            set_pixel(x, y, fill)
                if kind == "line":
                    line(x0, y0, x1, y1, ink)
                else:
                    line(x0, y0, x1, y0, ink)
                    line(x0, y0, x0, y1, ink)
                    line(x1, y0, x1, y1, ink)
                    line(x0, y1, x1, y1, ink)
            else:
                if isinstance(value, Mapping):
                    cx = (float(value["x0"]) + float(value.get("x1", value["x0"]))) / 2
                    cy = (float(value["top"]) + float(value.get("bottom", value["top"]))) / 2
                    default_radius = value.get("radius", value.get("r", 5))
                else:
                    cx, cy = value
                    default_radius = 5
                center = point(float(cx), float(cy))
                radius = float(options.get("radius", default_radius))
                ink = color(options.get("stroke"), (255, 0, 0))
                for angle in range(360):
                    radians = math.radians(angle)
                    set_pixel(
                        round(center[0] + radius * scale_x * math.cos(radians)),
                        round(center[1] + radius * scale_y * math.sin(radians)),
                        ink,
                    )

    def _repr_png_(self) -> bytes:
        stream = BytesIO()
        self.save(stream)
        return stream.getvalue()

    def show(self) -> None:
        try:
            from PIL import Image
        except ImportError as error:
            raise ImportError(
                "PageImage.show() requires Pillow. Install it with: "
                "python -m pip install 'core-pdf-compat[pdfplumber]'"
            ) from error

        with Image.open(BytesIO(self._repr_png_())) as image:
            image.show()


def _quantized_png(png: bytes) -> bytes:
    # Like upstream, limit output to 256 colors. Without Pillow, the PNG stays as it is.
    try:
        from PIL import Image
    except ImportError:
        return png
    with Image.open(BytesIO(png)) as image:
        quantized = image.convert("RGB").quantize(256)
        stream = BytesIO()
        quantized.save(stream, "PNG")
        return stream.getvalue()


def _encode_png(width: int, height: int, channels: int, pixels: bytes | bytearray) -> bytes:
    stride = width * channels
    scanlines = b"".join(
        b"\x00" + pixels[row * stride : (row + 1) * stride] for row in range(height)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 6 if channels == 4 else 2, 0, 0, 0)
    return (
        PNG_SIGNATURE
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(scanlines))
        + png_chunk(b"IEND", b"")
    )


class PDF(ClosingMixin, Container):
    @property
    def edges(self) -> list[ObjectDict]:
        return [edge for page in self.pages for edge in page.edges]

    @property
    def horizontal_edges(self) -> list[ObjectDict]:
        return [edge for page in self.pages for edge in page.horizontal_edges]

    @property
    def vertical_edges(self) -> list[ObjectDict]:
        return [edge for page in self.pages for edge in page.vertical_edges]

    def __init__(
        self,
        document: PdfDocument | PdfInput,
        source: PdfInput | None = None,
        pages: Iterable[int] | None = None,
        laparams: Any = None,
        unicode_norm: str | None = None,
        raise_unicode_errors: bool = True,
    ) -> None:
        self._unicode_norm = unicode_norm
        self.raise_unicode_errors = raise_unicode_errors
        self._latest_capture = LatestCapture()
        opened_here = source is None
        if source is None:
            source = cast(PdfInput, document)
            document = _source(source)
        self._document = cast(PdfDocument, document)
        self._opened_as_pdfminer = opened_here
        self.doc = DocumentView(self._document)
        self.source: PdfInput = source
        self.stream = source
        self.path = source if isinstance(source, (str, bytes)) else None
        raw_metadata = self._document.get_metadata()
        info = raw_metadata.get("info") if isinstance(raw_metadata, dict) else None
        self.metadata = dict(info) if isinstance(info, dict) else {}
        self.laparams = laparams
        self._page_selection = tuple(pages) if pages is not None else None
        self._pages: list[Page] | None = None
        self._objects: dict[str, list[ObjectDict]] | None = None
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
        repair: bool = False,
        gs_path: str | None = None,
        repair_setting: RepairSetting = "default",
        raise_unicode_errors: bool = True,
        **_: Any,
    ) -> PDF:
        if repair:
            # The repaired copy carries no encryption, so the password is not reused.
            source = repair_stream(
                source,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                password=password,
                gs_path=gs_path,
                setting=repair_setting,
            )
            password = ""
        pdf = cls(
            _source(source, password), source, pages, laparams, unicode_norm, raise_unicode_errors
        )
        pdf._opened_as_pdfminer = True
        return pdf

    @property
    def pages(self) -> list[Page]:
        if self._pages is None:
            try:
                from ..pdfminer._pages import pdfminer_resolvable_pages

                doctop = 0.0
                self._pages = []
                resolvable = tuple(pdfminer_resolvable_pages(self._document))
                selected = set(self._page_selection) if self._page_selection is not None else None
                for page_number, (index, engine_page) in enumerate(resolvable, 1):
                    if selected is not None and page_number not in selected:
                        continue
                    page = Page(self, index, doctop, engine_page)
                    self._pages.append(page)
                    doctop += page.height
            except IndexError, PdfminerException:
                raise
            except Exception as exc:
                raise PdfminerException(exc) from exc
        return self._pages

    @property
    def objects(self) -> dict[str, list[ObjectDict]]:
        if self._objects is None:
            result: dict[str, list[ObjectDict]] = {}
            for page in self.pages:
                for kind, values in page.objects.items():
                    result.setdefault(kind, []).extend(values)
            self._objects = result
        return self._objects

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
        try:
            return [element.to_dict() for element in structure.PDFStructTree(self)]
        except structure.StructTreeMissing:
            return []

    def to_dict(self, object_types: list[str] | None = None) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "pages": [page.to_dict(object_types) for page in self.pages],
        }

    def close(self) -> None:
        self.flush_cache()
        self._document.close()

    def flush_cache(self, *_: Any) -> None:
        for page in self._pages or ():
            page.flush_cache()
        self.flush_merged_objects()

    def flush_merged_objects(self) -> None:
        self._objects = None
        self._rect_edges = None
        self._curve_edges = None


open = PDF.open


def extract_text(source: PdfInput, **kwargs: Any) -> str:
    with PDF.open(source, **kwargs) as pdf:
        return "\f".join(page.extract_text(**kwargs) for page in pdf.pages) + "\f"


def within_bbox(objs: Iterable[ObjectDict], bbox: BBox) -> list[ObjectDict]:
    return [obj for obj in objs if bbox_contains(bbox, obj_to_bbox(obj))]


def outside_bbox(objs: Iterable[ObjectDict], bbox: BBox) -> list[ObjectDict]:
    return [obj for obj in objs if bbox_overlap(obj_to_bbox(obj), bbox) is None]


def bbox_overlap(left: BBox, right: BBox) -> BBox | None:
    x0, top = max(left[0], right[0]), max(left[1], right[1])
    x1, bottom = min(left[2], right[2]), min(left[3], right[3])
    if x1 < x0 or bottom < top or (x1 == x0 and bottom == top):
        return None
    return (x0, top, x1, bottom)


def obj_to_bbox(obj: ObjectDict) -> BBox:
    return (obj["x0"], obj["top"], obj["x1"], obj["bottom"])


def intersects_bbox(obj: ObjectDict | Iterable[ObjectDict], bbox: BBox) -> bool | list[ObjectDict]:
    if not isinstance(obj, Mapping):
        return [item for item in obj if intersects_bbox(item, bbox)]
    obj = obj
    return bbox_overlap(bbox, obj_to_bbox(obj)) is not None  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def crop_to_bbox(objs: Iterable[ObjectDict], bbox: BBox) -> list[ObjectDict]:
    clipped = []
    for obj in objs:
        overlap = bbox_overlap(obj_to_bbox(obj), bbox)
        if overlap is None:
            continue
        x0, top, x1, bottom = overlap
        result = {
            **obj,
            "x0": x0,
            "top": top,
            "x1": x1,
            "bottom": bottom,
            "width": x1 - x0,
            "height": bottom - top,
        }
        if "doctop" in obj:
            result["doctop"] += top - obj["top"]
        clipped.append(result)
    return clipped


def cluster_list(values: Iterable[float], tolerance: float = 0) -> list[list[float]]:
    ordered = sorted(values)
    if tolerance == 0:
        return [[value] for value in ordered]
    return cluster_by(ordered, lambda value: value, tolerance)


cluster_objects = cluster_by


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
        if "doctop" in result:
            result["doctop"] += value
        if "y0" in result:
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
            if "y1" in result:
                result["y1"] += old - value
        elif key == "bottom" and "y0" in result:
            result["y0"] += old - value
    return result


def filter_edges(
    edges: Iterable[ObjectDict],
    orientation: str | None = None,
    edge_type: str | None = None,
    min_length: float = 1,
) -> list[ObjectDict]:
    if orientation not in {"h", "v", None}:
        raise ValueError("orientation must be 'h' or 'v'")
    return [
        edge
        for edge in edges
        if (orientation is None or edge.get("orientation") == orientation)
        and (edge_type is None or edge.get("object_type") == edge_type)
        and edge["height" if edge["orientation"] == "v" else "width"] >= min_length
    ]


def merge_edges(
    edges: Iterable[ObjectDict],
    snap_x_tolerance: float = 3,
    snap_y_tolerance: float = 3,
    join_x_tolerance: float = 3,
    join_y_tolerance: float = 3,
) -> list[ObjectDict]:
    values = list(edges)
    if any(edge["orientation"] not in {"h", "v"} for edge in values):
        raise ValueError("orientation must be 'h' or 'v'")
    if snap_x_tolerance > 0 or snap_y_tolerance > 0:
        values = snap_objects(
            [edge for edge in values if edge["orientation"] == "v"], "x0", snap_x_tolerance
        ) + snap_objects(
            [edge for edge in values if edge["orientation"] == "h"], "top", snap_y_tolerance
        )

    def group_key(edge: ObjectDict) -> tuple[str, float]:
        orientation = edge["orientation"]
        return orientation, edge["top" if orientation == "h" else "x0"]

    result: list[ObjectDict] = []
    for (orientation, _), group in groupby(sorted(values, key=group_key), group_key):
        start, end, tolerance = (
            ("x0", "x1", join_x_tolerance)
            if orientation == "h"
            else ("top", "bottom", join_y_tolerance)
        )
        joined: list[ObjectDict] = []
        for edge in sorted(group, key=lambda item: item[start]):
            if joined and edge[start] <= joined[-1][end] + tolerance:
                if edge[end] > joined[-1][end]:
                    joined[-1] = resize_object(joined[-1], end, edge[end])
            else:
                joined.append(edge)
        result.extend(joined)
    return result


def snap_objects(objects: Iterable[ObjectDict], attr: str, tolerance: float) -> list[ObjectDict]:
    axis = {"x0": "h", "x1": "h", "top": "v", "bottom": "v"}[attr]
    result: list[ObjectDict] = []
    for cluster in cluster_objects(objects, attr, tolerance):
        target = sum(item[attr] for item in cluster) / len(cluster)
        result.extend(move_object(item, axis, target - item[attr]) for item in cluster)
    return result


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
    if isinstance(value, list):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict("records")
    return list(value)


def _edges_for(objs: Iterable[ObjectDict]) -> list[ObjectDict]:
    return [edge for obj in objs for edge in _edges(obj)]


def _line_edge(line: ObjectDict) -> ObjectDict:
    return {**line, "orientation": "h" if line["top"] == line["bottom"] else "v"}


def _rect_edges(rect: ObjectDict) -> list[ObjectDict]:
    rect_edge = {**rect, "object_type": "rect_edge"}
    return [
        {**rect_edge, "height": 0, "y0": rect["y1"], "bottom": rect["top"], "orientation": "h"},
        {
            **rect_edge,
            "height": 0,
            "y1": rect["y0"],
            "top": rect["top"] + rect["height"],
            "doctop": rect["doctop"] + rect["height"],
            "orientation": "h",
        },
        {**rect_edge, "width": 0, "x1": rect["x0"], "orientation": "v"},
        {**rect_edge, "width": 0, "x0": rect["x1"], "orientation": "v"},
    ]


def _curve_edges(curve: ObjectDict) -> list[ObjectDict]:
    return [
        {
            "object_type": "curve_edge",
            "x0": min(start[0], end[0]),
            "x1": max(start[0], end[0]),
            "top": min(start[1], end[1]),
            "doctop": min(start[1], end[1]) + (curve["doctop"] - curve["top"]),
            "bottom": max(start[1], end[1]),
            "width": abs(start[0] - end[0]),
            "height": abs(start[1] - end[1]),
            "orientation": "v" if start[0] == end[0] else "h" if start[1] == end[1] else None,
        }
        for start, end in pairwise(curve.get("pts", ()))
    ]


def _edges(obj: ObjectDict) -> list[ObjectDict]:
    kind = obj["object_type"]
    if "_edge" in kind:
        return [obj]
    if kind == "line":
        return [_line_edge(obj)]
    return _rect_edges(obj) if kind == "rect" else _curve_edges(obj)


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
    "repair",
    "within_bbox",
    "utils",
    "PdfminerException",
)

# Upstream exposes these as attributes after a bare `import pdfplumber`.
import core_pdf_compat.pdfplumber.page  # noqa: E402, F401

from . import (  # noqa: E402
    structure,
    utils,
)
from .table import Table, TableFinder, TableSettings  # noqa: E402
from .utils.text import extract_words  # noqa: E402
