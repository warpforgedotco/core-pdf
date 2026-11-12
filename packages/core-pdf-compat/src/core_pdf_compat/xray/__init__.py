from __future__ import annotations

import asyncio
import json
import mmap
import re
import struct
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Iterator
from http import HTTPStatus
from math import floor, isfinite
from os import PathLike
from pathlib import Path
from typing import Any

from core_pdf import PdfDocument
from core_pdf._vendor.fontTools.ttLib import TTLibError
from core_pdf.impl.capture_page import capture_annotation_appearances
from core_pdf.impl.capture_program import DEFAULT_CAPTURE, CaptureOptions, PageProgram
from core_pdf.impl.capture_recording import TextState
from core_pdf.impl.exceptions import PdfPasswordError, PdfUnsupportedError
from core_pdf.impl.fonts_cmap import ToUnicodeCMap
from core_pdf.impl.geometry import (
    bbox_intersects,
    overlap_ratio_of,
)
from core_pdf.impl.render_page import compose_page
from core_pdf.impl.spec.s_07_content.model import PATH_OPERAND_COUNTS, PATH_RECT, PdfPath
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf.impl.text import collapse_ws
from core_pdf.impl.types import PdfReference, RecordType, ReplaceFields, ReprFields

from .._shared import parse_indirect_object_at

_FLOAT32 = struct.Struct("f")
_UINT32 = struct.Struct("I")
_DATE = re.compile(r"[0-3]?\d[/\-][0-3]?\d[/\-]\d{2,4}")
_OK_WORDS = re.compile(
    r"confidential|name +redacted|privileged?|re|red|reda|redac|redact|"
    r"redacte|redacted|redacted +and +publicly +filed|",
    re.IGNORECASE | re.MULTILINE,
)


class XrayDocument(PdfDocument):
    @property
    def font_semantic_context(self) -> None:
        return None


class _Rectangle(ReprFields, ReplaceFields, metaclass=RecordType, frozen=False):
    bbox: tuple[float, float, float, float]
    seqno: int
    fill: tuple[float, ...]
    fitz_box: tuple[float, float, float, float]
    allow_same_fill: bool = False

    __hash__ = None  # type: ignore[assignment]


class _Character(ReprFields, ReplaceFields, metaclass=RecordType, frozen=False):
    bbox: tuple[float, float, float, float]
    text: str
    seqno: int
    fill: tuple[float, ...] | None

    __hash__ = None  # type: ignore[assignment]


class _RecoveredFont(ReprFields, ReplaceFields, metaclass=RecordType, frozen=False):
    cmap: ToUnicodeCMap
    first_char: int
    widths: tuple[float, ...]

    __hash__ = None  # type: ignore[assignment]


def _occluded(character: _Character, rectangle: _Rectangle, threshold: float) -> bool:
    if not bbox_intersects(character.bbox, rectangle.bbox) or not (
        rectangle.seqno > character.seqno
        or (rectangle.allow_same_fill and rectangle.fill == character.fill)
    ):
        return False
    return overlap_ratio_of(character.bbox, rectangle.bbox) > threshold


class _XrayCapture(TextState):
    """Core's page capture, keeping the operands of each re operator a drawing paints.

    Flattening adds an re operator's width and height to its corner in float64, where MuPDF
    adds them in float32 (``fz_rectto(x, y, x + w, y + h)`` in pdf-op-run.c), so the corner
    MuPDF reports cannot be recovered from the flattened points.
    """

    def __init__(
        self, document: Any, *, page_clip: Any, options: CaptureOptions = DEFAULT_CAPTURE
    ) -> None:
        super().__init__(
            document,
            hidden_layers=document.oc_hidden_layers(),
            page_clip=page_clip,
            options=options,
        )
        # The (x, y, w, h) operands of each re in a drawing painted with an identity CTM,
        # by the drawing's sequence number.
        self.rect_operands: dict[int, tuple[tuple[float, float, float, float], ...]] = {}

    def paint_path(self, state: object, source: PdfPath, kind: str, fill_rule: str) -> None:
        drawing_count = len(self.drawings)
        operands: list[tuple[float, float, float, float]] = []
        if PATH_RECT in source.ops and self.graphics.ctm == IDENTITY_MATRIX:
            coords = source.coords
            position = 0
            for op in source.ops:
                if op == PATH_RECT:
                    x, y, w, h = coords[position : position + 4]
                    operands.append((x, y, w, h))
                position += PATH_OPERAND_COUNTS[op]
        super().paint_path(state, source, kind, fill_rule)
        if operands and len(self.drawings) > drawing_count:
            self.rect_operands[self.drawings[-1].seqno] = tuple(operands)


_FILL_KINDS = frozenset({"fill", "fillstroke"})


class _RectangleFound(BaseException):
    """Raised out of the content stream by _DrawingsProbe at the first candidate rectangle; a
    BaseException so content recovery, which catches Exception, does not swallow it."""


class _DrawingsProbe(_XrayCapture):
    """The page's drawings without its text, as x-ray's get_drawings() looks for rectangles
    before it reads any text, stopping at the first candidate. Shown text changes the drawings
    only through a Type3 glyph's procedure or a clipping render mode, so only those shows still
    run. A skipped show leaves the text position where it was, so a show that runs after one,
    before BT resets the position, could paint elsewhere than in the full capture: the probe
    answers yes there instead."""

    def __init__(self, page: Any, crop_box: tuple[float, ...], user_unit: float) -> None:
        super().__init__(
            page.document, page_clip=page.effective_page_clip(), options=_DRAWINGS_CAPTURE
        )
        self.page = page
        self.crop_box = crop_box
        self.user_unit = user_unit
        self.skipped_show = False

    def paint_path(self, state: object, source: PdfPath, kind: str, fill_rule: str) -> None:
        drawing_count = len(self.drawings)
        super().paint_path(state, source, kind, fill_rule)
        # _path_rectangles' first test, on the captured drawings, before building records.
        painted = [
            drawing
            for drawing in self.drawings[drawing_count:]
            if drawing.kind in _FILL_KINDS
            and drawing.fill is not None
            and drawing.fill_opacity == 1
            and drawing.path is not None
        ]
        if painted and any(
            _path_rectangles(drawing, self.crop_box, self.user_unit, self.rect_operands)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            for drawing in self.page.drawing_records(painted)
        ):
            raise _RectangleFound

    def runs_show(self, operands: Any) -> bool:
        if operands and (4 <= self.graphics.render_mode <= 7 or self.get_decoder().is_type3):
            if self.skipped_show:
                raise _RectangleFound
            return True
        self.skipped_show = True
        return False

    def op_BT(self, operands: Any, depth: int) -> None:
        super().op_BT(operands, depth)
        self.skipped_show = False

    def op_Tj(self, operands: Any, depth: int) -> None:
        if self.runs_show(operands):
            super().op_Tj(operands, depth)

    def op_TJ(self, operands: Any, depth: int) -> None:
        if self.runs_show(operands):
            super().op_TJ(operands, depth)

    def op_quote(self, operands: Any, depth: int) -> None:
        if self.runs_show(operands):
            super().op_quote(operands, depth)

    def op_double_quote(self, operands: Any, depth: int) -> None:
        if self.runs_show(operands):
            super().op_double_quote(operands, depth)


# The probe reads only filled paths: no text, and no image's soft mask decoded for its alpha.
_DRAWINGS_CAPTURE = CaptureOptions(glyphs=False, text_runs=False, soft_mask_alpha=False)


def _capture_page(page: Any) -> tuple[Any, dict[int, Any]]:
    """``page.get_page_program()``, with the re operands ``_XrayCapture`` keeps."""
    state = _XrayCapture(page.document, page_clip=page.effective_page_clip())
    try:
        page.consume_contents(state)
        state.run_accumulator.flush()
        body = state.captured_program()
        program = PageProgram(body=body, appearances=capture_annotation_appearances(page, state))
        return program, state.rect_operands
    finally:
        state.release()


def _mupdf_rect_boxes(
    operands: tuple[tuple[float, float, float, float], ...],
) -> dict[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Each re's box as core flattens it, mapped to the box MuPDF's float32 corners make."""
    boxes = {}
    for x, y, w, h in operands:
        x1 = x + w
        y1 = y + h
        mupdf_x = _float32(x)
        mupdf_y = _float32(y)
        mupdf_x1 = _float32(mupdf_x + _float32(w))
        mupdf_y1 = _float32(mupdf_y + _float32(h))
        boxes[(min(x, x1), min(y, y1), max(x, x1), max(y, y1))] = (
            min(mupdf_x, mupdf_x1),
            min(mupdf_y, mupdf_y1),
            max(mupdf_x, mupdf_x1),
            max(mupdf_y, mupdf_y1),
        )
    return boxes


def _path_rectangles(
    drawing: Any,
    crop_box: tuple[float, float, float, float],
    user_unit: float = 1.0,
    rect_operands: dict[int, Any] | None = None,
) -> list[_Rectangle]:
    if (
        drawing.kind not in {"fill", "fillstroke"}
        or drawing.fill is None
        or drawing.fill_opacity != 1
        or drawing.path is None
    ):
        return []
    operands = rect_operands.get(drawing.seqno) if rect_operands is not None else None
    mupdf_boxes = _mupdf_rect_boxes(operands) if operands else {}
    output: list[_Rectangle] = []
    for subpath in drawing.path.subpaths:
        points = subpath.points
        if not subpath.closed or len(points) != 4:
            continue
        xs = {point[0] for point in points}
        ys = {point[1] for point in points}
        if len(xs) != 2 or len(ys) != 2:
            continue
        bbox = (min(xs), min(ys), max(xs), max(ys))
        mupdf_box = mupdf_boxes.get(bbox)
        top_box = (
            _fitz_box(bbox, crop_box, user_unit)
            if mupdf_box is None
            else _fitz_box(mupdf_box, crop_box, user_unit, exact=True)
        )
        if top_box[3] <= 43 or top_box[2] - top_box[0] <= 4 or top_box[3] - top_box[1] <= 4:
            continue
        # Text the same color as the rectangle is invisible even when drawn on top of it.
        output.append(_Rectangle(bbox, drawing.seqno, drawing.fill, top_box, True))
    return output


class _PageRaster:
    __slots__ = ("page", "program", "rendered")

    def __init__(self, page: Any, program: PageProgram) -> None:
        self.page = page
        self.program = program
        self.rendered: Any = None

    def rasterize(self, box: tuple[float, float, float, float]) -> Any:
        if self.rendered is None:
            self.rendered = compose_page(
                self.page,
                page_program=self.program,
                semantic_context=self.page.document.resolver.semantic_context,
            )
        return self.rendered.rasterize(scale=1.0, crop=box)


def _uniform(raster_page: _PageRaster, box: tuple[float, float, float, float]) -> bool:
    raster = raster_page.rasterize(box)
    pixels = memoryview(raster.pixels).cast("B")
    if not pixels or raster.channels <= 0:
        return False
    data = pixels.tobytes()
    first = data[: raster.channels]
    count, remainder = divmod(len(data), raster.channels)
    return remainder == 0 and data == first * count


def _fitz_box(
    box: tuple[float, float, float, float],
    crop_box: tuple[float, float, float, float],
    user_unit: float = 1.0,
    *,
    exact: bool = False,
) -> tuple[float, float, float, float]:
    """The box in PyMuPDF's page space; ``exact`` takes the box's corners as MuPDF's own."""
    crop_x0, _crop_y0, _crop_x1, crop_y1 = crop_box
    user_unit = _float32(user_unit)
    crop_x0 = _float32(_fitz_coordinate(crop_x0) * user_unit)
    crop_y1 = _float32(_fitz_coordinate(crop_y1) * user_unit)
    x0, y0, x1, y1 = box if exact else (_fitz_coordinate(value) for value in box)
    return (
        _float32(_float32(x0 * user_unit) - crop_x0),
        _float32(crop_y1 - _float32(y1 * user_unit)),
        _float32(_float32(x1 * user_unit) - crop_x0),
        _float32(crop_y1 - _float32(y0 * user_unit)),
    )


def _float32(value: float) -> float:
    return _FLOAT32.unpack(_FLOAT32.pack(value))[0]


def _fitz_coordinate(value: float) -> float:
    millipoint = round(value, 3)
    if abs(value - millipoint) < 0.00005:
        value = millipoint
    return _float32(value)


def _next_float32(value: float) -> float:
    bits = _UINT32.unpack(_FLOAT32.pack(value))[0]
    return _FLOAT32.unpack(_UINT32.pack(bits + 1))[0]


def _parse_object_at(document: Any, offset: int) -> object | None:
    try:
        return parse_indirect_object_at(
            document.raw_data,
            offset,
            reference_resolver=document.resolver.resolve,
            decipher=document.decipher,
        )
    except TypeError, ValueError:
        return None


def _recover_font(page: Any, font_name: str) -> _RecoveredFont | None:
    document = page.document
    raw_data = bytes(document.raw_data)
    pattern = rb"/" + re.escape(font_name.encode("latin-1")) + rb"\s+(\d+)\s+(\d+)\s+R\b"
    references = reversed(re.findall(pattern, raw_data))
    for object_number, generation_number in references:
        object_header = object_number + rb"\s+" + generation_number + rb"\s+obj\b"
        match = re.search(object_header, raw_data)
        if match is None:
            continue
        font = _parse_object_at(document, match.start())
        if not isinstance(font, dict):
            continue
        to_unicode = font.get("ToUnicode")
        if not isinstance(to_unicode, PdfReference):
            continue
        stream_match = re.search(
            str(to_unicode.object_number).encode()
            + rb"\s+"
            + str(to_unicode.generation_number).encode()
            + rb"\s+obj\b",
            raw_data,
        )
        if stream_match is None:
            continue
        stream = _parse_object_at(document, stream_match.start())
        if not isinstance(stream, PdfStream):
            continue
        try:
            first_char = font.get("FirstChar")
            widths = font.get("Widths")
            if not isinstance(first_char, int) or not isinstance(widths, list):
                continue
            return _RecoveredFont(
                ToUnicodeCMap(stream.data),
                first_char,
                tuple(float(width) for width in widths),
            )
        except TypeError, ValueError:
            continue
    return None


class _DocumentRecovery:
    __slots__ = ("document", "fonts", "overrides")

    def __init__(self, document: PdfDocument) -> None:
        self.document = document
        self.fonts: dict[str, _RecoveredFont | None] = {}
        self.overrides: dict[bytes, bytes] | None = None

    def font(self, page: Any, font_name: str) -> _RecoveredFont | None:
        if font_name not in self.fonts:
            self.fonts[font_name] = _recover_font(page, font_name)
        return self.fonts[font_name]

    def operand_overrides(self) -> dict[bytes, bytes]:
        if self.overrides is None:
            self.overrides = _operand_overrides(self.document.raw_data)
        return self.overrides


class _BoxGrid:
    CELL = 64.0
    MAX_CELLS = 1024

    def __init__(self, boxes: list[tuple[float, float, float, float]]) -> None:
        self.count = len(boxes)
        self.everywhere: list[int] = []
        self.cells: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
        for index, box in enumerate(boxes):
            span = self.span(box)
            if span is None:
                self.everywhere.append(index)
                continue
            for cell in span:
                self.cells[cell].append(index)

    @classmethod
    def span(cls, box: tuple[float, float, float, float]) -> list[tuple[int, int]] | None:
        x0, y0, x1, y1 = box
        if not (isfinite(x0) and isfinite(y0) and isfinite(x1) and isfinite(y1)):
            return None
        column_start, column_stop = floor(x0 / cls.CELL), floor(x1 / cls.CELL) + 1
        row_start, row_stop = floor(y0 / cls.CELL), floor(y1 / cls.CELL) + 1
        columns = max(0, column_stop - column_start)
        rows = max(0, row_stop - row_start)
        if not columns or not rows:
            return []
        if columns * rows > cls.MAX_CELLS:
            return None
        return [
            (column, row)
            for column in range(column_start, column_stop)
            for row in range(row_start, row_stop)
        ]

    def candidates(self, box: tuple[float, float, float, float]) -> list[int]:
        span = self.span(box)
        if span is None:
            return list(range(self.count))
        found = set(self.everywhere)
        for cell in span:
            found.update(self.cells.get(cell, ()))
        return sorted(found)


# Two or more hex strings shown with Tj. Written as one string then more, not as a repeated
# group, the pattern starts with a literal "<" the regex engine searches for quickly; a
# repeated group made it try every byte of the file.
HEX_STRINGS_SHOWN = re.compile(rb"<[0-9A-Fa-f]+>(?:<[0-9A-Fa-f]+>)+\s*Tj\b")
HEX_STRING = re.compile(rb"<([0-9A-Fa-f]+)>")
WORD_BYTES = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")
ASCII_WHITESPACE = frozenset(b" \t\n\r\x0b\x0c")
TRAILER_ROOT = re.compile(rb"/Root\s+([^>]+)")


def _operand_overrides(raw_data: bytes | mmap.mmap) -> dict[bytes, bytes]:
    return {
        bytes.fromhex(groups[0].decode()): bytes.fromhex(groups[-1].decode())
        for match in HEX_STRINGS_SHOWN.finditer(raw_data)
        if len(groups := HEX_STRING.findall(match.group())) > 1
    }


def _trailer_keyword_ends(raw_data: bytes) -> Iterator[int]:
    """Where each match of rb"\\btrailer\\b" ends, found without the regex.

    A leading \\b keeps the regex engine from searching for the literal, so it tried every byte
    of the file. A bytes pattern's word characters are [A-Za-z0-9_], and "trailer" cannot
    overlap itself, so every delimited occurrence is a match.
    """
    start = raw_data.find(b"trailer")
    while start >= 0:
        end = start + 7
        if (start == 0 or raw_data[start - 1] not in WORD_BYTES) and (
            end == len(raw_data) or raw_data[end] not in WORD_BYTES
        ):
            yield end
        start = raw_data.find(b"trailer", start + 7)


def _has_rectangles(page: Any, crop_box: tuple[float, ...], user_unit: float) -> bool:
    """Whether the page may paint a rectangle that could be a redaction, found without
    capturing its text. A wrong yes costs only time, as _page_redactions finds the rectangles
    again; a wrong no would lose redactions. A page with a Highlight annotation always counts:
    a font fontTools rejects while the text is captured sends it to _raw_highlight_redactions."""
    if any(annotation.subtype == "Highlight" for annotation in page.get_annotations()):
        return True
    state = _DrawingsProbe(page, crop_box, user_unit)
    try:
        page.consume_contents(state)
        state.run_accumulator.flush()
        body = state.captured_program()
        drawings = PageProgram(
            body=body, appearances=capture_annotation_appearances(page, state)
        ).drawings
    except _RectangleFound:
        return True
    finally:
        state.release()
    return any(
        _path_rectangles(drawing, crop_box, user_unit, state.rect_operands)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        for drawing in page.drawing_records(drawings)
    )


def _page_redactions(page: Any, recovery: _DocumentRecovery) -> list[dict[str, object]]:
    source_crop_box = page.crop_box or page.media_box
    user_unit = page.user_unit
    crop_box = tuple(float(value) for value in source_crop_box)
    if not _has_rectangles(page, crop_box, user_unit):
        return []
    program, rect_operands = _capture_page(page)
    drawings = page.drawing_records(program.drawings)
    rectangles = [
        rectangle
        for drawing in drawings
        for rectangle in _path_rectangles(drawing, crop_box, user_unit, rect_operands)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    ]
    if not rectangles:
        return []
    annotations = page.get_annotations()
    annotation_boxes = {
        tuple(annotation.rect) for annotation in annotations if annotation.subtype == "Highlight"
    }
    non_annotation_rectangles = [
        rectangle for rectangle in rectangles if rectangle.bbox not in annotation_boxes
    ]
    rectangle_grid = _BoxGrid([rectangle.bbox for rectangle in rectangles])
    non_annotation_grid = _BoxGrid([rectangle.bbox for rectangle in non_annotation_rectangles])
    recovered_positions: dict[tuple[str, int], float] = {}

    def recover_font(font_name: str) -> _RecoveredFont | None:
        return recovery.font(page, font_name)

    def occluded(character: _Character, candidates: list[_Rectangle], grid: _BoxGrid) -> bool:
        return any(
            _occluded(character, candidates[index], 0.8)
            for index in grid.candidates(character.bbox)
        )

    operand_overrides = recovery.operand_overrides()
    glyphs = program.glyphs
    raster_page = _PageRaster(page, program)
    overridden_sequences: dict[int, bytes] = {}
    if operand_overrides:
        sequence_parts: defaultdict[int, list[bytes]] = defaultdict(list)
        for glyph in glyphs:
            sequence_parts[glyph.seqno].append(glyph.code_bytes)
        overridden_sequences = {
            seqno: operand_overrides[codes]
            for seqno, parts in sequence_parts.items()
            if (codes := b"".join(parts)) in operand_overrides
        }
    emitted_overrides: set[int] = set()
    characters: list[_Character] = []
    for glyph in glyphs:
        glyph_box = glyph.ink_bbox
        if glyph_box is None:
            continue
        override = overridden_sequences.get(glyph.seqno)
        if override is not None:
            if glyph.seqno in emitted_overrides:
                continue
            emitted_overrides.add(glyph.seqno)
            recovered_font = recover_font(glyph.font_name)
            if recovered_font is not None:
                x0 = glyph.advance_bbox[0]
                for code in override:
                    width_index = code - recovered_font.first_char
                    if not 0 <= width_index < len(recovered_font.widths):
                        continue
                    x1 = x0 + recovered_font.widths[width_index] * glyph.font_size * 0.001
                    character = _Character(
                        (x0, glyph_box[1], x1, glyph_box[3]),
                        recovered_font.cmap.decode(bytes((code,))),
                        glyph.seqno,
                        glyph.fill,
                    )
                    if occluded(character, rectangles, rectangle_grid):
                        characters.append(character)
                    x0 = x1
                continue
        text = glyph.text
        if any(ord(character) < 32 for character in text):
            recovered_font = recover_font(glyph.font_name)
            if recovered_font is not None:
                text = recovered_font.cmap.decode(glyph.code_bytes)
                code = glyph.char_code
                width_index = code - recovered_font.first_char if code is not None else -1
                if 0 <= width_index < len(recovered_font.widths):
                    position_key = (glyph.font_name, glyph.seqno)
                    x0 = recovered_positions.get(position_key, glyph.advance_bbox[0])
                    x1 = x0 + recovered_font.widths[width_index] * glyph.font_size * 0.001
                    recovered_positions[position_key] = x1
                    glyph_box = (x0, glyph_box[1], x1, glyph_box[3])
        character = _Character(glyph_box, text, glyph.seqno, glyph.fill)
        if glyph.font_size * user_unit == 1.0:
            is_occluded = occluded(character, non_annotation_rectangles, non_annotation_grid)
        else:
            is_occluded = occluded(character, rectangles, rectangle_grid)
        if is_occluded:
            characters.append(character)

    redactions: list[dict[str, object]] = []
    character_grid = _BoxGrid([character.bbox for character in characters])
    taken = [False] * len(characters)
    for rectangle in sorted(rectangles, key=lambda item: item.seqno, reverse=True):
        covered = []
        for index in character_grid.candidates(rectangle.bbox):
            if not taken[index] and bbox_intersects(characters[index].bbox, rectangle.bbox):
                taken[index] = True
                covered.append(characters[index])
        text = "".join(character.text for character in covered)
        if len(text) > 1 and len(set(text)) == 1:
            continue
        if not text.strip() or re.search(r"[\d\w]", text) is None:
            continue
        if not _OK_WORDS.sub("", collapse_ws(text)):
            continue
        fitz_box = rectangle.fitz_box
        if rectangle.bbox in annotation_boxes:
            fitz_box = (*fitz_box[:2], _next_float32(fitz_box[2]), fitz_box[3])
        outside_page = (
            fitz_box[2] <= 0
            or fitz_box[3] <= 0
            or fitz_box[0] >= (crop_box[2] - crop_box[0]) * user_unit
            or fitz_box[1] >= (crop_box[3] - crop_box[1]) * user_unit
        )
        # Edge pixels are only partly covered by a fractional rectangle, so they are not
        # evidence of visible text.
        raster_box = (
            rectangle.bbox[0] + 1.0,
            rectangle.bbox[1] + 1.0,
            rectangle.bbox[2] - 1.0,
            rectangle.bbox[3] - 1.0,
        )
        if not page.rotation and not outside_page and not _uniform(raster_page, raster_box):
            continue
        center = (
            (rectangle.bbox[0] + rectangle.bbox[2]) * 0.5,
            (rectangle.bbox[1] + rectangle.bbox[3]) * 0.5,
        )
        later_nonrect_overlay = False
        for drawing in drawings:
            if (
                drawing.seqno <= rectangle.seqno
                or drawing.kind not in {"fill", "fillstroke"}
                or drawing.rect is None
                or not (
                    drawing.rect[0] <= center[0] <= drawing.rect[2]
                    and drawing.rect[1] <= center[1] <= drawing.rect[3]
                )
            ):
                continue
            drawing_rectangles = _path_rectangles(drawing, crop_box, user_unit, rect_operands)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            if not any(
                candidate.bbox[0] <= center[0] <= candidate.bbox[2]
                and candidate.bbox[1] <= center[1] <= candidate.bbox[3]
                for candidate in drawing_rectangles
            ):
                later_nonrect_overlay = True
                break
        if later_nonrect_overlay:
            continue
        redactions.append(
            {
                "bbox": fitz_box,
                "text": text,
            }
        )
    return redactions


def _validate_mupdf_structure(document: PdfDocument) -> None:
    raw_data = bytes(document.raw_data)
    eof = raw_data.rfind(b"%%EOF")
    if eof >= 0:
        trailing = raw_data[eof + 5 :]
        if (
            b"%PDF-" in trailing
            or re.search(rb"\d+\s+\d+\s+obj\s*$", trailing)
            or re.search(rb"[0-9A-Fa-f]{64}\"[0-9A-Fa-f]{64}", trailing)
        ):
            raise PdfUnsupportedError("invalid trailing PDF data")
    if any(
        not raw_data.startswith(b"<<", _skip_whitespace(raw_data, end))
        for end in _trailer_keyword_ends(raw_data)
    ):
        raise PdfUnsupportedError("invalid trailer dictionary")
    if eof < 0:
        for end in _trailer_keyword_ends(raw_data):
            # The pattern has no anchor, so searching from the end is searching the rest.
            root = TRAILER_ROOT.search(raw_data, end)
            if root is not None and re.match(rb"\d+\s+\d+\s+R\b", root.group(1)) is None:
                raise PdfUnsupportedError("invalid trailer root")
        if re.search(rb"\xff{32,}", raw_data[-256:]):
            raise PdfUnsupportedError("invalid binary trailer data")


def _skip_whitespace(raw_data: bytes, position: int) -> int:
    """Where bytes.lstrip() would leave raw_data[position:] starting."""
    while position < len(raw_data) and raw_data[position] in ASCII_WHITESPACE:
        position += 1
    return position


def _raw_highlight_redactions(page: Any, recovery: _DocumentRecovery) -> list[dict[str, object]]:
    highlights = [
        tuple(annotation.rect)
        for annotation in page.get_annotations()
        if annotation.subtype == "Highlight"
    ]
    if not highlights:
        return []
    raw_data = bytes(page.document.raw_data)
    resources = page.resources
    resource_fonts = resources.get("Font")
    font_names = (
        {str(name) for name in resource_fonts} if isinstance(resource_fonts, dict) else set()
    )
    font_match = next(
        (
            match
            for match in re.finditer(rb"/([A-Za-z0-9_.-]+)\s+[-+.\d]+\s+Tf\b", raw_data)
            if match.group(1).decode("latin-1") in font_names
        ),
        None,
    )
    if font_match is None:
        return []
    font_name = font_match.group(1).decode("latin-1")
    recovered_font = recovery.font(page, font_name)
    if recovered_font is None:
        return []
    begin = raw_data.rfind(b"BT", 0, font_match.start())
    end = raw_data.find(b"ET", font_match.end())
    if begin < 0 or end < 0:
        return []
    stream = raw_data[begin:end]
    token_pattern = re.compile(
        rb"([-+.\d]+)\s+([-+.\d]+)\s+Td\b|"
        rb"/([A-Za-z0-9_.-]+)\s+([-+.\d]+)\s+Tf\b|"
        rb"((?:<[0-9A-Fa-f]+>)+)\s*Tj\b"
    )
    x = y = line_x = line_y = 0.0
    font_size = 0.0
    font_active = False
    recovered: list[_Character] = []
    for token in token_pattern.finditer(stream):
        if token.group(1) is not None:
            line_x += float(token.group(1))
            line_y += float(token.group(2))
            x, y = line_x, line_y
            continue
        if token.group(3) is not None:
            font_active = token.group(3).decode("latin-1") == font_name
            font_size = float(token.group(4))
            continue
        if not font_active or token.group(5) is None:
            continue
        operands = re.findall(rb"<([0-9A-Fa-f]+)>", token.group(5))
        if not operands:
            continue
        codes = bytes.fromhex(operands[-1].decode())
        for code in codes:
            width_index = code - recovered_font.first_char
            if not 0 <= width_index < len(recovered_font.widths):
                continue
            x1 = x + recovered_font.widths[width_index] * font_size * 0.001
            recovered.append(
                _Character(
                    (x, y - font_size * 0.2, x1, y + font_size * 0.8),
                    recovered_font.cmap.decode(bytes((code,))),
                    0,
                    None,
                )
            )
            x = x1
    crop_box = tuple(float(value) for value in (page.crop_box or page.media_box))
    output: list[dict[str, object]] = []
    for highlight in highlights:
        text = "".join(
            character.text for character in recovered if bbox_intersects(character.bbox, highlight)
        )
        if text.strip():
            fitz_box = _fitz_box(highlight, crop_box, page.user_unit)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            output.append(
                {
                    "bbox": (*fitz_box[:2], _next_float32(fitz_box[2]), fitz_box[3]),
                    "text": text,
                }
            )
    return output


def _source_bytes(source: object) -> bytes | None:
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, PathLike)):
        try:
            return Path(source).read_bytes()
        except OSError:
            return None
    return None


# x-ray opens the file with PyMuPDF, whose errors are RuntimeErrors of its own. These stand in
# for them under the same names and messages, so code catching them by built-in class still does.
class FileDataError(RuntimeError):
    """PyMuPDF cannot open the file."""


class FileNotFoundError(RuntimeError):  # noqa: A001
    """PyMuPDF's: the named file does not exist."""


class EmptyFileError(FileDataError):
    """PyMuPDF's: the file or stream is empty."""


def _check_source(source: object) -> None:
    """Raise what PyMuPDF's Document raises before it reads a missing, odd or empty source."""
    if isinstance(source, bytes):
        if not source:
            raise EmptyFileError("Cannot open empty stream.")
        return
    if not isinstance(source, (str, PathLike)):
        return
    filename = str(source)
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"no such file: '{filename}'")
    if not path.is_file():
        raise FileDataError(f"'{filename}' is no file")
    if path.stat().st_size == 0:
        raise EmptyFileError(f"Cannot open empty file: {filename=}.")


def _open_failure(source: object) -> FileDataError:
    """The error PyMuPDF's Document raises for a source it cannot open."""
    if isinstance(source, bytes):
        return FileDataError("Failed to open stream")
    return FileDataError(f"Failed to open file {str(source)!r}.")


def _open(source: Any) -> XrayDocument | None:
    """The document as PyMuPDF opens it, None when it opens with nothing to inspect."""
    _check_source(source)
    try:
        document = XrayDocument.open(source)
    except PdfUnsupportedError as error:
        # A page tree inside object streams is out of MuPDF's reach when it cannot read them,
        # and the document opens with no pages.
        raw_data = _source_bytes(source)
        if (
            raw_data is not None
            and re.search(rb"/Type\s*/ObjStm\b", raw_data)
            and not re.search(rb"/Type\s*/Page\b", raw_data)
        ):
            return None
        if isinstance(error, PdfPasswordError):
            # PyMuPDF opens a file it cannot decrypt, then refuses to load its pages.
            raise ValueError("document closed or encrypted") from None
        raise _open_failure(source) from error
    try:
        _validate_mupdf_structure(document)
        if document.trailer_dict.get("Root") is None:
            # MuPDF fails to open a file in which repair finds no catalog.
            raise PdfUnsupportedError("missing catalog root")
    except PdfUnsupportedError as error:
        document.close()
        raise _open_failure(source) from error
    return document


# x-ray 0.4.0 downloads an https URL with httpx2, whose errors are classes of its own. These
# stand in for the ones a download raises, under the same names, bases and messages.
class HTTPError(Exception):
    """httpx2's: the base of its request and response errors."""


class HTTPStatusError(HTTPError):
    """httpx2's: the server answered with a status that is not a success."""


class RequestError(HTTPError):
    """httpx2's: the request could not be sent or answered."""


class TransportError(RequestError):
    """httpx2's: the connection failed."""


class TimeoutException(TransportError):
    """httpx2's: the request took longer than its timeout."""


class NetworkError(TransportError):
    """httpx2's: the network failed while connecting or reading."""


class ConnectError(NetworkError):
    """httpx2's: the connection could not be made."""


_STATUS_ERROR_TYPES = {
    1: "Informational response",
    3: "Redirect response",
    4: "Client error",
    5: "Server error",
}


def _status_message(status: int, reason: str, url: str) -> str:
    """The message httpx2's raise_for_status gives an unsuccessful response."""
    error_type = _STATUS_ERROR_TYPES.get(status // 100, "Invalid status code")
    return (
        f"{error_type} '{status} {reason}' for url '{url}'\n"
        f"For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/{status}"
    )


def _download(url: str) -> bytes:
    """GET the URL as x-ray 0.4.0 does: following redirects, a 10-second timeout, success only."""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310
            return response.read()
    except urllib.error.HTTPError as error:
        try:
            reason = HTTPStatus(error.code).phrase
        except ValueError:
            reason = str(error.reason)
        raise HTTPStatusError(_status_message(error.code, reason, error.url)) from None
    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise TimeoutException(str(error.reason)) from None
        raise ConnectError(str(error.reason)) from None
    except TimeoutError as error:
        raise TimeoutException(str(error)) from None


# Ported from x-ray 0.4.0's inspect and cli, Copyright (c) Free Law Project, under the BSD
# 2-Clause License. The PDF is read with core-pdf rather than PyMuPDF.
async def inspect(file: str | bytes | Path) -> dict[int, list[dict[str, object]]]:
    """
    Inspect a file for bad redactions and return a Dict with their info

    URL downloads are asynchronous; PDF analysis runs synchronously.

    :file: The PDF to process, as bytes if you have the file in memory (useful
    if it's coming from the network), as a unicode string if you know the
    path to the file on your local disk, or as a pathlib.Path object.
    :return: A dict with the bad redaction information. If no bad redactions
    are found, returns an empty dict.
    """
    if isinstance(file, str) and file.startswith("https://"):
        file = await asyncio.to_thread(_download, file)
    return _inspect(file)


async def cli(args: list[str] | None = None) -> None:
    """Process command line arguments."""
    if not args:
        args = sys.argv[1:]
    file = args[0]
    print(json.dumps(await inspect(file), indent=2))  # noqa: T201


def _inspect(source: Any) -> dict[int, list[dict[str, object]]]:
    output: dict[int, list[dict[str, object]]] = {}
    document = _open(source)
    if document is None:
        return {}
    with document:
        recovery = _DocumentRecovery(document)
        try:
            for page in document.pages:
                try:
                    redactions = _page_redactions(page, recovery)
                except TTLibError:
                    redactions = _raw_highlight_redactions(page, recovery)
                except KeyError, TypeError, ValueError:
                    redactions = []
                if redactions:
                    output[page.page_number] = redactions
        except RecursionError, TTLibError:
            return {}
    if output and all(
        not _DATE.sub("", str(item["text"])) for findings in output.values() for item in findings
    ):
        return {}
    return output


__all__ = ("cli", "inspect")
