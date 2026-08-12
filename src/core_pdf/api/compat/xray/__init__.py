from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from math import ceil, floor
from os import PathLike
from pathlib import Path
from typing import Any

from core_pdf import PdfDocument
from core_pdf._vendor.fontTools.ttLib import TTLibError
from core_pdf.impl.engine.layout.geometry import flip_rect_vertical
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.engine.spec.s_09_fonts.cmap_tounicode import ToUnicodeCMap
from core_pdf.impl.exceptions import PdfUnsupportedError
from core_pdf.impl.objects import PdfReference, PdfStream

_DATE = re.compile(r"[0-3]?\d[/\-][0-3]?\d[/\-]\d{2,4}")
_OK_WORDS = re.compile(
    r"confidential|name +redacted|privileged?|re|red|reda|redac|redact|"
    r"redacte|redacted|redacted +and +publicly +filed|",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(slots=True)
class _Rectangle:
    bbox: tuple[float, float, float, float]
    seqno: int
    fill: tuple[float, ...]
    allow_same_fill: bool = False


@dataclass(slots=True)
class _Character:
    bbox: tuple[float, float, float, float]
    text: str
    seqno: int
    fill: tuple[float, ...] | None


@dataclass(slots=True)
class _RecoveredFont:
    cmap: ToUnicodeCMap
    first_char: int
    widths: tuple[float, ...]


def _intersection_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _occluded(character: _Character, rectangle: _Rectangle, threshold: float) -> bool:
    overlap = _intersection_area(character.bbox, rectangle.bbox)
    if overlap <= 0 or not (
        rectangle.seqno > character.seqno
        or (rectangle.allow_same_fill and rectangle.fill == character.fill)
    ):
        return False
    area = max(0.0, character.bbox[2] - character.bbox[0]) * max(
        0.0, character.bbox[3] - character.bbox[1]
    )
    return bool(area and overlap / area > threshold)


def _path_rectangles(drawing: Any, crop_box: tuple[float, float, float, float]) -> list[_Rectangle]:
    if (
        drawing.kind not in {"fill", "fillstroke"}
        or drawing.fill is None
        or drawing.fill_opacity != 1
        or drawing.path is None
    ):
        return []
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
        top_box = _fitz_box(bbox, crop_box)
        if top_box[3] <= 43 or top_box[2] - top_box[0] <= 4 or top_box[3] - top_box[1] <= 4:
            continue
        outside_page = top_box[2] <= 0 or top_box[3] <= 0
        same_color_text_is_hidden = drawing.fill == (1.0, 1.0, 1.0)
        output.append(
            _Rectangle(bbox, drawing.seqno, drawing.fill, outside_page or same_color_text_is_hidden)
        )
    return output


def _uniform(page: Any, box: tuple[float, float, float, float]) -> bool:
    raster = page.render().rasterize(scale=1.0, crop=box)
    pixels = memoryview(raster.pixels).cast("B")
    if not pixels or raster.channels <= 0:
        return False
    first = pixels[: raster.channels].tobytes()
    return all(
        pixels[index : index + raster.channels].tobytes() == first
        for index in range(0, len(pixels), raster.channels)
    )


def _pixmap_crop(
    box: tuple[float, float, float, float], page_height: float
) -> tuple[float, float, float, float]:
    top_box = flip_rect_vertical(box, page_height)
    pixel_box = (
        float(floor(top_box[0])) - 1.0,
        float(floor(top_box[1])) - 1.0,
        float(ceil(top_box[2])) + 1.0,
        float(ceil(top_box[3])) + 1.0,
    )
    return flip_rect_vertical(pixel_box, page_height)


def _fitz_box(
    box: tuple[float, float, float, float],
    crop_box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Convert engine PDF coordinates to PyMuPDF's cropped, top-origin space."""
    crop_x0, _crop_y0, _crop_x1, crop_y1 = crop_box
    crop_x0 = _fitz_coordinate(crop_x0)
    crop_y1 = _fitz_coordinate(crop_y1)
    return (
        _float32(_fitz_coordinate(box[0]) - crop_x0),
        _float32(crop_y1 - _fitz_coordinate(box[3])),
        _float32(_fitz_coordinate(box[2]) - crop_x0),
        _float32(crop_y1 - _fitz_coordinate(box[1])),
    )


def _fitz_coordinate(value: float) -> float:
    millipoint = round(value, 3)
    if abs(value - millipoint) < 0.00005:
        value = millipoint
    return _float32(value)


def _float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _next_float32(value: float) -> float:
    bits = struct.unpack("I", struct.pack("f", value))[0]
    return struct.unpack("f", struct.pack("I", bits + 1))[0]


def _recover_font(page: Any, font_name: str) -> _RecoveredFont | None:
    """Recover a CMap from raw objects when a damaged xref hides the font resource."""
    document = page.document
    raw_data = bytes(document.raw_data)
    pattern = rb"/" + re.escape(font_name.encode("latin-1")) + rb"\s+(\d+)\s+(\d+)\s+R\b"
    references = reversed(re.findall(pattern, raw_data))
    for object_number, generation_number in references:
        object_header = object_number + rb"\s+" + generation_number + rb"\s+obj\b"
        match = re.search(object_header, raw_data)
        if match is None:
            continue
        lexer = PdfLexer(
            document.raw_data,
            reference_resolver=document.resolver.resolve,
            decipher=document.decipher,
        )
        try:
            lexer.rewind(match.start())
            font = lexer.parse_indirect_object()
        except (TypeError, ValueError):
            continue
        finally:
            lexer.close()
        if not isinstance(font, dict):
            continue
        to_unicode = lookup_dict_key(font, "ToUnicode")
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
        lexer = PdfLexer(
            document.raw_data,
            reference_resolver=document.resolver.resolve,
            decipher=document.decipher,
        )
        try:
            lexer.rewind(stream_match.start())
            stream = lexer.parse_indirect_object()
            if isinstance(stream, PdfStream):
                first_char = lookup_dict_key(font, "FirstChar")
                widths = lookup_dict_key(font, "Widths")
                if not isinstance(first_char, int) or not isinstance(widths, list):
                    continue
                return _RecoveredFont(
                    ToUnicodeCMap(stream.data),
                    first_char,
                    tuple(float(width) for width in widths),
                )
        except (TypeError, ValueError):
            continue
        finally:
            lexer.close()
    return None


def _page_redactions(page: Any) -> list[dict[str, object]]:
    source_crop_box = page.crop_box or page.media_box
    crop_box = tuple(float(value) for value in source_crop_box)
    drawings = tuple(page.get_drawings())
    rectangles = [
        rectangle for drawing in drawings for rectangle in _path_rectangles(drawing, crop_box)
    ]
    if not rectangles:
        return []
    annotations = page.get_annotations()
    annotation_boxes = {
        tuple(annotation.rect) for annotation in annotations if annotation.subtype == "Highlight"
    }
    widget_boxes = {
        tuple(annotation.rect) for annotation in annotations if annotation.subtype == "Widget"
    }
    recovered_fonts: dict[str, _RecoveredFont | None] = {}
    recovered_positions: dict[tuple[str, int], float] = {}
    raw_data = bytes(page.document.raw_data)
    operand_overrides = {
        bytes.fromhex(groups[0].decode()): bytes.fromhex(groups[-1].decode())
        for match in re.finditer(rb"(?:<[0-9A-Fa-f]+>){2,}\s*Tj\b", raw_data)
        if len(groups := re.findall(rb"<([0-9A-Fa-f]+)>", match.group())) > 1
    }
    glyphs = tuple(page.get_page_program().products.glyphs)
    sequence_codes: dict[int, bytes] = {}
    for glyph in glyphs:
        sequence_codes[glyph.seqno] = sequence_codes.get(glyph.seqno, b"") + glyph.code_bytes
    overridden_sequences = {
        seqno: operand_overrides[codes]
        for seqno, codes in sequence_codes.items()
        if codes in operand_overrides
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
            recovered_font = recovered_fonts.setdefault(
                glyph.font_name,
                _recover_font(page, glyph.font_name),
            )
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
                    if any(_occluded(character, rectangle, 0.8) for rectangle in rectangles):
                        characters.append(character)
                    x0 = x1
                continue
        text = glyph.text
        if any(ord(character) < 32 for character in text):
            recovered_font = recovered_fonts.setdefault(
                glyph.font_name,
                _recover_font(page, glyph.font_name),
            )
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
        matching_rectangles = [
            rectangle
            for rectangle in rectangles
            if not (rectangle.bbox in annotation_boxes and glyph.font_size == 1.0)
        ]
        if any(_occluded(character, rectangle, 0.8) for rectangle in matching_rectangles):
            characters.append(character)

    redactions: list[dict[str, object]] = []
    remaining = characters.copy()
    for rectangle in sorted(rectangles, key=lambda item: item.seqno, reverse=True):
        covered = [
            character
            for character in remaining
            if _intersection_area(character.bbox, rectangle.bbox) > 0
        ]
        for character in covered:
            remaining.remove(character)
        text = "".join(character.text for character in covered)
        if len(text) > 1 and len(set(text)) == 1:
            continue
        if not text.strip() or re.search(r"[\d\w]", text) is None:
            continue
        if not _OK_WORDS.sub("", " ".join(text.strip().split())):
            continue
        fitz_box = _fitz_box(rectangle.bbox, crop_box)
        if rectangle.bbox in annotation_boxes:
            fitz_box = (*fitz_box[:2], _next_float32(fitz_box[2]), fitz_box[3])
        integer_aligned = all(abs(value * 2 - round(value * 2)) < 0.0001 for value in fitz_box)
        is_widget = rectangle.bbox in widget_boxes
        widget_has_later_content = is_widget and any(
            glyph.seqno >= rectangle.seqno
            and glyph.ink_bbox is not None
            and _intersection_area(glyph.ink_bbox, rectangle.bbox) > 0
            for glyph in page.get_page_program().products.glyphs
        )
        outside_page = (
            fitz_box[2] <= 0
            or fitz_box[3] <= 0
            or fitz_box[0] >= crop_box[2] - crop_box[0]
            or fitz_box[1] >= crop_box[3] - crop_box[1]
        )
        raster_box = (
            rectangle.bbox
            if page.rotation
            or (integer_aligned and not widget_has_later_content)
            or (is_widget and not widget_has_later_content)
            else _pixmap_crop(rectangle.bbox, float(page.height))
        )
        if not page.rotation and not outside_page and not _uniform(page, raster_box):
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
            drawing_rectangles = _path_rectangles(drawing, crop_box)
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
    """Apply the strict top-level syntax assumptions made by MuPDF."""
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
        not raw_data[match.end() :].lstrip().startswith(b"<<")
        for match in re.finditer(rb"\btrailer\b", raw_data)
    ):
        raise PdfUnsupportedError("invalid trailer dictionary")
    if eof < 0:
        for match in re.finditer(rb"\btrailer\b", raw_data):
            trailer_data = raw_data[match.end() :]
            root = re.search(rb"/Root\s+([^>]+)", trailer_data)
            if root is not None and re.match(rb"\d+\s+\d+\s+R\b", root.group(1)) is None:
                raise PdfUnsupportedError("invalid trailer root")
        if re.search(rb"\xff{32,}", raw_data[-256:]):
            raise PdfUnsupportedError("invalid binary trailer data")


def _raw_highlight_redactions(page: Any) -> list[dict[str, object]]:
    """Recover simple hidden text when a corrupt embedded font aborts page capture."""
    highlights = [
        tuple(annotation.rect)
        for annotation in page.get_annotations()
        if annotation.subtype == "Highlight"
    ]
    if not highlights:
        return []
    raw_data = bytes(page.document.raw_data)
    resources = page.resolve_resources()
    resource_fonts = lookup_dict_key(resources, "Font")
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
    recovered_font = _recover_font(page, font_name)
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
            character.text
            for character in recovered
            if _intersection_area(character.bbox, highlight) > 0
        )
        if text.strip():
            fitz_box = _fitz_box(highlight, crop_box)
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
    if isinstance(source, (str, PathLike)) and not str(source).startswith("https://"):
        try:
            return Path(source).read_bytes()
        except OSError:
            return None
    return None


def _requires_password(document: PdfDocument) -> bool:
    """Match MuPDF's implicit empty-user-password authentication for AES-256 files."""
    if document.decipher is None:
        return False
    handler = document.decipher.__self__
    if getattr(handler, "r", 0) < 5:
        return False
    empty_hash = handler.password_hash(b"", handler.u_validation_salt)
    return empty_hash != handler.u_hash


def inspect(source: Any) -> dict[int, list[dict[str, object]]]:
    """Return x-ray-shaped bad-redaction findings from engine evidence."""
    output: dict[int, list[dict[str, object]]] = {}
    try:
        document = PdfDocument.open(source)
    except PdfUnsupportedError:
        raw_data = _source_bytes(source)
        if (
            raw_data is not None
            and re.search(rb"/Type\s*/ObjStm\b", raw_data)
            and not re.search(rb"/Type\s*/Page\b", raw_data)
        ):
            return {}
        raise
    with document:
        _validate_mupdf_structure(document)
        if _requires_password(document):
            raise PdfUnsupportedError("document closed or encrypted")
        try:
            for page in document.pages:
                try:
                    redactions = _page_redactions(page)
                except TTLibError:
                    redactions = _raw_highlight_redactions(page)
                except (KeyError, TypeError, ValueError):
                    redactions = []
                if redactions:
                    output[page.page_number] = redactions
        except (RecursionError, TTLibError):
            return {}
    if output and all(
        not _DATE.sub("", str(item["text"])) for findings in output.values() for item in findings
    ):
        return {}
    return output


__all__ = ("inspect",)
