"""High-level PyMuPDF-shaped facade backed by core-pdf."""

from __future__ import annotations

import json
import struct
import zlib
from collections.abc import Mapping
from dataclasses import replace
from html import escape
from os import PathLike
from pathlib import Path
from typing import Any, cast

from core_pdf.api.compat.pypdf import (
    PdfInput,
    PdfPageObject,
    PdfReader,
    StructuredState,
)
from core_pdf.impl.engine.layout.geometry import bbox_intersects, rect_tuple
from core_pdf.impl.engine.structured import (
    Annotation,
    Block,
    BlockKind,
    Figure,
    FormField,
    Link,
    TextLine,
)
from core_pdf.impl.engine.structured import (
    Page as StructuredPage,
)

BBox = tuple[float, float, float, float]


def _float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def coerce_bbox(value: object) -> BBox:
    box = rect_tuple(value)
    if box is None:
        raise ValueError(f"value does not describe a rectangle: {value!r}")
    return box


def synthesize_characters(text: str, box: BBox) -> list[tuple[str, BBox]]:
    x0, y0, x1, y1 = box
    width = (x1 - x0) / max(1, len(text))
    return [
        (character, (x0 + index * width, y0, x0 + (index + 1) * width, y1))
        for index, character in enumerate(text)
    ]


def write_bytes(target: str | PathLike[str] | Any, data: bytes) -> None:
    if isinstance(target, (str, PathLike)):
        Path(target).write_bytes(data)
    else:
        target.write(data)


def internal_png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def encode_png(width: int, height: int, channels: int, pixels: bytes) -> bytes:
    if channels not in (3, 4):
        raise ValueError("PNG output requires RGB or RGBA pixels")
    stride = width * channels
    scanlines = b"".join(
        b"\x00" + pixels[row * stride : (row + 1) * stride] for row in range(height)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 6 if channels == 4 else 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + internal_png_chunk(b"IHDR", header)
        + internal_png_chunk(b"IDAT", zlib.compress(scanlines))
        + internal_png_chunk(b"IEND", b"")
    )


class Matrix:
    """Small PyMuPDF-compatible scale matrix for raster requests."""

    def __init__(self, sx: float = 1.0, sy: float | None = None, *_: float) -> None:
        self.a = float(sx)
        self.d = float(sx if sy is None else sy)


class Pixmap:
    def __init__(self, data: bytes, width: int, height: int, channels: int, dpi: float) -> None:
        self.samples = data
        self.width = width
        self.height = height
        self.n = channels
        self.alpha = channels == 4
        self.xres = dpi
        self.yres = dpi

    def tobytes(self, output: str = "png", *args: object, **kwargs: object) -> bytes:
        del args, kwargs
        if output.casefold() != "png":
            raise ValueError("only PNG pixmaps are supported")
        return encode_png(self.width, self.height, self.n, self.samples)

    def save(self, filename: object, output: str = "png", **kwargs: object) -> None:
        del kwargs
        write_bytes(cast(Any, filename), self.tobytes(output))


class Annot:
    def __init__(
        self, document: "Document", page_number: int, annotation: Annotation, index: int
    ) -> None:
        self._document = document
        self._page_number = page_number
        self._annotation = annotation
        self._index = index
        self.rect = annotation.bbox
        self.type = (annotation.subtype, annotation.subtype)
        self.info: dict[str, object] = {"content": annotation.contents}

    def update(self) -> None:
        self._document._replace_annotation(
            self._page_number,
            self._index,
            replace(
                self._annotation,
                bbox=self.rect,
                contents=str(self.info.get("content", "")),
            ),
        )

    def set_info(self, info: dict[str, object]) -> None:
        self.info.update(info)
        self.update()

    def set_rect(self, rect: tuple[float, float, float, float]) -> None:
        self.rect = coerce_bbox(rect)
        self.update()


class Widget:
    def __init__(
        self, document: "Document", page_number: int, field: FormField, index: int
    ) -> None:
        self._document = document
        self._page_number = page_number
        self._index = index
        self._field = field
        self.field_name = field.name
        self.field_type = field.field_type
        self.field_value = field.value_text
        self.rect = field.bbox

    def update(self) -> bool:
        self._document._replace_form_field(
            self._page_number,
            self._index,
            replace(self._field, value_text=str(self.field_value), bbox=self.rect),
        )
        return True


class Page(PdfPageObject):
    def __init__(
        self, document: StructuredState, page: Any, owner: "Document | None" = None
    ) -> None:
        super().__init__(document, page)
        self._owner = owner

    @property
    def rect(self) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = self.cropbox
        return (0.0, 0.0, _float32(x1 - x0), _float32(y1 - y0))

    def _native_text_view(self) -> Any:
        """Return only text represented by PDF text operators, as MuPDF does by default."""
        view = self._page.text_view
        blocks: list[Block] = []
        for block in view.blocks:
            lines = tuple(line for line in block.lines if line.source != "ocr")
            if lines:
                blocks.append(replace(block, lines=lines))
        return type(view)(tuple(blocks), page_number=self._page.page_number)

    def _mupdf_plain_text(self) -> str:
        runs = (
            self._document.capability_page(self._page.page_number).get_page_program().products.runs
        )
        output = ""
        previous: Any | None = None
        for run in runs:
            if not run.text:
                continue
            if previous is not None:
                current_height = max(
                    run.advance_bbox[3] - run.advance_bbox[1], run.coords[run.FONT_SIZE]
                )
                previous_height = max(
                    previous.advance_bbox[3] - previous.advance_bbox[1],
                    previous.coords[previous.FONT_SIZE],
                )
                degenerate = (
                    run.advance_bbox[3] == run.advance_bbox[1]
                    or previous.advance_bbox[3] == previous.advance_bbox[1]
                )
                vertically_separate = (
                    run.baseline is not None
                    and previous.baseline is not None
                    and abs(run.baseline[1] - previous.baseline[1])
                    > (
                        max(current_height, previous_height)
                        if degenerate
                        else min(current_height, previous_height)
                    )
                    * 0.8
                )
                if vertically_separate and "\f" not in run.text and not output.endswith("\x00"):
                    output += "\n"
                elif (
                    not output.endswith((" ", "\n"))
                    and not run.text.startswith(" ")
                    and run.advance_bbox[0] - previous.advance_bbox[2]
                    > (run.advance_bbox[3] - run.advance_bbox[1]) * 0.2
                ):
                    output += " "
            if not output and run.text.startswith(" "):
                output = " \n" + run.text[1:]
            else:
                output += run.text
            previous = run
        return output + ("\n" if output else "")

    def _mupdf_words(self) -> list[tuple[float, float, float, float, str, int, int, int]]:
        page = self._document.capability_page(self._page.page_number)
        products = page.get_page_program().products
        crop_box = page.crop_box or page.media_box
        crop_x0, _crop_y0, _crop_x1, crop_y1 = crop_box
        glyphs_by_sequence: dict[int, list[Any]] = {}
        for glyph in products.glyphs:
            glyphs_by_sequence.setdefault(glyph.seqno, []).append(glyph)
        indexed_words = list(self._native_text_view().words)
        search_from = 0
        output: list[tuple[float, float, float, float, str, int, int, int]] = []
        fallback_block = 0
        previous_raw_x1: float | None = None
        previous_word_y: tuple[float, float] | None = None
        previous_baseline: float | None = None
        previous_mupdf_baseline: float | None = None
        for run_index, run in enumerate(products.runs):
            if not run.text or not run.visible or run.baseline is None:
                continue
            font_size = run.coords[run.FONT_SIZE]
            ascent = (run.advance_bbox[3] - run.baseline[1]) / font_size
            descent = (run.baseline[1] - run.advance_bbox[1]) / font_size
            metric_height = ascent + descent
            baseline = run.baseline[1]
            mupdf_baseline = (
                _float32(baseline)
                if previous_baseline is None or previous_mupdf_baseline is None
                else _float32(previous_mupdf_baseline + _float32(baseline - previous_baseline))
            )
            previous_baseline = baseline
            previous_mupdf_baseline = mupdf_baseline
            next_sequence = (
                products.runs[run_index + 1].seqno
                if run_index + 1 < len(products.runs)
                else float("inf")
            )
            run_glyphs = [
                glyph
                for sequence, sequence_glyphs in glyphs_by_sequence.items()
                if run.seqno <= sequence < next_sequence
                for glyph in sequence_glyphs
            ]
            origin_y = _float32(_float32(float(crop_y1)) - mupdf_baseline)
            if metric_height:
                y0 = _float32(origin_y - _float32(font_size * _float32(ascent / metric_height)))
                y1 = _float32(origin_y + _float32(font_size * _float32(descent / metric_height)))
            else:
                y0 = _float32(crop_y1 - max(glyph.ink_bbox[3] for glyph in run_glyphs))
                y1 = _float32(crop_y1 - min(glyph.ink_bbox[1] for glyph in run_glyphs))
            current: list[Any] = []
            for glyph in (*run_glyphs, None):
                if glyph is not None and not glyph.text.isspace():
                    current.append(glyph)
                    continue
                if not current:
                    continue
                text = "".join(item.text for item in current)
                raw_x0 = current[0].advance_bbox[0] - crop_x0
                same_line = previous_word_y == (y0, y1)
                if same_line and previous_raw_x1 is not None and output:
                    x0 = _float32(output[-1][2] + _float32(raw_x0 - previous_raw_x1))
                else:
                    x0 = _float32(raw_x0)
                x1 = x0
                raw_previous_x1 = raw_x0
                for item_index, item in enumerate(current):
                    if item_index:
                        x1 = _float32(
                            x1 + _float32(item.advance_bbox[0] - crop_x0 - raw_previous_x1)
                        )
                    width_units = round(
                        (item.advance_bbox[2] - item.advance_bbox[0]) / font_size * 1000
                    )
                    advance = _float32(
                        _float32(font_size) * _float32(width_units * _float32(0.001))
                    )
                    x1 = _float32(x1 + advance)
                    raw_previous_x1 = item.advance_bbox[2] - crop_x0
                if len({item.seqno for item in current}) > 1:
                    x1 = _float32(current[-1].advance_bbox[2] - crop_x0)
                indices = None
                for index in range(search_from, len(indexed_words)):
                    candidate = indexed_words[index]
                    if candidate.text == text:
                        indices = (
                            candidate.block_index,
                            candidate.line_index,
                            candidate.word_index,
                        )
                        search_from = index + 1
                        break
                if indices is None:
                    if same_line and output:
                        indices = (output[-1][5], output[-1][6], output[-1][7] + 1)
                    else:
                        indices = (fallback_block, 0, 0)
                        fallback_block += 1
                if same_line and output and x0 - output[-1][2] <= (y1 - y0) * 0.1:
                    previous = output[-1]
                    output[-1] = (
                        *previous[:2],
                        x1,
                        previous[3],
                        previous[4] + text,
                        *previous[5:],
                    )
                else:
                    output.append((x0, y0, x1, y1, text, *indices))
                previous_raw_x1 = current[-1].advance_bbox[2] - crop_x0
                previous_word_y = (y0, y1)
                current = []
        indexed: list[tuple[float, float, float, float, str, int, int, int]] = []
        block_index = 0
        line_index = 0
        word_index = 0
        previous_line: tuple[float, float] | None = None
        for x0, y0, x1, y1, text, *_indices in output:
            line = (y0, y1)
            if previous_line is not None and line != previous_line:
                gap = y0 - previous_line[1]
                if gap > min(y1 - y0, previous_line[1] - previous_line[0]) * 0.25:
                    block_index += 1
                    line_index = 0
                else:
                    line_index += 1
                word_index = 0
            indexed.append((x0, y0, x1, y1, text, block_index, line_index, word_index))
            word_index += 1
            previous_line = line
        return indexed

    def get_text(self, kind: str = "text", *args: object, **kwargs: object) -> object:
        del args
        clip = kwargs.pop("clip", None)
        sort = bool(kwargs.pop("sort", False))
        for option in ("flags", "textpage"):
            kwargs.pop(option, None)
        if kwargs:
            raise TypeError(f"unsupported text options: {', '.join(kwargs)}")
        text_view = self._native_text_view()
        if clip is not None:
            clip_bbox = cast(tuple[float, float, float, float], clip)
            elements = tuple(
                element
                for element in text_view.elements
                if element.bbox is not None and bbox_intersects(clip_bbox, element.bbox)
            )
            text_view = type(text_view)(elements, page_number=self._page.page_number)
        else:
            text_view = self._native_text_view()
        text = text_view.text
        if sort:
            text = "\n".join(sorted(text.splitlines(), key=str.casefold))
        if kind in {"text", "plain"}:
            return self._mupdf_plain_text() if clip is None and not sort else text
        if kind == "html":
            return "<div>" + escape(text).replace("\n", "<br>\n") + "</div>"
        if kind == "xhtml":
            return (
                '<div xmlns="http://www.w3.org/1999/xhtml">'
                + escape(text).replace("\n", "<br />\n")
                + "</div>"
            )
        if kind == "xml":
            return "<page>" + escape(text) + "</page>"
        if kind == "blocks":
            return [
                (
                    *(block.bbox or self.mediabox),
                    block.text,
                    index,
                    0,
                )
                for index, block in enumerate(text_view.blocks)
            ]
        if kind in {"dict", "rawdict", "json", "rawjson"}:

            def line_payload(line: Any, fallback: object) -> dict[str, object]:
                bbox = line.bbox or fallback
                payload: dict[str, object] = {"text": line.text}
                if kind in {"rawdict", "rawjson"}:
                    payload["chars"] = [
                        {"c": character, "bbox": box}
                        for character, box in synthesize_characters(
                            line.text, cast(tuple[float, float, float, float], bbox)
                        )
                    ]
                return payload

            blocks = [
                {
                    "type": 0,
                    "number": index,
                    "bbox": block.bbox or self.mediabox,
                    "text": block.text,
                    "lines": [
                        {
                            "bbox": line.bbox or block.bbox or self.mediabox,
                            "spans": [line_payload(line, block.bbox or self.mediabox)],
                        }
                        for line in block.lines
                    ],
                }
                for index, block in enumerate(text_view.blocks)
            ]
            payload = {"width": self._page.width, "height": self._page.height, "blocks": blocks}
            return json.dumps(payload) if kind in {"json", "rawjson"} else payload
        if kind == "words":
            if clip is None:
                return self._mupdf_words()
            words = self._native_text_view().words
            if clip is not None:
                x0, y0, x1, y1 = cast(tuple[float, float, float, float], clip)
                words = tuple(
                    word
                    for word in words
                    if word.bbox is not None
                    and word.bbox[0] < x1
                    and word.bbox[2] > x0
                    and word.bbox[1] < y1
                    and word.bbox[3] > y0
                )
            return [
                (
                    *(word.bbox if word.bbox is not None else (0.0, 0.0, 0.0, 0.0)),
                    word.text,
                    word.block_index,
                    word.line_index,
                    word.word_index,
                )
                for word in words
            ]
        raise ValueError(f"unsupported text extraction kind: {kind}")

    def get_textbox(self, rect: tuple[float, float, float, float], *args: object) -> str:
        del args
        return cast(str, self.get_text("text", clip=rect))

    def get_text_length(self, text: str, fontname: str = "helv", fontsize: float = 11.0) -> float:
        del fontname
        return sum(fontsize * (0.5 if character.isspace() else 0.55) for character in text)

    def insert_text(
        self,
        point: tuple[float, float],
        text: str,
        fontsize: float = 11.0,
        fontname: str = "helv",
        **kwargs: object,
    ) -> int:
        del fontname, kwargs
        if self._owner is None:
            raise RuntimeError("text insertion requires a document-owned page")
        x, y = point
        lines = text.splitlines() or [""]
        line_height = fontsize * 1.2
        line_records = tuple(
            TextLine(
                line,
                break_before=index > 0,
                bbox=(
                    x,
                    y + index * line_height,
                    x + self.get_text_length(line, fontsize=fontsize),
                    y + (index + 1) * line_height,
                ),
            )
            for index, line in enumerate(lines)
        )
        block = Block(
            order=max((item.order for item in self._page.elements), default=-1) + 1,
            kind=BlockKind.PARAGRAPH,
            lines=line_records,
            bbox=(
                x,
                y,
                x
                + max((self.get_text_length(line, fontsize=fontsize) for line in lines), default=0),
                y + line_height * len(lines),
            ),
        )
        self._owner._append_block(self._page.page_number, block)
        return len(text)

    def insert_textbox(
        self,
        rect: tuple[float, float, float, float],
        buffer: str,
        fontsize: float = 11.0,
        fontname: str = "helv",
        **kwargs: object,
    ) -> int:
        del fontname, kwargs
        return self.insert_text((rect[0], rect[1]), buffer, fontsize=fontsize)

    def draw_rect(
        self,
        rect: tuple[float, float, float, float],
        color: tuple[float, ...] | None = None,
        fill: tuple[float, ...] | None = None,
        width: float = 1.0,
        **kwargs: object,
    ) -> "Page":
        del kwargs
        if self._owner is None:
            raise RuntimeError("drawing requires a document-owned page")
        self._owner._append_figure(
            self._page.page_number,
            Figure(
                order=max((item.order for item in self._page.elements), default=-1) + 1,
                bbox=coerce_bbox(rect),
                kind="rect",
                metadata={"color": color, "fill": fill, "width": width},
            ),
        )
        self._page = self._owner._document.pages[self._page.page_number - 1]
        return self

    def draw_line(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        color: tuple[float, ...] | None = None,
        width: float = 1.0,
        **kwargs: object,
    ) -> "Page":
        del kwargs
        if self._owner is None:
            raise RuntimeError("drawing requires a document-owned page")
        self._owner._append_figure(
            self._page.page_number,
            Figure(
                order=max((item.order for item in self._page.elements), default=-1) + 1,
                bbox=(p1[0], p1[1], p2[0], p2[1]),
                kind="line",
                metadata={"p1": p1, "p2": p2, "color": color, "width": width},
            ),
        )
        self._page = self._owner._document.pages[self._page.page_number - 1]
        return self

    def get_textpage(self, *args: object, **kwargs: object) -> "TextPage":
        del args, kwargs
        return TextPage(self)

    def get_pixmap(
        self,
        matrix: Matrix | None = None,
        dpi: float | None = None,
        clip: tuple[float, float, float, float] | None = None,
        alpha: bool = False,
        **kwargs: object,
    ) -> Pixmap:
        del kwargs
        scale = matrix.a if matrix is not None else 1.0
        if matrix is not None and matrix.d != matrix.a:
            raise ValueError("non-uniform pixmap matrices are not supported")
        requested_dpi = float(dpi if dpi is not None else 72.0 * scale)
        engine_page = self._document.capability_page(self._page.page_number)
        raster = engine_page.render().rasterize(scale=max(0.01, requested_dpi / 72.0), crop=clip)
        data = bytes(raster.pixels)
        if not alpha and raster.channels == 4:
            data = b"".join(data[index : index + 3] for index in range(0, len(data), 4))
        channels = 3 if not alpha and raster.channels == 4 else raster.channels
        return Pixmap(data, raster.width, raster.height, channels, requested_dpi)

    def search_for(
        self, needle: str, *args: object, **kwargs: object
    ) -> list[tuple[float, float, float, float]]:
        del args
        if not needle:
            return []
        clip = kwargs.get("clip")
        query = needle.casefold()
        results = [
            cast(BBox, item.bbox)
            for item in self._page.elements
            if item.bbox is not None and query in str(getattr(item, "text", "")).casefold()
        ]
        if clip is None:
            return list(results)
        x0, y0, x1, y1 = cast(tuple[float, float, float, float], clip)
        return [
            bbox
            for bbox in results
            if bbox[0] < x1 and bbox[2] > x0 and bbox[1] < y1 and bbox[3] > y0
        ]

    def get_links(self) -> list[dict[str, object]]:
        links = self._page.links
        return [{"uri": link.url, "kind": link.link_type, "from": link.bbox} for link in links]

    def insert_link(self, link: Mapping[str, object]) -> None:
        owner = self._owner
        if owner is None:
            raise RuntimeError("link mutation requires a document-owned page")
        bbox = cast(tuple[float, float, float, float], link.get("from", self.mediabox))
        url = link.get("uri")
        owner._replace_links(
            self._page.page_number,
            (*self._page.links, Link(bbox=bbox, url=str(url) if url is not None else None)),
        )

    def update_link(self, link: Mapping[str, object]) -> None:
        owner = self._owner
        if owner is None:
            raise RuntimeError("link mutation requires a document-owned page")
        bbox = cast(tuple[float, float, float, float], link.get("from", self.mediabox))
        links = list(self._page.links)
        index = next((index for index, item in enumerate(links) if item.bbox == bbox), None)
        if index is None:
            raise ValueError("link was not found")
        url = link.get("uri")
        links[index] = replace(links[index], bbox=bbox, url=str(url) if url is not None else None)
        owner._replace_links(self._page.page_number, tuple(links))

    def delete_link(self, link: Mapping[str, object]) -> None:
        owner = self._owner
        if owner is None:
            raise RuntimeError("link mutation requires a document-owned page")
        bbox = cast(tuple[float, float, float, float], link.get("from", self.mediabox))
        links = list(self._page.links)
        index = next((index for index, item in enumerate(links) if item.bbox == bbox), None)
        if index is None:
            raise ValueError("link was not found")
        del links[index]
        owner._replace_links(self._page.page_number, tuple(links))

    def get_drawings(self) -> list[dict[str, object]]:
        page = self._document.capability_page(self._page.page_number).structured_view
        figures = self._page.figures or page.figures
        drawings = [
            {"type": figure.kind, "bbox": figure.bbox, **dict(figure.metadata)}
            for figure in figures
        ]
        for annotation in self._page.annotations:
            if annotation.subtype != "CoreFigure":
                continue
            try:
                marker = json.loads(annotation.contents)
            except json.JSONDecodeError:
                continue
            if isinstance(marker, dict):
                drawings.append(
                    {
                        "type": marker.get("kind", "figure"),
                        "bbox": marker.get("bbox", annotation.bbox),
                        **(
                            marker.get("metadata", {})
                            if isinstance(marker.get("metadata", {}), dict)
                            else {}
                        ),
                    }
                )
        return drawings

    def get_images(self, full: bool = False) -> list[dict[str, object]]:
        del full
        images = self._document.capability_page(self._page.page_number).extract_images()
        return [
            {
                "bbox": image.rect or image.image_clip,
                "width": image.image_metadata.width if image.image_metadata else 0,
                "height": image.image_metadata.height if image.image_metadata else 0,
            }
            for image in images
        ]

    def get_image_info(self, hashes: bool = False, xrefs: bool = False) -> list[dict[str, object]]:
        del hashes, xrefs
        images = self._document.capability_page(self._page.page_number).extract_images()
        return [
            {
                "bbox": image.rect or image.image_clip,
                "width": image.image_metadata.width if image.image_metadata else 0,
                "height": image.image_metadata.height if image.image_metadata else 0,
            }
            for image in images
        ]

    def get_image_rects(self, image: object, transform: bool = False) -> list[object]:
        del transform
        return [
            record["bbox"]
            for record in self.get_images(full=True)
            if image in (record.get("xref"), record.get("name"), record.get("id"))
        ]

    def annots(self) -> list[object]:
        document = self._owner
        if document is None:
            raise RuntimeError("annotation mutation requires a document-owned page")
        return [
            Annot(document, self._page.page_number, annotation, index)
            for index, annotation in enumerate(self._page.annotations)
            if annotation.subtype != "CoreFigure"
        ]

    def widgets(self) -> list[Widget]:
        if self._owner is None:
            raise RuntimeError("widget access requires a document-owned page")
        return [
            Widget(self._owner, self._page.page_number, field, index)
            for index, field in enumerate(self._page.form_fields)
        ]

    def _add_annotation(self, subtype: str, rect: tuple[float, float, float, float]) -> Annot:
        owner = self._owner
        if owner is None:
            raise RuntimeError("annotation creation requires a document-owned page")
        annotations = (
            *self._page.annotations,
            Annotation(
                subtype=subtype,
                bbox=coerce_bbox(rect),
            ),
        )
        owner._replace_page_annotations(self._page.page_number, annotations)
        self._page = owner._document.pages[self._page.page_number - 1]
        return Annot(
            owner,
            self._page.page_number,
            self._page.annotations[-1],
            len(self._page.annotations) - 1,
        )

    def add_text_annot(self, point: tuple[float, float], text: str, **kwargs: object) -> Annot:
        del kwargs
        x, y = point
        annot = self._add_annotation("Text", (x, y, x + 20, y + 20))
        annot.info["content"] = text
        annot.update()
        return annot

    def add_highlight_annot(self, rect: tuple[float, float, float, float]) -> Annot:
        return self._add_annotation("Highlight", rect)

    def add_underline_annot(self, rect: tuple[float, float, float, float]) -> Annot:
        return self._add_annotation("Underline", rect)

    def add_rect_annot(self, rect: tuple[float, float, float, float]) -> Annot:
        return self._add_annotation("Square", rect)

    def redact(self, bbox: tuple[float, float, float, float]) -> None:
        pending = getattr(self._document, "_pending_redactions", {})
        pending.setdefault(self._page.page_number, []).append(tuple(bbox))
        cast(Any, self._document)._pending_redactions = pending

    def apply_redactions(self) -> None:
        self._document.apply_redactions()


class Document(PdfReader):
    def __init__(self, stream: PdfInput, password: str | None = None) -> None:
        super().__init__(stream, password)
        self.pages = cast(tuple[Page, ...], self.pages)
        self._pending_redactions: dict[int, list[tuple[float, float, float, float]]] = {}
        self._toc_override: list[list[object]] | None = None
        self._embedded_files = {
            item.filename: item.data for item in self._document.source_pdf.embedded_files()
        }

    def __getitem__(self, index: int) -> Page:
        return Page(self._document, self._document.pages[index], self)

    def load_page(self, index: int) -> Page:
        return self[index]

    def new_page(
        self,
        pno: int = -1,
        width: float = 595.0,
        height: float = 842.0,
    ) -> Page:
        if pno < 0:
            index = len(self._document.pages)
        else:
            index = min(pno, len(self._document.pages))
        blank = StructuredPage(
            page_number=index + 1,
            width=float(width),
            height=float(height),
        )
        pages = (*self._document.pages[:index], blank, *self._document.pages[index:])
        pages = tuple(
            replace(page, page_number=page_index + 1) for page_index, page in enumerate(pages)
        )
        self._document = self._document.replace_pages(pages)
        typed_pages = tuple(Page(self._document, page, self) for page in pages)
        self.pages = typed_pages
        return typed_pages[index]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def __len__(self) -> int:
        return self.page_count

    def get_page_text(self, page_number: int) -> str:
        return str(self.load_page(page_number).get_text())

    def get_page_images(self, page_number: int, full: bool = False) -> list[dict[str, object]]:
        return self.load_page(page_number).get_images(full=full)

    def get_page_pixmap(
        self,
        page_number: int,
        matrix: Matrix | None = None,
        dpi: float | None = None,
        clip: tuple[float, float, float, float] | None = None,
        alpha: bool = False,
    ) -> Pixmap:
        return self.load_page(page_number).get_pixmap(matrix, dpi, clip, alpha)

    def embfile_names(self) -> list[str]:
        return list(self._embedded_files)

    def embfile_get(self, name: str) -> bytes:
        return self._embedded_files[name]

    def embfile_add(self, name: str, buffer: bytes, **kwargs: object) -> None:
        del kwargs
        self._embedded_files[str(name)] = bytes(buffer)

    def embfile_del(self, name: str) -> None:
        del self._embedded_files[name]

    def get_toc(self, simple: bool = True) -> list[list[object]]:
        """Return PyMuPDF-style ``[level, title, page]`` outline rows."""
        if self._toc_override is not None:
            rows = [list(row[:3]) for row in self._toc_override]
            if not simple:
                for toc_row in rows:
                    toc_row.append({"kind": "goto", "page": toc_row[2]})
            return rows
        outlines = self._document.source_pdf.iter_outlines()
        result: list[list[object]] = []
        for item in outlines:
            page = (item.page_index + 1) if item.page_index is not None else 0
            row: list[object] = [item.level + 1, item.title, page]
            if not simple:
                row.append({"kind": "goto", "page": page})
            result.append(row)
        return result

    def set_toc(self, toc: list[list[object]]) -> None:
        """Replace the in-memory outline used by subsequent ``get_toc`` calls."""
        normalized: list[list[object]] = []
        for row in toc:
            if len(row) < 3:
                raise ValueError("each table-of-contents row needs level, title, and page")
            level, title, page = row[:3]
            if not isinstance(level, int) or level < 1:
                raise ValueError("outline levels must be positive integers")
            if not isinstance(title, str):
                raise TypeError("outline titles must be strings")
            if not isinstance(page, int) or page < 0:
                raise ValueError("outline page numbers must be non-negative integers")
            normalized.append([level, title, page, *row[3:]])
        self._toc_override = normalized

    def set_metadata(self, metadata: dict[str, object]) -> None:
        self._document = self._document.update_metadata(metadata)
        self.metadata.update(metadata)
        self.pages = tuple(Page(self._document, page, self) for page in self._document.pages)

    def _replace_annotation(self, page_number: int, index: int, annotation: Annotation) -> None:
        pages = list(self._document.pages)
        page = pages[page_number - 1]
        annotations = list(page.annotations)
        annotations[index] = annotation
        pages[page_number - 1] = replace(page, annotations=tuple(annotations))
        self._document = self._document.replace_pages(pages)
        self.pages = tuple(Page(self._document, item, self) for item in pages)

    def _replace_page_annotations(
        self, page_number: int, annotations: tuple[Annotation, ...]
    ) -> None:
        pages = list(self._document.pages)
        pages[page_number - 1] = replace(pages[page_number - 1], annotations=annotations)
        self._document = self._document.replace_pages(pages)
        self.pages = tuple(Page(self._document, item, self) for item in pages)

    def _replace_form_field(self, page_number: int, index: int, field: FormField) -> None:
        pages = list(self._document.pages)
        page = pages[page_number - 1]
        fields = list(page.form_fields)
        fields[index] = field
        pages[page_number - 1] = replace(page, form_fields=tuple(fields))
        self._document = self._document.replace_pages(pages)
        self.pages = tuple(Page(self._document, item, self) for item in pages)

    def _replace_links(self, page_number: int, links: tuple[Link, ...]) -> None:
        pages = list(self._document.pages)
        pages[page_number - 1] = replace(pages[page_number - 1], links=links)
        self._document = self._document.replace_pages(pages)
        self.pages = tuple(Page(self._document, item, self) for item in pages)

    def _append_block(self, page_number: int, block: Block) -> None:
        pages = list(self._document.pages)
        index = page_number - 1
        pages[index] = replace(pages[index], blocks=(*pages[index].blocks, block))
        self._document = self._document.replace_pages(pages)
        self.pages = tuple(Page(self._document, item, self) for item in pages)

    def _append_figure(self, page_number: int, figure: Figure) -> None:
        pages = list(self._document.pages)
        index = page_number - 1
        pages[index] = replace(pages[index], figures=(*pages[index].figures, figure))
        self._document = self._document.replace_pages(pages)
        self.pages = tuple(Page(self._document, item, self) for item in pages)

    def select(self, pages: list[int] | tuple[int, ...]) -> None:
        selected = tuple(self._document.pages[index] for index in pages)
        self._document = self._document.replace_pages(selected)
        self.pages = tuple(Page(self._document, page, self) for page in selected)

    def delete_page(self, page_number: int) -> None:
        self._document = self._document.delete_page(page_number + 1)
        self.pages = tuple(Page(self._document, page, self) for page in self._document.pages)

    def insert_pdf(
        self,
        source: "Document",
        from_page: int = 0,
        to_page: int = -1,
    ) -> None:
        end = source.page_count if to_page < 0 else to_page + 1
        inserted = tuple(source._document.pages[from_page:end])
        combined = (*self._document.pages, *inserted)
        self._document = self._document.replace_pages(combined)
        self.pages = tuple(Page(self._document, page, self) for page in combined)

    def apply_redactions(self) -> None:
        pages = list(self._document.pages)
        for page_number, boxes in self._pending_redactions.items():
            index = page_number - 1
            page = pages[index]
            annotations = page.annotations + tuple(Annotation("Redact", box) for box in boxes)
            pages[index] = replace(page, annotations=annotations)
        self._document = self._document.replace_pages(pages)
        self._pending_redactions.clear()

    def save(self, filename: object, **kwargs: object) -> bytes:
        supported_options = {
            "ascii",
            "clean",
            "deflate",
            "deflate_fonts",
            "deflate_images",
            "encryption",
            "expand",
            "garbage",
            "incremental",
            "linear",
            "pretty",
        }
        unsupported = set(kwargs) - supported_options
        if unsupported:
            raise TypeError(f"unsupported save options: {sorted(unsupported)[0]}")
        self.apply_redactions()
        return self._document.write_redacted(
            cast(str, filename), outlines=self._toc_override, attachments=self._embedded_files
        )


class TextPage:
    """Reusable PyMuPDF text view backed by a local page facade."""

    def __init__(self, page: Page) -> None:
        self._page = page

    def extractText(self, *args: object, **kwargs: object) -> str:
        del args
        return cast(str, self._page.get_text("text", textpage=self, **kwargs))

    def extractWORDS(self, *args: object, **kwargs: object) -> object:
        del args
        return self._page.get_text("words", textpage=self, **kwargs)

    def extractBLOCKS(self, *args: object, **kwargs: object) -> object:
        del args
        return self._page.get_text("blocks", textpage=self, **kwargs)

    def extractDICT(self, *args: object, **kwargs: object) -> object:
        del args
        return self._page.get_text("dict", textpage=self, **kwargs)

    def extractRAWDICT(self, *args: object, **kwargs: object) -> object:
        del args
        return self._page.get_text("rawdict", textpage=self, **kwargs)

    def extractHTML(self, *args: object, **kwargs: object) -> str:
        del args
        return cast(str, self._page.get_text("html", textpage=self, **kwargs))

    def extractXHTML(self, *args: object, **kwargs: object) -> str:
        del args
        return cast(str, self._page.get_text("xhtml", textpage=self, **kwargs))

    def extractXML(self, *args: object, **kwargs: object) -> str:
        del args
        return cast(str, self._page.get_text("xml", textpage=self, **kwargs))


def open(stream: object, *args: object, **kwargs: object) -> Document:
    del args, kwargs
    return Document(cast(PdfInput, stream))


__all__ = ("Document", "Matrix", "Page", "Pixmap", "TextPage", "Widget", "open")
