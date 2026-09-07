"""High-level PyMuPDF-shaped facade backed by core-pdf."""

from __future__ import annotations

import json
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from html import escape
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, cast, overload

from core_pdf.api.compat._shared import (
    BBox,
    ClosingMixin,
    coerce_bbox,
    encode_png,
    float32,
    write_bytes,
)
from core_pdf.api.compat.pymupdf.geometry import IRect, Matrix, Point, Quad, Rect
from core_pdf.api.compat.pymupdf.text import TextProjection, capture_text
from core_pdf.api.compat.pypdf import (
    PdfPageObject,
    StructuredState,
)
from core_pdf.api.document import PdfDocument
from core_pdf.api.document import PdfPage as NativePdfPage
from core_pdf.impl._impl.model.geometry import bbox_intersects
from core_pdf.impl._impl.output.model import (
    Annotation,
    Block,
    BlockKind,
    Figure,
    FormField,
    Link,
    TextLine,
)
from core_pdf.impl._impl.output.model import (
    Document as StructuredDocument,
)
from core_pdf.impl._impl.output.model import (
    Page as StructuredPage,
)


def synthesize_characters(text: str, box: BBox) -> list[tuple[str, BBox]]:
    x0, y0, x1, y1 = box
    width = (x1 - x0) / max(1, len(text))
    return [
        (character, (x0 + index * width, y0, x0 + (index + 1) * width, y1))
        for index, character in enumerate(text)
    ]


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
    _lazy_source: NativePdfPage | None

    def __init__(
        self, document: StructuredState, page: Any, owner: "Document | None" = None
    ) -> None:
        super().__init__(document, page)
        self._page_number = page.page_number
        self._owner = owner
        self._generation = owner._page_generation if owner is not None else 0
        self._number: int | tuple[int, int] | list[int] = page.page_number - 1
        self.mediabox = Rect(self.mediabox)
        raw_crop = Rect(self.cropbox)
        self.cropbox = Rect(
            raw_crop.x0,
            self.mediabox.y1 - raw_crop.y1,
            raw_crop.x1,
            self.mediabox.y1 - raw_crop.y0,
        )
        self.rotation %= 360
        if self.rotation % 90:
            self.rotation = 0
        self._user_unit = 1.0
        if document.pdf is not None:
            source = document.pdf.pages[page.page_number - 1]
            unit = document.pdf.resolver.resolve(source.page_dict.get("UserUnit"))
            if isinstance(unit, (int, float)):
                self._user_unit = float32(unit)

    @property
    def _page(self) -> StructuredPage:
        if self._lazy_source is not None:
            self._page_storage = self._lazy_source.structured_view
            self._lazy_source = None
        return self._page_storage

    @_page.setter
    def _page(self, value: StructuredPage) -> None:
        self._page_storage = value
        self._lazy_source = None

    @property
    def mediabox(self) -> Any:
        if hasattr(self, "_owner") and self.parent is None:
            raise AssertionError("page is None")
        return Rect(self._mediabox)

    @mediabox.setter
    def mediabox(self, value: Any) -> None:
        self._mediabox = Rect(value)

    @property
    def cropbox(self) -> Any:
        if hasattr(self, "_owner") and self.parent is None:
            raise AssertionError("page is None")
        return Rect(self._cropbox)

    @cropbox.setter
    def cropbox(self, value: Any) -> None:
        self._cropbox = Rect(value)

    @property
    def parent(self) -> Document | None:
        if self._owner is not None and (
            self._owner.is_closed or self._generation != self._owner._page_generation
        ):
            return None
        return self._owner

    @property
    def number(self) -> int | tuple[int, int] | list[int] | None:
        return self._number if self.parent is not None else None

    @property
    def rect(self) -> Rect:
        if self.parent is None:
            raise AssertionError("page is None")
        x0, y0, x1, y1 = self.cropbox
        width = float32(float32(x1 - x0) * abs(self._user_unit))
        height = float32(float32(y1 - y0) * abs(self._user_unit))
        if self.rotation % 180:
            width, height = height, width
        return Rect(0.0, 0.0, width, height)

    def bound(self) -> Rect:
        return self.rect

    @property
    def cropbox_position(self) -> Point:
        return self.cropbox.tl

    def _additional_box(self, name: str) -> Rect:
        if self.parent is None:
            raise AssertionError("page is None")
        if self._document.pdf is not None:
            source = self._document.pdf.pages[self._page_number - 1]
            box = source.resolve_box(name)
            if box is not None:
                x0, y0, x1, y1 = map(float32, box)
                return Rect(x0, self.mediabox.y1 - y1, x1, self.mediabox.y1 - y0)
        return Rect(self.cropbox)

    @property
    def bleedbox(self) -> Rect:
        return self._additional_box("BleedBox")

    @property
    def trimbox(self) -> Rect:
        return self._additional_box("TrimBox")

    @property
    def artbox(self) -> Rect:
        return self._additional_box("ArtBox")

    @property
    def transformation_matrix(self) -> Matrix:
        if self.parent is None:
            raise AssertionError("page is None")
        crop = self.cropbox
        if self.rotation:
            return Matrix(1, 0, 0, -1, 0, crop.height)
        unit = self._user_unit
        bottom, top = self.mediabox.y1 - crop.y1, self.mediabox.y1 - crop.y0
        if unit < 0:
            tx, ty = -crop.x1 * unit, bottom * unit
        else:
            tx, ty = -crop.x0 * unit, top * unit
        return Matrix(unit, 0, 0, -unit, float32(tx), float32(ty))

    @property
    def rotation_matrix(self) -> Matrix:
        if self.parent is None:
            raise AssertionError("page is None")
        width, height = self.cropbox.width, self.cropbox.height
        return Matrix(
            {
                0: (1, 0, 0, 1, 0, 0),
                90: (0, 1, -1, 0, height, 0),
                180: (-1, 0, 0, -1, width, height),
                270: (0, -1, 1, 0, 0, width),
            }[self.rotation]
        )

    @property
    def derotation_matrix(self) -> Matrix:
        if self.parent is None:
            raise TypeError(
                "Wrong number or type of arguments for overloaded function "
                "'Page_derotate_matrix'.\n"
                "  Possible C/C++ prototypes are:\n"
                "    Page_derotate_matrix(mupdf::PdfPage &)\n"
                "    Page_derotate_matrix(mupdf::FzPage &)\n"
            )
        return ~self.rotation_matrix

    def _native_text_view(self) -> Any:
        """Return only text represented by PDF text operators, as MuPDF does by default."""
        return self._page.text_view

    def get_text(self, kind: str = "text", *args: object, **kwargs: object) -> object:
        if self.parent is None:
            raise AssertionError("page is None")
        del args
        clip = kwargs.pop("clip", None)
        sort = bool(kwargs.pop("sort", False))
        flags = kwargs.pop("flags", None)
        textpage = kwargs.pop("textpage", None)
        delimiters_value = kwargs.pop("delimiters", None)
        delimiters = "" if delimiters_value is None else str(delimiters_value)
        if kwargs:
            raise TypeError(f"unsupported text options: {', '.join(kwargs)}")
        if textpage is not None:
            snapshot = cast(TextPage, textpage)
            if snapshot.parent != self:
                raise ValueError("not a textpage of this page")
            if kind in {"text", "plain"}:
                return snapshot.extractText(sort=sort)
            if kind == "words":
                words = snapshot.extractWORDS(delimiters=delimiters)
                if clip is not None:
                    clip_rect = Rect(clip)
                    words = [
                        word
                        for word in words
                        if (Rect(word[:4]) & clip_rect).get_area() * 2 >= Rect(word[:4]).get_area()
                    ]
                return sorted(words, key=lambda word: (word[3], word[0])) if sort else words
            if kind == "blocks":
                snapshot_blocks = snapshot.extractBLOCKS()
                return (
                    sorted(snapshot_blocks, key=lambda block: (block[3], block[0]))
                    if sort
                    else snapshot_blocks
                )
        if self._document.pdf is not None and kind in {"text", "plain", "words", "blocks"}:
            projection = capture_text(
                self._document.capability_page(self._page_number),
                flags=195 if flags is None else int(cast(Any, flags)),
                clip=clip,
            )
            if kind == "words":
                return projection.words(sort=sort, delimiters=delimiters)
            if kind == "blocks":
                return projection.block_records(sort=sort)
            return projection.text(sort=sort)
        text_view = self._native_text_view()
        if clip is not None:
            clip_bbox = cast(tuple[float, float, float, float], clip)
            elements = tuple(
                element
                for element in text_view.elements
                if element.bbox is not None and bbox_intersects(clip_bbox, element.bbox)
            )
            text_view = type(text_view)(elements, page_number=self._page_number)
        text = text_view.text
        if sort:
            text = "\n".join(sorted(text.splitlines(), key=str.casefold))
        if kind in {"text", "plain"}:
            return text + ("\n" if text else "")
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
            return [
                (
                    *(word.bbox or (0.0, 0.0, 0.0, 0.0)),
                    word.text,
                    word.block_index,
                    word.line_index,
                    word.word_index,
                )
                for word in text_view.words
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
        self._owner._append_block(self._page_number, block)
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
            self._page_number,
            Figure(
                order=max((item.order for item in self._page.elements), default=-1) + 1,
                bbox=coerce_bbox(rect),
                kind="rect",
                metadata={"color": color, "fill": fill, "width": width},
            ),
        )
        self._page = self._owner._document.pages[self._page_number - 1]
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
            self._page_number,
            Figure(
                order=max((item.order for item in self._page.elements), default=-1) + 1,
                bbox=(p1[0], p1[1], p2[0], p2[1]),
                kind="line",
                metadata={"p1": p1, "p2": p2, "color": color, "width": width},
            ),
        )
        self._page = self._owner._document.pages[self._page_number - 1]
        return self

    def get_textpage(
        self, clip: object = None, flags: int = 0, matrix: Matrix | None = None
    ) -> "TextPage":
        if self.parent is None:
            raise AssertionError("page is None")
        return TextPage(self, clip=clip, flags=flags, matrix=matrix)

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
        engine_page = self._document.capability_page(self._page_number)
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
            item.bbox
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
            self._page_number,
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
        owner._replace_links(self._page_number, tuple(links))

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
        owner._replace_links(self._page_number, tuple(links))

    def get_drawings(self) -> list[dict[str, object]]:
        page = self._document.capability_page(self._page_number).structured_view
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
        images = self._document.capability_page(self._page_number).extract_images()
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
        return self.get_images(full=True)

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
            Annot(document, self._page_number, annotation, index)
            for index, annotation in enumerate(self._page.annotations)
            if annotation.subtype != "CoreFigure"
        ]

    def widgets(self) -> list[Widget]:
        if self._owner is None:
            raise RuntimeError("widget access requires a document-owned page")
        return [
            Widget(self._owner, self._page_number, field, index)
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
        owner._replace_page_annotations(self._page_number, annotations)
        self._page = owner._document.pages[self._page_number - 1]
        return Annot(
            owner,
            self._page_number,
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
        pending.setdefault(self._page_number, []).append(tuple(bbox))
        cast(Any, self._document)._pending_redactions = pending

    def apply_redactions(self) -> None:
        self._document.apply_redactions()


class FileDataError(RuntimeError):
    """A document stream could not be read."""


class EmptyFileError(FileDataError):
    """An input file or stream is empty."""


class FileNotFoundError(RuntimeError):
    """A requested document does not exist."""


class Document(ClosingMixin):
    def __init__(
        self,
        filename: str | PathLike[str] | None = None,
        stream: bytes | bytearray | BytesIO | None = None,
        filetype: str | None = None,
    ) -> None:
        if filename is not None and not isinstance(filename, (str, PathLike)):
            raise TypeError("bad filename")
        if stream is not None and not isinstance(stream, (bytes, bytearray, BytesIO)):
            raise TypeError("bad stream")
        self.name = str(filename) if filename is not None else None
        self.is_closed = False
        self.is_encrypted = False
        self._page_generation = 0
        self._source_document: PdfDocument | None = None
        self._pending_redactions: dict[int, list[tuple[float, float, float, float]]] = {}
        self._toc_override: list[list[object]] | None = None
        self._embedded_files: dict[str, bytes] = {}
        self.metadata: dict[str, Any] = {
            "format": "PDF 1.7",
            **dict.fromkeys(
                (
                    "title",
                    "author",
                    "subject",
                    "keywords",
                    "creator",
                    "producer",
                    "creationDate",
                    "modDate",
                    "trapped",
                ),
                "",
            ),
            "encryption": None,
        }
        if stream is None and not filename:
            self._document = StructuredState.synthetic(StructuredDocument())
            return
        source: bytes | str
        if stream is not None:
            source = stream.getvalue() if isinstance(stream, BytesIO) else bytes(stream)
            if not source:
                raise EmptyFileError("Cannot open empty stream.")
        else:
            source = str(filename)
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"no such file: '{source}'")
            if not path.is_file():
                raise FileDataError(f"'{source}' is no file")
            if not path.stat().st_size:
                raise EmptyFileError(f"Cannot open empty file: filename='{source}'.")
        if filetype is not None and filetype.lower() != "pdf":
            raise NotImplementedError("non-PDF input formats are not implemented")
        pdf = PdfDocument.open(source)
        try:
            self._document = StructuredState(pdf)
            info = pdf.get_metadata().get("info", {})
            if isinstance(info, Mapping):
                for key in self.metadata:
                    if key not in {"format", "encryption"}:
                        pdf_key = key[0].upper() + key[1:]
                        self.metadata[key] = str(info.get(pdf_key, ""))
            header = bytes(pdf.raw_data[:32]).splitlines()[0]
            if header.startswith(b"%PDF-"):
                self.metadata["format"] = "PDF " + header[5:].decode("ascii", errors="replace")
            self._embedded_files = {item.filename: item.data for item in pdf.embedded_files()}
        except Exception:
            pdf.close()
            raise
        self._source_document = pdf

    def _check_open(self, *, check_encrypted: bool = False) -> None:
        if check_encrypted and (self.is_closed or self.is_encrypted):
            raise ValueError("document closed or encrypted")
        if self.is_closed:
            raise ValueError("document closed")

    def close(self) -> None:
        self._check_open()
        if self._source_document is not None:
            self._source_document.close()
        self.is_closed = True

    @property
    def is_pdf(self) -> bool:
        self._check_open()
        return True

    @property
    def chapter_count(self) -> int:
        self._check_open()
        return 1

    def chapter_page_count(self, chapter: int = 0) -> int:
        self._check_open()
        if chapter != 0:
            raise ValueError("bad chapter number")
        return self.page_count

    def __iter__(self) -> Iterator[Page]:
        return self.pages()

    def __contains__(self, index: object) -> bool:
        count = self.page_count
        if type(index) is int:
            return count > 0 and index < count
        if isinstance(index, (tuple, list)) and len(index) == 2:
            chapter, page = index
            return type(chapter) is int and chapter == 0 and type(page) is int and 0 <= page < count
        return False

    @overload
    def __getitem__(self, index: int | tuple[int, int]) -> Page: ...

    @overload
    def __getitem__(self, index: slice) -> list[Page]: ...

    def __getitem__(self, index: int | tuple[int, int] | slice) -> Page | list[Page]:
        self._check_open()
        if isinstance(index, slice):
            return [self.load_page(i) for i in range(*index.indices(self.page_count))]
        if not isinstance(index, int) and not (isinstance(index, tuple) and len(index) == 2):
            raise AssertionError(f"Invalid item number: i={index!r}.")
        if index not in self:
            raise IndexError(f"page {index} not in document")
        return self.load_page(index)

    def load_page(self, index: int | tuple[int, int] | list[int]) -> Page:
        self._check_open(check_encrypted=True)
        if index not in self:
            raise ValueError("page not in document")
        page_index = index if isinstance(index, int) else index[1]
        page_index %= self.page_count
        if self._document.pdf is not None:
            source = self._document.pdf.pages[page_index]
            media = source.media_box or (0.0, 0.0, 612.0, 792.0)
            model = StructuredPage(
                page_number=page_index + 1,
                width=media[2] - media[0],
                height=media[3] - media[1],
                rotation=source.rotation,
            )
            page = Page(self._document, model, self)
            page._lazy_source = source
        else:
            page = Page(self._document, self._document.pages[page_index], self)
        page._number = page_index if isinstance(index, int) else index
        return page

    def pages(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Iterator[Page]:
        self._check_open()
        count = self.page_count
        start = 0 if start is None else start
        if count and start < 0:
            start %= count
        if start < 0 or (count and start >= count):
            raise ValueError("bad start page number")
        stop = count if stop is None else min(count, stop)
        step = (1 if start < stop else -1) if step is None else step
        if not step:
            raise ValueError("arg 3 must not be zero")
        for index in range(start, stop, step):
            yield self.load_page(index)

    def new_page(
        self,
        pno: int = -1,
        width: float = 595.0,
        height: float = 842.0,
    ) -> Page:
        self._check_open(check_encrypted=True)
        if pno < -1:
            raise RuntimeError("bad page number(s)")
        if pno > self.page_count:
            raise RuntimeError("code=4: cannot insert page beyond end of page tree")
        index = self.page_count if pno == -1 else pno
        blank = StructuredPage(
            page_number=index + 1,
            width=float(width),
            height=float(height),
        )
        pages = (*self._document.pages[:index], blank, *self._document.pages[index:])
        pages = tuple(
            replace(page, page_number=page_index + 1) for page_index, page in enumerate(pages)
        )
        self._page_generation += 1
        return self._set_pages(pages)[index]

    @property
    def page_count(self) -> int:
        self._check_open()
        if self._document.pdf is not None:
            return len(self._document.pdf.pages)
        return len(self._document.pages)

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
        self._check_open()
        if self._toc_override is not None:
            rows = [list(row[:3]) for row in self._toc_override]
            if not simple:
                for toc_row in rows:
                    toc_row.append({"kind": "goto", "page": toc_row[2]})
            return rows
        if self._document.pdf is None:
            return []
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
        self._set_document(self._document.update_metadata(metadata))
        self.metadata.update(metadata)

    def _set_document(self, document: StructuredState) -> tuple[Page, ...]:
        """Adopt ``document`` and rebuild the facade page objects from it."""
        self._document = document
        typed_pages = tuple(Page(document, page, self) for page in document.pages)
        return typed_pages

    def _set_pages(self, pages: Sequence[StructuredPage]) -> tuple[Page, ...]:
        return self._set_document(self._document.replace_pages(tuple(pages)))

    def _mutate_page(
        self, page_number: int, mutate: Callable[[StructuredPage], StructuredPage]
    ) -> None:
        pages = list(self._document.pages)
        pages[page_number - 1] = mutate(pages[page_number - 1])
        self._set_pages(pages)

    def _replace_annotation(self, page_number: int, index: int, annotation: Annotation) -> None:
        def mutate(page: StructuredPage) -> StructuredPage:
            annotations = list(page.annotations)
            annotations[index] = annotation
            return replace(page, annotations=tuple(annotations))

        self._mutate_page(page_number, mutate)

    def _replace_page_annotations(
        self, page_number: int, annotations: tuple[Annotation, ...]
    ) -> None:
        self._mutate_page(page_number, lambda page: replace(page, annotations=annotations))

    def _replace_form_field(self, page_number: int, index: int, field: FormField) -> None:
        def mutate(page: StructuredPage) -> StructuredPage:
            fields = list(page.form_fields)
            fields[index] = field
            return replace(page, form_fields=tuple(fields))

        self._mutate_page(page_number, mutate)

    def _replace_links(self, page_number: int, links: tuple[Link, ...]) -> None:
        self._mutate_page(page_number, lambda page: replace(page, links=links))

    def _append_block(self, page_number: int, block: Block) -> None:
        self._mutate_page(page_number, lambda page: replace(page, blocks=(*page.blocks, block)))

    def _append_figure(self, page_number: int, figure: Figure) -> None:
        self._mutate_page(page_number, lambda page: replace(page, figures=(*page.figures, figure)))

    def select(self, pages: list[int] | tuple[int, ...]) -> None:
        self._set_pages(tuple(self._document.pages[index] for index in pages))

    def delete_page(self, page_number: int) -> None:
        self._set_document(self._document.delete_page(page_number + 1))

    def insert_pdf(
        self,
        source: "Document",
        from_page: int = 0,
        to_page: int = -1,
    ) -> None:
        end = source.page_count if to_page < 0 else to_page + 1
        inserted = tuple(source._document.pages[from_page:end])
        self._set_pages((*self._document.pages, *inserted))

    def apply_redactions(self) -> None:
        pages = list(self._document.pages)
        for page_number, boxes in self._pending_redactions.items():
            index = page_number - 1
            page = pages[index]
            annotations = page.annotations + tuple(Annotation("Redact", box) for box in boxes)
            pages[index] = replace(page, annotations=annotations)
        self._document = self._document.replace_pages(pages)
        self._pending_redactions.clear()


class TextPage:
    """Text snapshot captured from native page operations."""

    def __init__(
        self, page: Page, *, clip: object = None, flags: int = 0, matrix: Matrix | None = None
    ) -> None:
        self.parent = weakref.proxy(page)
        self._page_ref = weakref.ref(page)
        self._rect = (
            Rect(clip)
            if clip is not None
            else Rect(
                0, 0, page.cropbox.width * page._user_unit, page.cropbox.height * page._user_unit
            ).normalize()
        )
        self._projection: TextProjection | None = None
        self._legacy: dict[str, Any] = {}
        if page._document.pdf is not None:
            self._projection = capture_text(
                page._document.capability_page(page._page_number),
                flags=flags,
                clip=clip,
                matrix=matrix,
            )
        else:
            self._legacy = {
                kind: page.get_text(kind, flags=flags, clip=clip)
                for kind in ("text", "words", "blocks")
            }

    @property
    def _page(self) -> Page:
        page = self._page_ref()
        if page is None:
            raise ReferenceError("weakly-referenced object no longer exists")
        return page

    @property
    def rect(self) -> Rect:
        return Rect(self._rect)

    def extractText(self, sort: bool = False) -> str:
        if self._projection is not None:
            return self._projection.text(sort=sort)
        return cast(str, self._legacy["text"])

    extractTEXT = extractText

    def extractWORDS(self, delimiters: str | None = None) -> list[tuple[Any, ...]]:
        if self._projection is not None:
            return self._projection.words(delimiters=delimiters or "")
        return cast(list[tuple[Any, ...]], deepcopy(self._legacy["words"]))

    def extractBLOCKS(self) -> list[tuple[Any, ...]]:
        if self._projection is not None:
            return self._projection.block_records()
        return cast(list[tuple[Any, ...]], deepcopy(self._legacy["blocks"]))

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


open = Document


__all__ = (
    "Document",
    "EmptyFileError",
    "FileDataError",
    "FileNotFoundError",
    "IRect",
    "Matrix",
    "Page",
    "Pixmap",
    "Point",
    "Quad",
    "Rect",
    "TextPage",
    "Widget",
    "open",
)
