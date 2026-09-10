"""High-level PyMuPDF-shaped facade backed by core-pdf."""

from __future__ import annotations

import json
import weakref
from base64 import b64encode
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
from core_pdf.api.compat.pymupdf.objects import ObjectAccess, format_object, internal_metadata_value
from core_pdf.api.compat.pymupdf.text import TextProjection, capture_text
from core_pdf.api.compat.pypdf import StructuredState
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
from core_pdf.impl.exceptions import PdfUnsupportedError
from core_pdf.impl.spec.s_07_security.standard import create_standard_security_handler
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import coerce_to_bytes
from core_pdf.impl.types import PdfReference


def synthesize_characters(text: str, box: BBox) -> list[tuple[str, BBox]]:
    x0, y0, x1, y1 = box
    width = (x1 - x0) / max(1, len(text))
    return [
        (character, (x0 + index * width, y0, x0 + (index + 1) * width, y1))
        for index, character in enumerate(text)
    ]


def internal_text_json(payload: dict[str, Any]) -> str:
    def encode(value: object) -> str | None:
        if isinstance(value, (bytes, bytearray)):
            return b64encode(value).decode("ascii")
        return None

    return json.dumps(payload, separators=(",", ":"), indent=1, default=encode)


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


class Page:
    _lazy_source: NativePdfPage | None

    def __init__(
        self, document: StructuredState, page: Any, owner: "Document | None" = None
    ) -> None:
        self._document = document
        self._page = page
        self._mediabox = Rect(0, 0, page.width, page.height)
        self._cropbox = Rect(page.cropbox or self._mediabox)
        self._rotation = page.rotation
        if document.pdf is not None:
            source = document.pdf.pages[page.page_number - 1]
            self._mediabox = Rect(tuple(map(float32, source.media_box or self._mediabox)))
            self._cropbox = Rect(tuple(map(float32, source.crop_box or self._mediabox)))
            rotation = source.inherited_values.get("Rotate")
            self._rotation = (
                int(rotation) if isinstance(rotation, (int, float)) else source.rotation
            )
        self._page_number = page.page_number
        self._owner = owner
        self._generation = owner._page_generation if owner is not None else 0
        self._object_revision = owner._object_revision if owner is not None else 0
        self._number: int | tuple[int, int] | list[int] = page.page_number - 1
        self.mediabox = Rect(self.mediabox)
        raw_crop = Rect(self.cropbox)
        self.cropbox = Rect(
            raw_crop.x0,
            float32(self.mediabox.y1 - raw_crop.y1),
            raw_crop.x1,
            float32(self.mediabox.y1 - raw_crop.y0),
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
        self._refresh_native()
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

    def _refresh_native(self) -> None:
        owner = getattr(self, "_owner", None)
        if (
            owner is not None
            and not owner.is_closed
            and self._generation == owner._page_generation
            and self._object_revision != owner._object_revision
        ):
            replacement = owner.load_page(self._number)
            self.__dict__.update(replacement.__dict__)

    @property
    def parent(self) -> Document | None:
        if self._owner is not None and (
            self._owner.is_closed or self._generation != self._owner._page_generation
        ):
            return None
        self._refresh_native()
        return self._owner

    @property
    def xref(self) -> int:
        if self.parent is None:
            raise AssertionError("page is None")
        return self.parent.page_xref(self._page_number - 1)

    @property
    def number(self) -> int | tuple[int, int] | list[int] | None:
        return self._number if self.parent is not None else None

    @property
    def rotation(self) -> int:
        if hasattr(self, "_owner") and self.parent is None:
            raise AssertionError("page is None")
        return self._rotation

    @rotation.setter
    def rotation(self, value: int) -> None:
        self._rotation = value

    def set_rotation(self, rotation: int) -> None:
        parent = self.parent
        if parent is None:
            raise AssertionError("page is None")
        rotation = rotation % 360 if rotation % 90 == 0 else 0
        if not isinstance(rotation, int):
            raise TypeError("in method 'pdf_dict_put_int', argument 3 of type 'int64_t'")
        parent.xref_set_key(self.xref, "Rotate", str(rotation))
        self._refresh_native()

    def set_mediabox(self, rect: object) -> None:
        parent = self.parent
        if parent is None:
            raise AssertionError("page is None")
        box = Rect(rect)
        if box.is_empty or box.is_infinite:
            raise ValueError("rect is infinite or empty")
        xref = self.xref
        access = parent._objects
        obj = access.get(xref)
        if not isinstance(obj, dict):
            raise ValueError("invalid page dictionary")
        updated = {
            key: value
            for key, value in obj.items()
            if key not in {"CropBox", "BleedBox", "TrimBox", "ArtBox"}
        }
        updated["MediaBox"] = list(map(float32, box))
        access.overrides[xref] = updated
        parent._object_revision += 1
        self._refresh_native()

    def _set_box(self, name: str, rect: object) -> None:
        parent = self.parent
        if parent is None:
            raise ValueError("orphaned object: parent is None")
        box = Rect(rect)
        media = self.mediabox
        bounds = Rect(media.x0, 0, media.x1, media.height)
        if box.is_empty or box.is_infinite or box not in bounds:
            raise ValueError(f"{name} not in MediaBox")
        values = [box.x0, media.y1 - box.y1, box.x1, media.y1 - box.y0]
        parent.xref_set_key(self.xref, name, format_object(list(map(float32, values))))
        self._refresh_native()

    def set_cropbox(self, rect: object) -> None:
        self._set_box("CropBox", rect)

    def set_bleedbox(self, rect: object) -> None:
        self._set_box("BleedBox", rect)

    def set_trimbox(self, rect: object) -> None:
        self._set_box("TrimBox", rect)

    def set_artbox(self, rect: object) -> None:
        self._set_box("ArtBox", rect)

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
                return Rect(x0, float32(self.mediabox.y1 - y1), x1, float32(self.mediabox.y1 - y0))
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
        if kind in {"dict", "rawdict", "json", "rawjson"} and (
            textpage is not None or self._document.pdf is not None
        ):
            native_snapshot = (
                cast(TextPage, textpage)
                if textpage is not None
                else self.get_textpage(
                    clip=clip, flags=199 if flags is None else int(cast(Any, flags))
                )
            )
            cb = self.cropbox if clip is None else None
            payload = (
                native_snapshot.extractRAWDICT(cb=cb, sort=sort)
                if kind in {"rawdict", "rawjson"}
                else native_snapshot.extractDICT(cb=cb, sort=sort)
            )
            return internal_text_json(payload) if kind in {"json", "rawjson"} else payload
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

    def get_textbox(self, rect: object, textpage: TextPage | None = None) -> str:
        if textpage is None:
            textpage = self.get_textpage()
        elif textpage.parent != self:
            raise ValueError("not a textpage of this page")
        return textpage.extractTextbox(rect)

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
        if matrix is not None and (
            matrix.d != scale
            or scale <= 0.0
            or any(value != 0.0 for value in (matrix.b, matrix.c, matrix.e, matrix.f))
        ):
            raise ValueError("only positive uniform scale pixmap matrices are supported")
        requested_dpi = float(dpi if dpi is not None else 72.0 * scale)
        engine_page = self._document.capability_page(self._page_number)
        raster = engine_page.render().rasterize(scale=max(0.01, requested_dpi / 72.0), crop=clip)
        data = bytes(raster.pixels)
        if not alpha and raster.channels == 4:
            data = b"".join(data[index : index + 3] for index in range(0, len(data), 4))
        channels = 3 if not alpha and raster.channels == 4 else raster.channels
        return Pixmap(data, raster.width, raster.height, channels, requested_dpi)

    def search_for(
        self,
        needle: str,
        *,
        clip: object = None,
        quads: bool = False,
        flags: int = 210,
        textpage: TextPage | None = None,
    ) -> list[Quad] | list[Rect] | None:
        if textpage is None:
            textpage = self.get_textpage(clip=clip, flags=flags)
        elif textpage.parent != self:
            raise ValueError("not a textpage of this page")
        return textpage.search(needle, quads=quads)

    def get_links(self) -> list[dict[str, object]]:
        links = self._page.links
        return [{"uri": link.url, "kind": link.link_type, "from": link.bbox} for link in links]

    def internal_link_owner(self) -> Document:
        """The document backing this page, for a mutation that cannot go anywhere else."""
        owner = self._owner
        if owner is None:
            raise RuntimeError("link mutation requires a document-owned page")
        return owner

    def internal_link_index(
        self, bbox: tuple[float, float, float, float]
    ) -> tuple[list[Link], int]:
        """This page's links, and the position of the one ``bbox`` addresses."""
        links = list(self._page.links)
        index = next((index for index, item in enumerate(links) if item.bbox == bbox), None)
        if index is None:
            raise ValueError("link was not found")
        return links, index

    def insert_link(self, link: Mapping[str, object]) -> None:
        owner = self.internal_link_owner()
        bbox = cast(tuple[float, float, float, float], link.get("from", self.mediabox))
        url = link.get("uri")
        owner._replace_links(
            self._page_number,
            (*self._page.links, Link(bbox=bbox, url=str(url) if url is not None else None)),
        )

    def update_link(self, link: Mapping[str, object]) -> None:
        owner = self.internal_link_owner()
        bbox = cast(tuple[float, float, float, float], link.get("from", self.mediabox))
        links, index = self.internal_link_index(bbox)
        url = link.get("uri")
        links[index] = replace(links[index], bbox=bbox, url=str(url) if url is not None else None)
        owner._replace_links(self._page_number, tuple(links))

    def delete_link(self, link: Mapping[str, object]) -> None:
        owner = self.internal_link_owner()
        bbox = cast(tuple[float, float, float, float], link.get("from", self.mediabox))
        links, index = self.internal_link_index(bbox)
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


def internal_authentication_status(pdf: PdfDocument, password: str) -> int:
    document_id = pdf.resolver.resolve(pdf.trailer_dict.get("ID", [b""]))
    encrypt = pdf.resolver.resolve(pdf.trailer_dict.get("Encrypt"))
    assert isinstance(document_id, (list, tuple))
    assert isinstance(encrypt, dict)
    return create_standard_security_handler(
        document_id, cast(PdfDict, encrypt), password, retain_authentication_status=True
    ).authentication_status


class internal_PreviewPdfDocument(PdfDocument):
    """Retain an encrypted file's structural preview until it is authenticated."""

    internal_locked: bool = False

    def init_security(self, password: str) -> None:
        self.internal_locked = False
        encrypt = self.resolver.resolve(self.trailer_dict.get("Encrypt"))
        if isinstance(encrypt, dict):
            if encrypt.get("R") == 7:
                raise FileDataError("unsupported encryption revision")
            if encrypt.get("R") in (5, 6):
                for key, size in (("O", 48), ("OE", 32)):
                    try:
                        value = coerce_to_bytes(encrypt.get(key))
                    except TypeError as error:
                        raise FileDataError("invalid owner password entry") from error
                    if len(value) != size:
                        raise FileDataError("invalid owner password entry")
        try:
            super().init_security(password)
        except PdfUnsupportedError as error:
            if str(error) != "Incorrect password":
                raise
            self.internal_locked = True


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
        self.needs_pass = 0
        self._page_generation = 0
        self._objects_invalidated = False
        self._object_access: ObjectAccess | None = None
        self._object_revision = 0
        self._native_revision = 0
        self._working_document: PdfDocument | None = None
        self._source_document: PdfDocument | None = None
        self._pending_redactions: dict[int, list[tuple[float, float, float, float]]] = {}
        self._toc_override: list[list[object]] | None = None
        self._embedded_files: dict[str, bytes] = {}
        self.metadata: dict[str, Any] | None = ObjectAccess(None).metadata()
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
        pdf = internal_PreviewPdfDocument.open(source)
        try:
            if not pdf.internal_locked and pdf.trailer_dict.get("Encrypt") is not None:
                # An empty owner password permits explicit authentication, but only
                # an empty user password makes the reference open without prompting.
                pdf.internal_locked = not bool(internal_authentication_status(pdf, "") & 2)
            self.internal_bind_source(pdf)
        except Exception:
            pdf.close()
            raise
        self.needs_pass = int(self.is_encrypted)

    def internal_bind_source(self, pdf: internal_PreviewPdfDocument) -> None:
        metadata = None if pdf.internal_locked else ObjectAccess(pdf).metadata()
        embedded = (
            {}
            if pdf.internal_locked
            else {item.filename: item.data for item in pdf.embedded_files()}
        )
        self._document = StructuredState(pdf)
        self._source_document = pdf
        self._object_access = None
        self.is_encrypted = pdf.internal_locked
        self.metadata = metadata
        self._embedded_files = embedded

    def authenticate(self, password: str) -> int:
        self._check_open()
        previous = self._source_document
        if previous is None or previous.trailer_dict.get("Encrypt") is None:
            return 1
        try:
            status = internal_authentication_status(previous, password)
        except PdfUnsupportedError as error:
            if str(error) != "Incorrect password":
                raise
            return 0
        except UnicodeEncodeError:
            return 0
        if not self.is_encrypted:
            return status
        authenticated = internal_PreviewPdfDocument.open(
            bytes(previous.raw_data), password=password
        )
        if authenticated.internal_locked:
            authenticated.close()
            return 0
        try:
            self.internal_bind_source(authenticated)
        except Exception:
            authenticated.close()
            raise
        previous.close()
        return status

    def _check_open(self, *, check_encrypted: bool = False) -> None:
        if check_encrypted and (self.is_closed or self.is_encrypted):
            raise ValueError("document closed or encrypted")
        if self.is_closed:
            raise ValueError("document closed")

    def close(self) -> None:
        self._check_open()
        if self._working_document is not None:
            self._working_document.close()
        if self._source_document is not None:
            self._source_document.close()
        self.is_closed = True

    @property
    def is_pdf(self) -> bool:
        self._check_open()
        return True

    @property
    def _objects(self) -> ObjectAccess:
        self._check_open()
        if self._objects_invalidated or (
            self._document.pdf is None and self._source_document is not None
        ):
            raise NotImplementedError("object access for structured edits is not implemented")
        if self._object_access is None:
            self._object_access = ObjectAccess(self._source_document)
        return self._object_access

    def _refresh_native(self) -> None:
        if self._native_revision == self._object_revision:
            return
        assert self._object_access is not None
        data = self._object_access.tobytes(no_new_id=True)
        document = PdfDocument.open(data)
        previous = self._working_document
        self._working_document = document
        self._document = StructuredState(document)
        self._native_revision = self._object_revision
        if previous is not None:
            previous.close()

    def page_xref(self, pno: int) -> int:
        references = self._objects.page_references()
        if not references or pno >= len(references):
            raise ValueError("bad page number(s)")
        return references[pno % len(references)].object_number

    def get_new_xref(self) -> int:
        result = self._objects.allocate()
        self._object_revision += 1
        return result

    def update_object(self, xref: int, text: str, page: Page | None = None) -> None:
        del page
        self._objects.update(xref, text)
        self._object_revision += 1

    def xref_set_key(self, xref: int, key: str, value: str) -> None:
        self._objects.set_key(xref, key, value)
        self._object_revision += 1

    def update_stream(
        self,
        xref: int = 0,
        stream: bytes | bytearray | None = None,
        new: bool = True,
        compress: bool = True,
    ) -> None:
        del new
        self._check_open(check_encrypted=True)
        if not isinstance(stream, (bytes, bytearray)):
            raise ValueError("bad type: 'stream'")
        self._objects.update_stream(xref, bytes(stream), compress=compress)
        self._object_revision += 1

    def tobytes(self, *args: object, **kwargs: object) -> bytes:
        if args:
            raise TypeError("positional save options are not implemented")
        self._check_open(check_encrypted=True)
        access = self._objects
        if not self.page_count:
            raise ValueError("cannot save with zero pages")
        no_new_id = bool(kwargs.pop("no_new_id", False))
        defaults = {
            "garbage": 0,
            "clean": 0,
            "deflate": 0,
            "deflate_images": 0,
            "deflate_fonts": 0,
            "incremental": 0,
            "ascii": 0,
            "expand": 0,
            "linear": 0,
            "appearance": 0,
            "pretty": 0,
            "encryption": 1,
            "permissions": 4095,
            "owner_pw": None,
            "user_pw": None,
            "preserve_metadata": 1,
            "use_objstms": 0,
            "compression_effort": 0,
            "raise_on_repair": False,
            "reproducible": False,
        }
        for key, value in kwargs.items():
            if key not in defaults:
                raise TypeError(f"Document.write() got an unexpected keyword argument '{key}'")
            if value != defaults[key]:
                raise NotImplementedError(f"save option is not implemented: {key}")
        version = str((self.metadata or {}).get("format", "PDF 1.7")).removeprefix("PDF ")
        return access.tobytes(no_new_id=no_new_id, version=version)

    def save(self, filename: object, **kwargs: object) -> None:
        self._check_open()
        if (
            isinstance(filename, (str, PathLike))
            and self.name is not None
            and Path(cast("str | PathLike[str]", filename)).resolve() == Path(self.name).resolve()
            and not kwargs.get("incremental")
        ):
            raise ValueError("save to original must be incremental")
        data = self.tobytes(**kwargs)
        write_bytes(cast(Any, filename), data)

    def xref_length(self) -> int:
        return self._objects.length

    def pdf_catalog(self) -> int:
        root = self._objects.trailer.get("Root")
        return root.object_number if isinstance(root, PdfReference) else 0

    def pdf_trailer(self, compressed: bool = False, ascii: bool = False) -> str:
        return self.xref_object(-1, compressed=compressed, ascii=ascii)

    def xref_object(self, xref: int, compressed: bool = False, ascii: bool = False) -> str:
        access = self._objects
        try:
            obj = access.get(xref)
        except ValueError as error:
            raise RuntimeError("bad xref") from error
        if obj is None and xref not in access.overrides:
            raise RuntimeError(f"code=7: cannot find object in xref ({xref} 0 R)")
        return format_object(obj, compressed=compressed, ascii_only=ascii)

    def xref_get_keys(self, xref: int) -> list[str]:
        return self._objects.keys(xref)

    def xref_get_key(self, xref: int, key: str) -> tuple[str, str]:
        return self._objects.key(xref, key)

    def xref_stream(self, xref: int) -> bytes | None:
        self._check_open(check_encrypted=True)
        obj = self._objects.get(xref)
        return obj.data if isinstance(obj, PdfStream) else None

    def xref_stream_raw(self, xref: int) -> bytes | None:
        self._check_open(check_encrypted=True)
        obj = self._objects.get(xref)
        return bytes(obj.raw_data) if isinstance(obj, PdfStream) else None

    def xref_is_stream(self, xref: int = 0) -> bool:
        access = self._objects
        if not 0 < xref < access.length:
            return False
        return isinstance(access.get(xref), PdfStream)

    def xref_is_font(self, xref: int) -> bool:
        self._check_open(check_encrypted=True)
        return self.xref_get_key(xref, "Type") == ("name", "/Font")

    def xref_is_image(self, xref: int) -> bool:
        self._check_open(check_encrypted=True)
        return self.xref_get_key(xref, "Subtype") == ("name", "/Image")

    def xref_is_xobject(self, xref: int) -> bool:
        self._check_open(check_encrypted=True)
        return self.xref_get_key(xref, "Subtype") == ("name", "/Form")

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
        self._refresh_native()
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
        if not self._objects_invalidated:
            self._objects.insert_blank_page(index, float(width), float(height))
            self._object_revision += 1
            self._page_generation += 1
            self._refresh_native()
            return self.load_page(index)
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
        self._refresh_native()
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
        self._check_open(check_encrypted=True)
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

    def xref_xml_metadata(self) -> int:
        kind, reference = self.xref_get_key(self.pdf_catalog(), "Metadata")
        return int(reference.split()[0]) if kind == "xref" else 0

    def get_xml_metadata(self) -> str:
        xref = self.xref_xml_metadata()
        data = self.xref_stream(xref) if xref else None
        return data.split(b"\0", 1)[0].decode("utf-8", errors="replace") if data is not None else ""

    def set_xml_metadata(self, metadata: str) -> None:
        self._check_open(check_encrypted=True)
        data = metadata.encode("utf-8")
        xref = self.xref_xml_metadata()
        if not xref:
            xref = self.get_new_xref()
            self.update_object(xref, "<<>>")
            self.xref_set_key(self.pdf_catalog(), "Metadata", f"{xref} 0 R")
        self.update_stream(xref, data, compress=False)
        self.xref_set_key(xref, "Type", "/Metadata")
        self.xref_set_key(xref, "Subtype", "/XML")

    def del_xml_metadata(self) -> None:
        self._check_open(check_encrypted=True)
        access = self._objects
        xref = self.pdf_catalog()
        catalog = access.get(xref)
        if isinstance(catalog, dict) and "Metadata" in catalog:
            access.overrides[xref] = {
                key: value for key, value in catalog.items() if key != "Metadata"
            }
            self._object_revision += 1

    def set_metadata(self, metadata: dict[str, object] | None = None) -> None:
        if self.is_closed:
            raise AttributeError("'NoneType' object has no attribute 'm_internal'")
        self._check_open(check_encrypted=True)
        if metadata is None:
            metadata = {}
        if type(metadata) is not dict:
            raise ValueError("bad metadata")
        assert self.metadata is not None
        fields = {
            key: key[0].upper() + key[1:]
            for key in self.metadata
            if key not in {"format", "encryption"}
        }
        unknown = set(metadata) - fields.keys() - {"format", "encryption"}
        if unknown:
            raise ValueError(f"bad dict key(s): {unknown}")
        kind, reference = self.xref_get_key(-1, "Info")
        if kind != "xref":
            if not metadata:
                return
            xref = self.get_new_xref()
            self.update_object(xref, "<<>>")
            self.xref_set_key(-1, "Info", f"{xref} 0 R")
        else:
            xref = int(reference.split()[0])
            if not metadata:
                self.xref_set_key(-1, "Info", "null")
        for key, value in metadata.items():
            if key in fields:
                self.xref_set_key(xref, fields[key], internal_metadata_value(value))
        self.metadata = {
            **self.metadata,
            **{key: self._objects.metadata_text(pdf_key) for key, pdf_key in fields.items()},
        }

    def _set_document(self, document: StructuredState) -> tuple[Page, ...]:
        """Adopt ``document`` and rebuild the facade page objects from it."""
        self._objects_invalidated = True
        self._document = document
        return tuple(Page(document, page, self) for page in document.pages)

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
        self._objects_invalidated = True
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
                for kind in ("text", "words", "blocks", "dict", "rawdict")
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

    def search(
        self, needle: str, hit_max: int = 0, quads: bool = True
    ) -> list[Quad] | list[Rect] | None:
        del hit_max
        projection = self._projection or TextProjection.from_rawdict(self._legacy["rawdict"])
        return projection.search(needle, quads=quads)

    def extractTextbox(self, rect: object) -> str:
        projection = self._projection or TextProjection.from_rawdict(self._legacy["rawdict"])
        return projection.textbox(Rect(rect))

    def extractSelection(self, pointa: object, pointb: object) -> str:
        projection = self._projection or TextProjection.from_rawdict(self._legacy["rawdict"])
        return projection.selection(Point(pointa), Point(pointb))

    def extractWORDS(self, delimiters: str | None = None) -> list[tuple[Any, ...]]:
        if self._projection is not None:
            return self._projection.words(delimiters=delimiters or "")
        return cast(list[tuple[Any, ...]], deepcopy(self._legacy["words"]))

    def extractBLOCKS(self) -> list[tuple[Any, ...]]:
        if self._projection is not None:
            return self._projection.block_records()
        return cast(list[tuple[Any, ...]], deepcopy(self._legacy["blocks"]))

    def extractDICT(self, cb: Rect | None = None, sort: bool = False) -> dict[str, Any]:
        return self._dictionary(cb, sort=sort, raw=False)

    def extractRAWDICT(self, cb: Rect | None = None, sort: bool = False) -> dict[str, Any]:
        return self._dictionary(cb, sort=sort, raw=True)

    def _dictionary(self, cb: Rect | None, *, sort: bool, raw: bool) -> dict[str, Any]:
        bounds = self._rect if cb is None else cb
        if self._projection is not None:
            return self._projection.dictionary(bounds.width, bounds.height, raw=raw, sort=sort)
        result = cast(dict[str, Any], deepcopy(self._legacy["rawdict" if raw else "dict"]))
        result.update(width=bounds.width, height=bounds.height)
        if sort:
            result["blocks"].sort(key=lambda block: (block["bbox"][3], block["bbox"][0]))
        return result

    def extractJSON(self, cb: Rect | None = None, sort: bool = False) -> str:
        return internal_text_json(self.extractDICT(cb=cb, sort=sort))

    def extractRAWJSON(self, cb: Rect | None = None, sort: bool = False) -> str:
        return internal_text_json(self.extractRAWDICT(cb=cb, sort=sort))

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
