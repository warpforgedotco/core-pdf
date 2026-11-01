# Follows pypdf's annotations, Copyright (c) 2006-2008, Mathieu Fenniak,
# under its BSD-style license.
"""pypdf's annotation types. Each is a dictionary with the entries of its subtype."""

from __future__ import annotations

import logging
import uuid
from abc import ABC
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from .constants import AnnotationFlag
from .generic import (
    PAGE_FIT,
    ArrayObject,
    BooleanObject,
    DictionaryObject,
    Fit,
    FloatObject,
    IndirectObject,
    NameObject,
    NumberObject,
    RectangleObject,
    TextStringObject,
    hex_to_rgb,
)

logger = logging.getLogger(__name__)

NO_FLAGS = AnnotationFlag(0)

Vertex = tuple[float, float]
Rect = RectangleObject | tuple[float, float, float, float]


def _set(target: DictionaryObject, entries: dict[str, Any]) -> None:
    for key, value in entries.items():
        target[NameObject(key)] = value


def _name(value: str) -> NameObject:
    return NameObject(value)


def _floats(values: Iterable[float]) -> ArrayObject:
    return ArrayObject(FloatObject(value) for value in values)


def _bounds(vertices: Sequence[Vertex]) -> RectangleObject:
    xs = [x for x, _ in vertices]
    ys = [y for _, y in vertices]
    return RectangleObject((min(xs), min(ys), max(xs), max(ys)))


def _vertices(vertices: Sequence[Vertex]) -> ArrayObject:
    return ArrayObject(NumberObject(value) for vertex in vertices for value in vertex)


class AnnotationDictionary(DictionaryObject, ABC):
    def __init__(self) -> None:
        super().__init__()
        # /Rect is not set here, as Polygon and PolyLine work it out themselves.
        self[NameObject("/Type")] = NameObject("/Annot")

    @property
    def flags(self) -> AnnotationFlag:
        return AnnotationFlag(self.get_resolved("/F", 0))

    @flags.setter
    def flags(self, value: AnnotationFlag) -> None:
        self[NameObject("/F")] = NumberObject(value)


class MarkupAnnotation(AnnotationDictionary, ABC):
    """Base of the markup annotations, which can be a reply to another annotation."""

    def __init__(
        self,
        *,
        title_bar: str | None = None,
        in_reply_to: DictionaryObject | IndirectObject | None = None,
        reply_type: Literal["R", "Group"] = "R",
        annotation_name: str | None = None,
    ) -> None:
        super().__init__()
        if title_bar is not None:
            self[NameObject("/T")] = TextStringObject(title_bar)
        if annotation_name is not None and in_reply_to is None:
            raise ValueError("annotation_name is only supported when in_reply_to is set")
        if reply_type != "R" and in_reply_to is None:
            raise ValueError("reply_type is only meaningful when in_reply_to is set")
        if in_reply_to is None:
            return
        reference = (
            in_reply_to
            if isinstance(in_reply_to, IndirectObject)
            else getattr(in_reply_to, "indirect_reference", None)
        )
        if not isinstance(reference, IndirectObject):
            raise ValueError(
                "in_reply_to must be a registered annotation "
                "(added via writer.add_annotation() first)"
            )
        self[NameObject("/IRT")] = reference
        self[NameObject("/RT")] = NameObject(f"/{reply_type}")
        name = str(uuid.uuid4()) if annotation_name is None else annotation_name
        self[NameObject("/NM")] = TextStringObject(name)


class Text(MarkupAnnotation):
    def __init__(
        self, *, rect: Rect, text: str, open: bool = False, flags: int = NO_FLAGS, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        _set(
            self,
            {
                "/Subtype": _name("/Text"),
                "/Rect": RectangleObject(rect),
                "/Contents": TextStringObject(text),
                "/Open": BooleanObject(open),
                "/Flags": NumberObject(flags),
            },
        )


class FreeText(MarkupAnnotation):
    def __init__(
        self,
        *,
        text: str,
        rect: Rect,
        font: str = "Helvetica",
        bold: bool = False,
        italic: bool = False,
        font_size: str = "14pt",
        font_color: str = "000000",
        border_color: str | None = "000000",
        background_color: str | None = "ffffff",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Table 225 of the 1.7 reference: CSS2 style attributes used in rich text strings.
        style = "italic" if italic else "normal"
        weight = "bold" if bold else "normal"
        css = f"font: {style} {weight} {font_size} {font};text-align:left;color:#{font_color}"
        appearance = ""
        if border_color:
            appearance = "".join(f"{part} " for part in hex_to_rgb(border_color)) + "rg"
        _set(
            self,
            {
                "/Subtype": _name("/FreeText"),
                "/Rect": RectangleObject(rect),
                "/Contents": TextStringObject(text),
                "/DS": TextStringObject(css),
                "/DA": TextStringObject(appearance),
            },
        )
        if border_color is None:
            # A width of 0 means no border.
            self[NameObject("/BS")] = DictionaryObject({NameObject("/W"): NumberObject(0)})
        if background_color is not None:
            self[NameObject("/C")] = _floats(hex_to_rgb(background_color))


class Line(MarkupAnnotation):
    def __init__(self, p1: Vertex, p2: Vertex, rect: Rect, text: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _set(
            self,
            {
                "/Subtype": _name("/Line"),
                "/Rect": RectangleObject(rect),
                "/L": _floats((*p1, *p2)),
                "/LE": ArrayObject([_name("/None"), _name("/None")]),
                "/IC": _floats((0.5, 0.5, 0.5)),
                "/Contents": TextStringObject(text),
            },
        )


class PolyLine(MarkupAnnotation):
    def __init__(self, vertices: list[Vertex], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not vertices:
            raise ValueError("A polyline needs at least 1 vertex with two coordinates")
        _set(
            self,
            {
                "/Subtype": _name("/PolyLine"),
                "/Vertices": _vertices(vertices),
                "/Rect": _bounds(vertices),
            },
        )


class Rectangle(MarkupAnnotation):
    def __init__(self, rect: Rect, *, interior_color: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _set(self, {"/Subtype": _name("/Square"), "/Rect": RectangleObject(rect)})
        if interior_color:
            self[NameObject("/IC")] = _floats(hex_to_rgb(interior_color))


class Highlight(MarkupAnnotation):
    def __init__(
        self,
        *,
        rect: Rect,
        quad_points: ArrayObject,
        highlight_color: str = "ff0000",
        printing: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        _set(
            self,
            {
                "/Subtype": _name("/Highlight"),
                "/Rect": RectangleObject(rect),
                "/QuadPoints": quad_points,
                "/C": _floats(hex_to_rgb(highlight_color)),
            },
        )
        if printing:
            self.flags = AnnotationFlag.PRINT


class Ellipse(MarkupAnnotation):
    def __init__(self, rect: Rect, *, interior_color: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _set(self, {"/Subtype": _name("/Circle"), "/Rect": RectangleObject(rect)})
        if interior_color:
            self[NameObject("/IC")] = _floats(hex_to_rgb(interior_color))


class Polygon(MarkupAnnotation):
    def __init__(self, vertices: list[Vertex], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not vertices:
            raise ValueError("A polygon needs at least 1 vertex with two coordinates")
        _set(
            self,
            {
                "/Subtype": _name("/Polygon"),
                "/Vertices": _vertices(vertices),
                "/IT": _name("/PolygonCloud"),
                "/Rect": _bounds(vertices),
            },
        )


class Link(AnnotationDictionary):
    def __init__(
        self,
        *,
        rect: Rect,
        border: ArrayObject | None = None,
        url: str | None = None,
        target_page_index: int | None = None,
        fit: Fit = PAGE_FIT,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if url is None and target_page_index is None:
            raise ValueError(
                "Either 'url' or 'target_page_index' have to be provided. Both were None."
            )
        if url is not None and target_page_index is not None:
            raise ValueError(
                "Either 'url' or 'target_page_index' have to be provided. "
                f"{url=}, {target_page_index=}"
            )
        _set(
            self,
            {
                "/Subtype": _name("/Link"),
                "/Rect": RectangleObject(rect),
                "/Border": border_array(border, 0),
            },
        )
        if url is not None:
            self[NameObject("/A")] = DictionaryObject(
                {
                    NameObject("/S"): NameObject("/URI"),
                    NameObject("/Type"): NameObject("/Action"),
                    NameObject("/URI"): TextStringObject(url),
                }
            )
        else:
            # add_annotation turns this into a destination, once the page is known.
            deferred = DictionaryObject()
            dict.__setitem__(deferred, "target_page_index", NumberObject(target_page_index))
            dict.__setitem__(deferred, "fit", NameObject(fit.fit_type))
            dict.__setitem__(deferred, "fit_args", fit.fit_args)
            self[NameObject("/Dest")] = deferred


def border_array(border: Any, default: int) -> ArrayObject:
    if border is None:
        return ArrayObject([NumberObject(default)] * 3)
    output = ArrayObject(NumberObject(item) for item in border[:3])
    if len(border) == 4:
        output.append(ArrayObject(NumberObject(item) for item in border[3]))
    return output


class Popup(AnnotationDictionary):
    def __init__(
        self,
        *,
        rect: Rect,
        parent: DictionaryObject | None = None,
        open: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        _set(
            self,
            {
                "/Subtype": _name("/Popup"),
                "/Rect": RectangleObject(rect),
                "/Open": BooleanObject(open),
            },
        )
        if parent:
            # The parent has to be an indirect object.
            reference = getattr(parent, "indirect_reference", None)
            if reference is None:
                logger.warning("Unregistered Parent object : No Parent field set")
            else:
                self[NameObject("/Parent")] = reference


__all__ = [
    "NO_FLAGS",
    "AnnotationDictionary",
    "Ellipse",
    "FreeText",
    "Highlight",
    "Line",
    "Link",
    "MarkupAnnotation",
    "PolyLine",
    "Polygon",
    "Popup",
    "Rectangle",
    "Text",
]
