# Follows pypdf's page merging, Copyright (c) 2006-2008, Mathieu Fenniak,
# under its BSD-style license.
"""pypdf's page transformations and page merging, over the writer's generic objects."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

from .generic import (
    ArrayObject,
    ContentStream,
    DictionaryObject,
    FloatObject,
    NameObject,
    RectangleObject,
    is_null_or_none,
)

Matrix = tuple[float, float, float, float, float, float]

RESOURCE_KINDS = (
    "/ExtGState",
    "/ColorSpace",
    "/Pattern",
    "/Shading",
    "/XObject",
    "/Font",
    "/Properties",
)


def _multiply(m: Matrix, n: Matrix) -> Matrix:
    return (
        m[0] * n[0] + m[1] * n[2],
        m[0] * n[1] + m[1] * n[3],
        m[2] * n[0] + m[3] * n[2],
        m[2] * n[1] + m[3] * n[3],
        m[4] * n[0] + m[5] * n[2] + n[4],
        m[4] * n[1] + m[5] * n[3] + n[5],
    )


class Transformation:
    """A 2D transformation matrix, stored as PDF's six numbers [a b c d e f]."""

    def __init__(self, ctm: Matrix = (1, 0, 0, 1, 0, 0)) -> None:
        self.ctm = ctm

    @property
    def matrix(self) -> tuple[tuple[float, float, int], ...]:
        a, b, c, d, e, f = self.ctm
        return ((a, b, 0), (c, d, 0), (e, f, 1))

    @staticmethod
    def compress(matrix: Any) -> Matrix:
        return (matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1], matrix[2][0], matrix[2][1])

    def _to_cm(self) -> str:
        return " ".join(f"{value:.4f}" for value in self.ctm) + " cm"

    def transform(self, m: Transformation) -> Transformation:
        return Transformation(_multiply(self.ctm, m.ctm))

    def translate(self, tx: float = 0, ty: float = 0) -> Transformation:
        a, b, c, d, e, f = self.ctm
        return Transformation((a, b, c, d, e + tx, f + ty))

    def scale(self, sx: float | None = None, sy: float | None = None) -> Transformation:
        if sx is None and sy is None:
            raise ValueError("Either sx or sy must be specified")
        x = sy if sx is None else sx
        y = sx if sy is None else sy
        assert x is not None
        assert y is not None
        return Transformation(_multiply(self.ctm, (x, 0, 0, y, 0, 0)))

    def rotate(self, rotation: float) -> Transformation:
        angle = math.radians(rotation)
        cos, sin = math.cos(angle), math.sin(angle)
        return Transformation(_multiply(self.ctm, (cos, sin, -sin, cos, 0, 0)))

    def __repr__(self) -> str:
        return f"Transformation(ctm={self.ctm})"

    def apply_on(self, pt: Any, as_object: bool = False) -> Any:
        make = FloatObject if as_object else float
        x, y = float(pt[0]), float(pt[1])
        a, b, c, d, e, f = self.ctm
        point = (make(x * a + y * c + e), make(x * b + y * d + f))
        return list(point) if isinstance(pt, list) else point


def as_matrix(ctm: Transformation | Matrix) -> Matrix:
    return ctm.ctm if isinstance(ctm, Transformation) else ctm


def page_dictionary(page: Any) -> DictionaryObject:
    return page if isinstance(page, DictionaryObject) else page._generic_dictionary()


def _box(page: DictionaryObject, *keys: str) -> RectangleObject:
    for key in keys:
        box = page.get_inherited(key)
        if not is_null_or_none(box):
            return RectangleObject(box)
    raise ValueError("The page has no /MediaBox")


def _corners(box: RectangleObject) -> list[tuple[float, float]]:
    left, bottom, right, top = map(float, box)
    return [(left, bottom), (left, top), (right, top), (right, bottom)]


def _bounds(points: Iterable[tuple[float, float]]) -> RectangleObject:
    xs, ys = zip(*points, strict=True)
    return RectangleObject((min(xs), min(ys), max(xs), max(ys)))


def _transformed(box: RectangleObject, ctm: Matrix | None) -> list[tuple[float, float]]:
    corners = _corners(box)
    if ctm is None:
        return corners
    a, b, c, d, e, f = map(float, ctm)
    transformation = Transformation((a, b, c, d, e, f))
    return [transformation.apply_on(corner) for corner in corners]


def add_cm(content: ContentStream, ctm: Matrix) -> ContentStream:
    content.operations.insert(0, ([FloatObject(value) for value in ctm], b"cm"))
    return content


def transform_page(page: Any, ctm: Matrix, expand: bool) -> None:
    content = page.get_contents()
    if content is not None:
        add_cm(content, ctm)
        content.isolate_graphics_state()
        page.replace_contents(content)
    if expand:
        page[NameObject("/MediaBox")] = _bounds(_transformed(page.mediabox, ctm))


def scale_page(page: Any, sx: float, sy: float) -> None:
    transform_page(page, (sx, 0, 0, sy, 0, 0), False)
    # Boxes that were left out default to the crop box, so that one is scaled last of them.
    for key, box in (
        ("/BleedBox", page.bleedbox),
        ("/TrimBox", page.trimbox),
        ("/ArtBox", page.artbox),
        ("/CropBox", page.cropbox),
        ("/MediaBox", page.mediabox),
    ):
        page[NameObject(key)] = box.scale(sx, sy)
    annotations = page.get_resolved("/Annots")
    if isinstance(annotations, ArrayObject):
        for item in annotations:
            annotation = item.get_object()
            rect = (
                annotation.get_resolved("/Rect")
                if isinstance(annotation, DictionaryObject)
                else None
            )
            if isinstance(rect, ArrayObject) and len(rect) >= 4:
                rect[:4] = [
                    FloatObject(float(rect[index]) * (sx if index % 2 == 0 else sy))
                    for index in range(4)
                ]
    viewports = page.get_resolved("/VP")
    # pypdf scales the first of several viewports. Each one covers its own part of the page.
    for viewport in viewports if isinstance(viewports, ArrayObject) else [viewports]:
        if isinstance(viewport, DictionaryObject) and "/BBox" in viewport:
            viewport[NameObject("/BBox")] = RectangleObject(viewport.get_resolved("/BBox")).scale(
                sx, sy
            )


class _Merge:
    """One merge of a source page into a page held by a writer."""

    def __init__(self, page: Any, source: DictionaryObject) -> None:
        self.page = page
        self.source = source
        self.writer = page.writer
        reference = source.indirect_reference
        self.owner = getattr(reference, "pdf", None)
        tracked = self.writer is not None and self.owner is not None
        self.copies = self.writer._copies.setdefault(id(self.owner), {}) if tracked else {}

    def bring(self, value: Any) -> Any:
        if self.writer is None:
            return value
        return self.writer._import(value, self.owner, self.copies)

    def own(self, holder: DictionaryObject, key: str, empty: Any) -> Any:
        """The value under key. A page outside a writer may only share it, so it gets a copy."""
        value = holder.get_resolved(key)
        if value is None:
            value = empty
        elif self.writer is None:
            value = type(empty)(value)
        else:
            return value
        holder[NameObject(key)] = value
        return value

    def resources(self) -> dict[str, NameObject]:
        """Copy the source's resources in, and return the names that had to change."""
        theirs = self.source.get_resolved("/Resources")
        if not isinstance(theirs, DictionaryObject):
            return {}
        ours = self.own(self.page, "/Resources", DictionaryObject())
        rename: dict[str, NameObject] = {}
        for kind in RESOURCE_KINDS:
            incoming = theirs.get_resolved(kind)
            if not isinstance(incoming, DictionaryObject):
                continue
            rename.update(self._merge_kind(self.own(ours, kind, DictionaryObject()), incoming))
        self._merge_proc_set(ours, theirs)
        return rename

    def _merge_kind(
        self, ours: DictionaryObject, incoming: DictionaryObject
    ) -> dict[str, NameObject]:
        rename = {}
        for key, raw in list(dict.items(incoming)):
            value = self.bring(raw)
            name = key
            attempt = 0
            while name in ours and dict.__getitem__(ours, name) != value:
                name = f"{key}-{attempt}"
                attempt += 1
            if name != key:
                rename[key] = NameObject(name)
            dict.__setitem__(ours, NameObject(name), value)
        ordered = sorted(dict.items(ours))
        ours.clear()
        for key, value in ordered:
            dict.__setitem__(ours, key, value)
        return rename

    def _merge_proc_set(self, ours: DictionaryObject, theirs: DictionaryObject) -> None:
        incoming = theirs.get_resolved("/ProcSet")
        if not isinstance(incoming, ArrayObject):
            return
        names = self.own(ours, "/ProcSet", ArrayObject())
        names.extend(item for item in list(incoming) if item not in names)
        names.sort()

    def annotations(self, ctm: Matrix | None) -> None:
        incoming = self.source.get_resolved("/Annots")
        if not isinstance(incoming, ArrayObject) or not incoming:
            return
        if is_null_or_none(self.page.get_resolved("/Annots")):
            self.page[NameObject("/Annots")] = ArrayObject()
        annots = self.page.get_resolved("/Annots")
        if self.writer is None:
            annots = self.page[NameObject("/Annots")] = ArrayObject(annots)
            annots.extend(incoming)
            return
        transformation = Transformation() if ctm is None else Transformation(ctm)
        # A copy, as merging a page into itself would otherwise never finish.
        for item in list(incoming):
            annotation = item.get_object()
            if isinstance(annotation, DictionaryObject):
                annots.append(self._annotation(annotation, transformation))

    def _detached(self, value: DictionaryObject, *dropped: str) -> DictionaryObject:
        """A new copy that leaves out the entries tying the object to its old page."""
        kept = DictionaryObject()
        for key, item in dict.items(value):
            if key not in dropped:
                dict.__setitem__(kept, key, item)
        return self.bring(kept)

    def _annotation(self, annotation: DictionaryObject, transformation: Transformation) -> Any:
        copy = self._detached(annotation, "/P", "/StructParent", "/Parent", "/Popup")
        reference = self.writer._add_object(copy)
        rect = annotation.get_resolved("/Rect")
        if isinstance(rect, ArrayObject) and len(rect) >= 4:
            points = [transformation.apply_on((rect[i], rect[i + 1])) for i in (0, 2)]
            copy[NameObject("/Rect")] = ArrayObject(_bounds(points))
        quads = annotation.get_resolved("/QuadPoints")
        if isinstance(quads, ArrayObject):
            moved = ArrayObject()
            for index in range(0, len(quads) - 1, 2):
                moved.extend(transformation.apply_on((quads[index], quads[index + 1]), True))
            copy[NameObject("/QuadPoints")] = moved
        popup = annotation.get_resolved("/Popup")
        if isinstance(popup, DictionaryObject):
            popup_copy = self._detached(popup, "/P", "/Parent")
            popup_copy[NameObject("/Parent")] = reference
            popup_copy[NameObject("/P")] = self.page.indirect_reference
            copy[NameObject("/Popup")] = self.writer._add_object(popup_copy)
        copy[NameObject("/P")] = self.page.indirect_reference
        return reference

    def content(
        self, rename: dict[str, NameObject], transform: Callable[[Any], Any] | None
    ) -> ContentStream | None:
        contents = self.source.get_resolved("/Contents")
        if is_null_or_none(contents):
            return None
        stream = ContentStream(contents, self.writer)
        left, bottom, right, top = map(float, _box(self.source, "/CropBox", "/MediaBox"))
        clip = [FloatObject(value) for value in (left, bottom, right - left, top - bottom)]
        stream.operations[:0] = [(clip, b"re"), ([], b"W"), ([], b"n")]
        if transform is not None:
            stream = transform(stream)
        for operands, _operator in stream.operations:
            if isinstance(operands, list):
                for index, item in enumerate(operands):
                    if isinstance(item, NameObject):
                        operands[index] = rename.get(item, item)
        stream.isolate_graphics_state()
        return stream


def merge_resources(
    page: Any, res1: DictionaryObject, res2: DictionaryObject, resource: str, new_res1: bool
) -> tuple[DictionaryObject, dict[str, NameObject]]:
    ours = res1.get_resolved(resource)
    if new_res1:
        ours = DictionaryObject() if ours is None else DictionaryObject(ours)
    incoming = res2.get_resolved(resource)
    incoming = DictionaryObject() if incoming is None else incoming
    return ours, _Merge(page, DictionaryObject())._merge_kind(ours, incoming)


def merge_page(
    page: Any,
    other: Any,
    transform: Callable[[Any], Any] | None = None,
    ctm: Matrix | None = None,
    over: bool = True,
    expand: bool = False,
) -> None:
    source = page_dictionary(other)
    merge = _Merge(page, source)
    rename = merge.resources()
    merge.annotations(ctm)
    # Read both before replacing, as the source may be this page.
    original = page.get_contents()
    incoming = merge.content(rename, transform)
    merged = ArrayObject()
    if original is not None:
        original.isolate_graphics_state()
        merged.append(original)
    if incoming is not None:
        merged.insert(len(merged) if over else 0, incoming)
    if expand:
        corners = _corners(page.mediabox) + _transformed(_box(source, "/MediaBox"), ctm)
        page[NameObject("/MediaBox")] = _bounds(corners)
    page.replace_contents(merged)


__all__ = [
    "Transformation",
    "add_cm",
    "as_matrix",
    "merge_page",
    "merge_resources",
    "scale_page",
    "transform_page",
]
