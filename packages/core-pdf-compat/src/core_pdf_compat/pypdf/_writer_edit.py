# Follows pypdf's writer, Copyright (c) 2006-2008, Mathieu Fenniak,
# under its BSD-style license.
"""The parts of PdfWriter that change what is on the pages: annotations, labels, removals."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from core_pdf_spec.s_07_syntax.serialize import serialize_object

from .annotations import border_array
from .constants import ImageType, ObjectDeletionFlag
from .generic import (
    ArrayObject,
    ContentStream,
    Destination,
    DictionaryObject,
    Fit,
    FloatObject,
    IndirectObject,
    NameObject,
    NullObject,
    NumberObject,
    PdfObject,
    RectangleObject,
    StreamObject,
    TextStringObject,
    TreeObject,
    is_null_or_none,
    to_core,
)

logger = logging.getLogger(__name__)

_UNSET = object()
ATTACHMENT_ANNOTATIONS = ("/FileAttachment", "/Sound", "/Movie", "/Screen")
DRAWING_OPERATORS = tuple(b"w J j M d i W W* b b* B B* S s f f* F n m l c v y h re sh".split())
TEXT_OPERATORS = (b"Tj", b"TJ", b"'", b'"')


def pdf_objectify(value: Any) -> Any:
    if isinstance(value, PdfObject):
        return value
    if isinstance(value, dict):
        output = DictionaryObject()
        for key, item in value.items():
            output[NameObject(key)] = pdf_objectify(item)
        return output
    if isinstance(value, str):
        return NameObject(value) if value.startswith("/") else TextStringObject(value)
    if isinstance(value, (float, int)):
        return FloatObject(value)
    if isinstance(value, list):
        return ArrayObject(pdf_objectify(item) for item in value)
    raise NotImplementedError(f"{type(value)=} could not be cast to a PdfObject")


def _nums_insert(key: NumberObject, value: DictionaryObject, nums: ArrayObject) -> None:
    index = len(nums)
    while index != 0 and key <= nums[index - 2]:
        index -= 2
    if index < len(nums) and key == nums[index]:
        nums[index + 1] = value
    else:
        nums.insert(index, key)
        nums.insert(index + 1, value)


def _nums_clear_range(key: NumberObject, last: int, nums: ArrayObject) -> None:
    index = nums.index(key) + 2
    while index < len(nums) and nums[index] <= last:
        del nums[index : index + 2]


def _font_ids(resources: Any, names: list[str]) -> list[Any]:
    """The resource names, such as /F1, of fonts with the given names."""
    found: dict[str, list[Any]] = {}

    def walk(value: Any, key: Any, seen: set[int]) -> None:
        value = value.get_object() if isinstance(value, IndirectObject) else value
        if isinstance(value, dict):
            if id(value) in seen:
                return
            seen.add(id(value))
            if value.get("/Type") == "/Font":
                name = str(value.get("/BaseFont", "")).lstrip("/").split("+")[-1]
                ids = found.setdefault(name, [])
                if key not in ids:
                    ids.append(key)
            for child, item in dict.items(value):
                walk(item, child, seen)
        elif isinstance(value, list):
            for item in list.__iter__(value):
                walk(item, None, seen)

    walk(resources, None, set())
    return [identifier for name in names for identifier in found.get(name, ())]


class EditWriter:
    _objects: list[Any]
    _root_object: DictionaryObject
    _info: Any
    _ID: Any
    _idnum_hash: dict[bytes, Any]
    pages: Any
    _add_object: Any

    def add_annotation(self, page_number: Any, annotation: Any) -> DictionaryObject:
        from ._writer import PageObject

        page = self.pages[page_number] if isinstance(page_number, int) else page_number
        if not isinstance(page, PageObject):
            raise TypeError("page: invalid type")
        added = pdf_objectify(annotation)
        added[NameObject("/P")] = page.indirect_reference
        if page.annotations is None:
            page[NameObject("/Annots")] = ArrayObject()
        if added.get_resolved("/Subtype") == "/Link" and "/Dest" in added:
            deferred = added["/Dest"]
            fit = Fit(fit_type=deferred["fit"], fit_args=dict.__getitem__(deferred, "fit_args"))
            target = Destination(NameObject("/LinkName"), deferred["target_page_index"], fit)
            added[NameObject("/Dest")] = target.dest_array
        page.annotations.append(self._add_object(added))
        if added.get_resolved("/Subtype") == "/Popup" and "/Parent" in added:
            added["/Parent"][NameObject("/Popup")] = added.indirect_reference
        return added

    def add_uri(self, page_number: int, uri: str, rect: Any, border: Any = None) -> None:
        page = self.pages[page_number]
        if isinstance(rect, str):
            rect = RectangleObject(float(item) for item in rect.strip("[] ").split())
        elif not isinstance(rect, RectangleObject):
            rect = RectangleObject(rect)
        link = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Annot"),
                NameObject("/Subtype"): NameObject("/Link"),
                NameObject("/P"): page.indirect_reference,
                NameObject("/Rect"): rect,
                NameObject("/H"): NameObject("/I"),
                NameObject("/Border"): border_array(border, 2),
                NameObject("/A"): DictionaryObject(
                    {
                        NameObject("/S"): NameObject("/URI"),
                        NameObject("/URI"): TextStringObject(uri),
                    }
                ),
            }
        )
        reference = self._add_object(link)
        if "/Annots" in page:
            page["/Annots"].append(reference)
        else:
            page[NameObject("/Annots")] = ArrayObject([reference])

    def set_page_label(
        self,
        page_index_from: int,
        page_index_to: int,
        style: Any = None,
        prefix: str | None = None,
        start: int | None = 0,
    ) -> None:
        if style is None and prefix is None:
            raise ValueError("At least one of style and prefix must be given")
        if page_index_from < 0:
            raise ValueError("page_index_from must be greater or equal than 0")
        if page_index_to < page_index_from:
            raise ValueError("page_index_to must be greater or equal than page_index_from")
        if page_index_to >= len(self.pages):
            raise ValueError("page_index_to exceeds number of pages")
        if start is not None and start != 0 and start < 1:
            raise ValueError("If given, start must be greater or equal than one")
        self._set_page_label(page_index_from, page_index_to, style, prefix, start)

    def _set_page_label(
        self,
        page_index_from: int,
        page_index_to: int,
        style: Any = None,
        prefix: str | None = None,
        start: int | None = 0,
    ) -> None:
        default = DictionaryObject({NameObject("/S"): NameObject("/D")})
        label = DictionaryObject()
        if style is not None:
            label[NameObject("/S")] = NameObject(style)
        if prefix is not None:
            label[NameObject("/P")] = TextStringObject(prefix)
        if start != 0:
            label[NameObject("/St")] = NumberObject(start)
        catalog = self._root_object
        if "/PageLabels" not in catalog:
            nums = ArrayObject()
            _nums_insert(NumberObject(0), default, nums)
            labels = TreeObject()
            labels[NameObject("/Nums")] = nums
            catalog[NameObject("/PageLabels")] = labels
        nums = catalog["/PageLabels"]["/Nums"]
        first = NumberObject(page_index_from)
        _nums_insert(first, label, nums)
        _nums_clear_range(first, page_index_to, nums)
        index = nums.index(first) + 2
        following = nums[index] if index < len(nums) else None
        if following != page_index_to + 1 and page_index_to + 1 < len(self.pages):
            _nums_insert(NumberObject(page_index_to + 1), default, nums)

    def remove_links(self) -> None:
        for page in self.pages:
            self.remove_objects_from_page(page, ObjectDeletionFlag.ALL_ANNOTATIONS)

    def remove_annotations(self, subtypes: Any) -> None:
        for page in self.pages:
            self._remove_annots_from_page(page, subtypes)

    def _remove_annots_from_page(self, page: Any, subtypes: Iterable[str] | None) -> None:
        page = page.get_object()
        if "/Annots" not in page:
            return
        annots = page["/Annots"]
        index = 0
        while index < len(annots):
            item = annots[index]
            target = item.get_object()
            if subtypes is None or target.get_resolved("/Subtype") in subtypes:
                if isinstance(item, IndirectObject):
                    self._objects[item.idnum - 1] = NullObject()
                del annots[index]
            else:
                index += 1

    def remove_objects_from_page(
        self, page: Any, to_delete: Any, text_filters: dict[str, Any] | None = None
    ) -> None:
        from ._writer import PageObject

        if isinstance(to_delete, (list, tuple)):
            for flag in to_delete:
                self.remove_objects_from_page(page, flag)
            return
        for flag, subtypes in (
            (ObjectDeletionFlag.LINKS, ("/Link",)),
            (ObjectDeletionFlag.ATTACHMENTS, ATTACHMENT_ANNOTATIONS),
            (ObjectDeletionFlag.OBJECTS_3D, ("/3D",)),
            (ObjectDeletionFlag.ALL_ANNOTATIONS, None),
        ):
            if to_delete & flag:
                self._remove_annots_from_page(page, subtypes)
                return
        operators: tuple[bytes, ...] = ()
        if to_delete & ObjectDeletionFlag.DRAWING_IMAGES:
            operators = DRAWING_OPERATORS
        if to_delete & ObjectDeletionFlag.TEXT:
            operators = TEXT_OPERATORS
        if not isinstance(page, PageObject):
            page = PageObject(self, page.indirect_reference)
        if "/Contents" not in page:
            return
        content = page.get_contents()
        images, forms = self._clean_forms(page, [], operators, to_delete, text_filters)
        self._clean_content(content, images, forms, operators, to_delete, text_filters)
        page.replace_contents(content)

    def _clean_content(
        self,
        content: Any,
        images: list[str],
        forms: list[str],
        operators: tuple[bytes, ...],
        to_delete: Any,
        text_filters: dict[str, Any] | None,
    ) -> None:
        del forms
        deleting_text = bool(to_delete & ObjectDeletionFlag.TEXT)
        fonts = (text_filters or {}).get("font_ids", []) if deleting_text else []
        font = None
        kept = []
        for operation in content.operations:
            operands, operator = operation
            if operator == b"Tf":
                font = operands[0]
            matches = (
                (operator == b"INLINE IMAGE" and to_delete & ObjectDeletionFlag.INLINE_IMAGES)
                or operator in operators
                or (
                    operator == b"Do"
                    and to_delete & ObjectDeletionFlag.XOBJECT_IMAGES
                    and operands[0] in images
                )
            )
            removed = matches and (not deleting_text or not text_filters or font in fonts)
            if not removed:
                kept.append(operation)
        content.operations = kept
        content.get_data()

    def _clean_forms(
        self,
        element: Any,
        stack: list[Any],
        operators: tuple[bytes, ...],
        to_delete: Any,
        text_filters: dict[str, Any] | None,
    ) -> tuple[list[str], list[str]]:
        reference = element.indirect_reference
        if any(
            item is element or (reference is not None and item.indirect_reference == reference)
            for item in stack
        ):
            return [], []
        resources = element.get_resolved("/Resources")
        xobjects = {} if resources is None else resources.get_resolved("/XObject", {})
        images: list[str] = []
        forms: list[str] = []
        for key, value in list(xobjects.items()):
            target = value.get_object()
            if not isinstance(target, DictionaryObject):
                continue
            replacement: Any = None
            subtype = target.get_resolved("/Subtype")
            if to_delete & ObjectDeletionFlag.XOBJECT_IMAGES and subtype == "/Image":
                replacement = NullObject()
                images.append(key)
            if subtype == "/Form":
                forms.append(key)
                replacement = self._form_content(target)
                stack.append(element)
                self._clean_forms(replacement, stack, operators, to_delete, text_filters)
            if replacement is None:
                continue
            if isinstance(value, IndirectObject):
                self._objects[value.idnum - 1] = replacement
                replacement.indirect_reference = value
            else:
                xobjects[key] = self._add_object(replacement)
        for key in images:
            del xobjects[key]
        if isinstance(element, StreamObject):
            self._clean_content(element, images, forms, operators, to_delete, text_filters)
        return images, forms

    def _form_content(self, form: Any) -> ContentStream:
        if isinstance(form, ContentStream):
            return form
        content = ContentStream(form, self)
        for key, item in dict.items(form):
            if key not in {"/Length", "/Filter", "/DecodeParms"}:
                dict.__setitem__(content, key, item)
        content.indirect_reference = form.indirect_reference
        return content

    def remove_images(self, to_delete: Any = None) -> None:
        chosen = ImageType.ALL if to_delete is None or isinstance(to_delete, bool) else to_delete
        flags = ObjectDeletionFlag.NONE
        for name in ("XOBJECT_IMAGES", "INLINE_IMAGES", "DRAWING_IMAGES"):
            if chosen & ImageType[name]:
                flags |= ObjectDeletionFlag[name]
        for page in self.pages:
            self.remove_objects_from_page(page, flags)

    def remove_text(self, font_names: list[str] | None = None) -> None:
        for page in self.pages:
            filters = {}
            if font_names:
                filters["font_ids"] = _font_ids(page.get("/Resources"), font_names)
            self.remove_objects_from_page(page, ObjectDeletionFlag.TEXT, text_filters=filters)

    def compress_identical_objects(
        self,
        remove_identicals: Any = _UNSET,
        remove_orphans: Any = _UNSET,
        *,
        remove_duplicates: bool = True,
        remove_unreferenced: bool = True,
    ) -> None:
        from ._writer import PageObject

        if remove_identicals is not _UNSET:
            _deprecated("remove_identicals", "remove_duplicates")
            remove_duplicates = remove_identicals
        if remove_orphans is not _UNSET:
            _deprecated("remove_orphans", "remove_unreferenced")
            remove_unreferenced = remove_orphans
        first_seen: dict[bytes, IndirectObject] = {}
        moved: dict[int, IndirectObject] = {}
        for index, item in enumerate(self._objects):
            # A page stays apart: a file may not list one page twice.
            if is_null_or_none(item) or isinstance(item, PageObject):
                continue
            form = serialize_object(to_core(item))
            kept = first_seen.get(form)
            if remove_duplicates and kept is not None:
                moved[index + 1] = kept
                self._objects[index] = None
            else:
                first_seen[form] = item.indirect_reference
        self._idnum_hash = {}
        referenced: set[int] = set()

        def replace(value: Any) -> None:
            if isinstance(value, DictionaryObject):
                entries = list(dict.items(value))
            elif isinstance(value, ArrayObject):
                entries = list(enumerate(list.__iter__(value)))
            else:
                return
            for key, item in entries:
                if isinstance(item, IndirectObject):
                    referenced.add(item.idnum)
                    if item.idnum in moved:
                        target = moved[item.idnum]
                        referenced.add(target.idnum)
                        if isinstance(value, DictionaryObject):
                            dict.__setitem__(value, key, target)
                        else:
                            list.__setitem__(value, key, target)
                else:
                    replace(item)

        for item in self._objects:
            replace(item)
        if remove_unreferenced:
            roots = [self._root_object, self._info, self._ID]
            referenced.update(
                item.indirect_reference.idnum
                for item in roots
                if item is not None and item.indirect_reference is not None
            )
            for index in range(len(self._objects)):
                if index + 1 not in referenced:
                    self._objects[index] = None


def _deprecated(old: str, new: str) -> None:
    import warnings

    warnings.warn(
        f"{old} is deprecated and will be removed in pypdf 7.0.0. Use {new} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


__all__ = ["EditWriter", "pdf_objectify"]
