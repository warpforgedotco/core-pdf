# The shape of this API and its file formats follow pypdf._page and
# pypdf.generic._image_xobject, Copyright (c) 2006, Mathieu Fenniak, under its BSD-style
# license. core-pdf decodes the samples.
"""pypdf's page images, over core-pdf's image decoder."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from io import BytesIO
from typing import Any

from core_pdf_compat.pypdf.constants import (
    _INLINE_IMAGE_KEY_MAPPING,
    _INLINE_IMAGE_VALUE_MAPPING,
)
from core_pdf_compat.pypdf.errors import PdfReadError
from core_pdf_compat.pypdf.generic import (
    ArrayObject,
    DictionaryObject,
    IndirectObject,
    NameObject,
    NullObject,
    StreamObject,
    is_null_or_none,
)

logger = logging.getLogger(__name__)


def _pillow() -> Any:
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(
            "pillow is required to do image extraction. "
            "It can be installed via 'pip install pypdf[image]'"
        ) from error
    return Image


def xobject_image(
    x_object: Any,
    pillow_parameters: dict[str, Any] | None = None,
    visited: set[int] | None = None,
) -> tuple[str | None, bytes, Any]:
    """The file extension, file bytes and Pillow image for an image XObject, as pypdf makes
    them: core-pdf decodes the filters and pypdf's conversion builds the image.
    """
    _pillow()
    from core_pdf_compat.pypdf._image_xobject import _xobj_to_image  # noqa: PLC0415

    return _xobj_to_image(x_object, pillow_parameters, visited)


class ImageFile:
    def __init__(
        self,
        name: str = "",
        data: bytes = b"",
        image: Any = None,
        indirect_reference: IndirectObject | None = None,
        is_inline: bool = False,
        is_displayed: bool = False,
    ) -> None:
        self.name = name
        self.data = data
        self.image = image
        self.indirect_reference = indirect_reference
        self.is_inline = is_inline
        self.is_displayed = is_displayed

    def replace(self, new_image: Any, **kwargs: Any) -> None:
        from . import PdfReader  # noqa: PLC0415

        image = _pillow()
        reference = self.indirect_reference
        if reference is None:
            raise TypeError("Cannot update an inline image.")
        if not hasattr(reference.pdf, "_objects"):
            raise TypeError("Cannot update an image not belonging to a PdfWriter.")
        if not isinstance(new_image, image.Image):
            raise TypeError("new_image shall be a PIL Image")
        output = BytesIO()
        new_image.save(output, "PDF", **kwargs)
        replacement = PdfReader(output).pages[0].images[0]
        assert replacement.indirect_reference is not None
        reference.pdf._replace_object(reference, replacement.indirect_reference.get_object())
        extension, self.data, self.image = xobject_image(
            reference.get_object(), pillow_parameters=kwargs
        )
        self.name = self.name[: self.name.rfind(".")] + str(extension)

    def __str__(self) -> str:
        return f"{type(self).__name__}(name={self.name}, data: {len(self.data)} Byte)"

    def __repr__(self) -> str:
        return f"{str(self)[:-1]}, hash: {hash(self.data)})"


class VirtualListImages(Sequence[ImageFile]):
    def __init__(
        self, ids_function: Callable[[], list[Any]], get_function: Callable[[Any], ImageFile]
    ) -> None:
        self.ids_function = ids_function
        self.get_function = get_function

    def __len__(self) -> int:
        return len(self.ids_function())

    def keys(self) -> list[Any]:
        return self.ids_function()

    def items(self) -> list[tuple[Any, ImageFile]]:
        return [(key, self[key]) for key in self.ids_function()]

    def __getitem__(self, index: Any) -> Any:
        keys = self.ids_function()
        if isinstance(index, slice):
            selected = [keys[position] for position in range(*index.indices(len(keys)))]
            return type(self)(lambda: selected, self.get_function)
        if isinstance(index, (str, list, tuple)):
            return self.get_function(index)
        if not isinstance(index, int):
            raise TypeError("Invalid sequence indices type")
        position = index + len(keys) if index < 0 else index
        if not 0 <= position < len(keys):
            raise IndexError("Sequence index out of range")
        return self.get_function(keys[position])

    def __iter__(self) -> Iterator[ImageFile]:
        return (self[index] for index in range(len(self)))

    def __str__(self) -> str:
        names = (f"Image_{index}={key}" for index, key in enumerate(self.ids_function()))
        return f"[{', '.join(names)}]"


def _inline_value(page: Any, key: str, value: Any) -> Any:
    try:
        return NameObject(_INLINE_IMAGE_VALUE_MAPPING[value])
    except TypeError, KeyError:
        pass
    if isinstance(value, NameObject):
        try:
            return page["/Resources"]["/ColorSpace"][value]
        except KeyError as error:
            raise PdfReadError(f"Cannot find resource entry {value} for {key}") from error
    return value


def _inline_object(page: Any, operand: dict[str, Any]) -> StreamObject:
    fields: dict[str, Any] = {
        "__streamdata__": operand["data"],
        "/Length": len(operand["data"]),
    }
    for key, value in operand["settings"].items():
        if key in {"/Length", "/L"}:
            continue
        if isinstance(value, list):
            translated: Any = ArrayObject(_inline_value(page, key, item) for item in value)
        else:
            translated = _inline_value(page, key, value)
        if key not in _INLINE_IMAGE_KEY_MAPPING:
            logger.warning("Unknown inline image key %s, keeping it as-is.", key)
        fields.setdefault(NameObject(_INLINE_IMAGE_KEY_MAPPING.get(key, key)), translated)
    return StreamObject.initialize_from_dictionary(fields)


def content_stream_images(page: Any) -> dict[str, ImageFile | None]:
    """Names drawn by ``Do`` map to None, and inline images to their files."""
    content = page.get_contents()
    if is_null_or_none(content):
        return {}
    inline: list[dict[str, Any]] = []
    files: dict[str, ImageFile | None] = {}
    for operands, operator in content.operations:
        if operator == b"INLINE IMAGE":
            inline.append(operands)
        elif operator == b"Do" and operands:
            files[str(operands[0])] = None
    for number, operand in enumerate(inline):
        extension, data, image = xobject_image(_inline_object(page, operand))
        files[f"~{number}~"] = ImageFile(
            name=f"~{number}~{extension}",
            data=data,
            image=image,
            is_inline=True,
            is_displayed=True,
        )
    return files


def image_ids(
    page: Any,
    node: DictionaryObject | None = None,
    ancestors: list[str] | None = None,
    call_stack: list[Any] | None = None,
) -> list[Any]:
    """Each image's name, or the path to it through nested forms."""
    call_stack = [] if call_stack is None else call_stack
    reference = getattr(node, "indirect_reference", None)
    if reference in call_stack:
        return []
    call_stack.append(reference)
    if page._content_stream_images is None:
        page._content_stream_images = page._parse_images_from_content_stream()
    holder = page if node is None else node
    ancestors = [] if ancestors is None else ancestors
    inline = [name for name, image in page._content_stream_images.items() if image]
    resources = holder.get("/Resources", NullObject()).get_object()
    if not isinstance(resources, DictionaryObject):
        if not is_null_or_none(resources):
            logger.warning("Resources are not a dictionary: %s", resources)
        resources = DictionaryObject()
    if "/XObject" not in resources:
        return [] if ancestors else inline
    ids: list[Any] = []
    x_objects = resources["/XObject"].get_object()
    if not isinstance(x_objects, DictionaryObject):
        logger.warning("XObject resources are not a dictionary: %s", x_objects)
        return []
    for name in x_objects:
        x_object = x_objects[name]
        if not isinstance(x_object, StreamObject):
            continue
        if x_object["/Subtype"] == "/Image":
            ids.append(name if not ancestors else [*ancestors, name])
        else:
            ids.extend(image_ids(page, x_object, [*ancestors, name], call_stack))
    return [*ids, *(name for name in inline if name not in ids)]


def page_image(page: Any, key: Any, node: DictionaryObject | None = None) -> ImageFile:
    holder = page if node is None else node
    key = list(key) if isinstance(key, tuple) else key
    if isinstance(key, list) and len(key) == 1:
        key = key[0]
    is_inline = isinstance(key, str) and key[0] == "~" and key[-1] == "~"
    x_objects: Any = None
    resources = holder.get("/Resources", NullObject()).get_object()
    if isinstance(resources, DictionaryObject) and "/XObject" in resources:
        x_objects = resources["/XObject"]
    elif not is_inline:
        raise KeyError(f"Cannot access image object {key} without XObject resources")
    if not isinstance(key, str):
        return page_image(page, key[1:], x_objects[key[0]])
    if is_inline:
        if page._content_stream_images is None:
            page._content_stream_images = page._parse_images_from_content_stream()
        if key not in page._content_stream_images:
            raise KeyError(f"Image {key} not found")
        return page._content_stream_images[key]
    if key not in x_objects:
        raise KeyError(f"Image {key} not found")
    x_object = x_objects[key]
    if x_object.get("/Subtype", "") != "/Image":
        raise KeyError(f"XObject {key} is not an image")
    displayed = page._content_stream_images is not None and key in page._content_stream_images
    extension, data, image = xobject_image(x_object)
    return ImageFile(
        name=f"{key[1:]}{extension}",
        data=data,
        image=image,
        indirect_reference=x_object.indirect_reference,
        is_displayed=displayed,
    )
