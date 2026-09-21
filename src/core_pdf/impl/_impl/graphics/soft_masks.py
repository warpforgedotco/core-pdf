# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.image_spec import ImageSource, image_smask_in_data


def image_has_color_key_mask(dictionary: dict[Any, Any]) -> bool:
    return isinstance(dictionary.get("Mask"), (list, tuple))


def image_encodes_opacity(dictionary: dict[Any, Any]) -> bool:
    try:
        return image_smask_in_data(dictionary) != 0
    except ValueError:
        return False


def image_color_key_mask_is_shape(dictionary: dict[Any, Any]) -> bool:
    return image_has_color_key_mask(dictionary) and not image_encodes_opacity(dictionary)


def image_overrides_graphics_soft_mask(source: ImageSource) -> bool:
    return (
        source.soft_mask is not None
        or isinstance(source.dictionary.get("Mask"), PdfStream)
        or image_has_color_key_mask(source.dictionary)
        or image_encodes_opacity(source.dictionary)
    )
