# SPDX-License-Identifier: AGPL-3.0-only
"""Select an image's mask without evaluating an overridden graphics-state mask."""

from typing import Any

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.image_spec import ImageSource, image_smask_in_data


def image_has_color_key_mask(dictionary: dict[Any, Any]) -> bool:
    """Whether the image declares a colour-key Mask array, ISO 32000-2 8.9.6.4."""
    return isinstance(dictionary.get("Mask"), (list, tuple))


def image_encodes_opacity(dictionary: dict[Any, Any]) -> bool:
    """Whether JPX SMaskInData carries opacity; malformed values mean it does not."""
    try:
        return image_smask_in_data(dictionary) != 0
    except ValueError:
        return False


def image_color_key_mask_is_shape(dictionary: dict[Any, Any]) -> bool:
    """A colour-key Mask is intrinsic hard shape unless SMaskInData overrides it."""
    return image_has_color_key_mask(dictionary) and not image_encodes_opacity(dictionary)


def image_overrides_graphics_soft_mask(source: ImageSource) -> bool:
    """Image masks take precedence over graphics SMask, ISO 32000-2 11.6.4.3."""
    return (
        source.soft_mask is not None
        or isinstance(source.dictionary.get("Mask"), PdfStream)
        or image_has_color_key_mask(source.dictionary)
        or image_encodes_opacity(source.dictionary)
    )
