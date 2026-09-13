# SPDX-License-Identifier: AGPL-3.0-only
"""Select an image's mask without evaluating an overridden graphics-state mask."""

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.image_spec import ImageSource, image_smask_in_data


def image_overrides_graphics_soft_mask(source: ImageSource) -> bool:
    """Image masks take precedence over graphics SMask, ISO 32000-2 11.6.4.3."""
    if source.soft_mask is not None or isinstance(
        source.dictionary.get("Mask"), (PdfStream, list, tuple)
    ):
        return True
    try:
        return image_smask_in_data(source.dictionary) != 0
    except ValueError:
        return False
