# SPDX-License-Identifier: AGPL-3.0-only
"""Selected sRGB screen approximation for NChannel spot/process combinations.

ISO 32000-2, 8.6.6.5 permits reader-selected blending instead of the global tint
function. Multiplication here is a core output policy, not a PDF-mandated ink or
spectral equation. It does not retain separate plates for overprint/transparency.
"""

from collections.abc import Callable, Mapping

import numpy

from core_pdf.impl._impl.graphics.color_spec import ColorSpace, internal_nchannel_attributes
from core_pdf_spec.exceptions import PdfParseError, PdfUnsupportedError
from core_pdf_spec.s_07_filters.errors import FilterError


def internal_mix_nchannel(
    values: numpy.ndarray,
    space: ColorSpace,
    convert: Callable[[numpy.ndarray, ColorSpace], numpy.ndarray],
) -> numpy.ndarray | None:
    """Combine process and spot appearances, or request the whole global fallback.

    Only an empty/absent MixingHints dictionary and white zero-tint spot endpoints
    are supported. Other papers, ink opacities, laydown orders and dot gain need a
    different mixing policy; silently normalizing their endpoints would lose meaning.
    The callback carries rendering intent, BPC and recursion limits into every space.
    """
    attributes = internal_nchannel_attributes(space)
    if attributes is None:
        return None
    raw_attributes = space.params.get("Attributes")
    if isinstance(raw_attributes, Mapping) and raw_attributes.get("MixingHints"):
        return None
    process = attributes.process
    process_indices = (
        {index for index in process.component_indices if index is not None}
        if process is not None
        else set()
    )
    spots = tuple(
        (index, attributes.colorants[name])
        for index, name in enumerate(space.colorants)
        if index not in process_indices
    )
    if not spots:
        return None
    try:
        # A zero amount of every spot ink must leave the process baseline alone.
        # A nonwhite endpoint may encode a different paper; defer to the writer's
        # complete transform rather than guessing how to remove that paper.
        zero = numpy.zeros((1, 1), dtype=numpy.float64)
        for _, spot in spots:
            if not numpy.all(internal_rgb_appearance(convert(zero, spot)) == 1):
                return None
        mixed = numpy.ones((len(values), 3), dtype=numpy.float64)
        if process is not None:
            mapped = numpy.zeros((len(values), len(process.component_indices)), dtype=numpy.float64)
            for destination, source in enumerate(process.component_indices):
                if source is not None:
                    mapped[:, destination] = values[:, source]
            mixed = internal_rgb_appearance(convert(mapped, process.color_space))
        for source, spot in spots:
            mixed *= internal_rgb_appearance(convert(values[:, source : source + 1], spot))
        return numpy.rint(numpy.clip(mixed, 0, 1) * 255).astype(numpy.uint8)
    except (
        TypeError,
        ValueError,
        ArithmeticError,
        FilterError,
        PdfParseError,
        PdfUnsupportedError,
    ):
        # A damaged individual description must not leave a partially mixed result.
        # Authentication errors and unexpected decoder failures still propagate.
        return None


def internal_rgb_appearance(converted: numpy.ndarray) -> numpy.ndarray:
    if converted.ndim != 2 or converted.shape[1] not in {1, 3}:
        raise ValueError("invalid NChannel component appearance")
    rgb = converted.astype(numpy.float64) / 255
    return numpy.repeat(rgb, 3, axis=1) if rgb.shape[1] == 1 else rgb
