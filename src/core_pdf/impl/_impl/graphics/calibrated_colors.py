# SPDX-License-Identifier: AGPL-3.0-only

from functools import lru_cache
from typing import Any

import imagecodecs
import numpy

from core_pdf.impl._impl.graphics.color_math import d50_xyz_to_srgb
from core_pdf.impl._impl.graphics.icc_profiles import (
    IccProfileError,
    internal_cms_options,
    internal_srgb_profile,
)
from core_pdf_spec.s_08_graphics.color_math import compensate_black_point_xyz, xyz_to_lab_components
from core_pdf_spec.s_08_graphics.color_rendering import (
    DEFAULT_COLOR_RENDERING,
    ColorRendering,
    use_black_point_compensation,
)


@lru_cache(maxsize=64)
def internal_lab_profile(white: tuple[float, float, float]) -> bytes:
    total = sum(white)
    return bytes(
        imagecodecs.cms_profile("lab4", whitepoint=(white[0] / total, white[1] / total, white[1]))
    )


def calibrated_xyz_to_srgb(
    xyz: numpy.ndarray[Any, Any],
    white: tuple[float, float, float],
    black: tuple[float, float, float],
    rendering: ColorRendering,
) -> numpy.ndarray[Any, Any]:
    if rendering == DEFAULT_COLOR_RENDERING:
        return numpy.rint(
            numpy.clip(d50_xyz_to_srgb(xyz.astype(numpy.float32)), 0, 1) * 255
        ).astype(numpy.uint8)
    values = xyz
    if use_black_point_compensation(rendering, default=True) and any(black):
        values = compensate_black_point_xyz(values, white, black, (0.0, 0.0, 0.0))
    lab = xyz_to_lab_components(values, white)
    intent, flags = internal_cms_options(rendering)
    try:
        converted = imagecodecs.cms_transform(
            numpy.ascontiguousarray(lab).reshape(-1, 1, 3),
            internal_lab_profile(white),
            internal_srgb_profile(),
            colorspace="lab",
            outcolorspace="rgb",
            outdtype=numpy.uint8,
            intent=intent,
            flags=flags,
        )
    except imagecodecs.CmsError as exc:
        raise IccProfileError("invalid calibrated output profile") from exc
    return numpy.asarray(converted, dtype=numpy.uint8).reshape(-1, 3)
