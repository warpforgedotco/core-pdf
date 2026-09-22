# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Any

import imagecodecs
import numpy

from core_pdf.impl.graphics.color_spec import (
    ColorSpace,
    cs_param_floats,
    nchannel_attributes,
    nchannel_process,
    parse_color_space,
)
from core_pdf.impl.graphics.device_profiles import cmyk_components_to_srgb
from core_pdf.impl.graphics.functions import compile_pdf_function
from core_pdf.impl.graphics.icc_profiles import (
    IccProfileError,
    IccSampleError,
    cms_options,
    parse_icc_transform,
    srgb_profile,
)
from core_pdf.impl.runtime.scalars import parse_float
from core_pdf_spec.exceptions import PdfParseError, PdfUnsupportedError
from core_pdf_spec.s_07_filters.errors import FilterError
from core_pdf_spec.s_08_graphics.color_kernels import (
    color_key_alpha,
    decode_sample_values,
    unpack_image_samples,
)
from core_pdf_spec.s_08_graphics.color_math import (
    compensate_black_point_xyz,
    lab_components_to_xyz,
    xyz_to_lab_components,
)
from core_pdf_spec.s_08_graphics.color_rendering import (
    DEFAULT_COLOR_RENDERING,
    ColorRendering,
    use_black_point_compensation,
)
from core_pdf_spec.s_11_transparency.images import unblend_matte_components


def quantize(values: numpy.ndarray[Any, Any], maximum: int = 255) -> numpy.ndarray:
    return numpy.rint(numpy.clip(values, 0, 1) * maximum).astype(
        numpy.uint8 if maximum == 255 else numpy.uint16
    )


def convert_components(
    values: numpy.ndarray[Any, Any],
    space: ColorSpace,
    depth: int = 0,
    *,
    matte: tuple[float, ...] | None = None,
    alpha: numpy.ndarray[Any, Any] | None = None,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> numpy.ndarray:
    if depth > 8 or not space.component_ranges:
        raise ValueError("invalid image color space")
    if values.ndim != 2 or values.shape[1] != len(space.component_ranges):
        raise ValueError("invalid color component count")
    if space.kind == "Indexed" and not numpy.isfinite(values).all():
        raise ValueError("invalid Indexed color component")
    if matte is not None and space.kind != "Indexed":
        if alpha is None:
            raise ValueError("missing image matte opacity")
        values = unblend_matte_components(values, alpha, matte)
    low, high = numpy.asarray(space.component_ranges, dtype=numpy.float64).T
    values = numpy.clip(values, low, high)
    kind = space.kind
    process = nchannel_process(space)
    if process is not None:
        mapped = numpy.zeros((len(values), len(process.component_indices)), dtype=numpy.float64)
        for destination, source in enumerate(process.component_indices):
            if source is not None:
                mapped[:, destination] = values[:, source]
        return convert_components(mapped, process.color_space, depth + 1, rendering=rendering)
    if kind == "DeviceN":
        mixed = mix_nchannel(
            values,
            space,
            lambda components, target: convert_components(
                components, target, depth + 1, rendering=rendering
            ),
        )
        if mixed is not None:
            return mixed
    if kind in {"DeviceGray", "DeviceRGB"}:
        return quantize(values)
    if kind == "DeviceCMYK":
        return cmyk_components_to_srgb(values, rendering=rendering)
    if kind == "ICCBased":
        try:
            transform = parse_icc_transform(space.icc_profile) if space.icc_profile else None
            if transform is not None and transform.input_channels == values.shape[1]:
                return transform.apply_uint16(quantize(values, 65535), rendering=rendering)
        except IccProfileError, IccSampleError:
            pass
        if space.alternate is not None:
            return convert_components(values, space.alternate, depth + 1, rendering=rendering)
    if kind == "Indexed" and space.base is not None and space.lookup is not None:
        count = len(space.base.component_ranges)
        entries = numpy.frombuffer(space.lookup, dtype=numpy.uint8)
        if len(entries) < (space.hival + 1) * count:
            raise ValueError("invalid Indexed color lookup")
        table = entries[: (space.hival + 1) * count].reshape(-1, count)
        indices = numpy.floor(values[:, 0] + 0.5).astype(numpy.intp)
        base = decode_sample_values(table[indices], space.base.component_ranges, 255)
        return convert_components(
            base, space.base, depth + 1, matte=matte, alpha=alpha, rendering=rendering
        )
    if kind in {"Separation", "DeviceN"}:
        try:
            if space.alternate is None:
                raise ValueError("missing tint alternate")
            function = compile_pdf_function(space.tint_fn)
            distinct, inverse = numpy.unique(values, axis=0, return_inverse=True)
            tinted = numpy.asarray(
                [function(*(float(component) for component in row)) for row in distinct],
                dtype=numpy.float64,
            )
            if tinted.shape != (len(distinct), len(space.alternate.component_ranges)):
                raise ValueError("invalid tint transform output count")
            if not numpy.isfinite(tinted).all():
                raise ValueError("nonfinite tint transform output")
            return convert_components(tinted, space.alternate, depth + 1, rendering=rendering)[
                inverse
            ]
        except TypeError, ValueError, ArithmeticError:
            gray = quantize(1 - numpy.max(values, axis=1, keepdims=True))
            return numpy.repeat(gray, 3, axis=1)
    if kind in {"Lab", "CalGray", "CalRGB"}:
        white = cs_param_floats(space.params, "WhitePoint", 3, [0.9642, 1, 0.8249])
        if kind == "Lab":
            xyz = lab_components_to_xyz(
                values.astype(numpy.float32), (white[0], white[1], white[2])
            )
        elif kind == "CalGray":
            exponent = parse_float(space.params.get("Gamma", 1), None)
            if exponent is None or exponent <= 0:
                raise ValueError("invalid CalGray Gamma")
            xyz = (values**exponent * numpy.asarray(white)).astype(numpy.float32)
        else:
            gamma = cs_param_floats(space.params, "Gamma", 3, [1, 1, 1])
            matrix = cs_param_floats(space.params, "Matrix", 9, [1, 0, 0, 0, 1, 0, 0, 0, 1])
            xyz = ((values ** numpy.asarray(gamma)) @ numpy.asarray(matrix).reshape(3, 3)).astype(
                numpy.float32
            )
        black = cs_param_floats(space.params, "BlackPoint", 3, [0, 0, 0])
        return calibrated_xyz_to_srgb(
            xyz, (white[0], white[1], white[2]), (black[0], black[1], black[2]), rendering
        )
    raise ValueError("unsupported image color space")


def convert_integer_samples(
    samples: numpy.ndarray,
    dictionary: dict[Any, Any],
    *,
    bits_per_component: int = 16,
    matte: tuple[float, ...] | None = None,
    alpha: numpy.ndarray[Any, Any] | None = None,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> numpy.ndarray:
    space = parse_color_space(dictionary.get("ColorSpace"))
    count = len(space.component_ranges)
    if count <= 0:
        raise ValueError("invalid image color space")
    maximum = (1 << bits_per_component) - 1
    integers = numpy.asarray(samples, dtype=numpy.uint16).reshape(-1, count)
    decode = dictionary.get("Decode")
    if decode is None:
        pairs = ((0.0, float(maximum)),) if space.kind == "Indexed" else space.component_ranges
    else:
        numbers = numpy.asarray(decode, dtype=numpy.float64)
        if numbers.shape != (count * 2,) or not numpy.isfinite(numbers).all():
            raise ValueError("invalid image Decode array")
        pairs = tuple((float(low), float(high)) for low, high in numbers.reshape(-1, 2))
    values = decode_sample_values(integers, pairs, maximum)
    output = convert_components(values, space, matte=matte, alpha=alpha, rendering=rendering)
    mask = dictionary.get("Mask")
    if isinstance(mask, (list, tuple)) and dictionary.get("SMask") is None:
        output = numpy.column_stack((output, color_key_alpha(integers, tuple(mask), maximum)))
    return output


def convert_integer_image(
    data: bytes | memoryview,
    dictionary: dict[Any, Any],
    *,
    bits_per_component: int = 16,
    matte: tuple[float, ...] | None = None,
    alpha: numpy.ndarray[Any, Any] | None = None,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> numpy.ndarray:
    space = parse_color_space(dictionary.get("ColorSpace"))
    samples = unpack_image_samples(
        data,
        bits_per_component,
        int(dictionary["Width"]),
        int(dictionary["Height"]),
        len(space.component_ranges),
    )
    return convert_integer_samples(
        samples,
        dictionary,
        bits_per_component=bits_per_component,
        matte=matte,
        alpha=alpha,
        rendering=rendering,
    )


ColorSamples = numpy.ndarray[Any, numpy.dtype[numpy.float32]]
SRGB_MATRIX = numpy.asarray(
    (
        (3.2404542, -1.5371385, -0.4985314),
        (-0.9692660, 1.8760108, 0.0415560),
        (0.0556434, -0.2040259, 1.0572252),
    ),
    dtype=numpy.float32,
)
D50_TO_D65_MATRIX = numpy.asarray(
    (
        (0.955473, -0.023098, 0.063259),
        (-0.028369, 1.009995, 0.021300),
        (0.012314, -0.020507, 1.330365),
    ),
    dtype=numpy.float32,
)
D50_XYZ_TO_SRGB_MATRIX = (SRGB_MATRIX @ D50_TO_D65_MATRIX).astype(numpy.float32)


def linear_to_srgb(values: ColorSamples) -> ColorSamples:
    clipped = numpy.clip(values, 0.0, None)
    return numpy.where(
        clipped <= 0.0031308,
        12.92 * clipped,
        1.055 * numpy.power(clipped, 1.0 / 2.4) - 0.055,
    ).astype(numpy.float32, copy=False)


def d50_xyz_to_srgb(values: ColorSamples) -> ColorSamples:
    return linear_to_srgb(values @ D50_XYZ_TO_SRGB_MATRIX.T)


@lru_cache(maxsize=64)
def lab_profile(white: tuple[float, float, float]) -> bytes:
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
    intent, flags = cms_options(rendering)
    try:
        converted = imagecodecs.cms_transform(
            numpy.ascontiguousarray(lab).reshape(-1, 1, 3),
            lab_profile(white),
            srgb_profile(),
            colorspace="lab",
            outcolorspace="rgb",
            outdtype=numpy.uint8,
            intent=intent,
            flags=flags,
        )
    except imagecodecs.CmsError as exc:
        raise IccProfileError("invalid calibrated output profile") from exc
    return numpy.asarray(converted, dtype=numpy.uint8).reshape(-1, 3)


def mix_nchannel(
    values: numpy.ndarray,
    space: ColorSpace,
    convert: Callable[[numpy.ndarray, ColorSpace], numpy.ndarray],
) -> numpy.ndarray | None:
    attributes = nchannel_attributes(space)
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
        zero = numpy.zeros((1, 1), dtype=numpy.float64)
        for _, spot in spots:
            if not numpy.all(rgb_appearance(convert(zero, spot)) == 1):
                return None
        mixed = numpy.ones((len(values), 3), dtype=numpy.float64)
        if process is not None:
            mapped = numpy.zeros((len(values), len(process.component_indices)), dtype=numpy.float64)
            for destination, source in enumerate(process.component_indices):
                if source is not None:
                    mapped[:, destination] = values[:, source]
            mixed = rgb_appearance(convert(mapped, process.color_space))
        for source, spot in spots:
            mixed *= rgb_appearance(convert(values[:, source : source + 1], spot))
        return numpy.rint(numpy.clip(mixed, 0, 1) * 255).astype(numpy.uint8)
    except (
        TypeError,
        ValueError,
        ArithmeticError,
        FilterError,
        PdfParseError,
        PdfUnsupportedError,
    ):
        return None


def rgb_appearance(converted: numpy.ndarray) -> numpy.ndarray:
    if converted.ndim != 2 or converted.shape[1] not in {1, 3}:
        raise ValueError("invalid NChannel component appearance")
    rgb = converted.astype(numpy.float64) / 255
    return numpy.repeat(rgb, 3, axis=1) if rgb.shape[1] == 1 else rgb
