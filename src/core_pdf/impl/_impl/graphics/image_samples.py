# SPDX-License-Identifier: AGPL-3.0-only
"""Shared natural-component conversion and integer image sample decoding."""

from __future__ import annotations

from typing import Any

import numpy

from core_pdf.impl._impl.graphics.calibrated_colors import calibrated_xyz_to_srgb
from core_pdf.impl._impl.graphics.color_spec import (
    ColorSpace,
    cs_param_floats,
    internal_nchannel_process,
    parse_color_space,
)
from core_pdf.impl._impl.graphics.device_profiles import cmyk_components_to_srgb
from core_pdf.impl._impl.graphics.functions import internal_compile_pdf_function
from core_pdf.impl._impl.graphics.icc_profiles import (
    IccProfileError,
    IccSampleError,
    parse_icc_transform,
)
from core_pdf.impl._impl.graphics.nchannel import internal_mix_nchannel
from core_pdf.impl._impl.runtime.scalars import parse_float
from core_pdf_spec.s_08_graphics.color_kernels import (
    color_key_alpha,
    decode_sample_values,
    unpack_image_samples,
)
from core_pdf_spec.s_08_graphics.color_math import lab_components_to_xyz
from core_pdf_spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering
from core_pdf_spec.s_11_transparency.images import unblend_matte_components


def internal_quantize(values: numpy.ndarray[Any, Any], maximum: int = 255) -> numpy.ndarray:
    return numpy.rint(numpy.clip(values, 0, 1) * maximum).astype(
        numpy.uint8 if maximum == 255 else numpy.uint16
    )


def internal_convert_components(
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
    process = internal_nchannel_process(space)
    if process is not None:
        # PDF 8.6.6.5: process components use their natural values. Missing
        # CMYK components are unpainted inks, not copied from adjacent channels.
        mapped = numpy.zeros((len(values), len(process.component_indices)), dtype=numpy.float64)
        for destination, source in enumerate(process.component_indices):
            if source is not None:
                mapped[:, destination] = values[:, source]
        return internal_convert_components(
            mapped, process.color_space, depth + 1, rendering=rendering
        )
    if kind == "DeviceN":
        mixed = internal_mix_nchannel(
            values,
            space,
            lambda components, target: internal_convert_components(
                components, target, depth + 1, rendering=rendering
            ),
        )
        if mixed is not None:
            return mixed
    if kind in {"DeviceGray", "DeviceRGB"}:
        return internal_quantize(values)
    if kind == "DeviceCMYK":
        return cmyk_components_to_srgb(values, rendering=rendering)
    if kind == "ICCBased":
        try:
            transform = parse_icc_transform(space.icc_profile) if space.icc_profile else None
            if transform is not None and transform.input_channels == values.shape[1]:
                return transform.apply_uint16(internal_quantize(values, 65535), rendering=rendering)
        except IccProfileError, IccSampleError:
            pass
        if space.alternate is not None:
            return internal_convert_components(
                values, space.alternate, depth + 1, rendering=rendering
            )
    if kind == "Indexed" and space.base is not None and space.lookup is not None:
        count = len(space.base.component_ranges)
        entries = numpy.frombuffer(space.lookup, dtype=numpy.uint8)
        if len(entries) < (space.hival + 1) * count:
            raise ValueError("invalid Indexed color lookup")
        table = entries[: (space.hival + 1) * count].reshape(-1, count)
        indices = numpy.floor(values[:, 0] + 0.5).astype(numpy.intp)
        base = decode_sample_values(table[indices], space.base.component_ranges, 255)
        return internal_convert_components(
            base, space.base, depth + 1, matte=matte, alpha=alpha, rendering=rendering
        )
    if kind in {"Separation", "DeviceN"}:
        # Reader recovery for missing/unusable tint transforms is subtractive:
        # zero tint is unpainted, full tint is the darkest approximation.
        # Keep this policy identical for vector, low- and high-depth samples.
        try:
            if space.alternate is None:
                raise ValueError("missing tint alternate")
            function = internal_compile_pdf_function(space.tint_fn)
            distinct, inverse = numpy.unique(values, axis=0, return_inverse=True)
            tinted = numpy.asarray(
                [function(*(float(component) for component in row)) for row in distinct],
                dtype=numpy.float64,
            )
            if tinted.shape != (len(distinct), len(space.alternate.component_ranges)):
                raise ValueError("invalid tint transform output count")
            if not numpy.isfinite(tinted).all():
                raise ValueError("nonfinite tint transform output")
            return internal_convert_components(
                tinted, space.alternate, depth + 1, rendering=rendering
            )[inverse]
        except TypeError, ValueError, ArithmeticError:
            gray = internal_quantize(1 - numpy.max(values, axis=1, keepdims=True))
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
            # PDF stores the XYZ contributions for A, B, C consecutively.
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
    """Decode native unsigned words, then convert colours and source-sample masks.

    ISO 32000-1 8.9.3 and 8.9.6.4: Decode precedes colour conversion but colour
    key comparisons use the original integers, including their low eight bits.
    """
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
    output = internal_convert_components(
        values, space, matte=matte, alpha=alpha, rendering=rendering
    )
    mask = dictionary.get("Mask")
    # An SMask takes precedence over the colour key mask (Table 89).
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
    """Unpack packed integer samples and convert them with their colour-key mask."""
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
