# SPDX-License-Identifier: AGPL-3.0-only
"""Native PDF color conversion."""

from __future__ import annotations

import typing
from collections.abc import Sequence
from contextlib import suppress
from typing import Any, TypeAlias

import numpy

from core_pdf.impl._impl.graphics.color_math import (
    d50_xyz_to_srgb,
)
from core_pdf.impl._impl.graphics.color_spec import (
    ColorParams,
    ColorSpace,
    cs_param_floats,
    internal_nchannel_process,
    parse_color_space,
    recover_image_bits_per_component,
)
from core_pdf.impl._impl.graphics.device_profiles import (
    cmyk_bytes_to_srgb,
    cmyk_floats_to_srgb,
    internal_component_byte,
)
from core_pdf.impl._impl.graphics.functions import internal_compile_pdf_function
from core_pdf.impl._impl.graphics.icc_profiles import (
    IccProfileError,
    IccSampleError,
    parse_icc_transform,
)
from core_pdf.impl._impl.graphics.image_kernels import (
    apply_decode_array as apply_decode_array_kernel,
)
from core_pdf.impl._impl.graphics.image_kernels import (
    image_dimension,
    unpack_subbyte_image_samples,
)
from core_pdf.impl._impl.graphics.image_samples import (
    convert_16bit_image,
    convert_integer_samples,
    internal_convert_components,
)
from core_pdf.impl._impl.runtime.array_views import ByteBuffer, uint8_view
from core_pdf.impl._impl.runtime.scalars import parse_float
from core_pdf_spec.s_08_graphics.color import indexed_color_components
from core_pdf_spec.s_08_graphics.color_kernels import unpack_image_samples
from core_pdf_spec.s_08_graphics.color_math import lab_components_to_xyz
from core_pdf_spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering

ImageDict: TypeAlias = dict[str, object]
ColorComponents: TypeAlias = list[float]
ImageBuffer: TypeAlias = ByteBuffer


def internal_requires_component_conversion(space: ColorSpace) -> bool:
    return (
        space.kind in {"ICCBased", "CalGray", "CalRGB", "Lab"}
        or internal_nchannel_process(space) is not None
        or space.base is not None
        and internal_requires_component_conversion(space.base)
        or space.alternate is not None
        and internal_requires_component_conversion(space.alternate)
    )


def color_operands_to_srgb(
    spec: ColorSpace,
    components: Sequence[float],
    *,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> tuple[float, float, float] | None:
    """Convert one colour's `sc`/`scn` operands, in ``spec``'s space, to sRGB.

    Returns ``None`` for the device spaces, whose operands already carry their
    own component count and are handled downstream.

    ISO 32000-1 8.6.6.3 and 8.6.6.4: an Indexed operand is an index into the
    palette, and a Separation/DeviceN operand is a tint that the tint transform
    maps into the alternate space. Both were previously clamped to 0..1 and
    painted as if they were device components, so a spot colour rendered as an
    *inverted* grey ("a tint value of 0.0 denotes the lightest colour ... and
    1.0 is the darkest") and an index painted black or white.
    """
    kind = spec.kind
    if internal_requires_component_conversion(spec) or (
        rendering != DEFAULT_COLOR_RENDERING and kind not in {"DeviceGray", "DeviceRGB", "Pattern"}
    ):
        if kind == "DeviceCMYK":
            from core_pdf.impl._impl.graphics.device_profiles import cmyk_floats_to_srgb

            if len(components) != 4:
                return None
            red, green, blue = cmyk_floats_to_srgb(*components, rendering=rendering)
            return red / 255.0, green / 255.0, blue / 255.0
        try:
            converted = internal_convert_components(
                numpy.asarray([components], dtype=numpy.float64), spec, rendering=rendering
            )[0]
            if len(converted) == 1:
                converted = numpy.repeat(converted, 3)
            return float(converted[0]) / 255, float(converted[1]) / 255, float(converted[2]) / 255
        except (TypeError, ValueError):
            return None
    if kind == "Indexed":
        return internal_indexed_operand_to_srgb(spec, components)
    if kind in {"Separation", "DeviceN"}:
        return internal_tint_operands_to_srgb(spec, components)
    return None


def internal_indexed_operand_to_srgb(
    spec: ColorSpace, components: Sequence[float]
) -> tuple[float, float, float] | None:
    if spec.base is None or not components:
        return None
    base = internal_output_color_space(spec.base)
    try:
        values = indexed_color_components(spec, components[0])
    except ValueError:
        return None
    return internal_srgb_bytes_to_floats(internal_apply_alt_color(list(values), base.kind))


def internal_tint_operands_to_srgb(
    spec: ColorSpace, components: Sequence[float]
) -> tuple[float, float, float] | None:
    if not components:
        return None
    alt = internal_output_color_space(spec.alternate) if spec.alternate is not None else None
    tints = [max(0.0, min(1.0, float(value))) for value in components]
    if alt is not None and spec.tint_fn is not None:
        try:
            expected = len(alt.component_ranges)
            evaluated = internal_evaluate_tint(spec.tint_fn, tints)
            if len(evaluated) == expected:
                converted = internal_srgb_bytes_to_floats(
                    internal_apply_alt_color(list(evaluated), alt.kind)
                )
                if converted is not None:
                    return converted
        except (ValueError, TypeError, ZeroDivisionError):
            pass
    # No usable tint transform. 8.6.6.4 still fixes the direction: "a tint value
    # of 0.0 denotes the lightest colour that can be achieved with the given
    # colorant, and 1.0 is the darkest", so the ink is subtractive.
    ink = max(tints)
    level = 1.0 - ink
    return (level, level, level)


def internal_evaluate_tint(tint_fn: object, tints: list[float]) -> tuple[float, ...]:
    """Evaluate a tint transform, propagating unsupported or malformed functions."""
    return internal_compile_pdf_function(tint_fn)(*tints)


def internal_srgb_bytes_to_floats(rgb: bytes | None) -> tuple[float, float, float] | None:
    if rgb is None or len(rgb) < 3:
        return None
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)


def internal_output_color_space(space: ColorSpace) -> ColorSpace:
    """Use the reader's selected ICC alternate for device-only conversion paths."""
    while space.kind == "ICCBased" and space.alternate is not None:
        space = space.alternate
    return space


def internal_separation_rgb_lut(
    tint_fn: object,
    alternate: ColorSpace,
) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
    """Compile a one-input Separation function to an 8-bit RGB lookup table."""
    expected = len(alternate.component_ranges)
    alt_name = alternate.kind
    try:
        evaluate = internal_compile_pdf_function(tint_fn)
    except ValueError as exc:
        raise ValueError("invalid separation tint function") from exc
    table = numpy.empty((256, 3), dtype=numpy.uint8)
    for value in range(256):
        try:
            components = list(evaluate(value / 255.0))
        except Exception as exc:
            raise ValueError("invalid separation tint function") from exc
        if len(components) != expected:
            raise ValueError("invalid separation tint function")
        rgb = internal_apply_alt_color(components, alt_name)
        if rgb is None:
            raise ValueError("invalid Separation color space")
        table[value] = tuple(rgb)
    return table


def internal_convert_image_data(
    raw: ImageBuffer,
    image_dict: ImageDict,
    *,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> ImageBuffer | None:
    """Convert encoded PDF image samples to grayscale or sRGB bytes."""
    spec = parse_color_space(image_dict.get("ColorSpace"))
    bits_per_component = recover_image_bits_per_component(image_dict)
    if bits_per_component == 16:
        return convert_16bit_image(
            memoryview(raw).cast("B"), image_dict, rendering=rendering
        ).reshape(-1)
    if bits_per_component not in {1, 2, 4, 8} or not spec.component_ranges:
        return None

    if rendering != DEFAULT_COLOR_RENDERING or (
        spec.kind in {"Indexed", "Separation", "DeviceN"}
        and internal_requires_component_conversion(spec)
    ):
        source_samples = unpack_image_samples(
            memoryview(raw).cast("B"),
            bits_per_component,
            image_dimension(image_dict, "Width"),
            image_dimension(image_dict, "Height"),
            len(spec.component_ranges),
        )
        return convert_integer_samples(
            source_samples, image_dict, bits_per_component=bits_per_component, rendering=rendering
        ).reshape(-1)

    fast = internal_simple_device_color_fast_path(raw, spec, image_dict, bits_per_component)
    if fast is not None:
        return fast
    samples = internal_normalize_image_samples(raw, spec, image_dict, bits_per_component)
    if samples is None:
        return None
    return internal_convert_color_samples(samples, spec)


def internal_convert_color_samples(
    samples: ImageBuffer,
    spec: ColorSpace,
    depth: int = 0,
) -> ImageBuffer | None:
    if depth > 3:
        return None
    kind = spec.kind
    if kind == "DeviceRGB":
        return samples
    if kind == "DeviceGray":
        return internal_convert_gray(samples)
    if kind == "DeviceCMYK":
        return internal_convert_cmyk(samples)
    if kind == "Lab":
        return internal_convert_lab(samples, spec)
    if kind == "CalGray":
        return internal_convert_calgray(samples, spec.params)
    if kind == "CalRGB":
        return internal_convert_calrgb(samples, spec.params)
    if kind == "Indexed":
        return internal_convert_indexed(samples, spec)
    if kind == "Separation":
        return internal_convert_separation(samples, spec)
    if kind == "DeviceN":
        return internal_convert_devicen(samples, spec)
    if kind != "ICCBased":
        return None

    transform = None
    if spec.icc_profile is not None:
        with suppress(IccProfileError):
            transform = parse_icc_transform(spec.icc_profile)
    if transform is not None and transform.input_channels == len(spec.component_ranges):
        values = uint8_view(samples)
        if len(values) % transform.input_channels == 0:
            try:
                return transform.apply_uint8(values.reshape(-1, transform.input_channels)).reshape(
                    -1
                )
            except (IccProfileError, IccSampleError):
                pass
    fallback = spec.alternate
    if fallback is None:
        raise ValueError("invalid ICCBased color space")
    return internal_convert_color_samples(
        samples,
        fallback,
        depth + 1,
    )


def internal_normalize_image_samples(
    raw: ImageBuffer,
    spec: ColorSpace,
    image_dict: ImageDict,
    bits_per_component: int,
) -> ImageBuffer | None:
    if bits_per_component == 8:
        return internal_apply_decode_array(raw, spec, image_dict, bits_per_component)
    width = image_dimension(image_dict, "Width")
    height = image_dimension(image_dict, "Height")
    if width <= 0 or height <= 0:
        return None
    components = len(spec.component_ranges)
    unpacked = unpack_subbyte_image_samples(
        raw,
        bits_per_component,
        width,
        height,
        components,
    )
    if spec.kind == "Indexed":
        return unpacked
    return internal_apply_decode_array(unpacked, spec, image_dict, bits_per_component)


def internal_simple_device_color_fast_path(
    raw: ImageBuffer,
    spec: ColorSpace,
    image_dict: ImageDict,
    bits_per_component: int,
) -> ImageBuffer | None:
    if bits_per_component != 8:
        return None
    if spec.kind not in {"DeviceRGB", "DeviceGray"}:
        return None
    if image_dict.get("Decode") is not None:
        return None
    width = image_dimension(image_dict, "Width")
    height = image_dimension(image_dict, "Height")
    if width <= 0 or height <= 0:
        return None
    expected = width * height * (3 if spec.kind == "DeviceRGB" else 1)
    if len(raw) != expected:
        return None
    # Native grayscale samples stay one-channel for extraction consumers;
    # renderers expand them at the final compositing boundary.
    return raw


def internal_apply_decode_array(
    samples: ImageBuffer,
    spec: ColorSpace,
    image_dict: ImageDict,
    bits_per_component: int,
) -> ImageBuffer:
    if spec.kind == "Indexed":
        return samples
    components = len(spec.component_ranges)
    if components <= 0:
        return samples
    max_sample = (1 << bits_per_component) - 1
    if max_sample <= 0:
        return samples
    decode = image_dict.get("Decode")
    pairs: list[tuple[float, float]] = []
    if isinstance(decode, (list, tuple)) and len(decode) >= components * 2:
        for index in range(components):
            try:
                dmin = float(typing.cast(typing.Any, decode[index * 2]))
                dmax = float(typing.cast(typing.Any, decode[index * 2 + 1]))
            except (TypeError, ValueError):
                pairs = []
                break
            pairs.append((dmin, dmax))
    if not pairs:
        pairs = [(0.0, 1.0)] * components
    if bits_per_component == 8 and all(pair == (0.0, 1.0) for pair in pairs):
        return samples
    return apply_decode_array_kernel(samples, tuple(pairs), max_sample)


def internal_convert_separation(
    raw: ImageBuffer,
    color_space: ColorSpace,
) -> ImageBuffer | None:
    alternate = (
        internal_output_color_space(color_space.alternate)
        if color_space.alternate is not None
        else parse_color_space("DeviceGray")
    )
    alt_name = alternate.kind
    tint_fn = color_space.tint_fn
    if tint_fn is None and alt_name in {"DeviceGray", "DeviceRGB", "DeviceCMYK"}:
        samples = uint8_view(raw)
        result = numpy.empty((len(samples), 3), dtype=numpy.uint8)
        if alt_name in {"DeviceGray", "DeviceRGB"}:
            result[:] = samples[:, None]
            return result.reshape(-1)
        inks = numpy.zeros((len(samples), 4), dtype=numpy.uint8)
        inks[:, 0] = samples
        return cmyk_bytes_to_srgb(inks).reshape(-1)

    return internal_separation_rgb_lut(tint_fn, alternate)[uint8_view(raw)].reshape(-1)


def internal_convert_devicen(
    raw: ImageBuffer,
    color_space: ColorSpace,
) -> ImageBuffer | None:
    alternate = (
        internal_output_color_space(color_space.alternate)
        if color_space.alternate is not None
        else None
    )
    alt_name = alternate.kind if alternate is not None else ""
    channels = len(color_space.component_ranges)
    tint_fn = color_space.tint_fn
    if channels <= 0:
        raise ValueError("invalid DeviceN color space")
    if len(raw) % channels != 0:
        raise ValueError("invalid DeviceN color sample data")

    if tint_fn is None and alt_name in {"DeviceGray", "DeviceRGB", "DeviceCMYK"}:
        samples = uint8_view(raw).reshape(-1, channels)
        if alt_name == "DeviceGray":
            return numpy.repeat(samples[:, :1], 3, axis=1).reshape(-1)
        if alt_name == "DeviceRGB":
            result = numpy.empty((len(samples), 3), dtype=numpy.uint8)
            result[:, 0] = samples[:, 0]
            result[:, 1] = samples[:, 1] if channels >= 2 else samples[:, 0]
            result[:, 2] = samples[:, 2] if channels >= 3 else samples[:, 0]
            return result.reshape(-1)
        carried = min(channels, 4)
        inks = numpy.zeros((len(samples), 4), dtype=numpy.uint8)
        inks[:, :carried] = samples[:, :carried]
        return cmyk_bytes_to_srgb(inks).reshape(-1)

    if alt_name not in {"DeviceGray", "DeviceRGB", "DeviceCMYK"}:
        raise ValueError("invalid DeviceN color space")
    expected = len(alternate.component_ranges) if alternate is not None else 0
    samples = uint8_view(raw).reshape(-1, channels)
    distinct, inverse = numpy.unique(samples, axis=0, return_inverse=True)
    tinted = numpy.empty((len(distinct), expected), dtype=numpy.float64)
    try:
        evaluate = internal_compile_pdf_function(tint_fn)
    except ValueError as exc:
        raise ValueError("invalid DeviceN tint function") from exc
    for index, row in enumerate(distinct.tolist()):
        components: ColorComponents = [value / 255.0 for value in row]
        try:
            components = list(evaluate(*components))
        except Exception as exc:
            raise ValueError("invalid DeviceN tint function") from exc
        if len(components) != expected:
            raise ValueError("invalid DeviceN tint function")
        tinted[index] = components

    scaled = numpy.clip(numpy.rint(numpy.clip(tinted, 0.0, 1.0) * 255.0), 0.0, 255.0)
    inks = scaled.astype(numpy.uint8)
    if alt_name == "DeviceCMYK":
        converted = cmyk_bytes_to_srgb(inks)
    elif alt_name == "DeviceRGB":
        converted = inks
    else:
        converted = numpy.repeat(inks, 3, axis=1)
    return converted[numpy.asarray(inverse).reshape(-1)].reshape(-1)


def internal_convert_gray(
    raw: ImageBuffer,
) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
    return numpy.repeat(uint8_view(raw), 3)


def internal_convert_cmyk(
    raw: ImageBuffer,
) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
    if len(raw) % 4 != 0:
        raise ValueError("invalid color sample data")
    return cmyk_bytes_to_srgb(uint8_view(raw).reshape(-1, 4)).reshape(-1)


def internal_convert_indexed(raw: ImageBuffer, spec: ColorSpace) -> ImageBuffer | None:
    lookup = spec.lookup
    if lookup is None or spec.hival < 0:
        return None
    base = internal_output_color_space(spec.base) if spec.base is not None else None
    hival = spec.hival
    samples = uint8_view(raw)
    if numpy.any(samples > hival):
        samples = numpy.minimum(samples, hival)
    if base is not None and base.kind == "DeviceRGB":
        if len(lookup) < (hival + 1) * 3:
            raise ValueError("invalid Indexed color lookup")
        table = uint8_view(lookup).reshape(-1, 3)
        return table[samples].reshape(-1)
    if base is not None and base.kind == "DeviceGray":
        if len(lookup) < hival + 1:
            raise ValueError("invalid Indexed color lookup")
        values = uint8_view(lookup)[samples]
        return numpy.repeat(values[:, None], 3, axis=1).reshape(-1)
    if base is not None and base.kind == "DeviceCMYK":
        if len(lookup) < (hival + 1) * 4:
            raise ValueError("invalid Indexed color lookup")
        table = uint8_view(lookup)[: (hival + 1) * 4].reshape(-1, 4)
        return internal_convert_cmyk(table[samples].reshape(-1))
    raise ValueError("invalid Indexed color space")


def internal_convert_calgray(raw: ImageBuffer, params: ColorParams) -> ImageBuffer:
    # Gamma is parsed only to reject malformed parameters; the conversion
    # itself intentionally treats CalGray as DeviceGray.
    if parse_float(params.get("Gamma", 1.0), None) is None:
        raise ValueError("invalid color space parameters")
    return internal_convert_gray(raw)


def internal_convert_calrgb(raw: ImageBuffer, params: ColorParams) -> ImageBuffer:
    black_point = cs_param_floats(params, "BlackPoint", 3, [0.0, 0.0, 0.0])
    gamma = cs_param_floats(params, "Gamma", 3, [1.0, 1.0, 1.0])
    matrix = cs_param_floats(
        params,
        "Matrix",
        9,
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )
    samples = uint8_view(raw).reshape(-1, 3)
    values = samples.astype(numpy.float64) / 255.0
    for index, exponent in enumerate(gamma):
        if exponent != 1.0:
            values[:, index] = numpy.power(values[:, index], exponent)
    matrix_array = numpy.asarray(matrix, dtype=numpy.float64).reshape(3, 3)
    black_point_array = numpy.asarray(black_point, dtype=numpy.float64)
    xyz = (values @ matrix_array.T + black_point_array).astype(numpy.float32)
    rgb = d50_xyz_to_srgb(xyz)
    return numpy.clip(rgb * 255.0, 0.0, 255.0).astype(numpy.uint8).reshape(-1)


def internal_convert_lab(raw: ImageBuffer, space: ColorSpace) -> ImageBuffer:
    white_point = cs_param_floats(space.params, "WhitePoint", 3, [0.9505, 1.0, 1.089])
    range_a, range_b = space.component_ranges[1:]
    samples = uint8_view(raw).reshape(-1, 3)
    a_span = range_a[1] - range_a[0]
    lab = samples.astype(numpy.float32)
    lab[:, 0] = lab[:, 0] / 255.0 * 100.0
    lab[:, 1] = lab[:, 1] / 255.0 * a_span + range_a[0]
    lab[:, 2] = lab[:, 2] / 255.0 * (range_b[1] - range_b[0]) + range_b[0]
    xyz = lab_components_to_xyz(lab, (white_point[0], white_point[1], white_point[2]))
    rgb = d50_xyz_to_srgb(xyz)
    return numpy.clip(rgb * 255.0, 0.0, 255.0).astype(numpy.uint8).reshape(-1)


def internal_apply_alt_color(components: ColorComponents, alt_name: str) -> bytes | None:
    if alt_name == "DeviceGray":
        value = max(0.0, min(1.0, components[0] if components else 0.0))
        byte = internal_component_byte(value)
        return bytes([byte, byte, byte])
    if alt_name == "DeviceRGB":
        red = max(0.0, min(1.0, components[0] if len(components) >= 1 else 0.0))
        green = max(0.0, min(1.0, components[1] if len(components) >= 2 else red))
        blue = max(0.0, min(1.0, components[2] if len(components) >= 3 else red))
        return bytes(
            [
                internal_component_byte(red),
                internal_component_byte(green),
                internal_component_byte(blue),
            ]
        )
    if alt_name == "DeviceCMYK":
        cyan = components[0] if len(components) >= 1 else 0.0
        magenta = components[1] if len(components) >= 2 else 0.0
        yellow = components[2] if len(components) >= 3 else 0.0
        black = components[3] if len(components) >= 4 else 0.0
        return bytes(cmyk_floats_to_srgb(cyan, magenta, yellow, black))
    return None
