# SPDX-License-Identifier: AGPL-3.0-only
"""Native PDF color conversion."""

from __future__ import annotations

import typing
from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy

from core_pdf.impl.runtime.array_views import ByteBuffer, uint8_view
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import parse_float
from core_pdf.impl.spec.s_08_graphics.color_kernels import (
    apply_decode_array as apply_decode_array_kernel,
)
from core_pdf.impl.spec.s_08_graphics.color_kernels import (
    image_component_count,
    image_dimension,
    unpack_subbyte_image_samples,
)
from core_pdf.impl.spec.s_08_graphics.color_math import (
    d50_xyz_to_srgb,
    lab_to_xyz,
)
from core_pdf.impl.spec.s_08_graphics.color_spec import (
    ImageColorSpec,
    cs_param,
    cs_param_floats,
    normalize_image_color_spec,
)
from core_pdf.impl.spec.s_08_graphics.device_profiles import (
    cmyk_bytes_to_srgb,
    cmyk_floats_to_srgb,
    internal_component_byte,
)
from core_pdf.impl.spec.s_08_graphics.icc_profiles import IccProfileError, IccSampleError
from core_pdf.impl.spec.s_08_graphics.pdf_function import internal_compile_pdf_function

ImageDict: TypeAlias = dict[str, object]
ColorComponents: TypeAlias = list[float]
ImageBuffer: TypeAlias = ByteBuffer


def color_operands_to_srgb(
    spec: ImageColorSpec, components: Sequence[float]
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
    if kind == "Indexed":
        return internal_indexed_operand_to_srgb(spec, components)
    if kind in {"Separation", "DeviceN"}:
        return internal_tint_operands_to_srgb(spec, components)
    return None


def internal_indexed_operand_to_srgb(
    spec: ImageColorSpec, components: Sequence[float]
) -> tuple[float, float, float] | None:
    lookup = spec.lookup
    base = spec.base
    if lookup is None or base is None or not components:
        return None
    try:
        width = internal_alternate_color_component_count(base)
    except ValueError:
        return None
    # 8.6.6.3: "If the value is a real number, it shall be rounded to the
    # nearest integer; if it is outside the range 0 to hival, it shall be
    # adjusted to the nearest value within that range."
    index = max(0, min(spec.hival, int(round(float(components[0])))))
    start = index * width
    entry = lookup[start : start + width]
    if len(entry) < width:
        return None
    return internal_srgb_bytes_to_floats(
        internal_apply_alt_color([byte / 255.0 for byte in entry], base)
    )


def internal_tint_operands_to_srgb(
    spec: ImageColorSpec, components: Sequence[float]
) -> tuple[float, float, float] | None:
    if not components:
        return None
    alt = spec.alt
    tints = [max(0.0, min(1.0, float(value))) for value in components]
    if alt is not None and spec.tint_fn is not None:
        try:
            expected = internal_alternate_color_component_count(alt)
            evaluated = internal_evaluate_tint(spec.tint_fn, tints)
            if len(evaluated) == expected:
                converted = internal_srgb_bytes_to_floats(
                    internal_apply_alt_color(list(evaluated), alt)
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


def internal_alternate_color_component_count(alt_name: str) -> int:
    if alt_name == "DeviceGray":
        return 1
    if alt_name == "DeviceRGB":
        return 3
    if alt_name == "DeviceCMYK":
        return 4
    raise ValueError("invalid Separation color space")


def internal_separation_rgb_lut(
    tint_fn: object,
    alt_name: str,
) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
    """Compile a one-input Separation function to an 8-bit RGB lookup table."""
    expected = internal_alternate_color_component_count(alt_name)
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


def internal_convert_image_data(raw: ImageBuffer, image_dict: ImageDict) -> ImageBuffer | None:
    """Convert encoded PDF image samples to grayscale or sRGB bytes."""
    spec = normalize_image_color_spec(image_dict)
    if spec.bits_per_component not in {1, 2, 4, 8} or spec.kind is None:
        return None

    fast = internal_simple_device_color_fast_path(raw, spec, image_dict)
    if fast is not None:
        return fast
    samples = internal_normalize_image_samples(raw, spec, image_dict)
    if samples is None:
        return None
    return internal_convert_color_samples(samples, spec)


def internal_convert_color_samples(
    samples: ImageBuffer,
    spec: ImageColorSpec,
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
        return internal_convert_lab(samples, spec.params)
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

    transform = spec.icc_transform
    if transform is not None and transform.input_channels == spec.channels:
        values = uint8_view(samples)
        if len(values) % transform.input_channels == 0:
            try:
                return transform.apply_uint8(values.reshape(-1, transform.input_channels)).reshape(
                    -1
                )
            except (IccProfileError, IccSampleError):
                pass
    fallback = spec.alt or {1: "DeviceGray", 3: "DeviceRGB", 4: "DeviceCMYK"}.get(spec.channels)
    if fallback is None:
        raise ValueError("invalid ICCBased color space")
    return internal_convert_color_samples(
        samples,
        ImageColorSpec(kind=fallback, params={}),
        depth + 1,
    )


def internal_normalize_image_samples(
    raw: ImageBuffer,
    spec: ImageColorSpec,
    image_dict: ImageDict,
) -> ImageBuffer | None:
    bits_per_component = spec.bits_per_component
    if bits_per_component == 8:
        return internal_apply_decode_array(raw, spec, image_dict)
    width = image_dimension(image_dict, "Width")
    height = image_dimension(image_dict, "Height")
    if width <= 0 or height <= 0:
        return None
    components = image_component_count(spec)
    unpacked = unpack_subbyte_image_samples(
        raw,
        bits_per_component,
        width,
        height,
        components,
    )
    if spec.kind == "Indexed":
        return unpacked
    return internal_apply_decode_array(unpacked, spec, image_dict)


def internal_simple_device_color_fast_path(
    raw: ImageBuffer,
    spec: ImageColorSpec,
    image_dict: ImageDict,
) -> ImageBuffer | None:
    if spec.bits_per_component != 8:
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
    spec: ImageColorSpec,
    image_dict: ImageDict,
) -> ImageBuffer:
    if spec.kind == "Indexed":
        return samples
    components = image_component_count(spec)
    if components <= 0:
        return samples
    max_sample = (1 << spec.bits_per_component) - 1
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
    if spec.bits_per_component == 8 and all(pair == (0.0, 1.0) for pair in pairs):
        return samples
    return apply_decode_array_kernel(samples, tuple(pairs), max_sample)


def internal_convert_separation(
    raw: ImageBuffer,
    color_space: ImageColorSpec,
) -> ImageBuffer | None:
    alt_name = color_space.alt or "DeviceGray"
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

    return internal_separation_rgb_lut(tint_fn, alt_name)[uint8_view(raw)].reshape(-1)


def internal_convert_devicen(
    raw: ImageBuffer,
    color_space: ImageColorSpec,
) -> ImageBuffer | None:
    alt_name = color_space.alt or ""
    channels = color_space.channels
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
    expected = internal_alternate_color_component_count(alt_name)
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


def internal_convert_indexed(raw: ImageBuffer, spec: ImageColorSpec) -> ImageBuffer | None:
    lookup = spec.lookup
    if lookup is None or spec.hival < 0:
        return None
    hival = spec.hival
    samples = uint8_view(raw)
    if numpy.any(samples > hival):
        samples = numpy.minimum(samples, hival)
    if spec.base == "DeviceRGB":
        if len(lookup) < (hival + 1) * 3:
            raise ValueError("invalid Indexed color lookup")
        table = uint8_view(lookup).reshape(-1, 3)
        return table[samples].reshape(-1)
    if spec.base == "DeviceGray":
        if len(lookup) < hival + 1:
            raise ValueError("invalid Indexed color lookup")
        values = uint8_view(lookup)[samples]
        return numpy.repeat(values[:, None], 3, axis=1).reshape(-1)
    if spec.base == "DeviceCMYK":
        if len(lookup) < (hival + 1) * 4:
            raise ValueError("invalid Indexed color lookup")
        table = uint8_view(lookup)[: (hival + 1) * 4].reshape(-1, 4)
        return internal_convert_cmyk(table[samples].reshape(-1))
    raise ValueError("invalid Indexed color space")


def internal_convert_calgray(raw: ImageBuffer, params: object) -> ImageBuffer:
    # Gamma is parsed only to reject malformed parameters; the conversion
    # itself intentionally treats CalGray as DeviceGray.
    if parse_float(cs_param(params, "Gamma", 1.0), None) is None:
        raise ValueError("invalid color space parameters")
    return internal_convert_gray(raw)


def internal_convert_calrgb(raw: ImageBuffer, params: object) -> ImageBuffer:
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


def internal_convert_lab(raw: ImageBuffer, params: object) -> ImageBuffer:
    white_point = cs_param_floats(params, "WhitePoint", 3, [0.9505, 1.0, 1.089])
    range_a = cs_param_floats(params, "Range", 2, [-100.0, 100.0])
    samples = uint8_view(raw).reshape(-1, 3)
    a_span = range_a[1] - range_a[0]
    lab = samples.astype(numpy.float32)
    lab[:, 0] /= 255.0
    lab[:, 1] = (lab[:, 1] / 255.0 * a_span + range_a[0] + 128.0) / 255.0
    lab[:, 2] = (lab[:, 2] / 255.0 * a_span + range_a[0] + 128.0) / 255.0
    xyz = lab_to_xyz(lab, (white_point[0], white_point[1], white_point[2]))
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
