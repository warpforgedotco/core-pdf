# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias

lazy import numpy

from core_pdf.impl.graphics.color_spec import GRAY_RGB_KINDS, ColorSpace, parse_color_space
from core_pdf.impl.spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering
lazy from core_pdf.impl.base.arrays import ByteBuffer, uint8_view
lazy from core_pdf.impl.graphics.image_samples import (
    ImageHeader,
    convert_components,
    convert_integer_image,
    tint_function,
)
lazy from core_pdf.impl.spec.s_08_graphics.color_kernels import decode_sample_values

ImageDict: TypeAlias = dict[str, object]


def color_operands_to_srgb(
    spec: ColorSpace,
    components: Sequence[float],
    *,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> tuple[float, float, float] | None:
    if (
        spec.kind in GRAY_RGB_KINDS
        or spec.kind == "Pattern"
        or (spec.kind == "DeviceCMYK" and rendering == DEFAULT_COLOR_RENDERING)
    ):
        return None
    try:
        converted = convert_components(
            numpy.asarray([components], dtype=numpy.float64), spec, rendering=rendering
        )[0]
        if len(converted) == 1:
            converted = numpy.repeat(converted, 3)
        return float(converted[0]) / 255, float(converted[1]) / 255, float(converted[2]) / 255
    except TypeError, ValueError:
        return None


def color_operands_to_cmyk(
    spec: ColorSpace, components: Sequence[float]
) -> tuple[float, float, float, float] | None:
    """A colour's CMYK, for a colour space that comes down to CMYK -- DeviceCMYK, an
    ICCBased CMYK space, and a Separation, DeviceN or Indexed space whose alternate or base
    does -- or None: the colour a CMYK blending space takes it in, without passing through
    sRGB."""
    values = [float(value) for value in components]
    for _ in range(8):
        kind = spec.kind
        if kind == "DeviceCMYK" or (kind == "ICCBased" and len(spec.component_ranges) == 4):
            if len(values) != 4:
                return None
            cyan, magenta, yellow, black = (min(1.0, max(0.0, value)) for value in values)
            return cyan, magenta, yellow, black
        if kind in {"Separation", "DeviceN"} and spec.alternate is not None:
            try:
                values = [float(value) for value in tint_function(spec.tint_fn)(*values)]
            except TypeError, ValueError, ArithmeticError:
                return None
            spec = spec.alternate
        elif kind == "Indexed" and spec.base is not None and spec.lookup is not None:
            count = len(spec.base.component_ranges)
            index = int(numpy.floor(values[0] + 0.5)) if values else -1
            if not 0 <= index <= spec.hival or len(spec.lookup) < (index + 1) * count:
                return None
            entry = numpy.frombuffer(spec.lookup, dtype=numpy.uint8)[
                index * count : (index + 1) * count
            ]
            values = decode_sample_values(entry.reshape(1, -1), spec.base.component_ranges, 255)[
                0
            ].tolist()
            spec = spec.base
        else:
            return None
    return None


def convert_image_data(
    raw: ByteBuffer,
    image_dict: ImageDict,
    *,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
    header: ImageHeader | None = None,
) -> ByteBuffer | None:
    if header is None:
        header = ImageHeader(image_dict)
    spec = header.space()
    bits = header.checked_bits()
    if bits not in {1, 2, 4, 8, 16} or not spec.component_ranges:
        return None
    fast = simple_device_color_fast_path(raw, spec, header, bits)
    if fast is not None and header.mask is None:
        return fast
    dictionary = dict(image_dict)
    decode = dictionary.get("Decode")
    if bits != 16 and decode is not None:
        count = len(spec.component_ranges) * 2
        try:
            if not isinstance(decode, (tuple, list)) or len(decode) < count:
                raise ValueError("incomplete Decode")
            numbers = numpy.asarray(decode[:count], dtype=numpy.float64)
            if not numpy.isfinite(numbers).all():
                raise ValueError("nonfinite Decode")
            dictionary["Decode"] = numbers.tolist()
        except TypeError, ValueError:
            dictionary.pop("Decode")
    return convert_integer_image(
        memoryview(raw).cast("B"),
        dictionary,
        bits_per_component=bits,
        rendering=rendering,
        space=spec,
    ).reshape(-1)


def simple_device_color_fast_path(
    raw: ByteBuffer, spec: ColorSpace, header: ImageHeader, bits_per_component: int
) -> ByteBuffer | None:
    if bits_per_component != 8:
        return None
    if spec.kind not in GRAY_RGB_KINDS:
        return None
    if header.decode is not None:
        return None
    width = header.width
    height = header.height
    if width <= 0 or height <= 0:
        return None
    expected = width * height * (3 if spec.kind == "DeviceRGB" else 1)
    if len(raw) != expected:
        return None
    return raw


def convert_cmyk(raw: ByteBuffer) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
    if len(raw) % 4:
        raise ValueError("invalid color sample data")
    return convert_components(
        uint8_view(raw).reshape(-1, 4).astype(numpy.float64) / 255, parse_color_space("DeviceCMYK")
    ).reshape(-1)
