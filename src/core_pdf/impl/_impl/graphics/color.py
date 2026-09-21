# SPDX-License-Identifier: AGPL-3.0-only
"""Vector and packed-image adapters for the shared component converter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy

from core_pdf.impl._impl.graphics.color_spec import (
    ColorSpace,
    parse_color_space,
    recover_image_bits_per_component,
)
from core_pdf.impl._impl.graphics.image_samples import (
    convert_integer_image,
    internal_convert_components,
)
from core_pdf.impl._impl.runtime.array_views import ByteBuffer, uint8_view
from core_pdf.impl._impl.runtime.scalars import parse_int
from core_pdf_spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering

ImageDict: TypeAlias = dict[str, object]


def color_operands_to_srgb(
    spec: ColorSpace,
    components: Sequence[float],
    *,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> tuple[float, float, float] | None:
    """Convert natural PDF operands; device operands remain handled downstream."""
    if spec.kind in {"DeviceGray", "DeviceRGB", "Pattern"} or (
        spec.kind == "DeviceCMYK" and rendering == DEFAULT_COLOR_RENDERING
    ):
        return None
    try:
        converted = internal_convert_components(
            numpy.asarray([components], dtype=numpy.float64), spec, rendering=rendering
        )[0]
        if len(converted) == 1:
            converted = numpy.repeat(converted, 3)
        return float(converted[0]) / 255, float(converted[1]) / 255, float(converted[2]) / 255
    except TypeError, ValueError:
        return None


def internal_convert_image_data(
    raw: ByteBuffer,
    image_dict: ImageDict,
    *,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> ByteBuffer | None:
    """Unpack and decode once, keeping original integers for colour-key masks."""
    spec = parse_color_space(image_dict.get("ColorSpace"))
    bits = recover_image_bits_per_component(image_dict)
    if bits not in {1, 2, 4, 8, 16} or not spec.component_ranges:
        return None
    fast = internal_simple_device_color_fast_path(raw, spec, image_dict, bits)
    if fast is not None and image_dict.get("Mask") is None:
        return fast
    dictionary = dict(image_dict)
    # Preserve the low-depth reader's tolerance for malformed Decode arrays.
    # Valid arrays always decode into the colour space's natural ranges.
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
        memoryview(raw).cast("B"), dictionary, bits_per_component=bits, rendering=rendering
    ).reshape(-1)


def internal_simple_device_color_fast_path(
    raw: ByteBuffer,
    spec: ColorSpace,
    image_dict: ImageDict,
    bits_per_component: int,
) -> ByteBuffer | None:
    if bits_per_component != 8:
        return None
    if spec.kind not in {"DeviceRGB", "DeviceGray"}:
        return None
    if image_dict.get("Decode") is not None:
        return None
    width = internal_image_dimension(image_dict, "Width")
    height = internal_image_dimension(image_dict, "Height")
    if width <= 0 or height <= 0:
        return None
    expected = width * height * (3 if spec.kind == "DeviceRGB" else 1)
    if len(raw) != expected:
        return None
    # Native grayscale samples stay one-channel for extraction consumers;
    # renderers expand them at the final compositing boundary.
    return raw


def internal_convert_cmyk(
    raw: ByteBuffer,
) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
    if len(raw) % 4:
        raise ValueError("invalid color sample data")
    return internal_convert_components(
        uint8_view(raw).reshape(-1, 4).astype(numpy.float64) / 255,
        parse_color_space("DeviceCMYK"),
    ).reshape(-1)


def internal_image_dimension(image_dict: ImageDict, key: str) -> int:
    value = image_dict.get(key)
    if type(value) is bool:
        return 0
    return parse_int(value, 0) or 0
