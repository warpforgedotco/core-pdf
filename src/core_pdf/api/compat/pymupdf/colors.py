"""Color conversion selected by the PyMuPDF compatibility facade."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache, lru_cache
from importlib import resources
from typing import Any

import imagecodecs
import numpy

from core_pdf.impl._impl.graphics.icc_profiles import ByteSamples
from core_pdf.impl._impl.graphics.image_kernels import image_component_count
from core_pdf.impl._impl.model.glyphs import PaintColorSource
from core_pdf.impl.spec.s_08_graphics.color import indexed_color_components
from core_pdf.impl.spec.s_08_graphics.color_spec import ImageColorSpec

from .functions import compile_color_function

# MuPDF uses relative colorimetric conversion, black point compensation, and
# LCMS's low-resolution precalculation for its default device profiles.
internal_TRANSFORM_FLAGS = 0x2000 | 0x0800


@cache
def internal_profiles() -> tuple[bytes, bytes]:
    directory = resources.files("core_pdf._vendor.icc")
    return (
        directory.joinpath("Artifex-CMYK-SWOP.icc").read_bytes(),
        directory.joinpath("Artifex-sRGB.icc").read_bytes(),
    )


@cache
def internal_gray_profile() -> bytes:
    return resources.files("core_pdf._vendor.icc").joinpath("Artifex-Gray.icc").read_bytes()


def internal_profile_words_to_rgb(
    words: numpy.ndarray[Any, Any], source: bytes, color_space: str
) -> numpy.ndarray[Any, Any]:
    """Convert 16-bit scalar color samples without prematurely packing bytes."""
    converted = imagecodecs.cms_transform(
        numpy.ascontiguousarray(words).reshape(-1, 1, words.shape[-1]),
        source,
        internal_profiles()[1],
        colorspace=color_space,
        outcolorspace="rgb",
        outdtype=numpy.uint16,
        intent=1,
        flags=internal_TRANSFORM_FLAGS,
    )
    return numpy.asarray(converted, dtype=numpy.float32).reshape(-1, 3) / numpy.float32(65535)


def scalar_colors_to_rgb(
    components: numpy.ndarray[Any, Any], color_space: str
) -> numpy.ndarray[Any, Any]:
    """Preserve scalar ICC precision for a table of gray or CMYK colors."""
    values = numpy.clip(numpy.asarray(components, dtype=numpy.float32), 0, 1)
    words = (values * numpy.float32(65535)).astype(numpy.uint16)
    source = internal_gray_profile() if color_space == "gray" else internal_profiles()[0]
    return internal_profile_words_to_rgb(words, source, color_space)


def cmyk_bytes_to_rgb(samples: ByteSamples) -> ByteSamples:
    """Convert an array of four-channel byte pixels with MuPDF's device profiles."""
    if samples.dtype != numpy.dtype(numpy.uint8) or samples.shape[-1:] != (4,):
        raise ValueError("CMYK samples must be uint8 pixels with four channels")
    if not samples.size:
        return numpy.empty((*samples.shape[:-1], 3), dtype=numpy.uint8)
    source, destination = internal_profiles()
    converted = imagecodecs.cms_transform(
        numpy.ascontiguousarray(samples).reshape(-1, 1, 4),
        source,
        destination,
        colorspace="cmyk",
        outcolorspace="rgb",
        outdtype=numpy.uint8,
        intent=1,
        flags=internal_TRANSFORM_FLAGS,
    )
    return numpy.asarray(converted, dtype=numpy.uint8).reshape(*samples.shape[:-1], 3)


@lru_cache(maxsize=8192)
def internal_cmyk_words_to_rgb(components: tuple[int, ...]) -> tuple[int, int, int]:
    rgb = internal_profile_words_to_rgb(
        numpy.asarray(components, dtype=numpy.uint16).reshape(1, 4), internal_profiles()[0], "cmyk"
    )
    packed = numpy.floor(rgb.reshape(3) * numpy.float32(255) + numpy.float32(0.5))
    return int(packed[0]), int(packed[1]), int(packed[2])


def cmyk_floats_to_rgb(components: Sequence[float]) -> tuple[int, int, int]:
    """Preserve the 16-bit scalar conversion used for text and vector colors."""
    values = numpy.clip(numpy.asarray(components, dtype=numpy.float32), 0, 1)
    words = (values * numpy.float32(65535)).astype(numpy.uint16)
    return internal_cmyk_words_to_rgb(tuple(int(value) for value in words))


@lru_cache(maxsize=8192)
def internal_gray_to_rgb(component: float) -> tuple[int, int, int]:
    rgb = scalar_colors_to_rgb(numpy.asarray([[component]]), "gray").reshape(3)
    packed = numpy.floor(rgb * numpy.float32(255) + numpy.float32(0.5))
    return int(packed[0]), int(packed[1]), int(packed[2])


@lru_cache(maxsize=8192)
def internal_icc_to_rgb(
    components: tuple[float, ...], profile: bytes, color_space: str
) -> tuple[int, int, int]:
    values = numpy.clip(numpy.asarray(components, dtype=numpy.float32), 0, 1)
    words = (values * numpy.float32(65535)).astype(numpy.uint16).reshape(1, -1)
    rgb = internal_profile_words_to_rgb(words, profile, color_space).reshape(3)
    packed = numpy.floor(rgb * numpy.float32(255) + numpy.float32(0.5))
    return int(packed[0]), int(packed[1]), int(packed[2])


def color_int(components: Sequence[float], source: PaintColorSource | None = None) -> int:
    """Pack a captured paint color, retaining CMYK alternate-space evidence."""
    if source is not None and isinstance(source.specification, ImageColorSpec):
        spec = source.specification
        try:
            if (
                spec.kind == "ICCBased"
                and spec.icc_profile
                and spec.channels in {1, 3, 4}
                and not (spec.channels == 3 and spec.icc_profile == internal_profiles()[1])
            ):
                converted_rgb = internal_icc_to_rgb(
                    source.components,
                    spec.icc_profile,
                    {1: "gray", 3: "rgb", 4: "cmyk"}[spec.channels],
                )
                return (converted_rgb[0] << 16) | (converted_rgb[1] << 8) | converted_rgb[2]
            if spec.kind == "Indexed" and source.components:
                base = spec.base_spec or ImageColorSpec(kind=spec.base, params={})
                if base.kind in {"DeviceGray", "DeviceRGB", "DeviceCMYK", "Separation", "DeviceN"}:
                    operands = indexed_color_components(
                        spec, source.components[0], image_component_count(base)
                    )
                    return color_int(operands, PaintColorSource(base, operands))
            elif spec.kind in {"Separation", "DeviceN"} and spec.alt == "DeviceCMYK":
                try:
                    evaluate = compile_color_function(spec.tint_fn)
                except ValueError:
                    # A rejected tint-function load leaves a one-component
                    # color in the reader's fallback gray space.
                    if len(source.components) == 1:
                        components = source.components
                else:
                    components = evaluate(*source.components)
        except (ValueError, TypeError, ZeroDivisionError, imagecodecs.CmsError):
            # Keep the already recovered native color for malformed operands.
            pass
    rgb: tuple[int, ...]
    if len(components) == 4:
        rgb = cmyk_floats_to_rgb(components)
    elif len(components) == 1:
        rgb = internal_gray_to_rgb(components[0])
    else:
        rgb = tuple(
            int(numpy.clip(numpy.float32(value) * numpy.float32(255) + numpy.float32(0.5), 0, 255))
            for value in components[:3]
        )
    color = 0
    for value in rgb:
        color = (color << 8) | value
    return color


def indexed_bytes_to_rgb(samples: ByteSamples, spec: ImageColorSpec) -> ByteSamples | None:
    """Convert an indexed palette with the same scalar device colors as MuPDF."""
    base = spec.base_spec or ImageColorSpec(kind=spec.base, params={})
    if base.kind not in {"DeviceGray", "DeviceRGB", "DeviceCMYK", "Separation", "DeviceN"}:
        return None
    if base.kind in {"Separation", "DeviceN"} and base.alt != "DeviceCMYK":
        return None
    width = image_component_count(base)
    if base.kind in {"Separation", "DeviceN"}:
        evaluate = compile_color_function(base.tint_fn)
        tints = numpy.asarray(
            [
                evaluate(*indexed_color_components(spec, index, width))
                for index in range(spec.hival + 1)
            ],
            dtype=numpy.float32,
        )
        # DeviceN image palettes first expand into byte samples of their
        # alternate space. MuPDF truncates this intermediate ink quantization.
        inks = (numpy.clip(tints, 0, 1) * numpy.float32(255)).astype(numpy.uint8)
        palette = cmyk_bytes_to_rgb(inks)
        return palette[numpy.minimum(samples, spec.hival)]
    palette = numpy.empty((spec.hival + 1, 3), dtype=numpy.uint8)
    for index in range(spec.hival + 1):
        operands = indexed_color_components(spec, index, width)
        color = color_int(operands, PaintColorSource(base, operands))
        palette[index] = (color >> 16, (color >> 8) & 255, color & 255)
    return palette[numpy.minimum(samples, spec.hival)]
