# SPDX-License-Identifier: AGPL-3.0-only
"""Native image color conversion manager."""

from __future__ import annotations

import typing
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, TypeAlias, cast

import numpy

from core_pdf.impl.runtime.array_views import ByteBuffer, readonly, uint8_view
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import (
    parse_float,
)
from core_pdf.impl.spec.s_08_graphics.color_kernels import (
    apply_decode_array as internal_native_apply_decode_array,
)
from core_pdf.impl.spec.s_08_graphics.color_kernels import (
    evaluate_sampled_tint_function as internal_native_evaluate_sampled_tint_function,
)
from core_pdf.impl.spec.s_08_graphics.color_kernels import (
    image_component_count as internal_native_image_component_count,
)
from core_pdf.impl.spec.s_08_graphics.color_kernels import (
    image_dimension as internal_native_image_dimension,
)
from core_pdf.impl.spec.s_08_graphics.color_kernels import (
    unpack_subbyte_image_samples as internal_native_unpack_subbyte_image_samples,
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
        ImageColorManager.apply_alt_color([byte / 255.0 for byte in entry], base)
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
            if evaluated is not None and len(evaluated) == expected:
                converted = internal_srgb_bytes_to_floats(
                    ImageColorManager.apply_alt_color(list(evaluated), alt)
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


def internal_evaluate_tint(tint_fn: object, tints: list[float]) -> tuple[float, ...] | None:
    from core_pdf.impl.spec.s_08_graphics.shading import internal_evaluate_pdf_function

    if len(tints) > 1:
        if not isinstance(tint_fn, PdfStream):
            return None
        return tuple(internal_native_evaluate_sampled_tint_function(tint_fn, *tints))
    return internal_evaluate_pdf_function(tint_fn, tints[0])


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


def internal_sampled_separation_lut_key(
    tint_fn: PdfStream,
    alt_name: str,
) -> tuple[object, ...] | None:
    """A value key for the LUT, or ``None`` when the function is not cacheable.

    Keying the cache on the ``PdfStream`` itself pinned the stream -- and with it
    the ``memoryview`` into the document's source buffer -- for the lifetime of
    the process, so the buffer outlived the document that owned it. These are the
    only entries ``evaluate_sampled_tint_function`` reads, so together with the
    decoded samples they determine the LUT exactly.
    """
    dictionary = tint_fn.dictionary
    key: list[object] = [alt_name]
    for name in ("FunctionType", "BitsPerSample", "Size", "Domain", "Range", "Encode"):
        value = dictionary.get(name)
        if isinstance(value, (list, tuple)):
            if not all(isinstance(item, (int, float)) for item in value):
                return None
            key.append(tuple(value))
        elif isinstance(value, (int, float)) or value is None:
            key.append(value)
        else:
            return None
    key.append(bytes(tint_fn.data))
    return tuple(key)


def internal_sampled_separation_rgb_lut(
    tint_fn: PdfStream,
    alt_name: str,
) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
    """Compile an 8-bit sampled Separation function to a read-only RGB LUT."""
    cache_key = internal_sampled_separation_lut_key(tint_fn, alt_name)
    if cache_key is not None:
        cached = internal_separation_lut_cache.get(cache_key)
        if cached is not None:
            return cached
        lut = internal_build_sampled_separation_rgb_lut(tint_fn, alt_name)
        if len(internal_separation_lut_cache) < 256:
            internal_separation_lut_cache[cache_key] = lut
        return lut
    return internal_build_sampled_separation_rgb_lut(tint_fn, alt_name)


internal_separation_lut_cache: dict[
    tuple[object, ...], numpy.ndarray[Any, numpy.dtype[numpy.uint8]]
] = {}


def internal_build_sampled_separation_rgb_lut(
    tint_fn: PdfStream,
    alt_name: str,
) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
    """Compile an 8-bit sampled Separation function to a read-only RGB LUT."""
    expected = internal_alternate_color_component_count(alt_name)
    table = numpy.empty((256, 3), dtype=numpy.uint8)
    for value in range(256):
        components = internal_native_evaluate_sampled_tint_function(tint_fn, value / 255.0)
        if len(components) != expected:
            raise ValueError("invalid separation tint function")
        rgb = ImageColorManager.apply_alt_color(components, alt_name)
        if rgb is None:
            raise ValueError("invalid Separation color space")
        table[value] = tuple(rgb)
    return readonly(table)


@lru_cache(maxsize=256)
def internal_calrgb_parameter_arrays(
    matrix: tuple[float, ...],
    black_point: tuple[float, ...],
) -> tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any]]:
    matrix_array = readonly(numpy.asarray(matrix, dtype=numpy.float64).reshape(3, 3))
    black_point_array = readonly(numpy.asarray(black_point, dtype=numpy.float64))
    return matrix_array, black_point_array


class ImageColorManager:
    @staticmethod
    def convert_image_data(
        raw: ImageBuffer,
        image_dict: ImageDict | ImageColorSpec,
    ) -> ImageBuffer | None:
        current: ImageDict | ImageColorSpec = image_dict
        depth = 0

        while depth <= 3:
            spec = (
                current
                if isinstance(current, ImageColorSpec)
                else normalize_image_color_spec(current)
            )
            if spec.bits_per_component not in {1, 2, 4, 8}:
                return None
            cs_kind = spec.kind
            if cs_kind is None:
                return None
            if isinstance(current, dict):
                fast = ImageColorManager.simple_device_color_fast_path(
                    raw,
                    spec,
                    current,
                )
                if fast is not None:
                    return fast
            samples = ImageColorManager.normalize_image_samples(raw, spec, current)
            if samples is None:
                return None
            if cs_kind == "DeviceRGB":
                return samples
            if cs_kind == "DeviceGray":
                return ImageColorManager.convert_gray(samples)
            if cs_kind == "DeviceCMYK":
                return ImageColorManager.convert_cmyk(samples)
            if cs_kind == "Lab":
                return ImageColorManager.convert_lab_raw(samples, spec.params)
            if cs_kind == "CalGray":
                return ImageColorManager.convert_calgray(samples, spec.params)
            if cs_kind == "CalRGB":
                return ImageColorManager.convert_calrgb(samples, spec.params)
            if cs_kind == "Indexed":
                return ImageColorManager.convert_indexed(samples, spec)
            if cs_kind == "ICCBased":
                if spec.alt is not None:
                    current = ImageColorSpec(kind=spec.alt, params={})
                    raw = samples
                    depth += 1
                    continue
                if spec.channels == 1:
                    return ImageColorManager.convert_gray(samples)
                if spec.channels == 3:
                    current = ImageColorSpec(kind="DeviceRGB", params={})
                    raw = samples
                    depth += 1
                    continue
                if spec.channels == 4:
                    current = ImageColorSpec(kind="DeviceCMYK", params={})
                    raw = samples
                    depth += 1
                    continue
                raise ValueError("invalid ICCBased color space")
            if cs_kind == "Separation":
                return ImageColorManager.convert_separation(samples, spec)
            if cs_kind == "DeviceN":
                return ImageColorManager.convert_devicen(samples, spec)
            return None
        return None

    @staticmethod
    def normalize_image_samples(
        raw: ImageBuffer,
        spec: ImageColorSpec,
        image_dict: ImageDict | ImageColorSpec,
    ) -> ImageBuffer | None:
        bpc = spec.bits_per_component
        if bpc == 8:
            return ImageColorManager.apply_decode_array(raw, spec, image_dict)
        width = internal_native_image_dimension(image_dict, "Width")
        height = internal_native_image_dimension(image_dict, "Height")
        if width <= 0 or height <= 0:
            return None
        components = internal_native_image_component_count(spec)
        unpacked = internal_native_unpack_subbyte_image_samples(raw, bpc, width, height, components)
        if spec.kind == "Indexed":
            return unpacked
        return ImageColorManager.apply_decode_array(unpacked, spec, image_dict)

    @staticmethod
    def simple_device_color_fast_path(
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
        width = internal_native_image_dimension(image_dict, "Width")
        height = internal_native_image_dimension(image_dict, "Height")
        if width <= 0 or height <= 0:
            return None
        expected = width * height * (3 if spec.kind == "DeviceRGB" else 1)
        if len(raw) != expected:
            return None
        # Both kinds pass through untouched. Native grayscale samples are
        # preserved: consumers that need RGB expand them at the final
        # compositing boundary, and OCR consumes the one-channel buffer.
        return raw

    @staticmethod
    def apply_decode_array(
        samples: ImageBuffer,
        spec: ImageColorSpec,
        image_dict: ImageDict | ImageColorSpec,
    ) -> ImageBuffer:
        if spec.kind == "Indexed":
            return samples
        components = internal_native_image_component_count(spec)
        if components <= 0:
            return samples
        bpc = spec.bits_per_component
        max_sample = (1 << bpc) - 1
        if max_sample <= 0:
            return samples
        decode = image_dict.get("Decode") if isinstance(image_dict, dict) else None
        pairs: list[tuple[float, float]] = []
        if isinstance(decode, (list, tuple)) and len(decode) >= components * 2:
            for i in range(components):
                try:
                    dmin = float(typing.cast(typing.Any, decode[i * 2]))
                    dmax = float(typing.cast(typing.Any, decode[i * 2 + 1]))
                except (TypeError, ValueError):
                    pairs = []
                    break
                pairs.append((dmin, dmax))
        if not pairs:
            pairs = [(0.0, 1.0)] * components
        if bpc == 8 and all(pair == (0.0, 1.0) for pair in pairs):
            return samples
        return internal_native_apply_decode_array(samples, tuple(pairs), max_sample)

    @staticmethod
    def convert_separation(raw: ImageBuffer, color_space: ImageColorSpec) -> ImageBuffer | None:
        alt_name = color_space.alt or "DeviceGray"
        tint_fn = color_space.tint_fn

        expected = internal_alternate_color_component_count(alt_name)
        if isinstance(tint_fn, PdfStream):
            samples = uint8_view(raw)
            return internal_sampled_separation_rgb_lut(tint_fn, alt_name)[samples].reshape(-1)

        if tint_fn is None and alt_name in {"DeviceGray", "DeviceRGB", "DeviceCMYK"}:
            samples = uint8_view(raw)
            result = numpy.empty((len(samples), 3), dtype=numpy.uint8)
            if alt_name in {"DeviceGray", "DeviceRGB"}:
                result[:] = samples[:, None]
                return result.reshape(-1)
            inks = numpy.zeros((len(samples), 4), dtype=numpy.uint8)
            inks[:, 0] = samples
            return cmyk_bytes_to_srgb(inks).reshape(-1)

        fallback_result = bytearray()
        for byte in raw:
            v = byte / 255.0
            if tint_fn is None:
                components: ColorComponents = [v]
            elif isinstance(tint_fn, (list, tuple)):
                if len(tint_fn) >= 1 and callable(tint_fn[0]):
                    tint_callable = typing.cast(typing.Callable[..., object], tint_fn[0])
                    try:
                        components = cast(ColorComponents, tint_callable(v))
                    except Exception as exc:
                        raise ValueError("invalid separation tint function") from exc
                else:
                    components = cast(ColorComponents, list(tint_fn))
            else:
                components = [v]
            if len(components) != expected:
                raise ValueError("invalid separation tint function")
            rgb = ImageColorManager.apply_alt_color(components, alt_name)
            if rgb is not None:
                fallback_result.extend(rgb)
        return numpy.frombuffer(fallback_result, dtype=numpy.uint8)

    @staticmethod
    def convert_devicen(raw: ImageBuffer, color_space: ImageColorSpec) -> ImageBuffer | None:
        alt_name = color_space.alt or ""
        n = color_space.channels
        tint_fn = color_space.tint_fn
        if n <= 0:
            raise ValueError("invalid DeviceN color space")
        if len(raw) % n != 0:
            raise ValueError("invalid DeviceN color sample data")

        if tint_fn is None and alt_name in {"DeviceGray", "DeviceRGB", "DeviceCMYK"}:
            samples = uint8_view(raw).reshape(-1, n)
            if alt_name == "DeviceGray":
                return numpy.repeat(samples[:, :1], 3, axis=1).reshape(-1)
            if alt_name == "DeviceRGB":
                result = numpy.empty((len(samples), 3), dtype=numpy.uint8)
                result[:, 0] = samples[:, 0]
                result[:, 1] = samples[:, 1] if n >= 2 else samples[:, 0]
                result[:, 2] = samples[:, 2] if n >= 3 else samples[:, 0]
                return result.reshape(-1)
            # A DeviceN with no tint function is malformed; the salvage is to
            # read the first four channels as process inks and leave any the
            # colour space is short of at zero.
            carried = min(n, 4)
            inks = numpy.zeros((len(samples), 4), dtype=numpy.uint8)
            inks[:, :carried] = samples[:, :carried]
            return cmyk_bytes_to_srgb(inks).reshape(-1)

        if alt_name not in {"DeviceGray", "DeviceRGB", "DeviceCMYK"}:
            raise ValueError("invalid DeviceN color space")
        expected = internal_alternate_color_component_count(alt_name)

        # The tint transform is a pure function of one ink tuple, and an image
        # holds far fewer distinct tuples than pixels. Evaluating per distinct
        # tuple and gathering turns a per-pixel Python call -- which also
        # thrashed the 4096-entry CMYK cache behind apply_alt_color, since a
        # 2-channel DeviceN has up to 65536 inputs -- into one call per colour.
        samples = uint8_view(raw).reshape(-1, n)
        distinct, inverse = numpy.unique(samples, axis=0, return_inverse=True)
        tinted = numpy.empty((len(distinct), expected), dtype=numpy.float64)
        for index, row in enumerate(distinct.tolist()):
            components: ColorComponents = [value / 255.0 for value in row]
            if isinstance(tint_fn, (list, tuple)) and len(tint_fn) >= 1 and callable(tint_fn[0]):
                tint_callable = typing.cast(typing.Callable[..., object], tint_fn[0])
                try:
                    components = cast(ColorComponents, tint_callable(*components))
                except Exception as exc:
                    raise ValueError("invalid DeviceN tint function") from exc
            elif isinstance(tint_fn, PdfStream):
                components = internal_native_evaluate_sampled_tint_function(tint_fn, *components)
            if len(components) != expected:
                raise ValueError("invalid DeviceN tint function")
            tinted[index] = components

        # apply_alt_color one colour at a time would drop back into the
        # single-sample ICC path; the alternate spaces all convert a whole
        # block at once. round-half-to-even matches int(round(...)), and the
        # clamp order is the same, so this is byte-identical.
        scaled = numpy.clip(numpy.rint(numpy.clip(tinted, 0.0, 1.0) * 255.0), 0.0, 255.0)
        inks = scaled.astype(numpy.uint8)
        if alt_name == "DeviceCMYK":
            converted = cmyk_bytes_to_srgb(inks)
        elif alt_name == "DeviceRGB":
            converted = inks
        else:
            converted = numpy.repeat(inks, 3, axis=1)
        return converted[numpy.asarray(inverse).reshape(-1)].reshape(-1)

    @staticmethod
    def convert_gray(raw: ImageBuffer) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
        samples = uint8_view(raw)
        return numpy.repeat(samples, 3)

    @staticmethod
    def convert_cmyk(raw: ImageBuffer) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
        n = len(raw)
        if n % 4 != 0:
            raise ValueError("invalid color sample data")
        return cmyk_bytes_to_srgb(uint8_view(raw).reshape(-1, 4)).reshape(-1)

    @staticmethod
    def convert_indexed(raw: ImageBuffer, spec: ImageColorSpec) -> ImageBuffer | None:
        if spec.lookup is None:
            return None
        lookup = spec.lookup
        hival = spec.hival
        samples = uint8_view(raw)
        # ISO 32000-1 8.6.6.3: "if it is outside the range 0 to hival, it shall
        # be adjusted to the nearest value within that range." Raising sent the
        # caller down the recovery path, which reinterpreted the palette
        # *indexes* as DeviceGray samples -- one stray sample turned the whole
        # image into a near-black field.
        if hival < 0:
            return None
        if numpy.any(samples > hival):
            samples = numpy.minimum(samples, hival)
        if spec.base == "DeviceRGB":
            if len(lookup) < (hival + 1) * 3:
                raise ValueError("invalid Indexed color lookup")
            table = uint8_view(lookup).reshape(-1, 3)
            return table[samples].reshape(-1)
        elif spec.base == "DeviceGray":
            if len(lookup) < hival + 1:
                raise ValueError("invalid Indexed color lookup")
            values = uint8_view(lookup)[samples]
            return numpy.repeat(values[:, None], 3, axis=1).reshape(-1)
        elif spec.base == "DeviceCMYK":
            # A CMYK palette used to raise here, and the caller's recovery path
            # then handed the raw palette indexes on as DeviceGray -- index
            # numbers rendered as grey levels. Look the entry up, then convert
            # it the same way any other CMYK sample is converted.
            if len(lookup) < (hival + 1) * 4:
                raise ValueError("invalid Indexed color lookup")
            table = uint8_view(lookup)[: (hival + 1) * 4].reshape(-1, 4)
            return ImageColorManager.convert_cmyk(table[samples].reshape(-1))
        else:
            raise ValueError("invalid Indexed color space")

    @staticmethod
    def convert_calgray(raw: ImageBuffer, params: object) -> ImageBuffer:
        # Gamma is parsed only to reject malformed parameters; the conversion
        # itself intentionally treats CalGray as DeviceGray.
        gamma_raw = cs_param(params, "Gamma", 1.0)
        gamma = parse_float(gamma_raw, None)
        if gamma is None:
            raise ValueError("invalid color space parameters")

        return ImageColorManager.convert_gray(raw)

    @staticmethod
    def convert_calrgb(raw: ImageBuffer, params: object) -> ImageBuffer:
        bp = cs_param_floats(params, "BlackPoint", 3, [0.0, 0.0, 0.0])
        gamma = cs_param_floats(params, "Gamma", 3, [1.0, 1.0, 1.0])
        matrix = cs_param_floats(params, "Matrix", 9, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

        samples = uint8_view(raw).reshape(-1, 3)
        values = samples.astype(numpy.float64) / 255.0
        for index, exponent in enumerate(gamma):
            if exponent != 1.0:
                values[:, index] = numpy.power(values[:, index], exponent)
        matrix_array, black_point_array = internal_calrgb_parameter_arrays(tuple(matrix), tuple(bp))
        xyz = (values @ matrix_array.T + black_point_array).astype(numpy.float32)
        rgb = d50_xyz_to_srgb(xyz)
        return numpy.clip(rgb * 255.0, 0.0, 255.0).astype(numpy.uint8).reshape(-1)

    @staticmethod
    def convert_lab_raw(raw: ImageBuffer, params: object) -> ImageBuffer:
        wp = cs_param_floats(params, "WhitePoint", 3, [0.9505, 1.0, 1.089])
        range_a = cs_param_floats(params, "Range", 2, [-100.0, 100.0])
        samples = uint8_view(raw).reshape(-1, 3)
        a_span = range_a[1] - range_a[0]
        lab = samples.astype(numpy.float32)
        lab[:, 0] /= 255.0
        lab[:, 1] = (lab[:, 1] / 255.0 * a_span + range_a[0] + 128.0) / 255.0
        lab[:, 2] = (lab[:, 2] / 255.0 * a_span + range_a[0] + 128.0) / 255.0
        xyz = lab_to_xyz(lab, (wp[0], wp[1], wp[2]))
        rgb = d50_xyz_to_srgb(xyz)
        return numpy.clip(rgb * 255.0, 0.0, 255.0).astype(numpy.uint8).reshape(-1)

    @staticmethod
    def apply_alt_color(components: ColorComponents, alt_name: str) -> bytes | None:
        if alt_name == "DeviceGray":
            v = max(0.0, min(1.0, components[0] if components else 0.0))
            v_byte = internal_component_byte(v)
            return bytes([v_byte, v_byte, v_byte])
        if alt_name == "DeviceRGB":
            r = max(0.0, min(1.0, components[0] if len(components) >= 1 else 0.0))
            g = max(0.0, min(1.0, components[1] if len(components) >= 2 else r))
            b = max(0.0, min(1.0, components[2] if len(components) >= 3 else r))
            return bytes(
                [
                    internal_component_byte(r),
                    internal_component_byte(g),
                    internal_component_byte(b),
                ]
            )
        if alt_name == "DeviceCMYK":
            c = components[0] if len(components) >= 1 else 0.0
            m = components[1] if len(components) >= 2 else 0.0
            y = components[2] if len(components) >= 3 else 0.0
            k = components[3] if len(components) >= 4 else 0.0
            return bytes(cmyk_floats_to_srgb(c, m, y, k))
        return None
