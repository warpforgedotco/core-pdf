# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import typing
from functools import lru_cache
from typing import TypeAlias, cast

from core_pdf.impl.engine.spec.s_07_objects.coercion import (
    parse_float,
    parse_int,
)
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_08_graphics.color_math import (
    adapt_d50_to_d65,
    lab_to_xyz,
    xyz_to_srgb,
)
from core_pdf.impl.engine.spec.s_08_graphics.color_spec import (
    ImageColorSpec,
    cs_name,
    cs_param,
    cs_param_floats,
    normalize_image_color_spec,
)
from core_pdf.impl.engine.spec.s_08_graphics.icc_profiles import (
    convert_icc_profile_samples,
)
from core_pdf.impl.objects import PdfStream

if typing.TYPE_CHECKING:
    from collections.abc import Callable

ImageDict: TypeAlias = dict[str, object]
ColorSpaceSequence: TypeAlias = list[object] | tuple[object, ...]
ColorComponents: TypeAlias = list[float]


class ImageColorManager:
    @staticmethod
    def convert_image_data(
        raw: bytes,
        image_dict: ImageDict | ImageColorSpec,
        *,
        prefer_embedded_icc: bool = False,
    ) -> bytes | None:
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
                if prefer_embedded_icc and spec.icc_profile is not None:
                    converted = convert_icc_profile_samples(samples, spec.icc_profile)
                    if converted is not None:
                        if spec.channels == 1 and len(converted) == len(samples):
                            return ImageColorManager.convert_gray(converted)
                        return converted
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
        raw: bytes,
        spec: ImageColorSpec,
        image_dict: ImageDict | ImageColorSpec,
    ) -> bytes | None:
        bpc = spec.bits_per_component
        if bpc == 8:
            return ImageColorManager.apply_decode_array(raw, spec, image_dict)
        width = image_dimension(image_dict, "Width")
        height = image_dimension(image_dict, "Height")
        if width <= 0 or height <= 0:
            return None
        components = image_component_count(spec)
        unpacked = unpack_subbyte_image_samples(raw, bpc, width, height, components)
        if spec.kind == "Indexed":
            return unpacked
        return ImageColorManager.apply_decode_array(unpacked, spec, image_dict)

    @staticmethod
    def simple_device_color_fast_path(
        raw: bytes,
        spec: ImageColorSpec,
        image_dict: ImageDict,
    ) -> bytes | None:
        if spec.bits_per_component != 8:
            return None
        if spec.kind not in {"DeviceRGB", "DeviceGray"}:
            return None
        if lookup_dict_key(image_dict, "Decode") is not None:
            return None
        width = image_dimension(image_dict, "Width")
        height = image_dimension(image_dict, "Height")
        if width <= 0 or height <= 0:
            return None
        expected = width * height * (3 if spec.kind == "DeviceRGB" else 1)
        if len(raw) != expected:
            return None
        if spec.kind == "DeviceRGB":
            return raw
        return ImageColorManager.convert_gray(raw)

    @staticmethod
    def apply_decode_array(
        samples: bytes,
        spec: ImageColorSpec,
        image_dict: ImageDict | ImageColorSpec,
    ) -> bytes:
        if spec.kind == "Indexed":
            return samples
        components = image_component_count(spec)
        if components <= 0:
            return samples
        bpc = spec.bits_per_component
        max_sample = (1 << bpc) - 1
        if max_sample <= 0:
            return samples
        decode = lookup_dict_key(image_dict, "Decode") if isinstance(image_dict, dict) else None
        pairs: list[tuple[float, float]] = []
        if isinstance(decode, (list, tuple)) and len(decode) >= components * 2:
            for i in range(components):
                try:
                    dmin = float(decode[i * 2])
                    dmax = float(decode[i * 2 + 1])
                except (TypeError, ValueError):
                    pairs = []
                    break
                pairs.append((dmin, dmax))
        if not pairs:
            pairs = [(0.0, 1.0)] * components
        if bpc == 8 and all(pair == (0.0, 1.0) for pair in pairs):
            return samples
        if bpc == 8:
            return apply_decode_array_8bit(samples, tuple(pairs))
        return apply_decode_array_subbyte(samples, tuple(pairs), max_sample)

    @staticmethod
    def convert_separation(
        raw: bytes, color_space: ImageColorSpec | ColorSpaceSequence
    ) -> bytes | None:
        if isinstance(color_space, ImageColorSpec):
            alt_name = color_space.alt or "DeviceGray"
            tint_fn = color_space.tint_fn
        else:
            if len(color_space) < 4:
                raise ValueError("invalid Separation color space")
            alt_cs = color_space[2]
            if not isinstance(alt_cs, dict):
                raise ValueError("invalid Separation color space")
            alt_name = cs_name(lookup_dict_key(alt_cs, "ColorSpace"), "DeviceGray") or "DeviceGray"
            tint_fn = color_space[3] if len(color_space) > 3 else None

        result = bytearray()
        for byte in raw:
            v = byte / 255.0
            if tint_fn is None:
                components: ColorComponents = [v]
            elif isinstance(tint_fn, PdfStream):
                components = evaluate_sampled_tint_function(tint_fn, v)
            elif isinstance(tint_fn, (list, tuple)):
                if len(tint_fn) >= 1 and callable(tint_fn[0]):
                    try:
                        components = cast(ColorComponents, tint_fn[0](v))
                    except Exception as exc:
                        raise ValueError("invalid separation tint function") from exc
                else:
                    components = cast(ColorComponents, list(tint_fn))
            else:
                components = [v]
            expected = (
                1
                if alt_name == "DeviceGray"
                else 3
                if alt_name == "DeviceRGB"
                else 4
                if alt_name == "DeviceCMYK"
                else None
            )
            if expected is None:
                raise ValueError("invalid Separation color space")
            if len(components) != expected:
                raise ValueError("invalid separation tint function")
            rgb = ImageColorManager.apply_alt_color(components, alt_name)
            if rgb is not None:
                result.extend(rgb)
        return bytes(result)

    @staticmethod
    def convert_devicen(
        raw: bytes, color_space: ImageColorSpec | ColorSpaceSequence
    ) -> bytes | None:
        if isinstance(color_space, ImageColorSpec):
            alt_name = color_space.alt or ""
            n = color_space.channels
            tint_fn = color_space.tint_fn
        else:
            if len(color_space) < 4:
                raise ValueError("invalid DeviceN color space")
            names = color_space[1]
            alt_cs_name = color_space[2]
            alt_name = cs_name(alt_cs_name, "") or ""
            n = len(names) if isinstance(names, (list, tuple)) else 1
            tint_fn = color_space[3] if len(color_space) > 3 else None
        if n <= 0:
            raise ValueError("invalid DeviceN color space")
        if len(raw) % n != 0:
            raise ValueError("invalid DeviceN color sample data")

        result = bytearray()
        step = n
        for i in range(0, len(raw), step):
            components: ColorComponents = [raw[i + j] / 255.0 for j in range(step)]
            if isinstance(tint_fn, (list, tuple)) and len(tint_fn) >= 1 and callable(tint_fn[0]):
                try:
                    components = cast(ColorComponents, tint_fn[0](*components))
                except Exception as exc:
                    raise ValueError("invalid DeviceN tint function") from exc
            elif isinstance(tint_fn, PdfStream):
                components = evaluate_sampled_tint_function(tint_fn, *components)
            expected = (
                1
                if alt_name == "DeviceGray"
                else 3
                if alt_name == "DeviceRGB"
                else 4
                if alt_name == "DeviceCMYK"
                else None
            )
            if expected is None:
                raise ValueError("invalid DeviceN color space")
            if len(components) != expected:
                raise ValueError("invalid DeviceN tint function")
            rgb = ImageColorManager.apply_alt_color(components, alt_name)
            if rgb is not None:
                result.extend(rgb)
        return bytes(result)

    @staticmethod
    def convert_gray(raw: bytes) -> bytes:
        result = bytearray(len(raw) * 3)
        result[0::3] = raw
        result[1::3] = raw
        result[2::3] = raw
        return bytes(result)

    @staticmethod
    def convert_cmyk(raw: bytes) -> bytes:
        result = bytearray()
        n = len(raw)
        inv255 = 1.0 / 255.0
        if n % 4 != 0:
            raise ValueError("invalid color sample data")
        for i in range(0, n, 4):
            c, m, y, k = raw[i], raw[i + 1], raw[i + 2], raw[i + 3]
            r = int(255 * (1 - c * inv255) * (1 - k * inv255))
            g = int(255 * (1 - m * inv255) * (1 - k * inv255))
            b = int(255 * (1 - y * inv255) * (1 - k * inv255))
            result.extend([max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))])
        return bytes(result)

    @staticmethod
    def convert_indexed(raw: bytes, spec: ImageColorSpec) -> bytes | None:
        if spec.lookup is None:
            return None
        result = bytearray()
        lookup = spec.lookup
        hival = spec.hival
        if spec.base == "DeviceRGB":
            if len(lookup) < (hival + 1) * 3:
                raise ValueError("invalid Indexed color lookup")
            for byte in raw:
                if byte > hival:
                    raise ValueError("invalid Indexed color sample")
                start = byte * 3
                result.extend(lookup[start : start + 3])
        elif spec.base == "DeviceGray":
            if len(lookup) < hival + 1:
                raise ValueError("invalid Indexed color lookup")
            for byte in raw:
                if byte > hival:
                    raise ValueError("invalid Indexed color sample")
                val = lookup[byte]
                result.extend([val, val, val])
        else:
            raise ValueError("invalid Indexed color space")
        return bytes(result)

    @staticmethod
    def convert_to_rgb(
        raw: bytes, fn: Callable[..., tuple[float, float, float]], channels: int = 1
    ) -> bytes:
        if channels <= 0:
            raise ValueError("invalid color channel count")
        if len(raw) % channels != 0:
            raise ValueError("invalid color sample data")
        result = bytearray(len(raw) * 3)
        inv255 = 1.0 / 255.0
        for i in range(0, len(raw), channels):
            out = (i // channels) * 3
            if channels == 1:
                r, g, b = fn(raw[i] * inv255)
            elif i + channels - 1 < len(raw):
                r, g, b = fn(*(raw[i + j] * inv255 for j in range(channels)))
            else:
                raise ValueError("invalid color sample data")
            result[out] = max(0, min(255, int(r * 255.0)))
            result[out + 1] = max(0, min(255, int(g * 255.0)))
            result[out + 2] = max(0, min(255, int(b * 255.0)))
        return bytes(result)

    @staticmethod
    def convert_calgray(raw: bytes, params: object) -> bytes:
        gamma_raw = cs_param(params, "Gamma", 1.0)
        gamma = parse_float(gamma_raw, None)
        if gamma is None:
            raise ValueError("invalid color space parameters")

        return ImageColorManager.convert_gray(raw)

    @staticmethod
    def convert_calrgb(raw: bytes, params: object) -> bytes:
        bp = cs_param_floats(params, "BlackPoint", 3, [0.0, 0.0, 0.0])
        gamma = cs_param_floats(params, "Gamma", 3, [1.0, 1.0, 1.0])
        matrix = cs_param_floats(params, "Matrix", 9, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

        def fn(r: float, g: float, b: float) -> tuple[float, float, float]:
            rg = pow(r, gamma[0]) if gamma[0] != 1.0 else r
            gg = pow(g, gamma[1]) if gamma[1] != 1.0 else g
            bg = pow(b, gamma[2]) if gamma[2] != 1.0 else b
            x = bp[0] + matrix[0] * rg + matrix[1] * gg + matrix[2] * bg
            y = bp[1] + matrix[3] * rg + matrix[4] * gg + matrix[5] * bg
            z = bp[2] + matrix[6] * rg + matrix[7] * gg + matrix[8] * bg
            ax, ay, az = adapt_d50_to_d65(x, y, z)
            return xyz_to_srgb(ax, ay, az)

        return ImageColorManager.convert_to_rgb(raw, fn, channels=3)

    @staticmethod
    def convert_lab_raw(raw: bytes, params: object) -> bytes:
        wp = cs_param_floats(params, "WhitePoint", 3, [0.9505, 1.0, 1.089])
        range_a = cs_param_floats(params, "Range", 2, [-100.0, 100.0])

        def fn(l_byte: float, a_byte: float, b_byte: float) -> tuple[float, float, float]:
            l_star = l_byte * 100.0
            a_star = a_byte * (range_a[1] - range_a[0]) + range_a[0]
            b_star = b_byte * (range_a[1] - range_a[0]) + range_a[0]
            x, y, z = lab_to_xyz(l_star, a_star, b_star, wp)
            ax, ay, az = adapt_d50_to_d65(x, y, z)
            return xyz_to_srgb(ax, ay, az)

        return ImageColorManager.convert_to_rgb(raw, fn, channels=3)

    @staticmethod
    def apply_alt_color(components: ColorComponents, alt_name: str) -> bytes | None:
        if alt_name == "DeviceGray":
            v = max(0.0, min(1.0, components[0] if components else 0.0))
            v_byte = max(0, min(255, int(round(v * 255.0))))
            return bytes([v_byte, v_byte, v_byte])
        if alt_name == "DeviceRGB":
            r = max(0.0, min(1.0, components[0] if len(components) >= 1 else 0.0))
            g = max(0.0, min(1.0, components[1] if len(components) >= 2 else r))
            b = max(0.0, min(1.0, components[2] if len(components) >= 3 else r))
            return bytes(
                [
                    max(0, min(255, int(round(r * 255.0)))),
                    max(0, min(255, int(round(g * 255.0)))),
                    max(0, min(255, int(round(b * 255.0)))),
                ]
            )
        if alt_name == "DeviceCMYK":
            c = components[0] if len(components) >= 1 else 0.0
            m = components[1] if len(components) >= 2 else 0.0
            y = components[2] if len(components) >= 3 else 0.0
            k = components[3] if len(components) >= 4 else 0.0
            r = int(255 * (1 - c) * (1 - k))
            g_ = int(255 * (1 - m) * (1 - k))
            b_ = int(255 * (1 - y) * (1 - k))
            return bytes([max(0, min(255, r)), max(0, min(255, g_)), max(0, min(255, b_))])
        return None


@lru_cache(maxsize=1024)
def decode_translation_tables_8bit(
    pairs: tuple[tuple[float, float], ...],
) -> tuple[bytes, ...]:
    tables: list[bytes] = []
    for dmin, dmax in pairs:
        table = bytearray(256)
        span = dmax - dmin
        for value in range(256):
            decoded = dmin + (value / 255.0) * span
            table[value] = max(0, min(255, int(round(decoded * 255.0))))
        tables.append(bytes(table))
    return tuple(tables)


def apply_decode_array_8bit(
    samples: bytes,
    pairs: tuple[tuple[float, float], ...],
) -> bytes:
    tables = decode_translation_tables_8bit(pairs)
    if len(tables) == 1:
        return samples.translate(tables[0])
    components = len(tables)
    result = bytearray(len(samples))
    for index, table in enumerate(tables):
        result[index::components] = samples[index::components].translate(table)
    return bytes(result)


@lru_cache(maxsize=1024)
def decode_translation_tables_subbyte(
    max_sample: int,
    pairs: tuple[tuple[float, float], ...],
) -> tuple[bytes, ...]:
    tables: list[bytes] = []
    for dmin, dmax in pairs:
        table = bytearray(256)
        span = dmax - dmin
        for value in range(256):
            normalized = value / max_sample if max_sample > 0 else 0.0
            decoded = dmin + normalized * span
            table[value] = max(0, min(255, int(round(decoded * 255.0))))
        tables.append(bytes(table))
    return tuple(tables)


def apply_decode_array_subbyte(
    samples: bytes,
    pairs: tuple[tuple[float, float], ...],
    max_sample: int,
) -> bytes:
    tables = decode_translation_tables_subbyte(max_sample, pairs)
    if len(tables) == 1:
        return samples.translate(tables[0])
    components = len(tables)
    result = bytearray(len(samples))
    for index, table in enumerate(tables):
        result[index::components] = samples[index::components].translate(table)
    return bytes(result)


def image_dimension(image_dict: ImageDict | ImageColorSpec, key: str) -> int:
    if not isinstance(image_dict, dict):
        return 0
    value = lookup_dict_key(image_dict, key)
    if type(value) is bool:
        return 0
    return parse_int(value, 0) or 0


def image_component_count(spec: ImageColorSpec) -> int:
    if spec.kind in {"DeviceGray", "CalGray", "Indexed", "Separation"}:
        return 1
    if spec.kind in {"DeviceRGB", "Lab", "CalRGB"}:
        return 3
    if spec.kind == "DeviceCMYK":
        return 4
    if spec.kind in {"ICCBased", "DeviceN"}:
        return max(1, spec.channels)
    return 1


def evaluate_sampled_tint_function(function: PdfStream, *inputs: float) -> list[float]:
    dictionary = function.dictionary
    if parse_int(lookup_dict_key(dictionary, "FunctionType"), -1) != 0:
        raise ValueError("invalid sampled function")
    if parse_int(lookup_dict_key(dictionary, "BitsPerSample"), 0) != 8:
        raise ValueError("unsupported sampled function bit depth")

    size_obj = lookup_dict_key(dictionary, "Size")
    domain_obj = lookup_dict_key(dictionary, "Domain")
    range_obj = lookup_dict_key(dictionary, "Range")
    if not isinstance(size_obj, (list, tuple)):
        raise ValueError("invalid sampled function size")
    if not isinstance(domain_obj, (list, tuple)):
        raise ValueError("invalid sampled function domain")
    if not isinstance(range_obj, (list, tuple)) or len(range_obj) < 2:
        raise ValueError("invalid sampled function range")

    input_count = len(size_obj)
    if input_count != len(inputs):
        raise ValueError("invalid sampled function input count")
    if len(domain_obj) < input_count * 2:
        raise ValueError("invalid sampled function domain")
    output_count = len(range_obj) // 2

    sizes = [parse_int(value, 0) or 0 for value in size_obj]
    if any(size <= 0 for size in sizes):
        raise ValueError("invalid sampled function size")
    sample_count = 1
    for size in sizes:
        sample_count *= size
    samples = function.data
    if len(samples) < sample_count * output_count:
        raise ValueError("invalid sampled function data")

    encode_obj = lookup_dict_key(dictionary, "Encode")
    encoded_positions: list[int] = []
    for input_index, raw_input in enumerate(inputs):
        domain_min = parse_float(domain_obj[input_index * 2], None)
        domain_max = parse_float(domain_obj[input_index * 2 + 1], None)
        if domain_min is None or domain_max is None or domain_max == domain_min:
            raise ValueError("invalid sampled function domain")
        clipped = max(domain_min, min(domain_max, raw_input))
        normalized = (clipped - domain_min) / (domain_max - domain_min)
        encode_min = 0.0
        encode_max = float(sizes[input_index] - 1)
        if isinstance(encode_obj, (list, tuple)) and len(encode_obj) >= input_count * 2:
            parsed_min = parse_float(encode_obj[input_index * 2], None)
            parsed_max = parse_float(encode_obj[input_index * 2 + 1], None)
            if parsed_min is not None and parsed_max is not None:
                encode_min = parsed_min
                encode_max = parsed_max
        encoded = encode_min + normalized * (encode_max - encode_min)
        encoded_positions.append(max(0, min(sizes[input_index] - 1, int(round(encoded)))))

    sample_index = 0
    stride = 1
    for input_index, position in enumerate(encoded_positions):
        if input_index > 0:
            stride *= sizes[input_index - 1]
        sample_index += position * stride

    result: list[float] = []
    base = sample_index * output_count
    for output_index in range(output_count):
        range_min = parse_float(range_obj[output_index * 2], None)
        range_max = parse_float(range_obj[output_index * 2 + 1], None)
        if range_min is None or range_max is None:
            raise ValueError("invalid sampled function range")
        decoded = range_min + (samples[base + output_index] / 255.0) * (range_max - range_min)
        result.append(max(range_min, min(range_max, decoded)))
    return result


def unpack_subbyte_image_samples(
    data: bytes,
    bits_per_component: int,
    width: int,
    height: int,
    components: int,
) -> bytes:
    if bits_per_component not in {1, 2, 4}:
        return data
    if width <= 0 or height <= 0 or components <= 0:
        raise ValueError("invalid image dimensions")
    samples_per_row = width * components
    row_bytes = (samples_per_row * bits_per_component + 7) // 8
    if len(data) < row_bytes * height:
        raise ValueError("invalid image sample data")
    mask = (1 << bits_per_component) - 1
    output = bytearray(width * height * components)
    out = 0
    for row in range(height):
        row_start = row * row_bytes
        bit_offset = 0
        for ignored in range(samples_per_row):
            source = data[row_start + (bit_offset // 8)]
            shift = 8 - bits_per_component - (bit_offset % 8)
            output[out] = (source >> shift) & mask
            out += 1
            bit_offset += bits_per_component
    return bytes(output)
