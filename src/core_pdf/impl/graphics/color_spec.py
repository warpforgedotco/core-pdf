# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from contextlib import suppress
from math import isfinite
from types import MappingProxyType
from typing import cast

from core_pdf.impl.graphics.icc_profiles import (
    IccProfileError,
    parse_icc_transform,
)
from core_pdf.impl.model.pdf_values import coerce_to_bytes
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.runtime.scalars import parse_float, parse_int
from core_pdf_spec.exceptions import PdfParseError, PdfUnsupportedError
from core_pdf_spec.s_07_filters.errors import FilterError
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.color import color_space_paints
from core_pdf_spec.s_08_graphics.color_spec import (
    ColorParams,
    ColorSpace,
    DeviceNAttributes,
    DeviceNProcess,
    parse_device_n_attributes,
)
from core_pdf_spec.s_08_graphics.color_spec import (
    parse_color_space as parse_pdf_color_space,
)


def cs_param_floats(params: ColorParams, key: str, count: int, default: list[float]) -> list[float]:
    raw = params.get(key, default)
    if isinstance(raw, (list, tuple)) and len(raw) >= count:
        result: list[float] = []
        for value in raw[:count]:
            parsed = parse_float(value, None)
            if parsed is None or not isfinite(parsed):
                raise ValueError("invalid color space parameters")
            result.append(parsed)
        return result
    return default


def describe_color_space(value: object) -> str | None:
    prefixes: list[str] = []
    seen: set[int] = set()
    current = value
    while True:
        name = recover_pdf_name(current)
        if name is not None:
            return ":".join((*prefixes, name))
        if not isinstance(current, (list, tuple)) or not current:
            return ":".join(prefixes) if prefixes else None
        marker = id(current)
        if marker in seen:
            return ":".join(prefixes) if prefixes else None
        seen.add(marker)
        kind = recover_pdf_name(current[0])
        if kind == "Indexed":
            prefixes.append("Indexed")
            if len(current) <= 1:
                return ":".join(prefixes)
            current = current[1]
            continue
        if kind == "ICCBased":
            prefixes.append("ICCBased")
            if len(current) <= 1:
                return ":".join(prefixes)
            profile = current[1]
            if isinstance(profile, PdfStream):
                profile_dictionary = profile.dictionary
            elif isinstance(profile, dict):
                profile_dictionary = profile
            else:
                return ":".join(prefixes)
            alternate = profile_dictionary.get("Alternate")
            if alternate is None:
                return ":".join(prefixes)
            current = alternate
            continue
        if kind is None:
            return ":".join(prefixes) if prefixes else None
        return ":".join((*prefixes, kind))


def recover_image_bits_per_component(image_dict: object) -> int:
    dictionary = image_dict if isinstance(image_dict, dict) else {}
    raw = dictionary.get("BitsPerComponent", 8)
    value = parse_int(raw, None)
    if type(raw) is bool or value is None or value <= 0:
        raise ValueError("invalid image bits-per-component")
    return value


def parse_color_space(value: object) -> ColorSpace:
    return internal_parse_color_space(value, set())


def internal_color_space_paints(value: object) -> bool:
    seen: set[int] = set()
    while isinstance(value, (list, tuple)) and value:
        marker = id(value)
        if marker in seen:
            return True
        seen.add(marker)
        kind = recover_pdf_name(value[0])
        if kind == "Indexed" and len(value) == 4 or kind == "Pattern" and len(value) == 2:
            value = value[1]
            continue
        if kind == "Separation" and len(value) == 4:
            names: tuple[str | None, ...] = (recover_pdf_name(value[1]),)
        elif kind == "DeviceN" and len(value) in {4, 5}:
            raw_names = value[1]
            if not isinstance(raw_names, (list, tuple)) or not raw_names:
                return True
            names = tuple(recover_pdf_name(name) for name in raw_names)
        else:
            return True
        if any(name is None for name in names):
            return True
        return color_space_paints(ColorSpace(kind, (), colorants=cast(tuple[str, ...], names)))
    return True


def internal_nchannel_attributes(space: ColorSpace) -> DeviceNAttributes | None:
    attributes = space.devicen_attributes
    if space.kind != "DeviceN" or attributes is None or attributes.subtype != "NChannel":
        return None
    return attributes


def internal_nchannel_process(space: ColorSpace) -> DeviceNProcess | None:
    attributes = internal_nchannel_attributes(space)
    if attributes is None:
        return None
    process = attributes.process
    if process is None:
        return None
    mapped = {index for index in process.component_indices if index is not None}
    return process if mapped == set(range(len(space.colorants))) else None


def internal_parse_color_space(value: object, active: set[int]) -> ColorSpace:
    marker = id(value) if isinstance(value, (list, tuple)) else None
    if marker is not None:
        if marker in active:
            return ColorSpace("Unknown", ())
        active.add(marker)
    try:
        direct = recover_pdf_name(value)
        if direct in {"DeviceGray", "DeviceRGB", "DeviceCMYK", "Pattern"}:
            return parse_pdf_color_space(direct)
        if isinstance(value, (list, tuple)) and value:
            kind = recover_pdf_name(value[0])
            if len(value) == 1 and kind in {"DeviceGray", "DeviceRGB", "DeviceCMYK", "Pattern"}:
                return parse_pdf_color_space(kind)
            if kind == "Indexed" and len(value) >= 4:
                hival = parse_int(value[2], None)
                if type(value[2]) is bool or hival is None or hival < 0:
                    raise ValueError("invalid hival")
                try:
                    lookup = (
                        value[3].data
                        if isinstance(value[3], PdfStream)
                        else coerce_to_bytes(value[3])
                    )
                except TypeError:
                    lookup = None
                base = internal_parse_color_space(value[1], active)
                return ColorSpace(
                    "Indexed", ((0.0, float(hival)),), base=base, hival=hival, lookup=lookup
                )
            if kind == "ICCBased" and len(value) >= 2 and isinstance(value[1], (dict, PdfStream)):
                stream = value[1]
                source = cast(
                    dict[object, object],
                    stream.dictionary if isinstance(stream, PdfStream) else stream,
                )
                raw_count = source.get("N", 3)
                count = parse_int(raw_count, None)
                if type(raw_count) is bool or count is None or count <= 0:
                    raise ValueError("invalid ICCBased color space")
                profile = stream.data if isinstance(stream, PdfStream) else None
                alternate = (
                    internal_parse_color_space(source["Alternate"], active)
                    if source.get("Alternate") is not None
                    else None
                )
                if profile is not None and alternate is None:
                    with suppress(IccProfileError):
                        alternate = internal_parse_color_space(
                            parse_icc_transform(profile).alternate_color_space, active
                        )
                if alternate is None:
                    alternate = internal_parse_color_space(
                        {1: "DeviceGray", 3: "DeviceRGB", 4: "DeviceCMYK"}.get(count), active
                    )
                ranges = (
                    internal_recovery_ranges(source.get("Range"), count, (0.0, 1.0) * count)
                    if count in {1, 3, 4}
                    else ()
                )
                params = MappingProxyType(
                    {
                        str(key): item
                        for key, item in source.items()
                        if key not in {"N", "Range", "Alternate"}
                    }
                )
                return ColorSpace(
                    "ICCBased", ranges, params, alternate=alternate, icc_profile=profile
                )
            if (
                kind in {"Lab", "CalRGB", "CalGray"}
                and len(value) >= 2
                and isinstance(value[1], dict)
            ):
                source = dict(value[1])
                calibrated_params: dict[str, object] = {}
                for key, raw in source.items():
                    if key == "Range":
                        continue
                    if isinstance(raw, (list, tuple)):
                        calibrated_params[str(key)] = tuple(raw)
                    else:
                        calibrated_params[str(key)] = raw
                ranges = (
                    (
                        (0.0, 100.0),
                        *internal_recovery_ranges(source.get("Range"), 2, (-100.0, 100.0) * 2),
                    )
                    if kind == "Lab"
                    else ((0.0, 1.0),) * (1 if kind == "CalGray" else 3)
                )
                return ColorSpace(kind, ranges, MappingProxyType(calibrated_params))
            if kind == "Pattern" and len(value) == 2:
                pattern_base = internal_parse_color_space(value[1], active)
                return ColorSpace(
                    kind,
                    pattern_base.component_ranges,
                    base=pattern_base,
                )
            if kind in {"Separation", "DeviceN"} and len(value) >= 4:
                raw_names = value[1] if kind == "DeviceN" else [value[1]]
                if not isinstance(raw_names, (list, tuple)):
                    raise ValueError("invalid DeviceN color space")
                names = tuple(recover_pdf_name(item) or "" for item in raw_names)
                attributes = None
                devicen_params: dict[str, object] = {}
                if kind == "DeviceN" and len(value) >= 5:
                    raw_attributes = value[4]
                    devicen_params["Attributes"] = (
                        MappingProxyType(dict(raw_attributes))
                        if isinstance(raw_attributes, dict)
                        else raw_attributes
                    )
                    with suppress(
                        TypeError, ValueError, FilterError, PdfParseError, PdfUnsupportedError
                    ):
                        attributes = parse_device_n_attributes(raw_attributes, names)
                return ColorSpace(
                    kind,
                    ((0.0, 1.0),) * len(names),
                    MappingProxyType(devicen_params),
                    alternate=internal_parse_color_space(value[2], active),
                    colorants=names,
                    tint_fn=value[3],
                    devicen_attributes=attributes,
                )
        try:
            return parse_pdf_color_space(value)
        except TypeError, ValueError:
            name = recover_pdf_name(value)
            if name is None and isinstance(value, (list, tuple)) and value:
                name = recover_pdf_name(value[0])
            if name is not None:
                return ColorSpace(name, ())
            return ColorSpace("Unknown", ())
    finally:
        if marker is not None:
            active.remove(marker)


def internal_recovery_ranges(
    raw: object, count: int, default: tuple[float, ...]
) -> tuple[tuple[float, float], ...]:
    values = cs_param_floats(
        {"Range": default if raw is None else raw}, "Range", count * 2, list(default)
    )
    ranges = tuple(zip(values[::2], values[1::2], strict=True))
    if any(low > high for low, high in ranges):
        raise ValueError("invalid color component Range")
    return ranges
