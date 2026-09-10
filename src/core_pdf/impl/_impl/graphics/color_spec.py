# SPDX-License-Identifier: AGPL-3.0-only
"""Color-space input compatibility and selected ICC fallback policy."""

from __future__ import annotations

from typing import TypeAlias, cast

from core_pdf.impl._impl.graphics.icc_profiles import (
    IccProfileError,
    parse_icc_transform,
)
from core_pdf.impl._impl.model.pdf_values import coerce_to_bytes
from core_pdf.impl._impl.runtime.scalars import parse_float, parse_int
from core_pdf.impl.types import MISSING
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax_primitives.coercion import normalize_pdf_name
from core_pdf_spec.s_08_graphics.color_spec import (
    ImageColorSpec,
)
from core_pdf_spec.s_08_graphics.color_spec import (
    color_spec_from_value as parse_pdf_color_spec,
)

ColorParams: TypeAlias = dict[str, object]


def cs_param(params: object, key: str, default: object = None) -> object:
    if isinstance(params, dict):
        return params.get(key, default)
    return default


def cs_param_floats(params: object, key: str, count: int, default: list[float]) -> list[float]:
    raw = cs_param(params, key, default)
    if isinstance(raw, (list, tuple)) and len(raw) >= count:
        result: list[float] = []
        for value in raw[:count]:
            parsed = parse_float(value, None)
            if parsed is None:
                raise ValueError("invalid color space parameters")
            result.append(parsed)
        return result
    return default


def describe_color_space(value: object) -> str | None:
    """Return a compact name for an image or shading color-space value."""
    prefixes: list[str] = []
    seen: set[int] = set()
    current = value
    while True:
        name = normalize_pdf_name(current)
        if name is not None:
            return ":".join((*prefixes, name))
        if not isinstance(current, (list, tuple)) or not current:
            return ":".join(prefixes) if prefixes else None
        marker = id(current)
        if marker in seen:
            return ":".join(prefixes) if prefixes else None
        seen.add(marker)
        kind = normalize_pdf_name(current[0])
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


def normalize_indexed_base_color_space_name(value: object) -> str | None:
    direct = normalize_pdf_name(value)
    if direct is not None:
        return direct
    if not isinstance(value, (list, tuple)) or not value:
        return None
    kind = normalize_pdf_name(value[0])
    if kind == "ICCBased" and len(value) >= 2 and isinstance(value[1], (dict, PdfStream)):
        icc_stream = value[1]
        icc_dict = icc_stream.dictionary if isinstance(icc_stream, PdfStream) else icc_stream
        alt = normalize_pdf_name(icc_dict.get("Alternate"))
        if alt is not None:
            return alt
        n = cs_param(icc_dict, "N", 3)
        channels = parse_int(n, 3)
        if isinstance(icc_stream, PdfStream):
            try:
                alt = parse_icc_transform(icc_stream.data).alternate_color_space
            except IccProfileError:
                alt = None
            if alt is not None:
                return alt
        return {1: "DeviceGray", 3: "DeviceRGB", 4: "DeviceCMYK"}.get(channels or 3)
    return None


def normalize_image_color_spec(image_dict: object) -> ImageColorSpec:
    dictionary = image_dict if isinstance(image_dict, dict) else {}
    raw_bpc = dictionary.get("BitsPerComponent", MISSING)
    if raw_bpc is not MISSING:
        if type(raw_bpc) is bool:
            raise ValueError("invalid image bits-per-component")
        parsed_bpc = parse_int(raw_bpc, None)
        if parsed_bpc is None:
            raise ValueError("invalid image bits-per-component")
        bits_per_component = parsed_bpc
    else:
        bits_per_component = 8
    if bits_per_component <= 0:
        raise ValueError("invalid image bits-per-component")
    return color_spec_from_value(
        dictionary.get("ColorSpace"), bits_per_component=bits_per_component
    )


def color_spec_from_value(color_space: object, *, bits_per_component: int = 8) -> ImageColorSpec:
    """Apply legacy ICC fallback policy around the passive PDF description."""
    if isinstance(color_space, (list, tuple)) and color_space:
        kind = normalize_pdf_name(color_space[0])
        if kind == "Indexed" and len(color_space) >= 4:
            raw_hival = color_space[2]
            hival = parse_int(raw_hival, None)
            if type(raw_hival) is bool or hival is None or hival < 0:
                raise ValueError("invalid hival")
            lookup = color_space[3]
            lookup_bytes: bytes | None
            if isinstance(lookup, PdfStream):
                lookup_bytes = lookup.data
            else:
                try:
                    lookup_bytes = coerce_to_bytes(lookup)
                except TypeError:
                    lookup_bytes = lookup if isinstance(lookup, bytes) else None
            return ImageColorSpec(
                kind="Indexed",
                params={},
                bits_per_component=bits_per_component,
                base=normalize_indexed_base_color_space_name(color_space[1]),
                hival=hival,
                lookup=lookup_bytes,
            )
        if (
            kind == "ICCBased"
            and len(color_space) >= 2
            and isinstance(color_space[1], (dict, PdfStream))
        ):
            icc_stream = color_space[1]
            icc_dict = icc_stream.dictionary if isinstance(icc_stream, PdfStream) else icc_stream
            alt = normalize_pdf_name(icc_dict.get("Alternate"))
            n = cs_param(icc_dict, "N", 3)
            channels = parse_int(n, None)
            if type(n) is bool or channels is None or channels <= 0:
                raise ValueError("invalid ICCBased color space")
            # PdfStream.data re-runs the filter pipeline on every access, so a
            # compressed profile is decoded once while parsing this spec.
            icc_profile = icc_stream.data if isinstance(icc_stream, PdfStream) else None
            if icc_profile is not None:
                try:
                    parsed_transform = parse_icc_transform(icc_profile)
                except IccProfileError:
                    pass
                else:
                    if alt is None:
                        alt = parsed_transform.alternate_color_space
            return ImageColorSpec(
                kind="ICCBased",
                params=cast(ColorParams, icc_dict),
                bits_per_component=bits_per_component,
                alt=alt,
                channels=channels,
                icc_profile=icc_profile,
            )
    return parse_pdf_color_spec(color_space, bits_per_component=bits_per_component)
