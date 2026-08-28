# SPDX-License-Identifier: AGPL-3.0-only
"""Native image color-space specification parsing."""

from __future__ import annotations

from typing import TypeAlias, cast

from core_pdf.impl.engine.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.engine.spec.s_07_syntax_primitives.coercion import (
    coerce_to_bytes,
    normalize_pdf_name,
    parse_float,
    parse_int,
)
from core_pdf.impl.engine.spec.s_07_syntax_primitives.pdfdict import (
    lookup_dict_key,
    lookup_dict_key_default,
)
from core_pdf.impl.engine.spec.s_08_graphics.icc_profiles import (
    IccProfileError,
    parse_icc_transform,
)
from core_pdf.impl.primitives import MISSING

ColorParams: TypeAlias = dict[str, object]
ColorNameList: TypeAlias = list[object] | tuple[object, ...]


def cs_param(params: object, key: str, default: object = None) -> object:
    if isinstance(params, dict):
        value = lookup_dict_key_default(params, key, MISSING)
        return default if value is MISSING else value
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


class ImageColorSpec:
    __slots__ = (
        "kind",
        "params",
        "bits_per_component",
        "base",
        "hival",
        "lookup",
        "alt",
        "tint_fn",
        "names",
        "channels",
        "icc_profile",
    )

    kind: str | None
    params: ColorParams
    bits_per_component: int
    base: str | None
    hival: int
    lookup: bytes | None
    alt: str | None
    tint_fn: object
    names: ColorNameList | None
    channels: int
    icc_profile: bytes | None

    def __init__(
        self,
        kind: str | None,
        params: ColorParams,
        bits_per_component: int = 8,
        base: str | None = None,
        hival: int = 0,
        lookup: bytes | None = None,
        alt: str | None = None,
        tint_fn: object = None,
        names: ColorNameList | None = None,
        channels: int = 1,
        icc_profile: bytes | None = None,
    ) -> None:
        self.kind = kind
        self.params = params
        self.bits_per_component = bits_per_component
        self.base = base
        self.hival = hival
        self.lookup = lookup
        self.alt = alt
        self.tint_fn = tint_fn
        self.names = names
        self.channels = channels
        self.icc_profile = icc_profile


def normalize_color_space_name(value: object) -> str | None:
    result = normalize_pdf_name(value)
    if result is not None:
        return result
    return None


def normalize_indexed_base_color_space_name(value: object) -> str | None:
    direct = normalize_color_space_name(value)
    if direct is not None:
        return direct
    if not isinstance(value, (list, tuple)) or not value:
        return None
    kind = normalize_color_space_name(value[0])
    if kind == "ICCBased" and len(value) >= 2 and isinstance(value[1], (dict, PdfStream)):
        icc_stream = value[1]
        icc_dict = icc_stream.dictionary if isinstance(icc_stream, PdfStream) else icc_stream
        alt = normalize_color_space_name(lookup_dict_key(icc_dict, "Alternate"))
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
    raw_bpc = lookup_dict_key_default(image_dict, "BitsPerComponent", MISSING)
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

    def parse_indexed_hival(value: object) -> int:
        if type(value) is bool:
            raise ValueError("invalid hival")
        parsed = parse_int(value, None)
        if parsed is None:
            raise ValueError("invalid hival")
        if parsed < 0:
            raise ValueError("invalid hival")
        return parsed

    def parse_channel_count(value: object) -> int:
        if type(value) is bool:
            raise ValueError("invalid ICCBased color space")
        parsed = parse_int(value, None)
        if parsed is None:
            raise ValueError("invalid ICCBased color space")
        if parsed <= 0:
            raise ValueError("invalid ICCBased color space")
        return parsed

    color_space = lookup_dict_key(image_dict, "ColorSpace")
    if isinstance(color_space, (list, tuple)) and color_space:
        kind = normalize_color_space_name(color_space[0])
        if kind == "Indexed" and len(color_space) >= 4:
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
                hival=parse_indexed_hival(color_space[2]),
                lookup=lookup_bytes,
            )
        if kind == "Indexed":
            raise ValueError("invalid Indexed color space")
        if (
            kind == "ICCBased"
            and len(color_space) >= 2
            and isinstance(color_space[1], (dict, PdfStream))
        ):
            icc_stream = color_space[1]
            icc_dict = icc_stream.dictionary if isinstance(icc_stream, PdfStream) else icc_stream
            alt = normalize_color_space_name(lookup_dict_key(icc_dict, "Alternate"))
            n = cs_param(icc_dict, "N", 3)
            channels = parse_channel_count(n)
            icc_profile = icc_stream.data if isinstance(icc_stream, PdfStream) else None
            if alt is None and isinstance(icc_stream, PdfStream):
                try:
                    alt = parse_icc_transform(icc_profile).alternate_color_space
                except IccProfileError:
                    alt = None
            return ImageColorSpec(
                kind="ICCBased",
                params=cast(ColorParams, icc_dict),
                bits_per_component=bits_per_component,
                alt=alt,
                channels=channels,
                icc_profile=icc_profile,
            )
        if kind == "ICCBased":
            raise ValueError("invalid ICCBased color space")
        if (
            kind in {"Lab", "CalGray", "CalRGB"}
            and len(color_space) >= 2
            and isinstance(color_space[1], dict)
        ):
            base = "DeviceGray" if kind == "CalGray" else "DeviceRGB" if kind == "CalRGB" else None
            return ImageColorSpec(
                kind=kind,
                params=cast(ColorParams, color_space[1]),
                bits_per_component=bits_per_component,
                base=base,
            )
        if kind in {"Lab", "CalGray", "CalRGB"}:
            raise ValueError(f"invalid {kind} color space")
        if kind in {"Separation", "DeviceN"}:
            if len(color_space) < 4:
                raise ValueError(f"invalid {kind} color space")
            names = color_space[1] if isinstance(color_space[1], (list, tuple)) else None
            if kind == "DeviceN" and names is None:
                raise ValueError("invalid DeviceN color space")
            alt = normalize_color_space_name(color_space[2])
            return ImageColorSpec(
                kind=kind,
                params={},
                bits_per_component=bits_per_component,
                alt=alt,
                tint_fn=color_space[3] if len(color_space) >= 4 else None,
                names=list(names)
                if kind == "DeviceN" and isinstance(names, (list, tuple))
                else None,
                channels=len(names)
                if kind == "DeviceN" and isinstance(names, (list, tuple))
                else 1,
            )
        if kind in {"Separation", "DeviceN"}:
            raise ValueError(f"invalid {kind} color space")
        return ImageColorSpec(kind=kind, params={}, bits_per_component=bits_per_component)
    return ImageColorSpec(
        kind=normalize_color_space_name(color_space),
        params={},
        bits_per_component=bits_per_component,
    )
