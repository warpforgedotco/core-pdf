# SPDX-License-Identifier: AGPL-3.0-only
"""Native PDF color-space specification parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias, cast

from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import (
    coerce_to_bytes,
    normalize_pdf_name,
    parse_int,
)
from core_pdf.impl.types import MISSING

ColorParams: TypeAlias = dict[str, object]


def cs_param(params: object, key: str, default: object = None) -> object:
    if isinstance(params, dict):
        return params.get(key, default)
    return default


@dataclass(frozen=True, slots=True, eq=False)
class ImageColorSpec:
    kind: str | None
    params: ColorParams
    bits_per_component: int = 8
    base: str | None = None
    hival: int = 0
    lookup: bytes | None = None
    alt: str | None = None
    tint_fn: object = None
    channels: int = 1
    icc_profile: bytes | None = field(default=None, repr=False)
    base_spec: ImageColorSpec | None = None


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
        n = cs_param(icc_dict, "N")
        channels = parse_int(n, None)
        return (
            {1: "DeviceGray", 3: "DeviceRGB", 4: "DeviceCMYK"}.get(channels)
            if channels is not None
            else None
        )
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
        raise ValueError("missing image bits-per-component")
    if bits_per_component <= 0:
        raise ValueError("invalid image bits-per-component")
    return color_spec_from_value(
        dictionary.get("ColorSpace"), bits_per_component=bits_per_component
    )


def color_spec_from_value(color_space: object, *, bits_per_component: int = 8) -> ImageColorSpec:
    """Parse a colour-space name or array into its resolved description.

    Split out of ``normalize_image_color_spec`` so the content-stream
    interpreter can resolve the operand of ``cs``/``CS``. Nothing here is
    image-specific: ISO 32000-1 8.6 gives one colour-space grammar, used by both
    an image's /ColorSpace entry and a colour-space resource.
    """

    def parse_indexed_hival(value: object) -> int:
        if type(value) is bool:
            raise ValueError("invalid hival")
        parsed = parse_int(value, None)
        if parsed is None:
            raise ValueError("invalid hival")
        if not 0 <= parsed <= 255:
            raise ValueError("invalid hival")
        return parsed

    def parse_channel_count(value: object) -> int:
        if type(value) is bool:
            raise ValueError("invalid ICCBased color space")
        parsed = parse_int(value, None)
        if parsed is None:
            raise ValueError("invalid ICCBased color space")
        if parsed not in {1, 3, 4}:
            raise ValueError("invalid ICCBased color space")
        return parsed

    if isinstance(color_space, (list, tuple)) and color_space:
        kind = normalize_pdf_name(color_space[0])
        if kind == "Indexed" and len(color_space) >= 4:
            base_value = color_space[1]
            base_kind = normalize_pdf_name(
                base_value[0]
                if isinstance(base_value, (list, tuple)) and base_value
                else base_value
            )
            if base_kind in {"Indexed", "Pattern"}:
                raise ValueError("invalid Indexed base color space")
            lookup = color_space[3]
            lookup_bytes: bytes | None
            if isinstance(lookup, PdfStream):
                lookup_bytes = lookup.data
            else:
                lookup_bytes = coerce_to_bytes(lookup)
            return ImageColorSpec(
                kind="Indexed",
                params={},
                bits_per_component=bits_per_component,
                base=normalize_indexed_base_color_space_name(color_space[1]),
                hival=parse_indexed_hival(color_space[2]),
                lookup=lookup_bytes,
                base_spec=color_spec_from_value(base_value),
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
            alt = normalize_pdf_name(icc_dict.get("Alternate"))
            n = cs_param(icc_dict, "N")
            channels = parse_channel_count(n)
            # PdfStream.data re-runs the filter pipeline on every access, so a
            # compressed profile is decoded once while parsing this spec.
            icc_profile = icc_stream.data if isinstance(icc_stream, PdfStream) else None
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
            return ImageColorSpec(
                kind=kind,
                params=cast(ColorParams, color_space[1]),
                bits_per_component=bits_per_component,
            )
        if kind in {"Lab", "CalGray", "CalRGB"}:
            raise ValueError(f"invalid {kind} color space")
        if kind in {"Separation", "DeviceN"}:
            if len(color_space) < 4:
                raise ValueError(f"invalid {kind} color space")
            names = color_space[1] if isinstance(color_space[1], (list, tuple)) else None
            if kind == "DeviceN" and names is None:
                raise ValueError("invalid DeviceN color space")
            alt = normalize_pdf_name(color_space[2])
            return ImageColorSpec(
                kind=kind,
                params={},
                bits_per_component=bits_per_component,
                alt=alt,
                tint_fn=color_space[3],
                channels=len(names)
                if kind == "DeviceN" and isinstance(names, (list, tuple))
                else 1,
            )
        return ImageColorSpec(kind=kind, params={}, bits_per_component=bits_per_component)
    return ImageColorSpec(
        kind=normalize_pdf_name(color_space),
        params={},
        bits_per_component=bits_per_component,
    )
