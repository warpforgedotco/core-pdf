# SPDX-License-Identifier: AGPL-3.0-only
"""PDF color-space descriptions, independent of sample layout and output devices."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias, cast

from core_pdf_spec.exceptions import PdfUnsupportedError
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    coerce_to_bytes,
    decoded_name,
    require_pdf_integer,
    require_pdf_number,
    require_pdf_number_array,
)
from core_pdf_spec.standards import PdfVersion, SemanticContext

ColorParams: TypeAlias = Mapping[str, object]
ComponentRanges: TypeAlias = tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True, eq=False)
class ColorSpace:
    kind: str
    component_ranges: ComponentRanges
    params: ColorParams = field(default_factory=lambda: MappingProxyType({}))
    base: ColorSpace | None = None
    alternate: ColorSpace | None = None
    colorants: tuple[str, ...] = ()
    hival: int = 0
    lookup: bytes | None = None
    tint_fn: object = None
    icc_profile: bytes | None = field(default=None, repr=False)


DEVICE_GRAY = ColorSpace("DeviceGray", ((0.0, 1.0),))
DEVICE_RGB = ColorSpace("DeviceRGB", ((0.0, 1.0),) * 3)
DEVICE_CMYK = ColorSpace("DeviceCMYK", ((0.0, 1.0),) * 4)
PATTERN = ColorSpace("Pattern", ())
internal_DEVICE_SPACES = {
    space.kind: space for space in (DEVICE_GRAY, DEVICE_RGB, DEVICE_CMYK, PATTERN)
}


def internal_array(value: object, size: int, message: str) -> tuple[float, ...]:
    values = require_pdf_number_array(value, message)
    if len(values) != size:
        raise ValueError(message)
    return values


def internal_ranges(value: object, count: int) -> ComponentRanges:
    values = internal_array(value, 2 * count, "invalid color component Range")
    ranges = tuple(zip(values[::2], values[1::2], strict=True))
    if any(low > high for low, high in ranges):
        raise ValueError("invalid color component Range")
    return ranges


def internal_calibrated_params(kind: str, source: dict) -> ColorParams:
    params: dict[str, object] = {}
    white = internal_array(source.get("WhitePoint"), 3, "invalid color WhitePoint")
    if white[0] <= 0 or white[1] != 1 or white[2] <= 0:
        raise ValueError("invalid color WhitePoint")
    params["WhitePoint"] = white
    black = internal_array(
        (0, 0, 0) if source.get("BlackPoint") is None else source["BlackPoint"],
        3,
        "invalid color BlackPoint",
    )
    if any(value < 0 for value in black):
        raise ValueError("invalid color BlackPoint")
    params["BlackPoint"] = black
    if kind == "CalGray":
        raw = source.get("Gamma")
        gamma = require_pdf_number(1 if raw is None else raw, "invalid color Gamma")
        if gamma <= 0:
            raise ValueError("invalid color Gamma")
        params["Gamma"] = gamma
    elif kind == "CalRGB":
        gammas = internal_array(
            (1, 1, 1) if source.get("Gamma") is None else source["Gamma"], 3, "invalid color Gamma"
        )
        if any(value <= 0 for value in gammas):
            raise ValueError("invalid color Gamma")
        params["Gamma"] = gammas
        params["Matrix"] = internal_array(
            (1, 0, 0, 0, 1, 0, 0, 0, 1) if source.get("Matrix") is None else source["Matrix"],
            9,
            "invalid color Matrix",
        )
    return MappingProxyType(params)


def parse_color_space(value: object, *, context: SemanticContext | None = None) -> ColorSpace:
    """Parse a resolved color-space value without discarding nested spaces.

    ISO 32000-1, 8.6: component ranges belong to the color space; image sample
    bit depth does not. Parameter arrays are copied into immutable tuples.

    An explicit context enforces the version-dependent Indexed base constraint
    in Adobe PDF Reference 1.3, 4.5.5 (pp. 181-182). Omitting context preserves
    the historical all-version API. This does not validate feature availability
    for every kind of color space.
    """
    if context is not None and (context.version is None or not context.version.recognized):
        raise PdfUnsupportedError("color-space semantics require a recognized PDF version")
    version = context.version if context is not None else None
    return internal_parse_color_space(value, set(), version)


def internal_parse_color_space(
    value: object, active: set[int], version: PdfVersion | None
) -> ColorSpace:
    name = decoded_name(value)
    if name in internal_DEVICE_SPACES:
        return internal_DEVICE_SPACES[name]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("invalid color space")
    marker = id(value)
    if marker in active:
        raise ValueError("color space cycle detected")
    active.add(marker)
    try:
        kind = decoded_name(value[0])
        if kind in internal_DEVICE_SPACES and len(value) == 1:
            return internal_DEVICE_SPACES[kind]
        if kind == "Pattern" and len(value) == 2:
            base = internal_parse_color_space(value[1], active, version)
            if base.kind == "Pattern":
                raise ValueError("Pattern cannot be its own underlying color space")
            return ColorSpace(kind, base.component_ranges, base=base)
        if kind == "Indexed" and len(value) == 4:
            base = internal_parse_color_space(value[1], active, version)
            if base.kind in {"Indexed", "Pattern"}:
                raise ValueError("invalid Indexed base color space")
            # Adobe PDF Reference 1.3, 4.5.5 explicitly requires an error for
            # these bases in PDF 1.2; ISO 32000-2:2020, 8.6.6.3 retains 1.3
            # as the introduction of the broader Indexed base constraint.
            if (
                base.kind in {"Separation", "DeviceN"}
                and version is not None
                and version < PdfVersion(1, 3)
            ):
                raise ValueError("Indexed Separation and DeviceN bases require PDF 1.3")
            hival = require_pdf_integer(value[2], "invalid hival")
            if not 0 <= hival <= 255:
                raise ValueError("invalid hival")
            lookup = value[3].data if isinstance(value[3], PdfStream) else coerce_to_bytes(value[3])
            if len(lookup) != (hival + 1) * len(base.component_ranges):
                raise ValueError("invalid Indexed color lookup")
            return ColorSpace(kind, ((0.0, float(hival)),), base=base, hival=hival, lookup=lookup)
        if kind in {"Lab", "CalGray", "CalRGB"} and len(value) == 2 and isinstance(value[1], dict):
            source = cast(dict[object, object], value[1])
            params = internal_calibrated_params(kind, source)
            ranges = (
                (
                    (0.0, 100.0),
                    *internal_ranges(
                        (-100, 100, -100, 100) if source.get("Range") is None else source["Range"],
                        2,
                    ),
                )
                if kind == "Lab"
                else ((0.0, 1.0),) * (1 if kind == "CalGray" else 3)
            )
            return ColorSpace(kind, ranges, params)
        if kind == "ICCBased" and len(value) == 2 and isinstance(value[1], PdfStream):
            stream = value[1]
            source = stream.dictionary
            count = require_pdf_integer(source.get("N"), "invalid ICCBased channel count")
            if count not in {1, 3, 4}:
                raise ValueError("invalid ICCBased channel count")
            ranges = internal_ranges(
                (0, 1) * count if source.get("Range") is None else source["Range"], count
            )
            raw_alt = source.get("Alternate")
            alternate = (
                {1: DEVICE_GRAY, 3: DEVICE_RGB, 4: DEVICE_CMYK}[count]
                if raw_alt is None
                else internal_parse_color_space(raw_alt, active, version)
            )
            if alternate.kind in {"Pattern", "Indexed", "Separation", "DeviceN", "ICCBased"}:
                raise ValueError("invalid ICCBased alternate color space")
            if len(alternate.component_ranges) != count:
                raise ValueError("invalid ICCBased alternate component count")
            params = MappingProxyType(
                {
                    str(key): item
                    for key, item in source.items()
                    if key not in {"N", "Range", "Alternate"}
                }
            )
            return ColorSpace(kind, ranges, params, alternate=alternate, icc_profile=stream.data)
        if kind in {"Separation", "DeviceN"} and len(value) in (
            {4} if kind == "Separation" else {4, 5}
        ):
            names = [value[1]] if kind == "Separation" else value[1]
            if not isinstance(names, (list, tuple)) or not names:
                raise ValueError("invalid colorant names")
            colorants = tuple(decoded_name(item) for item in names)
            if any(item is None for item in colorants):
                raise ValueError("invalid colorant name")
            alternate = internal_parse_color_space(value[2], active, version)
            if alternate.kind in {"Pattern", "Indexed", "Separation", "DeviceN"}:
                raise ValueError("invalid alternate color space")
            if value[3] is None:
                raise ValueError("missing tint transform")
            params = {}
            if len(value) == 5:
                if not isinstance(value[4], dict):
                    raise ValueError("invalid DeviceN attributes")
                params["Attributes"] = MappingProxyType(dict(value[4]))
            return ColorSpace(
                kind,
                ((0.0, 1.0),) * len(colorants),
                MappingProxyType(params),
                alternate=alternate,
                colorants=tuple(item for item in colorants if item is not None),
                tint_fn=value[3],
            )
        raise ValueError(f"invalid {kind or ''} color space")
    finally:
        active.remove(marker)


__all__ = (
    "ColorParams",
    "ComponentRanges",
    "ColorSpace",
    "DEVICE_GRAY",
    "DEVICE_RGB",
    "DEVICE_CMYK",
    "PATTERN",
    "parse_color_space",
)
