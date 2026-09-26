# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar, TypeAlias

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    coerce_to_bytes,
    decoded_name,
    require_pdf_integer,
    require_pdf_number,
    require_pdf_number_array,
    require_pdf_number_pairs,
)
from core_pdf_spec.standards import PdfVersion, SemanticContext, require_recognized_version
from core_records import Record, frozen_setattr

ColorParams: TypeAlias = Mapping[str, object]
ComponentRanges: TypeAlias = tuple[tuple[float, float], ...]


class ColorSpace(Record):
    __slots__ = (
        "kind",
        "component_ranges",
        "params",
        "base",
        "alternate",
        "colorants",
        "hival",
        "lookup",
        "tint_fn",
        "icc_profile",
        "devicen_attributes",
    )

    kind: str
    component_ranges: ComponentRanges
    params: ColorParams
    base: ColorSpace | None
    alternate: ColorSpace | None
    colorants: tuple[str, ...]
    hival: int
    lookup: bytes | None
    tint_fn: object
    icc_profile: bytes | None
    devicen_attributes: DeviceNAttributes | None

    __fields__: ClassVar[tuple[str, ...]] = (
        "kind",
        "component_ranges",
        "params",
        "base",
        "alternate",
        "colorants",
        "hival",
        "lookup",
        "tint_fn",
        "icc_profile",
        "devicen_attributes",
    )
    __repr_fields__: ClassVar[tuple[str, ...]] = (
        "kind",
        "component_ranges",
        "params",
        "base",
        "alternate",
        "colorants",
        "hival",
        "lookup",
        "tint_fn",
        "devicen_attributes",
    )
    __match_args__ = (
        "kind",
        "component_ranges",
        "params",
        "base",
        "alternate",
        "colorants",
        "hival",
        "lookup",
        "tint_fn",
        "icc_profile",
        "devicen_attributes",
    )

    def __init__(
        self,
        kind: str,
        component_ranges: ComponentRanges,
        params: ColorParams | None = None,
        base: ColorSpace | None = None,
        alternate: ColorSpace | None = None,
        colorants: tuple[str, ...] = (),
        hival: int = 0,
        lookup: bytes | None = None,
        tint_fn: object = None,
        icc_profile: bytes | None = None,
        devicen_attributes: DeviceNAttributes | None = None,
    ) -> None:
        frozen_setattr(self, "kind", kind)
        frozen_setattr(self, "component_ranges", component_ranges)
        frozen_setattr(
            self, "params", (lambda: MappingProxyType({}))() if params is None else params
        )
        frozen_setattr(self, "base", base)
        frozen_setattr(self, "alternate", alternate)
        frozen_setattr(self, "colorants", colorants)
        frozen_setattr(self, "hival", hival)
        frozen_setattr(self, "lookup", lookup)
        frozen_setattr(self, "tint_fn", tint_fn)
        frozen_setattr(self, "icc_profile", icc_profile)
        frozen_setattr(self, "devicen_attributes", devicen_attributes)


class DeviceNProcess(Record):
    __slots__ = ("color_space", "components", "component_indices")

    color_space: ColorSpace
    components: tuple[str, ...]
    component_indices: tuple[int | None, ...]

    __fields__: ClassVar[tuple[str, ...]] = ("color_space", "components", "component_indices")
    __match_args__ = ("color_space", "components", "component_indices")

    def __init__(
        self,
        color_space: ColorSpace,
        components: tuple[str, ...],
        component_indices: tuple[int | None, ...],
    ) -> None:
        frozen_setattr(self, "color_space", color_space)
        frozen_setattr(self, "components", components)
        frozen_setattr(self, "component_indices", component_indices)


class DeviceNAttributes(Record):
    __slots__ = ("subtype", "process", "colorants")

    subtype: str
    process: DeviceNProcess | None
    colorants: Mapping[str, ColorSpace]

    __fields__: ClassVar[tuple[str, ...]] = ("subtype", "process", "colorants")
    __match_args__ = ("subtype", "process", "colorants")

    def __init__(
        self,
        subtype: str,
        process: DeviceNProcess | None,
        colorants: Mapping[str, ColorSpace],
    ) -> None:
        frozen_setattr(self, "subtype", subtype)
        frozen_setattr(self, "process", process)
        frozen_setattr(self, "colorants", colorants)


DEVICE_GRAY = ColorSpace("DeviceGray", ((0.0, 1.0),))
DEVICE_RGB = ColorSpace("DeviceRGB", ((0.0, 1.0),) * 3)
DEVICE_CMYK = ColorSpace("DeviceCMYK", ((0.0, 1.0),) * 4)
PATTERN = ColorSpace("Pattern", ())
DEVICE_SPACES = {space.kind: space for space in (DEVICE_GRAY, DEVICE_RGB, DEVICE_CMYK, PATTERN)}
CMYK_NAMES = ("Cyan", "Magenta", "Yellow", "Black")


def colorant_names(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("invalid colorant names")
    names = tuple(decoded_name(item) for item in value)
    if any(name is None for name in names):
        raise ValueError("invalid colorant name")
    return tuple(name for name in names if name is not None)


def device_n_names(names: tuple[str, ...], subtype: str) -> None:
    named = tuple(name for name in names if name != "None")
    if "All" in names or len(set(named)) != len(named):
        raise ValueError("invalid DeviceN colorant names")
    if subtype == "NChannel" and "None" in names:
        raise ValueError("None is not allowed in NChannel")


def parse_device_n_attributes(
    value: object,
    colorants: tuple[str, ...],
    *,
    context: SemanticContext | None = None,
) -> DeviceNAttributes:
    version = require_recognized_version(
        context, "color-space semantics require a recognized PDF version"
    )
    names = colorant_names(colorants)
    return parse_device_n_attributes_for(value, names, set(), version)


def device_n_process(
    value: object,
    names: tuple[str, ...],
    subtype: str,
    active: set[int],
    version: PdfVersion | None,
) -> DeviceNProcess:
    if not isinstance(value, dict):
        raise ValueError("invalid DeviceN Process dictionary")
    source = value
    space = parse_color_space_versioned(source.get("ColorSpace"), active, version)
    if space.kind not in {"DeviceGray", "DeviceRGB", "DeviceCMYK", "CalGray", "CalRGB", "ICCBased"}:
        raise ValueError("invalid DeviceN process color space")
    components = colorant_names(source.get("Components"))
    count = len(space.component_ranges)
    if len(components) != count or len(set(components)) != count:
        raise ValueError("invalid DeviceN process Components")
    cmyk = space.kind == "DeviceCMYK" or (space.kind == "ICCBased" and count == 4)
    if any(name in {"All", "None"} for name in components) or any(
        name in CMYK_NAMES and (not cmyk or name != CMYK_NAMES[index])
        for index, name in enumerate(components)
    ):
        raise ValueError("invalid reserved name in DeviceN process Components")
    indices: list[int | None] = [None] * count
    for index, name in enumerate(names):
        if name in CMYK_NAMES:
            if subtype == "NChannel" and not cmyk:
                raise ValueError("reserved CMYK colorant requires a CMYK process color space")
            if cmyk:
                component = CMYK_NAMES.index(name)
            elif name in components:
                component = components.index(name)
            else:
                continue
        elif name in components:
            component = components.index(name)
        else:
            continue
        if indices[component] is not None:
            raise ValueError("conflicting DeviceN process component aliases")
        indices[component] = index
    if subtype == "NChannel" and not cmyk:
        first = indices[0]
        if first is None or indices != list(range(first, first + count)):
            raise ValueError("NChannel process components must be complete and in natural order")
    return DeviceNProcess(space, components, tuple(indices))


def parse_device_n_attributes_for(
    value: object,
    names: tuple[str, ...],
    active: set[int],
    version: PdfVersion | None,
) -> DeviceNAttributes:
    if not isinstance(value, dict):
        raise ValueError("invalid DeviceN attributes")
    marker = id(value)
    if marker in active:
        raise ValueError("color space cycle detected")
    active.add(marker)
    try:
        source = value
        subtype = "DeviceN" if source.get("Subtype") is None else decoded_name(source["Subtype"])
        if subtype not in {"DeviceN", "NChannel"}:
            raise ValueError("invalid DeviceN attributes Subtype")
        device_n_names(names, subtype)
        raw_process = source.get("Process")
        process = (
            device_n_process(raw_process, names, subtype, active, version)
            if raw_process is not None
            else None
        )
        if subtype == "NChannel" and process is None and any(name in CMYK_NAMES for name in names):
            raise ValueError("NChannel process colorants require a Process dictionary")
        process_names = set(process.components) if process is not None else set()
        if process is not None and (
            process.color_space.kind == "DeviceCMYK"
            or (process.color_space.kind == "ICCBased" and len(process.components) == 4)
        ):
            process_names.update(CMYK_NAMES)
        raw_colorants = source.get("Colorants")
        colorant_spaces: dict[str, ColorSpace] = {}
        if raw_colorants is not None:
            if not isinstance(raw_colorants, dict):
                raise ValueError("invalid DeviceN Colorants dictionary")
            for raw_name, raw_space in raw_colorants.items():
                name = decoded_name(raw_name)
                if name is None:
                    raise ValueError("invalid DeviceN Colorants name")
                if name in process_names or raw_space is None:
                    continue
                space = parse_color_space_versioned(raw_space, active, version)
                if space.kind != "Separation" or space.colorants != (name,):
                    raise ValueError("DeviceN Colorants entry must match its Separation name")
                colorant_spaces[name] = space
        if subtype == "NChannel" and any(
            name not in process_names and name not in colorant_spaces for name in names
        ):
            raise ValueError("NChannel spot colorants require matching Colorants entries")
        if source.get("MixingHints") is not None and not isinstance(source["MixingHints"], dict):
            raise ValueError("invalid DeviceN MixingHints dictionary")
        return DeviceNAttributes(subtype, process, MappingProxyType(colorant_spaces))
    finally:
        active.remove(marker)


def array(value: object, size: int, message: str) -> tuple[float, ...]:
    values = require_pdf_number_array(value, message)
    if len(values) != size:
        raise ValueError(message)
    return values


def parse_component_ranges(value: object, count: int) -> ComponentRanges:
    ranges = require_pdf_number_pairs(value, "invalid color component Range", count=count)
    if any(low > high for low, high in ranges):
        raise ValueError("invalid color component Range")
    return ranges


def calibrated_params(kind: str, source: PdfDict) -> ColorParams:
    params: dict[str, object] = {}
    white = array(source.get("WhitePoint"), 3, "invalid color WhitePoint")
    if white[0] <= 0 or white[1] != 1 or white[2] <= 0:
        raise ValueError("invalid color WhitePoint")
    params["WhitePoint"] = white
    black = array(
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
        gammas = array(
            (1, 1, 1) if source.get("Gamma") is None else source["Gamma"], 3, "invalid color Gamma"
        )
        if any(value <= 0 for value in gammas):
            raise ValueError("invalid color Gamma")
        params["Gamma"] = gammas
        params["Matrix"] = array(
            (1, 0, 0, 0, 1, 0, 0, 0, 1) if source.get("Matrix") is None else source["Matrix"],
            9,
            "invalid color Matrix",
        )
    return MappingProxyType(params)


def parse_color_space(value: object, *, context: SemanticContext | None = None) -> ColorSpace:
    version = require_recognized_version(
        context, "color-space semantics require a recognized PDF version"
    )
    return parse_color_space_versioned(value, set(), version)


def parse_color_space_versioned(
    value: object, active: set[int], version: PdfVersion | None
) -> ColorSpace:
    name = decoded_name(value)
    if name in DEVICE_SPACES:
        return DEVICE_SPACES[name]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("invalid color space")
    marker = id(value)
    if marker in active:
        raise ValueError("color space cycle detected")
    active.add(marker)
    try:
        kind = decoded_name(value[0])
        if kind in DEVICE_SPACES and len(value) == 1:
            return DEVICE_SPACES[kind]
        if kind == "Pattern" and len(value) == 2:
            base = parse_color_space_versioned(value[1], active, version)
            if base.kind == "Pattern":
                raise ValueError("Pattern cannot be its own underlying color space")
            return ColorSpace(kind, base.component_ranges, base=base)
        if kind == "Indexed" and len(value) == 4:
            base = parse_color_space_versioned(value[1], active, version)
            if base.kind in {"Indexed", "Pattern"}:
                raise ValueError("invalid Indexed base color space")
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
            source = value[1]
            params = calibrated_params(kind, source)
            ranges = (
                (
                    (0.0, 100.0),
                    *parse_component_ranges(
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
            # Named apart from the calibrated branch's dictionary above: a
            # stream dictionary is typed more loosely than a PdfDict, because
            # PdfStream sits below the object types it would otherwise name.
            stream_dictionary = stream.dictionary
            count = require_pdf_integer(
                stream_dictionary.get("N"), "invalid ICCBased channel count"
            )
            if count not in {1, 3, 4}:
                raise ValueError("invalid ICCBased channel count")
            ranges = parse_component_ranges(
                (0, 1) * count
                if stream_dictionary.get("Range") is None
                else stream_dictionary["Range"],
                count,
            )
            raw_alt = stream_dictionary.get("Alternate")
            alternate = (
                {1: DEVICE_GRAY, 3: DEVICE_RGB, 4: DEVICE_CMYK}[count]
                if raw_alt is None
                else parse_color_space_versioned(raw_alt, active, version)
            )
            if alternate.kind in {"Pattern", "Indexed", "Separation", "DeviceN", "ICCBased"}:
                raise ValueError("invalid ICCBased alternate color space")
            if len(alternate.component_ranges) != count:
                raise ValueError("invalid ICCBased alternate component count")
            params = MappingProxyType(
                {
                    str(key): item
                    for key, item in stream_dictionary.items()
                    if key not in {"N", "Range", "Alternate"}
                }
            )
            return ColorSpace(kind, ranges, params, alternate=alternate, icc_profile=stream.data)
        if kind in {"Separation", "DeviceN"} and len(value) in (
            {4} if kind == "Separation" else {4, 5}
        ):
            names = [value[1]] if kind == "Separation" else value[1]
            colorants = colorant_names(names)
            if kind == "DeviceN":
                device_n_names(colorants, "DeviceN")
            alternate = parse_color_space_versioned(value[2], active, version)
            if alternate.kind in {"Pattern", "Indexed", "Separation", "DeviceN"}:
                raise ValueError("invalid alternate color space")
            if value[3] is None:
                raise ValueError("missing tint transform")
            params = {}
            attributes = None
            if len(value) == 5:
                attributes = parse_device_n_attributes_for(value[4], colorants, active, version)
                params["Attributes"] = MappingProxyType(dict(value[4]))
            return ColorSpace(
                kind,
                ((0.0, 1.0),) * len(colorants),
                MappingProxyType(params),
                alternate=alternate,
                colorants=colorants,
                tint_fn=value[3],
                devicen_attributes=attributes,
            )
        raise ValueError(f"invalid {kind or ''} color space")
    finally:
        active.remove(marker)


__all__ = (
    "ColorParams",
    "ComponentRanges",
    "ColorSpace",
    "DeviceNProcess",
    "DeviceNAttributes",
    "DEVICE_GRAY",
    "DEVICE_RGB",
    "DEVICE_CMYK",
    "PATTERN",
    "parse_color_space",
    "parse_device_n_attributes",
)
