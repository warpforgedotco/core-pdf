# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from core_jpeg.impl.errors import JpegParseError, JpegUnsupportedError

LRCP = 0
RLCP = 1
RPCL = 2
PCRL = 3
CPRL = 4

JPX_MAX_RESOLUTION_LEVELS = 33
JPX_CODING_STYLE_SUPPORTED = 0x01 | 0x02 | 0x04
JPX_CODEBLOCK_STYLE_LAZY = 0x01
JPX_CODEBLOCK_STYLE_RESET = 0x02
JPX_CODEBLOCK_STYLE_TERMALL = 0x04
JPX_CODEBLOCK_STYLE_VSC = 0x08
JPX_CODEBLOCK_STYLE_PTERM = 0x10
JPX_CODEBLOCK_STYLE_SEGSYM = 0x20
JPX_CODEBLOCK_STYLE_HT = 0x40
JPX_CODEBLOCK_STYLE_HT_MIXED = 0x80
JPX_SUPPORTED_CODEBLOCK_STYLE = (
    JPX_CODEBLOCK_STYLE_LAZY
    | JPX_CODEBLOCK_STYLE_RESET
    | JPX_CODEBLOCK_STYLE_TERMALL
    | JPX_CODEBLOCK_STYLE_VSC
    | JPX_CODEBLOCK_STYLE_PTERM
    | JPX_CODEBLOCK_STYLE_SEGSYM
)


@dataclass(frozen=True)
class JpxComponentCodingParams:
    levels: int
    codeblock_w: int
    codeblock_h: int
    codeblock_style: int
    precincts: list[Any]
    reversible: bool


@dataclass(frozen=True)
class JpxProgressionChange:
    resolution_start: int
    component_start: int
    layer_end: int
    resolution_end: int
    component_end: int
    progression_order: int


@dataclass(frozen=True)
class JpxCodingParams:
    levels: int
    codeblock_w: int
    codeblock_h: int
    codeblock_style: int
    prog_order: int
    num_layers: int
    packet_uses_sop: bool
    packet_uses_eph: bool
    multiple_component_transform: int
    precincts: list[Any]
    component_coding_params: list[JpxComponentCodingParams | None]
    progression_changes: list[JpxProgressionChange]
    roi_shift_by_component: list[int | None]
    quant_guard_bits: int
    quant_guard_bits_by_component: list[int | None]
    quant_style: int
    quant_style_by_component: list[int | None]
    quant_steps: list[list[tuple[int, int]]]
    reversible: bool


def copy_component_coding_params(
    params: JpxComponentCodingParams,
) -> JpxComponentCodingParams:
    return JpxComponentCodingParams(
        levels=params.levels,
        codeblock_w=params.codeblock_w,
        codeblock_h=params.codeblock_h,
        codeblock_style=params.codeblock_style,
        precincts=list(params.precincts),
        reversible=params.reversible,
    )


def copy_coding_params(params: JpxCodingParams) -> JpxCodingParams:
    return JpxCodingParams(
        levels=params.levels,
        codeblock_w=params.codeblock_w,
        codeblock_h=params.codeblock_h,
        codeblock_style=params.codeblock_style,
        prog_order=params.prog_order,
        num_layers=params.num_layers,
        packet_uses_sop=params.packet_uses_sop,
        packet_uses_eph=params.packet_uses_eph,
        multiple_component_transform=params.multiple_component_transform,
        precincts=list(params.precincts),
        component_coding_params=[
            copy_component_coding_params(component_params) if component_params is not None else None
            for component_params in params.component_coding_params
        ],
        progression_changes=list(params.progression_changes),
        roi_shift_by_component=list(params.roi_shift_by_component),
        quant_guard_bits=params.quant_guard_bits,
        quant_guard_bits_by_component=list(params.quant_guard_bits_by_component),
        quant_style=params.quant_style,
        quant_style_by_component=list(params.quant_style_by_component),
        quant_steps=[list(steps) for steps in params.quant_steps],
        reversible=params.reversible,
    )


def validate_jpx_coding_params(params: JpxCodingParams) -> None:
    if params.codeblock_style & ~JPX_SUPPORTED_CODEBLOCK_STYLE:
        raise JpegUnsupportedError("JPX code-block style includes unsupported HT mode")
    for component_params in params.component_coding_params:
        if (
            component_params is not None
            and component_params.codeblock_style & ~JPX_SUPPORTED_CODEBLOCK_STYLE
        ):
            raise JpegUnsupportedError(
                "JPX component code-block style includes unsupported HT mode"
            )


def validate_jpx_spcod_spcoc(levels: int, cb_w: int, cb_h: int) -> None:
    if levels + 1 > JPX_MAX_RESOLUTION_LEVELS:
        raise JpegParseError("too many JPX resolution levels")
    if cb_w + 2 > 10 or cb_h + 2 > 10 or cb_w + cb_h + 4 > 12:
        raise JpegParseError("invalid JPX code-block dimensions")


def validate_jpx_precinct_size(value: int, resolution_index: int) -> None:
    if resolution_index != 0 and ((value & 0x0F) == 0 or (value >> 4) == 0):
        raise JpegParseError("invalid JPX precinct size")


def default_component_coding_params(
    params: JpxCodingParams,
) -> JpxComponentCodingParams:
    return JpxComponentCodingParams(
        levels=params.levels,
        codeblock_w=params.codeblock_w,
        codeblock_h=params.codeblock_h,
        codeblock_style=params.codeblock_style,
        precincts=list(params.precincts),
        reversible=params.reversible,
    )


def coding_component_style_params(
    params: JpxCodingParams, component_index: int
) -> JpxComponentCodingParams:
    if component_index < len(params.component_coding_params):
        component_params = params.component_coding_params[component_index]
        if component_params is not None:
            return component_params
    return default_component_coding_params(params)


def coding_params_with_component_style(
    params: JpxCodingParams,
    component_index: int,
    component_params: JpxComponentCodingParams,
) -> JpxCodingParams:
    component_coding_params = [
        copy_component_coding_params(current) if current is not None else None
        for current in params.component_coding_params
    ]
    while len(component_coding_params) <= component_index:
        component_coding_params.append(None)
    component_coding_params[component_index] = copy_component_coding_params(
        component_params,
    )
    return replace(params, component_coding_params=component_coding_params)


def coding_params_with_component_quantization(
    params: JpxCodingParams,
    component_index: int,
    guard_bits: int,
    style: int,
    steps: list[tuple[int, int]],
) -> JpxCodingParams:
    quant_steps = [list(component_steps) for component_steps in params.quant_steps]
    while len(quant_steps) <= component_index:
        quant_steps.append([])
    quant_steps[component_index] = steps

    guard_bits_by_component = list(params.quant_guard_bits_by_component)
    while len(guard_bits_by_component) <= component_index:
        guard_bits_by_component.append(None)
    guard_bits_by_component[component_index] = guard_bits

    style_by_component = list(params.quant_style_by_component)
    while len(style_by_component) <= component_index:
        style_by_component.append(None)
    style_by_component[component_index] = style
    return replace(
        params,
        quant_steps=quant_steps,
        quant_guard_bits_by_component=guard_bits_by_component,
        quant_style_by_component=style_by_component,
    )


def coding_component_roi_shift(params: JpxCodingParams, component_index: int) -> int:
    if component_index < len(params.roi_shift_by_component):
        roi_shift = params.roi_shift_by_component[component_index]
        if roi_shift is not None:
            return roi_shift
    return 0


def coding_params_with_component_roi_shift(
    params: JpxCodingParams,
    component_index: int,
    roi_shift: int,
) -> JpxCodingParams:
    roi_shift_by_component = list(params.roi_shift_by_component)
    while len(roi_shift_by_component) <= component_index:
        roi_shift_by_component.append(None)
    roi_shift_by_component[component_index] = roi_shift
    return replace(params, roi_shift_by_component=roi_shift_by_component)


def coding_component_quant_steps(
    params: JpxCodingParams, component_index: int
) -> list[tuple[int, int]]:
    if component_index < len(params.quant_steps) and params.quant_steps[component_index]:
        return params.quant_steps[component_index]
    if not params.quant_steps or not params.quant_steps[0]:
        raise JpegParseError("missing JPX quantization defaults")
    return params.quant_steps[0]


def coding_component_quant_guard_bits(params: JpxCodingParams, component_index: int) -> int:
    if component_index < len(params.quant_guard_bits_by_component):
        guard_bits = params.quant_guard_bits_by_component[component_index]
        if guard_bits is not None:
            return guard_bits
    return params.quant_guard_bits


def coding_component_quant_style(params: JpxCodingParams, component_index: int) -> int:
    if component_index < len(params.quant_style_by_component):
        style = params.quant_style_by_component[component_index]
        if style is not None:
            return style
    return params.quant_style
