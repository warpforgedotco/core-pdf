# SPDX-License-Identifier: AGPL-3.0-only

"""ISO 32000-2 11.6.5: soft-mask dictionaries: their group, backdrop and transfer function.

A Luminosity mask (11.5.3) composites its group over an opaque backdrop of the BC colour,
C = (1 - ag) C0 + ag Cg, in the group's colour space, and takes the luminosity of C; the
transfer function then gives the mask value. Outside the group's bounding box the mask is
the BC colour's luminosity through the transfer function (11.6.5.1).

Where the clauses leave a choice open:

- A device space's luminosity is "implementation-defined"; this takes the formulas of
  11.5.3's example: a gray level as it is, 0.30 R + 0.59 G + 0.11 B, and
  1 - min(1, 0.3 C + 0.59 M + 0.11 Y + K).
- A CIE-based space (CalGray, CalRGB, ICCBased, and a device space a default space remaps
  to one) gives the CIE 1931 Y of C, relative to the space's white (colorimetric_luminosity),
  not a formula on its components. Core composites a CalGray, CalRGB or ICCBased gray or RGB
  group in sRGB, the colour every such colour converts to, and takes Y from that sRGB, which
  is the colour's own Y wherever sRGB holds it; an ICCBased CMYK group composites the CMYK of
  its profile, whose Y the profile gives, rescaled so that the space's black, 0 0 0 1, is 0
  and its white 1: a printing profile's black reflects a few percent, and Note 2 has a black
  backdrop leave zero luminosity outside the group's shape. An ICCBased space whose profile
  core cannot read converts through its Alternate, and its luminosity is the device one.
- BC's default is "the colour space's initial value, representing black": all components 0,
  which is black in an additive space, and for a CMYK space -- whose initial value under
  8.6.5.5 would be all components 0, white -- its black, 0 0 0 1, as DeviceCMYK's initial
  value is (default_backdrop_color).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.s_07_syntax.resolution import resolve_reference_chain
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict, PdfObject, PdfValueResolver
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import (
    require_pdf_integer,
    require_pdf_number,
    require_pdf_number_array,
)
from core_pdf.impl.spec.s_08_graphics.color import initial_color_components
from core_pdf.impl.spec.s_08_graphics.color_spec import ColorSpace, parse_color_space
from core_pdf.impl.spec.s_08_graphics.matrix import Matrix
from core_pdf.impl.spec.s_08_graphics.pdf_function import PdfFunctionEvaluator, compile_pdf_function
from core_pdf.impl.spec.types import PdfReference
from core_records import GeneratedRecord

if TYPE_CHECKING:
    from core_pdf.impl.spec.s_11_transparency.blending_space import BlendingSpace


class SoftMask(GeneratedRecord):
    subtype: Literal["Alpha", "Luminosity"]
    group: PdfStream
    ctm: Matrix
    transfer: PdfFunctionEvaluator | None = None
    backdrop_color: tuple[float, ...] | None = None
    color_space: ColorSpace | None = None


def resolve(value: object, resolver: PdfValueResolver) -> PdfObject:
    value = resolve_reference_chain(value, resolver.resolve)
    if isinstance(value, PdfReference):
        raise ValueError("cyclic soft-mask reference")
    return value  # type: ignore[return-value]  # ty: ignore[invalid-return-type]


def array(value: object, resolver: PdfValueResolver) -> PdfObject:
    value = resolve(value, resolver)
    if isinstance(value, (list, tuple)):
        return [resolve(item, resolver) for item in value]
    return value


def function(value: object, resolver: PdfValueResolver, active: set[int]) -> PdfDict | PdfStream:
    value = resolve(value, resolver)
    if not isinstance(value, (dict, PdfStream)):
        raise ValueError("soft-mask transfer must be a function or Identity")
    identity = id(value)
    if identity in active:
        raise ValueError("cyclic soft-mask transfer function")
    active.add(identity)
    try:
        dictionary: PdfDict = dict(value.dictionary if isinstance(value, PdfStream) else value)
        for key in ("FunctionType", "BitsPerSample", "Order", "N"):
            if key in dictionary:
                dictionary[key] = resolve(dictionary[key], resolver)
        for key in ("Domain", "Range", "Size", "Encode", "Decode", "C0", "C1", "Bounds"):
            if key in dictionary:
                dictionary[key] = array(dictionary[key], resolver)
        function_type = require_pdf_integer(dictionary.get("FunctionType"))
        domain = require_pdf_number_array(dictionary.get("Domain"))
        if len(domain) != 2 or domain[1] < domain[0]:
            raise ValueError("soft-mask transfer must have one input")
        if dictionary.get("Range") is not None:
            output_range = require_pdf_number_array(dictionary["Range"])
            if len(output_range) != 2 or output_range[1] < output_range[0]:
                raise ValueError("soft-mask transfer must have one output")
        elif function_type in {0, 4}:
            raise ValueError("soft-mask transfer requires a Range")
        if function_type == 2:
            for key, default in (("C0", (0.0,)), ("C1", (1.0,))):
                if len(require_pdf_number_array(dictionary.get(key, default))) != 1:
                    raise ValueError("soft-mask transfer must have one output")
        elif function_type == 3:
            children = resolve(dictionary.get("Functions"), resolver)
            if not isinstance(children, (list, tuple)) or not children:
                raise ValueError("invalid soft-mask stitching function")
            dictionary["Functions"] = [function(child, resolver, active) for child in children]
        elif function_type not in {0, 4}:
            raise ValueError("unsupported soft-mask transfer function")
        if function_type in {0, 4} and not isinstance(value, PdfStream):
            raise ValueError("soft-mask sampled and calculator functions require streams")
        return value.replace(dictionary=dictionary) if isinstance(value, PdfStream) else dictionary
    finally:
        active.remove(identity)


def transfer(
    value: object,
    resolver: PdfValueResolver,
    compile_function: Callable[[object], PdfFunctionEvaluator],
) -> PdfFunctionEvaluator | None:
    value = resolve(value, resolver)
    if value is None or resolver.resolve_name(value) == "Identity":
        return None
    evaluate = compile_function(function(value, resolver, set()))

    def transfer(*inputs: float) -> tuple[float, ...]:
        if len(inputs) != 1:
            raise ValueError("soft-mask transfer must have one input")
        outputs = evaluate(max(0.0, min(1.0, require_pdf_number(inputs[0]))))
        if len(outputs) != 1:
            raise ValueError("soft-mask transfer must have one output")
        output = require_pdf_number(outputs[0], "invalid soft-mask transfer output")
        return (max(0.0, min(1.0, output)),)

    return transfer


def resolve_soft_mask_color_space(value: object, resolver: PdfValueResolver) -> ColorSpace:
    value = array(value, resolver)
    kind = resolver.resolve_name(value[0] if isinstance(value, list) and value else value)
    if kind not in {"DeviceGray", "DeviceRGB", "DeviceCMYK", "CalGray", "CalRGB", "ICCBased"}:
        raise ValueError("invalid soft-mask blending color space")
    if isinstance(value, list) and len(value) == 2:
        source = value[1]
        dictionary = source.dictionary if isinstance(source, PdfStream) else source
        if isinstance(dictionary, dict):
            prepared = dict(dictionary)
            for key in ("WhitePoint", "BlackPoint", "Gamma", "Matrix", "Range"):
                if key in prepared:
                    prepared[key] = array(prepared[key], resolver)
            if "N" in prepared:
                prepared["N"] = resolve(prepared["N"], resolver)
            if "Alternate" in prepared:
                prepared["Alternate"] = array(prepared["Alternate"], resolver)
            value = (
                value[0],
                source.replace(dictionary=prepared) if isinstance(source, PdfStream) else prepared,
            )
    space = parse_color_space(value)
    if any(bounds != (0.0, 1.0) for bounds in space.component_ranges):
        raise ValueError("invalid soft-mask blending component range")
    if space.icc_profile is not None and space.icc_profile[16:20] == b"Lab ":
        raise ValueError("invalid soft-mask lightness-chromaticity blending space")
    return space


def parse_soft_mask(
    value: object,
    resolver: PdfValueResolver,
    *,
    ctm: Matrix,
    compile_function: Callable[[object], PdfFunctionEvaluator] = compile_pdf_function,
) -> SoftMask | None:
    value = resolve(value, resolver)
    if resolver.resolve_name(value) == "None":
        return None
    if not isinstance(value, dict):
        raise ValueError("soft mask must be a dictionary or None")
    mask_type = resolve(value.get("Type"), resolver)
    if mask_type is not None and resolver.resolve_name(mask_type) != "Mask":
        raise ValueError("invalid soft-mask dictionary Type")
    subtype = resolver.resolve_name(value.get("S"))
    if subtype not in {"Alpha", "Luminosity"}:
        raise ValueError("invalid soft-mask subtype")
    group = resolve(value.get("G"), resolver)
    if (
        not isinstance(group, PdfStream)
        or resolver.resolve_name(group.dictionary.get("Subtype")) != "Form"
    ):
        raise ValueError("soft mask requires a Form XObject")
    group_type = resolve(group.dictionary.get("Type"), resolver)
    if group_type is not None and resolver.resolve_name(group_type) != "XObject":
        raise ValueError("invalid soft-mask Form Type")
    attributes = resolve(group.dictionary.get("Group"), resolver)
    if (
        not isinstance(attributes, dict)
        or resolver.resolve_name(attributes.get("S")) != "Transparency"
    ):
        raise ValueError("soft mask requires a transparency group")
    for key in ("I", "K"):
        flag = resolve(attributes.get(key), resolver)
        if flag is not None and not isinstance(flag, bool):
            raise ValueError("invalid soft-mask group flag")
    bbox = require_pdf_number_array(array(group.dictionary.get("BBox"), resolver))
    if len(bbox) != 4:
        raise ValueError("soft-mask Form requires a BBox")
    matrix = array(group.dictionary.get("Matrix"), resolver)
    if matrix is not None:
        Matrix.from_operand(matrix)
    color_space = None
    backdrop_color = None
    if subtype == "Luminosity":
        color_space = resolve_soft_mask_color_space(attributes.get("CS"), resolver)
        backdrop = resolve(value.get("BC"), resolver)
        backdrop_color = (
            default_backdrop_color(color_space)
            if backdrop is None
            else require_pdf_number_array(array(backdrop, resolver))
        )
        if backdrop_color is None or len(backdrop_color) != len(color_space.component_ranges):
            raise ValueError("invalid soft-mask backdrop component count")
    return SoftMask(
        "Alpha" if subtype == "Alpha" else "Luminosity",
        group,
        ctm,
        transfer(value.get("TR"), resolver, compile_function),
        backdrop_color,
        color_space,
    )


@cites("ISO 32000-2", "11.6.5.1", introduced="1.4", rule="BC defaults to the space's black")
def default_backdrop_color(space: ColorSpace) -> tuple[float, ...] | None:
    """A Luminosity mask's BC when the dictionary has none (Table 142): the initial value of
    the group's space, which represents black, and for a CMYK space black, 0 0 0 1, though
    the initial value of an ICCBased CMYK space is 0 0 0 0, white."""
    if space.kind == "ICCBased" and len(space.component_ranges) == 4:
        return (0.0, 0.0, 0.0, 1.0)
    return initial_color_components(space)


@cites("ISO 32000-2", "11.5.3", introduced="1.4", rule="a CIE-based space's luminosity is Y")
def colorimetric_luminosity(space: ColorSpace | None) -> bool:
    """Whether a Luminosity mask whose group composites in space takes the CIE 1931 Y of its
    colour, as a CIE-based space gives (CalGray, CalRGB, ICCBased), rather than the gray
    level a device space's colour converts to (device_luminosity)."""
    return space is not None and space.kind in {"CalGray", "CalRGB", "ICCBased"}


@cites("ISO 32000-2", "11.5.3", introduced="1.4", rule="the colour a luminosity mask is taken from")
def luminosity_mask_color(
    backdrop: tuple[float, ...], color: tuple[float, ...], alpha: float
) -> tuple[float, ...]:
    """C = (1 - ag) x C0 + ag x Cg: the group's colour and alpha over the opaque backdrop C0,
    component by component (in a subtractive space, on the colorant amounts or their
    complements alike, since the weights sum to one)."""
    return tuple(
        (1.0 - alpha) * base + alpha * value for base, value in zip(backdrop, color, strict=True)
    )


@cites("ISO 32000-2", "11.5.3", introduced="1.4", rule="the luminosity of a device colour")
def device_luminosity(space: BlendingSpace, components: tuple[float, ...]) -> float:
    """The luminosity of a colour of a blending space, given as its own components: its gray
    level, or 11.5.3's formula for DeviceRGB or DeviceCMYK."""
    if space == "DeviceGray":
        (gray,) = components
        return gray
    if space == "DeviceRGB":
        red, green, blue = components
        return 0.3 * red + 0.59 * green + 0.11 * blue
    cyan, magenta, yellow, black = components
    return 1.0 - min(1.0, 0.3 * cyan + 0.59 * magenta + 0.11 * yellow + black)


@cites("ISO 32000-2", "11.6.5.1", introduced="1.4", rule="a luminosity mask outside its group")
def luminosity_mask_value(
    space: BlendingSpace, color: tuple[float, ...], transfer: PdfFunctionEvaluator | None = None
) -> float:
    """The mask value of a colour C of the group's space: its luminosity through the
    transfer function TR (None: Identity). With the BC colour, the value outside the group's
    bounding box."""
    luminosity = device_luminosity(space, color)
    return luminosity if transfer is None else transfer(luminosity)[0]


__all__ = (
    "SoftMask",
    "colorimetric_luminosity",
    "default_backdrop_color",
    "device_luminosity",
    "luminosity_mask_color",
    "luminosity_mask_value",
    "parse_soft_mask",
)
