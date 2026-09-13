# SPDX-License-Identifier: AGPL-3.0-only
"""Graphics-state soft-mask dictionaries, ISO 32000-2 11.5 and Table 142."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfValueResolver
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    require_pdf_integer,
    require_pdf_number,
    require_pdf_number_array,
)
from core_pdf_spec.s_08_graphics.color import initial_color_components
from core_pdf_spec.s_08_graphics.color_spec import ColorSpace, parse_color_space
from core_pdf_spec.s_08_graphics.matrix import Matrix
from core_pdf_spec.s_08_graphics.pdf_function import PdfFunctionEvaluator, compile_pdf_function
from core_pdf_spec.types import PdfReference


@dataclass(frozen=True, slots=True)
class SoftMask:
    """Resolved mask semantics with its original, unevaluated group stream.

    ``ctm`` is the invoking CTM at gs, before the Form's Matrix is concatenated.
    ``transfer=None`` means Identity; a supplied evaluator clips its single output
    to [0, 1]. Alpha masks ignore backdrop colour and blending colour space.
    Resources and group content remain demand-driven and preserve source identity.
    """

    subtype: Literal["Alpha", "Luminosity"]
    group: PdfStream
    ctm: Matrix
    transfer: PdfFunctionEvaluator | None = None
    backdrop_color: tuple[float, ...] | None = None
    color_space: ColorSpace | None = None


def internal_resolve(value: object, resolver: PdfValueResolver) -> object:
    seen: set[tuple[int, int]] = set()
    while isinstance(value, PdfReference):
        key = (value.object_number, value.generation_number)
        if key in seen:
            raise ValueError("cyclic soft-mask reference")
        seen.add(key)
        value = resolver.resolve(value)
    return value


def internal_array(value: object, resolver: PdfValueResolver) -> object:
    value = internal_resolve(value, resolver)
    if isinstance(value, (list, tuple)):
        return tuple(internal_resolve(item, resolver) for item in value)
    return value


def internal_function(
    value: object, resolver: PdfValueResolver, active: set[int]
) -> dict[object, object] | PdfStream:
    value = internal_resolve(value, resolver)
    if not isinstance(value, (dict, PdfStream)):
        raise ValueError("soft-mask transfer must be a function or Identity")
    identity = id(value)
    if identity in active:
        raise ValueError("cyclic soft-mask transfer function")
    active.add(identity)
    try:
        dictionary: dict[object, object] = dict(
            value.dictionary if isinstance(value, PdfStream) else cast(dict[object, object], value)
        )
        for key in ("FunctionType", "BitsPerSample", "Order", "N"):
            if key in dictionary:
                dictionary[key] = internal_resolve(dictionary[key], resolver)
        for key in ("Domain", "Range", "Size", "Encode", "Decode", "C0", "C1", "Bounds"):
            if key in dictionary:
                dictionary[key] = internal_array(dictionary[key], resolver)
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
            children = internal_resolve(dictionary.get("Functions"), resolver)
            if not isinstance(children, (list, tuple)) or not children:
                raise ValueError("invalid soft-mask stitching function")
            dictionary["Functions"] = tuple(
                internal_function(child, resolver, active) for child in children
            )
        elif function_type not in {0, 4}:
            raise ValueError("unsupported soft-mask transfer function")
        if function_type in {0, 4} and not isinstance(value, PdfStream):
            raise ValueError("soft-mask sampled and calculator functions require streams")
        return value.replace(dictionary=dictionary) if isinstance(value, PdfStream) else dictionary
    finally:
        active.remove(identity)


def internal_transfer(
    value: object,
    resolver: PdfValueResolver,
    compile_function: Callable[[object], PdfFunctionEvaluator],
) -> PdfFunctionEvaluator | None:
    value = internal_resolve(value, resolver)
    if value is None or resolver.resolve_name(value) == "Identity":
        return None
    evaluate = compile_function(internal_function(value, resolver, set()))

    def transfer(*inputs: float) -> tuple[float, ...]:
        if len(inputs) != 1:
            raise ValueError("soft-mask transfer must have one input")
        outputs = evaluate(max(0.0, min(1.0, require_pdf_number(inputs[0]))))
        if len(outputs) != 1:
            raise ValueError("soft-mask transfer must have one output")
        output = require_pdf_number(outputs[0], "invalid soft-mask transfer output")
        return (max(0.0, min(1.0, output)),)

    return transfer


def internal_color_space(value: object, resolver: PdfValueResolver) -> ColorSpace:
    value = internal_array(value, resolver)
    kind = resolver.resolve_name(value[0] if isinstance(value, tuple) and value else value)
    if kind not in {"DeviceGray", "DeviceRGB", "DeviceCMYK", "CalGray", "CalRGB", "ICCBased"}:
        raise ValueError("invalid soft-mask blending color space")
    if isinstance(value, tuple) and len(value) == 2:
        source = value[1]
        dictionary = source.dictionary if isinstance(source, PdfStream) else source
        if isinstance(dictionary, dict):
            prepared = dict(dictionary)
            for key in ("WhitePoint", "BlackPoint", "Gamma", "Matrix", "Range"):
                if key in prepared:
                    prepared[key] = internal_array(prepared[key], resolver)
            if "N" in prepared:
                prepared["N"] = internal_resolve(prepared["N"], resolver)
            # Group ICC spaces cannot use special alternates; resolve only the
            # selected colour description, never unrelated profile references.
            if "Alternate" in prepared:
                prepared["Alternate"] = internal_array(prepared["Alternate"], resolver)
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
    """Parse a selected SMask, preserving G identity and its unused resources.

    PDF /None clears the mask. Omitted/null graphics-state entries are handled
    by the caller. Function compilation is injectable; strict defaults support
    the function types implemented by ``compile_pdf_function``. This parser
    does not execute the group or select a raster/luminosity conversion policy.
    """
    value = internal_resolve(value, resolver)
    if resolver.resolve_name(value) == "None":
        return None
    if not isinstance(value, dict):
        raise ValueError("soft mask must be a dictionary or None")
    mask_type = internal_resolve(value.get("Type"), resolver)
    if mask_type is not None and resolver.resolve_name(mask_type) != "Mask":
        raise ValueError("invalid soft-mask dictionary Type")
    subtype = resolver.resolve_name(value.get("S"))
    if subtype not in {"Alpha", "Luminosity"}:
        raise ValueError("invalid soft-mask subtype")
    group = internal_resolve(value.get("G"), resolver)
    if (
        not isinstance(group, PdfStream)
        or resolver.resolve_name(group.dictionary.get("Subtype")) != "Form"
    ):
        raise ValueError("soft mask requires a Form XObject")
    group_type = internal_resolve(group.dictionary.get("Type"), resolver)
    if group_type is not None and resolver.resolve_name(group_type) != "XObject":
        raise ValueError("invalid soft-mask Form Type")
    attributes = internal_resolve(group.dictionary.get("Group"), resolver)
    if (
        not isinstance(attributes, dict)
        or resolver.resolve_name(attributes.get("S")) != "Transparency"
    ):
        raise ValueError("soft mask requires a transparency group")
    for key in ("I", "K"):
        flag = internal_resolve(attributes.get(key), resolver)
        if flag is not None and not isinstance(flag, bool):
            raise ValueError("invalid soft-mask group flag")
    bbox = require_pdf_number_array(internal_array(group.dictionary.get("BBox"), resolver))
    if len(bbox) != 4:
        raise ValueError("soft-mask Form requires a BBox")
    matrix = internal_array(group.dictionary.get("Matrix"), resolver)
    if matrix is not None:
        Matrix.from_operand(matrix)
    color_space = None
    backdrop_color = None
    if subtype == "Luminosity":
        color_space = internal_color_space(attributes.get("CS"), resolver)
        backdrop = internal_resolve(value.get("BC"), resolver)
        backdrop_color = (
            initial_color_components(color_space)
            if backdrop is None
            else require_pdf_number_array(internal_array(backdrop, resolver))
        )
        if backdrop_color is None or len(backdrop_color) != len(color_space.component_ranges):
            raise ValueError("invalid soft-mask backdrop component count")
    return SoftMask(
        "Alpha" if subtype == "Alpha" else "Luminosity",
        group,
        ctm,
        internal_transfer(value.get("TR"), resolver, compile_function),
        backdrop_color,
        color_space,
    )


__all__ = ("SoftMask", "parse_soft_mask")
