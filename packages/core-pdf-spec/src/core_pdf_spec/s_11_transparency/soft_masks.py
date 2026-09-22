# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Literal, NoReturn, Self, TypeAlias, cast

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict, PdfKey, PdfValueResolver
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

frozen_setattr = object.__setattr__


class SoftMask:
    __slots__ = ("subtype", "group", "ctm", "transfer", "backdrop_color", "color_space")

    subtype: Literal["Alpha", "Luminosity"]
    group: PdfStream
    ctm: Matrix
    transfer: PdfFunctionEvaluator | None
    backdrop_color: tuple[float, ...] | None
    color_space: ColorSpace | None

    __fields__: ClassVar[tuple[str, ...]] = (
        "subtype",
        "group",
        "ctm",
        "transfer",
        "backdrop_color",
        "color_space",
    )
    __match_args__ = ("subtype", "group", "ctm", "transfer", "backdrop_color", "color_space")

    def __init__(
        self,
        subtype: Literal["Alpha", "Luminosity"],
        group: PdfStream,
        ctm: Matrix,
        transfer: PdfFunctionEvaluator | None = None,
        backdrop_color: tuple[float, ...] | None = None,
        color_space: ColorSpace | None = None,
    ) -> None:
        frozen_setattr(self, "subtype", subtype)
        frozen_setattr(self, "group", group)
        frozen_setattr(self, "ctm", ctm)
        frozen_setattr(self, "transfer", transfer)
        frozen_setattr(self, "backdrop_color", backdrop_color)
        frozen_setattr(self, "color_space", color_space)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"subtype={self.subtype!r}, "
            f"group={self.group!r}, "
            f"ctm={self.ctm!r}, "
            f"transfer={self.transfer!r}, "
            f"backdrop_color={self.backdrop_color!r}, "
            f"color_space={self.color_space!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.subtype == other.subtype
            and self.group == other.group
            and self.ctm == other.ctm
            and self.transfer == other.transfer
            and self.backdrop_color == other.backdrop_color
            and self.color_space == other.color_space
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.subtype,
                self.group,
                self.ctm,
                self.transfer,
                self.backdrop_color,
                self.color_space,
            )
        )

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        subtype = changes.pop("subtype", self.subtype)
        group = changes.pop("group", self.group)
        ctm = changes.pop("ctm", self.ctm)
        transfer = changes.pop("transfer", self.transfer)
        backdrop_color = changes.pop("backdrop_color", self.backdrop_color)
        color_space = changes.pop("color_space", self.color_space)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(subtype, group, ctm, transfer, backdrop_color, color_space)


def resolve(value: object, resolver: PdfValueResolver) -> object:
    seen: set[tuple[int, int]] = set()
    while isinstance(value, PdfReference):
        key = (value.object_number, value.generation_number)
        if key in seen:
            raise ValueError("cyclic soft-mask reference")
        seen.add(key)
        value = resolver.resolve(value)
    return value


def array(value: object, resolver: PdfValueResolver) -> object:
    value = resolve(value, resolver)
    if isinstance(value, (list, tuple)):
        return tuple(resolve(item, resolver) for item in value)
    return value


# Arrays are normalised to tuples here, which the PDF object model spells as
# lists, so the dictionary being assembled is a staging structure rather than
# a PdfDict. It becomes one only where it is handed back to a stream.
FunctionDictionary: TypeAlias = dict[PdfKey, object]


def function(
    value: object, resolver: PdfValueResolver, active: set[int]
) -> FunctionDictionary | PdfStream:
    value = resolve(value, resolver)
    if not isinstance(value, (dict, PdfStream)):
        raise ValueError("soft-mask transfer must be a function or Identity")
    identity = id(value)
    if identity in active:
        raise ValueError("cyclic soft-mask transfer function")
    active.add(identity)
    try:
        dictionary: FunctionDictionary = dict(
            value.dictionary if isinstance(value, PdfStream) else cast(FunctionDictionary, value)
        )
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
            dictionary["Functions"] = tuple(function(child, resolver, active) for child in children)
        elif function_type not in {0, 4}:
            raise ValueError("unsupported soft-mask transfer function")
        if function_type in {0, 4} and not isinstance(value, PdfStream):
            raise ValueError("soft-mask sampled and calculator functions require streams")
        return (
            value.replace(dictionary=cast(PdfDict, dictionary))
            if isinstance(value, PdfStream)
            else dictionary
        )
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
            initial_color_components(color_space)
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


__all__ = ("SoftMask", "parse_soft_mask")
