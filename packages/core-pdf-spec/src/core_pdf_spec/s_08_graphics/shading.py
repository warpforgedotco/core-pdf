# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, NoReturn, Self, cast

from core_pdf_spec.s_07_syntax_primitives.coercion import require_pdf_number_array
from core_pdf_spec.s_08_graphics.pdf_function import (
    PdfFunctionEvaluator,
    compile_pdf_function,
)

internal_frozen_setattr = object.__setattr__


class ShadingSpec:
    __slots__ = (
        "shading_type",
        "coords",
        "domain",
        "extend_start",
        "extend_end",
        "color_space",
        "bbox",
        "evaluator",
    )

    shading_type: int
    coords: tuple[float, ...]
    domain: tuple[float, float]
    extend_start: bool
    extend_end: bool
    color_space: object
    bbox: tuple[float, float, float, float] | None
    evaluator: PdfFunctionEvaluator

    __fields__: ClassVar[tuple[str, ...]] = (
        "shading_type",
        "coords",
        "domain",
        "extend_start",
        "extend_end",
        "color_space",
        "bbox",
        "evaluator",
    )
    __match_args__ = (
        "shading_type",
        "coords",
        "domain",
        "extend_start",
        "extend_end",
        "color_space",
        "bbox",
        "evaluator",
    )

    def __init__(
        self,
        shading_type: int,
        coords: tuple[float, ...],
        domain: tuple[float, float],
        extend_start: bool,
        extend_end: bool,
        color_space: object,
        bbox: tuple[float, float, float, float] | None,
        evaluator: PdfFunctionEvaluator,
    ) -> None:
        internal_frozen_setattr(self, "shading_type", shading_type)
        internal_frozen_setattr(self, "coords", coords)
        internal_frozen_setattr(self, "domain", domain)
        internal_frozen_setattr(self, "extend_start", extend_start)
        internal_frozen_setattr(self, "extend_end", extend_end)
        internal_frozen_setattr(self, "color_space", color_space)
        internal_frozen_setattr(self, "bbox", bbox)
        internal_frozen_setattr(self, "evaluator", evaluator)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"shading_type={self.shading_type!r}, "
            f"coords={self.coords!r}, "
            f"domain={self.domain!r}, "
            f"extend_start={self.extend_start!r}, "
            f"extend_end={self.extend_end!r}, "
            f"color_space={self.color_space!r}, "
            f"bbox={self.bbox!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.shading_type == other.shading_type
            and self.coords == other.coords
            and self.domain == other.domain
            and self.extend_start == other.extend_start
            and self.extend_end == other.extend_end
            and self.color_space == other.color_space
            and self.bbox == other.bbox
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.shading_type,
                self.coords,
                self.domain,
                self.extend_start,
                self.extend_end,
                self.color_space,
                self.bbox,
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
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        shading_type = changes.pop("shading_type", self.shading_type)
        coords = changes.pop("coords", self.coords)
        domain = changes.pop("domain", self.domain)
        extend_start = changes.pop("extend_start", self.extend_start)
        extend_end = changes.pop("extend_end", self.extend_end)
        color_space = changes.pop("color_space", self.color_space)
        bbox = changes.pop("bbox", self.bbox)
        evaluator = changes.pop("evaluator", self.evaluator)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            shading_type,
            coords,
            domain,
            extend_start,
            extend_end,
            color_space,
            bbox,
            evaluator,
        )


def parse_shading(
    dictionary: object,
    *,
    compile_function: Callable[[object], PdfFunctionEvaluator] = compile_pdf_function,
) -> ShadingSpec:
    if not isinstance(dictionary, dict):
        raise ValueError("invalid shading dictionary")
    shading_type = dictionary.get("ShadingType")
    if type(shading_type) is not int or shading_type not in {2, 3}:
        raise ValueError("unsupported shading type")
    coords = require_pdf_number_array(dictionary.get("Coords"), "invalid shading coordinates")
    if len(coords) != (4 if shading_type == 2 else 6):
        raise ValueError("invalid shading coordinates")
    domain = require_pdf_number_array(
        dictionary.get("Domain", (0.0, 1.0)), "invalid shading domain"
    )
    if len(domain) != 2:
        raise ValueError("invalid shading domain")
    extend = dictionary.get("Extend", (False, False))
    if (
        not isinstance(extend, (list, tuple))
        or len(extend) != 2
        or any(type(value) is not bool for value in extend)
    ):
        raise ValueError("invalid shading extension")
    color_space = dictionary.get("ColorSpace")
    if color_space is None:
        raise ValueError("missing shading color space")
    bbox_obj = dictionary.get("BBox")
    bbox_values = (
        () if bbox_obj is None else require_pdf_number_array(bbox_obj, "invalid shading bbox")
    )
    if dictionary.get("BBox") is not None and len(bbox_values) != 4:
        raise ValueError("invalid shading bbox")
    bbox = (bbox_values[0], bbox_values[1], bbox_values[2], bbox_values[3]) if bbox_values else None
    return ShadingSpec(
        shading_type,
        coords,
        (domain[0], domain[1]),
        cast(bool, extend[0]),
        cast(bool, extend[1]),
        color_space,
        bbox,
        compile_function(dictionary.get("Function")),
    )


__all__ = (
    "ShadingSpec",
    "parse_shading",
)
