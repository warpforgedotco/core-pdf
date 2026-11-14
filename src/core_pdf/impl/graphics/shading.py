# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any, ClassVar

lazy import numpy

from core_pdf.impl.base.caches import IdentityCache
from core_pdf.impl.base.types import GeneratedRecord
from core_pdf.impl.base.values import lenient_int
from core_pdf.impl.graphics.color import color_operands_to_srgb
from core_pdf.impl.graphics.color_spec import (
    DEVICE_KINDS,
    parse_color_space,
    raw_color_space_paints,
)
from core_pdf.impl.graphics.functions import compile_pdf_function, number_array
from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.exceptions import PdfError
from core_pdf.impl.spec.s_07_filters.errors import FilterError
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering
from core_pdf.impl.spec.s_08_graphics.color_spec import ColorSpace
from core_pdf.impl.spec.s_08_graphics.pdf_function import PdfFunctionEvaluator, batch_evaluator
from core_pdf.impl.spec.s_08_graphics.shading import (
    SHADING_STREAM_KEY,
    parse_function_shading,
    parse_shading,
)
lazy from core_pdf.impl.graphics.image_samples import convert_components
lazy from core_pdf.impl.spec.s_08_graphics.shading_mesh import decode_mesh

# The colour components each device space has.
DEVICE_COMPONENTS = {"DeviceGray": 1, "DeviceRGB": 3, "DeviceCMYK": 4}


class FunctionGeometry(GeneratedRecord):
    """A type 1 shading's Domain, normalized to (xmin, xmax, ymin, ymax), and its Matrix from
    Domain to the shading's target space."""

    domain: tuple[float, float, float, float]
    matrix: tuple[float, float, float, float, float, float]


class MeshGeometry(GeneratedRecord):
    """A type 4 to 7 shading's triangles (T, 3, 2 + k), or its patches' control points
    (P, 16, 2) and corner values (P, 4, k), in its target space; each value's range, and
    whether the single value is the parametric t of a Function."""

    triangles: Any
    patches: Any
    corners: Any
    ranges: tuple[tuple[float, float], ...]
    parametric: bool


class PreparedShading(GeneratedRecord):
    shading_type: int
    coords: tuple[float, ...]
    domain: tuple[float, float]
    extend_start: bool
    extend_end: bool
    color_model: str
    bbox: tuple[float, float, float, float] | None
    # The colour of each function input, or for types 1 and 4 to 7 of each tuple of values.
    evaluator: Callable[..., tuple[float, ...]]
    color_rendering: ColorRendering = DEFAULT_COLOR_RENDERING
    geometry: FunctionGeometry | MeshGeometry | None = None
    # Background's colour components, converted as the evaluator converts its colours.
    background: tuple[float, ...] | None = None

    __repr_fields__: ClassVar[tuple[str, ...] | None] = (
        "shading_type",
        "coords",
        "domain",
        "extend_start",
        "extend_end",
        "color_model",
        "bbox",
        "color_rendering",
        "background",
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
            and self.color_model == other.color_model
            and self.bbox == other.bbox
            and self.color_rendering == other.color_rendering
            and self.geometry is other.geometry
            and self.background == other.background
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.shading_type,
                self.coords,
                self.domain,
                self.extend_start,
                self.extend_end,
                self.color_model,
                self.bbox,
                self.color_rendering,
                id(self.geometry),
                self.background,
            )
        )


type ShadingEvaluatorCache = IdentityCache[
    tuple[PdfFunctionEvaluator, Callable[..., tuple[float, ...]], str]
]


def prepare_shading(
    dictionary: object,
    *,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
    evaluators: ShadingEvaluatorCache | None = None,
) -> PreparedShading | None:
    if not isinstance(dictionary, dict):
        return None
    if not raw_color_space_paints(dictionary.get("ColorSpace")):
        return None
    shading_type = lenient_int(dictionary.get("ShadingType"), 0)
    if shading_type == 1:
        return prepare_function_shading(dictionary, rendering)
    if shading_type in {4, 5, 6, 7}:
        return prepare_mesh_shading(dictionary, shading_type, rendering)
    if shading_type not in {2, 3}:
        return None
    coords = number_array(dictionary.get("Coords"))
    if (shading_type == 2 and len(coords) < 4) or (shading_type == 3 and len(coords) < 6):
        return None
    domain_values = number_array(dictionary.get("Domain"))
    domain = (domain_values[0], domain_values[1]) if len(domain_values) >= 2 else (0.0, 1.0)
    extend = dictionary.get("Extend")
    extend_start = isinstance(extend, (list, tuple)) and len(extend) > 0 and extend[0] is True
    extend_end = isinstance(extend, (list, tuple)) and len(extend) > 1 and extend[1] is True
    bbox_values = number_array(dictionary.get("BBox"))
    bbox = (
        (bbox_values[0], bbox_values[1], bbox_values[2], bbox_values[3])
        if len(bbox_values) >= 4
        else None
    )
    normalized = dict(dictionary)
    normalized.update(
        ShadingType=shading_type,
        Coords=coords[: 4 if shading_type == 2 else 6],
        Domain=domain,
        Extend=(extend_start, extend_end),
        ColorSpace=dictionary.get("ColorSpace") or "DeviceRGB",
    )
    if bbox is None:
        normalized.pop("BBox", None)
    else:
        normalized["BBox"] = bbox
    function = normalized.get("Function")
    color_space = normalized["ColorSpace"]
    key = (id(function), id(color_space), rendering)
    cached = evaluators.get_key(function, key) if evaluators is not None else None
    if cached is not None:
        compiled, evaluator, color_model = cached
        try:
            spec = parse_shading(normalized, compile_function=lambda _function: compiled)
        except ValueError:
            return None
    else:
        try:
            spec = parse_shading(normalized, compile_function=compile_pdf_function)
            space = parse_color_space(spec.color_space)
        except ValueError:
            return None
        evaluator, color_model = shading_color_evaluator(spec.evaluator, space, rendering)
        if evaluators is not None:
            evaluators.put_key(
                function, key, (spec.evaluator, evaluator, color_model), (color_space,)
            )
    return PreparedShading(
        spec.shading_type,
        coords,
        spec.domain,
        spec.extend_start,
        spec.extend_end,
        color_model,
        spec.bbox,
        evaluator,
        rendering,
        background=background_components(dictionary.get("Background"), evaluator, color_model),
    )


@cites(
    "ISO 32000-2",
    "8.7.4.3",
    introduced="1.3",
    rule="Background holds one colour's components in the shading's colour space",
)
def background_components(
    value: object, evaluator: Callable[..., tuple[float, ...]], color_model: str
) -> tuple[float, ...] | None:
    """Background's components, converted the way the evaluator converts a colour; None when
    there is none, or it does not hold one colour of the shading's space."""
    values = number_array(value)
    if not values:
        return None
    if isinstance(evaluator, SpaceEvaluator):
        if len(values) != len(evaluator.space.component_ranges):
            return None
        try:
            converted = color_operands_to_srgb(
                evaluator.space, values, rendering=evaluator.rendering
            )
        except ValueError, ArithmeticError, PdfError:
            return None
        return converted if converted is not None else values
    return values if len(values) == DEVICE_COMPONENTS.get(color_model, len(values)) else None


def prepare_function_shading(
    dictionary: dict[Any, Any], rendering: ColorRendering
) -> PreparedShading | None:
    normalized = dict(dictionary)
    normalized.update(ShadingType=1, ColorSpace=dictionary.get("ColorSpace") or "DeviceRGB")
    try:
        spec = parse_function_shading(normalized, compile_function=compile_pdf_function)
        space = parse_color_space(spec.color_space)
    except ValueError, ArithmeticError, PdfError:
        return None
    evaluator, color_model = shading_color_evaluator(spec.evaluator, space, rendering)
    x0, x1, y0, y1 = spec.domain
    domain = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
    return PreparedShading(
        1,
        (),
        (0.0, 1.0),
        False,
        False,
        color_model,
        spec.bbox,
        evaluator,
        rendering,
        geometry=FunctionGeometry(domain, spec.matrix),
        background=background_components(dictionary.get("Background"), evaluator, color_model),
    )


def components_of(*components: float) -> tuple[float, ...]:
    """A mesh vertex's colour without a Function: its components."""
    return components


@cites(
    "ISO 32000-2",
    "8.7.4.4",
    introduced="1.3",
    rule="an Indexed shading's colours are converted to the base space before interpolation",
)
def indexed_values(values: Any, space: ColorSpace) -> Any:
    """Indexed colour values (..., 1) as the base space's components (..., n)."""
    base = space.base
    assert base is not None
    count = len(base.component_ranges)
    flat = numpy.zeros(((space.hival + 1) * count,), dtype=numpy.float64)
    lookup = numpy.frombuffer(space.lookup or b"", dtype=numpy.uint8)[: len(flat)]
    flat[: len(lookup)] = lookup
    table = flat.reshape(space.hival + 1, count)
    index = numpy.clip(numpy.rint(values[..., 0]), 0, space.hival).astype(numpy.intp)
    low, high = numpy.asarray(base.component_ranges, dtype=numpy.float64).T
    return low + table[index] / 255.0 * (high - low)


def prepare_mesh_shading(
    dictionary: dict[Any, Any], shading_type: int, rendering: ColorRendering
) -> PreparedShading | None:
    stream = dictionary.get(SHADING_STREAM_KEY)
    if not isinstance(stream, PdfStream):
        return None
    function_object = dictionary.get("Function")
    try:
        space = parse_color_space(dictionary.get("ColorSpace") or "DeviceRGB")
        function = compile_pdf_function(function_object) if function_object is not None else None
        value_count = 1 if function is not None else len(space.component_ranges)
        mesh = decode_mesh(shading_type, dictionary, stream.data, value_count)
    except ValueError, ArithmeticError, FilterError, PdfError:
        return None
    triangles, patches, corners, ranges = mesh.triangles, mesh.patches, mesh.corners, mesh.ranges
    if function is None and space.kind == "Indexed" and space.base is not None:
        if triangles is not None:
            triangles = numpy.concatenate(
                (triangles[..., :2], indexed_values(triangles[..., 2:], space)), axis=-1
            )
        if corners is not None:
            corners = indexed_values(corners, space)
        space = space.base
        ranges = tuple((float(low), float(high)) for low, high in space.component_ranges)
    evaluator, color_model = shading_color_evaluator(
        function if function is not None else components_of, space, rendering
    )
    bbox_values = number_array(dictionary.get("BBox"))
    bbox = (
        (bbox_values[0], bbox_values[1], bbox_values[2], bbox_values[3])
        if len(bbox_values) >= 4
        else None
    )
    geometry = MeshGeometry(
        None if triangles is None else numpy.ascontiguousarray(triangles, dtype=numpy.float64),
        None if patches is None else numpy.ascontiguousarray(patches, dtype=numpy.float64),
        None if corners is None else numpy.ascontiguousarray(corners, dtype=numpy.float64),
        ranges,
        function is not None,
    )
    return PreparedShading(
        shading_type,
        (),
        (0.0, 1.0),
        False,
        False,
        color_model,
        bbox,
        evaluator,
        rendering,
        geometry=geometry,
        background=background_components(dictionary.get("Background"), evaluator, color_model),
    )


class SpaceEvaluator:
    """A shading function whose colours are converted from a non-device colour space to sRGB:
    one at a time when called, or a batch of inputs at once with many."""

    __slots__ = ("function", "space", "rendering", "convert")

    def __init__(
        self, function: PdfFunctionEvaluator, space: ColorSpace, rendering: ColorRendering
    ) -> None:
        self.function = function
        self.space = space
        self.rendering = rendering
        self.convert = lru_cache(maxsize=8192)(self.convert_one)

    def __call__(self, *values: float) -> tuple[float, ...]:
        return self.convert(*values)

    def convert_one(self, *values: float) -> tuple[float, ...]:
        components = self.function(*values)
        return (
            color_operands_to_srgb(self.space, components, rendering=self.rendering) or components
        )

    def many(
        self, values: Sequence[float] | Sequence[tuple[float, ...]]
    ) -> list[tuple[float, ...]] | None:
        """What calling it on each value, or each tuple of values, gives, converting every
        colour in one pass; None when a value's colour fails, which calling one at a time
        then reports."""
        many = batch_evaluator(self.function) if values and type(values[0]) is not tuple else None
        try:
            if many is not None:
                rows = many(numpy.asarray(values, dtype=numpy.float64))
                if not numpy.isfinite(rows).all():
                    raise ValueError("a colour that is not finite is converted one at a time")
            else:
                rows = numpy.asarray(
                    [
                        self.function(*value) if type(value) is tuple else self.function(value)
                        for value in values
                    ],
                    dtype=numpy.float64,
                )
            converted = convert_components(rows, self.space, rendering=self.rendering)
        except TypeError, ValueError, ArithmeticError:
            return None
        if converted.ndim != 2 or len(converted) != len(rows):
            return None
        if converted.shape[1] == 1:
            converted = numpy.repeat(converted, 3, axis=1)
        scaled = (converted[:, :3].astype(numpy.float64) / 255).tolist()
        return [(red, green, blue) for red, green, blue in scaled]


def shading_color_evaluator(
    function: PdfFunctionEvaluator, space: ColorSpace, rendering: ColorRendering
) -> tuple[Callable[..., tuple[float, ...]], str]:
    if space.kind in DEVICE_KINDS:
        return function, space.kind
    return SpaceEvaluator(function, space, rendering), "DeviceRGB"


__all__ = (
    "FunctionGeometry",
    "MeshGeometry",
    "PreparedShading",
    "ShadingEvaluatorCache",
    "SpaceEvaluator",
    "prepare_shading",
)
