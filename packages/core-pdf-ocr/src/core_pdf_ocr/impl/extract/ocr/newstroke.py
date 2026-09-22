# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from typing import Any, ClassVar, NoReturn, Self, cast

import numpy

from core_pdf.impl.capture.records import CapturedDrawing, CapturedPath
from core_pdf.impl.model.runs import TextRun
from core_pdf_ocr._vendor.newstroke_data import NEWSTROKE_ASCII, NEWSTROKE_ASCII_ALTERNATES

internal_frozen_setattr = object.__setattr__


FIT_ERROR = 0.08
FIXED_ERROR = 0.10
CURSOR_ERROR = 0.75
RAW_SPACE_WIDTH = 16.0
MIN_FONT_SIZE = 1.0
MAX_FONT_SIZE = 40.0
MIN_CANDIDATE_SEGMENTS = 10_000
MIN_MATCHED_SEGMENTS = 10_000
MIN_MATCHED_COVERAGE = 0.70
MIN_CHARACTERS = 1_000
MIN_SEQUENCES = 100


class NewstrokeDecode:
    __slots__ = (
        "runs",
        "candidate_segments",
        "matched_segments",
        "glyphs",
        "characters",
        "sequences",
        "maximum_error",
    )

    runs: tuple[TextRun, ...]
    candidate_segments: int
    matched_segments: int
    glyphs: int
    characters: int
    sequences: int
    maximum_error: float

    __fields__: ClassVar[tuple[str, ...]] = (
        "runs",
        "candidate_segments",
        "matched_segments",
        "glyphs",
        "characters",
        "sequences",
        "maximum_error",
    )
    __match_args__ = (
        "runs",
        "candidate_segments",
        "matched_segments",
        "glyphs",
        "characters",
        "sequences",
        "maximum_error",
    )

    def __init__(
        self,
        runs: tuple[TextRun, ...] = (),
        candidate_segments: int = 0,
        matched_segments: int = 0,
        glyphs: int = 0,
        characters: int = 0,
        sequences: int = 0,
        maximum_error: float = 0.0,
    ) -> None:
        internal_frozen_setattr(self, "runs", runs)
        internal_frozen_setattr(self, "candidate_segments", candidate_segments)
        internal_frozen_setattr(self, "matched_segments", matched_segments)
        internal_frozen_setattr(self, "glyphs", glyphs)
        internal_frozen_setattr(self, "characters", characters)
        internal_frozen_setattr(self, "sequences", sequences)
        internal_frozen_setattr(self, "maximum_error", maximum_error)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"runs={self.runs!r}, "
            f"candidate_segments={self.candidate_segments!r}, "
            f"matched_segments={self.matched_segments!r}, "
            f"glyphs={self.glyphs!r}, "
            f"characters={self.characters!r}, "
            f"sequences={self.sequences!r}, "
            f"maximum_error={self.maximum_error!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.runs == other.runs
            and self.candidate_segments == other.candidate_segments
            and self.matched_segments == other.matched_segments
            and self.glyphs == other.glyphs
            and self.characters == other.characters
            and self.sequences == other.sequences
            and self.maximum_error == other.maximum_error
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.runs,
                self.candidate_segments,
                self.matched_segments,
                self.glyphs,
                self.characters,
                self.sequences,
                self.maximum_error,
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
        runs = changes.pop("runs", self.runs)
        candidate_segments = changes.pop("candidate_segments", self.candidate_segments)
        matched_segments = changes.pop("matched_segments", self.matched_segments)
        glyphs = changes.pop("glyphs", self.glyphs)
        characters = changes.pop("characters", self.characters)
        sequences = changes.pop("sequences", self.sequences)
        maximum_error = changes.pop("maximum_error", self.maximum_error)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            runs,
            candidate_segments,
            matched_segments,
            glyphs,
            characters,
            sequences,
            maximum_error,
        )

    @property
    def matched_coverage(self) -> float:
        return self.matched_segments / max(1, self.candidate_segments)

    @property
    def trusted(self) -> bool:
        return (
            self.candidate_segments >= MIN_CANDIDATE_SEGMENTS
            and self.matched_segments >= MIN_MATCHED_SEGMENTS
            and self.matched_coverage >= MIN_MATCHED_COVERAGE
            and self.characters >= MIN_CHARACTERS
            and self.sequences >= MIN_SEQUENCES
            and self.maximum_error <= FIXED_ERROR
        )


class internal_Template:
    __slots__ = (
        "char",
        "width",
        "segments",
        "continuity",
        "solver",
        "points",
        "centroid_x",
        "centroid_y",
    )

    char: str
    width: float
    segments: numpy.ndarray[Any, numpy.dtype[numpy.float64]]
    continuity: tuple[bool, ...]
    solver: numpy.ndarray[Any, numpy.dtype[numpy.float64]]
    points: numpy.ndarray[Any, numpy.dtype[numpy.float64]]
    centroid_x: float
    centroid_y: float

    __fields__: ClassVar[tuple[str, ...]] = (
        "char",
        "width",
        "segments",
        "continuity",
        "solver",
        "points",
        "centroid_x",
        "centroid_y",
    )
    __match_args__ = (
        "char",
        "width",
        "segments",
        "continuity",
        "solver",
        "points",
        "centroid_x",
        "centroid_y",
    )

    def __init__(
        self,
        char: str,
        width: float,
        segments: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
        continuity: tuple[bool, ...],
        solver: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
        points: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
        centroid_x: float,
        centroid_y: float,
    ) -> None:
        internal_frozen_setattr(self, "char", char)
        internal_frozen_setattr(self, "width", width)
        internal_frozen_setattr(self, "segments", segments)
        internal_frozen_setattr(self, "continuity", continuity)
        internal_frozen_setattr(self, "solver", solver)
        internal_frozen_setattr(self, "points", points)
        internal_frozen_setattr(self, "centroid_x", centroid_x)
        internal_frozen_setattr(self, "centroid_y", centroid_y)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"char={self.char!r}, "
            f"width={self.width!r}, "
            f"segments={self.segments!r}, "
            f"continuity={self.continuity!r}, "
            f"solver={self.solver!r}, "
            f"points={self.points!r}, "
            f"centroid_x={self.centroid_x!r}, "
            f"centroid_y={self.centroid_y!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.char == other.char
            and self.width == other.width
            and self.segments == other.segments
            and self.continuity == other.continuity
            and self.solver == other.solver
            and self.points == other.points
            and self.centroid_x == other.centroid_x
            and self.centroid_y == other.centroid_y
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.char,
                self.width,
                self.segments,
                self.continuity,
                self.solver,
                self.points,
                self.centroid_x,
                self.centroid_y,
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
        char = changes.pop("char", self.char)
        width = changes.pop("width", self.width)
        segments = changes.pop("segments", self.segments)
        continuity = changes.pop("continuity", self.continuity)
        solver = changes.pop("solver", self.solver)
        points = changes.pop("points", self.points)
        centroid_x = changes.pop("centroid_x", self.centroid_x)
        centroid_y = changes.pop("centroid_y", self.centroid_y)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            char,
            width,
            segments,
            continuity,
            solver,
            points,
            centroid_x,
            centroid_y,
        )


class internal_TemplateSet:
    __slots__ = ("all", "robust", "by_first_delta")

    all: tuple[internal_Template, ...]
    robust: tuple[internal_Template, ...]
    by_first_delta: dict[tuple[int, int], tuple[internal_Template, ...]]

    __fields__: ClassVar[tuple[str, ...]] = ("all", "robust", "by_first_delta")
    __match_args__ = ("all", "robust", "by_first_delta")

    def __init__(
        self,
        all: tuple[internal_Template, ...],
        robust: tuple[internal_Template, ...],
        by_first_delta: dict[tuple[int, int], tuple[internal_Template, ...]],
    ) -> None:
        internal_frozen_setattr(self, "all", all)
        internal_frozen_setattr(self, "robust", robust)
        internal_frozen_setattr(self, "by_first_delta", by_first_delta)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"all={self.all!r}, "
            f"robust={self.robust!r}, "
            f"by_first_delta={self.by_first_delta!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.all == other.all
            and self.robust == other.robust
            and self.by_first_delta == other.by_first_delta
        )

    def __hash__(self) -> int:
        return hash((self.all, self.robust, self.by_first_delta))

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
        all = changes.pop("all", self.all)
        robust = changes.pop("robust", self.robust)
        by_first_delta = changes.pop("by_first_delta", self.by_first_delta)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(all, robust, by_first_delta)


class internal_Segment:
    __slots__ = ("x0", "y0", "x1", "y1", "style", "line_width")

    x0: float
    y0: float
    x1: float
    y1: float
    style: int
    line_width: float

    __fields__: ClassVar[tuple[str, ...]] = ("x0", "y0", "x1", "y1", "style", "line_width")
    __match_args__ = ("x0", "y0", "x1", "y1", "style", "line_width")

    def __init__(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        style: int,
        line_width: float,
    ) -> None:
        internal_frozen_setattr(self, "x0", x0)
        internal_frozen_setattr(self, "y0", y0)
        internal_frozen_setattr(self, "x1", x1)
        internal_frozen_setattr(self, "y1", y1)
        internal_frozen_setattr(self, "style", style)
        internal_frozen_setattr(self, "line_width", line_width)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"x0={self.x0!r}, "
            f"y0={self.y0!r}, "
            f"x1={self.x1!r}, "
            f"y1={self.y1!r}, "
            f"style={self.style!r}, "
            f"line_width={self.line_width!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.x0 == other.x0
            and self.y0 == other.y0
            and self.x1 == other.x1
            and self.y1 == other.y1
            and self.style == other.style
            and self.line_width == other.line_width
        )

    def __hash__(self) -> int:
        return hash((self.x0, self.y0, self.x1, self.y1, self.style, self.line_width))

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
        x0 = changes.pop("x0", self.x0)
        y0 = changes.pop("y0", self.y0)
        x1 = changes.pop("x1", self.x1)
        y1 = changes.pop("y1", self.y1)
        style = changes.pop("style", self.style)
        line_width = changes.pop("line_width", self.line_width)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(x0, y0, x1, y1, style, line_width)


class internal_Transform:
    __slots__ = ("matrix", "inverse", "scale", "x_scale", "y_scale")

    matrix: numpy.ndarray[Any, numpy.dtype[numpy.float64]]
    inverse: numpy.ndarray[Any, numpy.dtype[numpy.float64]]
    scale: float
    x_scale: float
    y_scale: float

    __fields__: ClassVar[tuple[str, ...]] = ("matrix", "inverse", "scale", "x_scale", "y_scale")
    __match_args__ = ("matrix", "inverse", "scale", "x_scale", "y_scale")

    def __init__(
        self,
        matrix: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
        inverse: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
        scale: float,
        x_scale: float,
        y_scale: float,
    ) -> None:
        internal_frozen_setattr(self, "matrix", matrix)
        internal_frozen_setattr(self, "inverse", inverse)
        internal_frozen_setattr(self, "scale", scale)
        internal_frozen_setattr(self, "x_scale", x_scale)
        internal_frozen_setattr(self, "y_scale", y_scale)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"matrix={self.matrix!r}, "
            f"inverse={self.inverse!r}, "
            f"scale={self.scale!r}, "
            f"x_scale={self.x_scale!r}, "
            f"y_scale={self.y_scale!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.matrix == other.matrix
            and self.inverse == other.inverse
            and self.scale == other.scale
            and self.x_scale == other.x_scale
            and self.y_scale == other.y_scale
        )

    def __hash__(self) -> int:
        return hash((self.matrix, self.inverse, self.scale, self.x_scale, self.y_scale))

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
        matrix = changes.pop("matrix", self.matrix)
        inverse = changes.pop("inverse", self.inverse)
        scale = changes.pop("scale", self.scale)
        x_scale = changes.pop("x_scale", self.x_scale)
        y_scale = changes.pop("y_scale", self.y_scale)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(matrix, inverse, scale, x_scale, y_scale)


class internal_Match:
    __slots__ = ("char", "start", "stop", "width", "transform", "translation", "error")

    char: str
    start: int
    stop: int
    width: float
    transform: internal_Transform
    translation: numpy.ndarray[Any, numpy.dtype[numpy.float64]]
    error: float

    __fields__: ClassVar[tuple[str, ...]] = (
        "char",
        "start",
        "stop",
        "width",
        "transform",
        "translation",
        "error",
    )
    __match_args__ = ("char", "start", "stop", "width", "transform", "translation", "error")

    def __init__(
        self,
        char: str,
        start: int,
        stop: int,
        width: float,
        transform: internal_Transform,
        translation: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
        error: float,
    ) -> None:
        internal_frozen_setattr(self, "char", char)
        internal_frozen_setattr(self, "start", start)
        internal_frozen_setattr(self, "stop", stop)
        internal_frozen_setattr(self, "width", width)
        internal_frozen_setattr(self, "transform", transform)
        internal_frozen_setattr(self, "translation", translation)
        internal_frozen_setattr(self, "error", error)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"char={self.char!r}, "
            f"start={self.start!r}, "
            f"stop={self.stop!r}, "
            f"width={self.width!r}, "
            f"transform={self.transform!r}, "
            f"translation={self.translation!r}, "
            f"error={self.error!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.char == other.char
            and self.start == other.start
            and self.stop == other.stop
            and self.width == other.width
            and self.transform == other.transform
            and self.translation == other.translation
            and self.error == other.error
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.char,
                self.start,
                self.stop,
                self.width,
                self.transform,
                self.translation,
                self.error,
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
        char = changes.pop("char", self.char)
        start = changes.pop("start", self.start)
        stop = changes.pop("stop", self.stop)
        width = changes.pop("width", self.width)
        transform = changes.pop("transform", self.transform)
        translation = changes.pop("translation", self.translation)
        error = changes.pop("error", self.error)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(char, start, stop, width, transform, translation, error)


def internal_templates() -> internal_TemplateSet:
    templates: list[internal_Template] = []
    by_first_delta: dict[tuple[int, int], list[internal_Template]] = {}
    encoded_glyphs = (
        *((chr(offset + 32), encoded) for offset, encoded in enumerate(NEWSTROKE_ASCII)),
        *NEWSTROKE_ASCII_ALTERNATES.items(),
    )
    for character, encoded in encoded_glyphs:
        start_x = ord(encoded[0]) - ord("R")
        end_x = ord(encoded[1]) - ord("R")
        strokes: list[list[tuple[float, float]]] = []
        stroke: list[tuple[float, float]] = []
        for index in range(2, len(encoded), 2):
            pair = encoded[index : index + 2]
            if pair == " R":
                if stroke:
                    strokes.append(stroke)
                stroke = []
            else:
                stroke.append(
                    (
                        float(ord(pair[0]) - ord("R") - start_x),
                        float(ord(pair[1]) - ord("R") - 8),
                    )
                )
        if stroke:
            strokes.append(stroke)
        raw_segments = [
            (left, right)
            for points in strokes
            for left, right in zip(points, points[1:], strict=False)
        ]
        if not raw_segments:
            continue
        segments = numpy.asarray(raw_segments, dtype=numpy.float64)
        continuity = tuple(
            bool(numpy.array_equal(segments[index, 1], segments[index + 1, 0]))
            for index in range(len(segments) - 1)
        )
        source = segments.reshape((-1, 2))
        design = numpy.column_stack((source, numpy.ones(len(source))))
        template = internal_Template(
            character,
            float(end_x - start_x),
            segments,
            continuity,
            numpy.linalg.pinv(design),
            source,
            float(numpy.mean(source[:, 0])),
            float(numpy.mean(source[:, 1])),
        )
        templates.append(template)
        delta = segments[0, 1] - segments[0, 0]
        by_first_delta.setdefault((int(delta[0]), int(delta[1])), []).append(template)
    all_templates = tuple(templates)
    return internal_TemplateSet(
        all_templates,
        tuple(template for template in all_templates if len(template.segments) >= 5),
        {key: tuple(value) for key, value in by_first_delta.items()},
    )


def internal_drawing_style(drawing: CapturedDrawing) -> tuple[object, ...] | None:
    style = drawing.stroke_style_key()
    if style is None or style[1] <= 0.0 or style[2] <= 0.0:
        return None
    return style


def internal_segments(
    drawings: tuple[CapturedDrawing, ...],
) -> tuple[tuple[internal_Segment | None, ...], tuple[tuple[object, ...], ...], int]:
    segments: list[internal_Segment | None] = []
    style_ids: dict[tuple[object, ...], int] = {}
    styles: list[tuple[object, ...]] = []
    candidate_count = 0
    for drawing in drawings:
        path = drawing.path
        if (
            drawing.kind != "stroke"
            or type(path) is not CapturedPath
            or len(path.subpaths) != 1
            or path.subpaths[0].closed
            or len(path.subpaths[0].points) != 2
        ):
            segments.append(None)
            continue
        style = internal_drawing_style(drawing)
        if style is None:
            segments.append(None)
            continue
        (x0, y0), (x1, y1) = path.subpaths[0].points
        if abs(x1 - x0) <= 1e-9 and abs(y1 - y0) <= 1e-9:
            segments.append(None)
            continue
        style_id = style_ids.get(style)
        if style_id is None:
            style_id = len(styles)
            style_ids[style] = style_id
            styles.append(style)
        segments.append(
            internal_Segment(
                float(x0),
                float(y0),
                float(x1),
                float(y1),
                style_id,
                cast(float, style[2]),
            )
        )
        candidate_count += 1
    return tuple(segments), tuple(styles), candidate_count


def internal_continuity(segments: tuple[internal_Segment | None, ...]) -> tuple[bool, ...]:
    result: list[bool] = []
    for left, right in zip(segments, segments[1:], strict=False):
        if left is None or right is None or left.style != right.style:
            result.append(False)
            continue
        tolerance = max(0.01, min(left.line_width, right.line_width) * 0.1)
        dx = left.x1 - right.x0
        dy = left.y1 - right.y0
        result.append(dx * dx + dy * dy <= tolerance * tolerance)
    return tuple(result)


def internal_window(
    segments: tuple[internal_Segment | None, ...],
    start: int,
    size: int,
    style: int,
    point_data: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    style_data: numpy.ndarray[Any, numpy.dtype[numpy.int16]],
) -> numpy.ndarray[Any, numpy.dtype[numpy.float64]] | None:
    stop = start + size
    if stop > len(segments) or not numpy.all(style_data[start:stop] == style):
        return None
    return point_data[start:stop]


def internal_fit_match(
    segments: tuple[internal_Segment | None, ...],
    start: int,
    template: internal_Template,
    point_data: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    style_data: numpy.ndarray[Any, numpy.dtype[numpy.int16]],
) -> internal_Match | None:
    first = segments[start]
    if first is None:
        return None
    actual = internal_window(
        segments, start, len(template.segments), first.style, point_data, style_data
    )
    if actual is None:
        return None
    source = template.points
    target = actual.reshape((-1, 2))
    coefficients = template.solver @ target
    matrix = coefficients[:2]
    translation = coefficients[2]
    a = float(matrix[0, 0])
    b = float(matrix[0, 1])
    c = float(matrix[1, 0])
    d = float(matrix[1, 1])
    x_scale = math.hypot(a, b)
    y_scale = math.hypot(c, d)
    scale = max(x_scale, y_scale)
    nominal_size = scale * 21.0
    if not (MIN_FONT_SIZE <= nominal_size <= MAX_FONT_SIZE):
        return None
    min_scale = min(x_scale, y_scale)
    if min_scale <= 0.0 or scale / min_scale > 4.0:
        return None
    orthogonality = abs(a * c + b * d) / (x_scale * y_scale)
    if orthogonality > 0.25:
        return None
    predicted = source @ matrix + translation
    residual = predicted - target
    error = (
        math.sqrt(
            float(numpy.max(residual[:, 0] * residual[:, 0] + residual[:, 1] * residual[:, 1]))
        )
        / scale
    )
    if error > FIT_ERROR:
        return None
    determinant = a * d - b * c
    inverse = numpy.asarray(((d, -b), (-c, a)), dtype=numpy.float64) / determinant
    transform = internal_Transform(matrix, inverse, scale, x_scale, y_scale)
    return internal_Match(
        template.char,
        start,
        start + len(template.segments),
        template.width,
        transform,
        translation,
        error,
    )


def internal_fixed_template_match(
    segments: tuple[internal_Segment | None, ...],
    continuity: tuple[bool, ...],
    template: internal_Template,
    start: int,
    transform: internal_Transform,
    style: int,
    point_data: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    style_data: numpy.ndarray[Any, numpy.dtype[numpy.int16]],
) -> internal_Match | None:
    size = len(template.segments)
    if continuity[start : start + size - 1] != template.continuity:
        return None
    actual = internal_window(segments, start, size, style, point_data, style_data)
    if actual is None:
        return None
    source = template.points
    target = actual.reshape((-1, 2))
    target_x = float(numpy.mean(target[:, 0]))
    target_y = float(numpy.mean(target[:, 1]))
    matrix = transform.matrix
    translation = numpy.asarray(
        (
            target_x - template.centroid_x * matrix[0, 0] - template.centroid_y * matrix[1, 0],
            target_y - template.centroid_x * matrix[0, 1] - template.centroid_y * matrix[1, 1],
        ),
        dtype=numpy.float64,
    )
    predicted = source @ transform.matrix + translation
    residual = predicted - target
    error = (
        math.sqrt(
            float(numpy.max(residual[:, 0] * residual[:, 0] + residual[:, 1] * residual[:, 1]))
        )
        / transform.scale
    )
    if error > FIXED_ERROR:
        return None
    return internal_Match(
        template.char,
        start,
        start + size,
        template.width,
        transform,
        translation,
        error,
    )


def internal_fixed_match(
    segments: tuple[internal_Segment | None, ...],
    continuity: tuple[bool, ...],
    templates: internal_TemplateSet,
    start: int,
    transform: internal_Transform,
    style: int,
    point_data: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    style_data: numpy.ndarray[Any, numpy.dtype[numpy.int16]],
) -> internal_Match | None:
    first = segments[start]
    if first is None:
        return None
    dx = first.x1 - first.x0
    dy = first.y1 - first.y0
    inverse = transform.inverse
    raw_dx = dx * inverse[0, 0] + dy * inverse[1, 0]
    raw_dy = dx * inverse[0, 1] + dy * inverse[1, 1]
    rounded_dx = round(raw_dx)
    rounded_dy = round(raw_dy)
    if max(abs(raw_dx - rounded_dx), abs(raw_dy - rounded_dy)) > 0.20:
        return None
    candidates: list[internal_Match] = []
    for template in templates.by_first_delta.get((rounded_dx, rounded_dy), ()):
        candidate = internal_fixed_template_match(
            segments,
            continuity,
            template,
            start,
            transform,
            style,
            point_data,
            style_data,
        )
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return None
    candidates.sort(key=lambda candidate: (-candidate.stop, candidate.error))
    best = candidates[0]
    if (
        len(candidates) > 1
        and candidates[1].stop == best.stop
        and candidates[1].char != best.char
        and candidates[1].error - best.error < 0.01
    ):
        return None
    return best


def internal_cursor_follows(previous: internal_Match, current: internal_Match) -> bool:
    expected = previous.translation + previous.width * previous.transform.matrix[0]
    offset = (current.translation - expected) @ previous.transform.inverse
    y_offset = abs(float(offset[1]))
    x_offset = float(offset[0])
    if y_offset > CURSOR_ERROR or x_offset < -CURSOR_ERROR:
        return False
    if x_offset > RAW_SPACE_WIDTH * 2.0 + CURSOR_ERROR:
        return False
    spaces = round(x_offset / RAW_SPACE_WIDTH)
    return 0 <= spaces <= 2 and abs(x_offset - spaces * RAW_SPACE_WIDTH) <= CURSOR_ERROR


def internal_decode_forward(
    segments: tuple[internal_Segment | None, ...],
    continuity: tuple[bool, ...],
    templates: internal_TemplateSet,
    seed: internal_Match,
    point_data: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    style_data: numpy.ndarray[Any, numpy.dtype[numpy.int16]],
) -> tuple[internal_Match, ...]:
    result = [seed]
    first = segments[seed.start]
    if first is None:
        return ()
    position = seed.stop
    while position < len(segments):
        candidate = internal_fixed_match(
            segments,
            continuity,
            templates,
            position,
            seed.transform,
            first.style,
            point_data,
            style_data,
        )
        if candidate is None or not internal_cursor_follows(result[-1], candidate):
            break
        result.append(candidate)
        position = candidate.stop
    return tuple(result)


def internal_decode_around(
    segments: tuple[internal_Segment | None, ...],
    continuity: tuple[bool, ...],
    templates: internal_TemplateSet,
    seed: internal_Match,
    minimum_start: int,
    point_data: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    style_data: numpy.ndarray[Any, numpy.dtype[numpy.int16]],
) -> tuple[internal_Match, ...]:
    result = list(
        internal_decode_forward(segments, continuity, templates, seed, point_data, style_data)
    )
    first = segments[seed.start]
    if first is None:
        return ()
    position = seed.start
    prepend: list[internal_Match] = []
    while position > minimum_start:
        candidates: list[internal_Match] = []
        for template in templates.all:
            start = position - len(template.segments)
            if start < minimum_start:
                continue
            candidate = internal_fixed_template_match(
                segments,
                continuity,
                template,
                start,
                seed.transform,
                first.style,
                point_data,
                style_data,
            )
            anchor = prepend[-1] if prepend else result[0]
            if candidate is not None and internal_cursor_follows(candidate, anchor):
                candidates.append(candidate)
        if not candidates:
            break
        candidates.sort(key=lambda candidate: (candidate.start, candidate.error))
        best = candidates[0]
        if (
            len(candidates) > 1
            and candidates[1].start == best.start
            and candidates[1].char != best.char
            and candidates[1].error - best.error < 0.01
        ):
            break
        prepend.append(best)
        position = best.start
    prepend.reverse()
    return tuple(prepend) + tuple(result)


def internal_sequence_text(matches: tuple[internal_Match, ...]) -> str:
    if not matches:
        return ""
    parts = [matches[0].char]
    transform = matches[0].transform
    for previous, current in zip(matches, matches[1:], strict=False):
        expected = previous.translation + previous.width * transform.matrix[0]
        offset = (current.translation - expected) @ transform.inverse
        if float(offset[0]) > RAW_SPACE_WIDTH * 0.5:
            parts.append(" " * max(1, round(float(offset[0]) / RAW_SPACE_WIDTH)))
        parts.append(current.char)
    return "".join(parts)


def internal_sequence_run(
    matches: tuple[internal_Match, ...],
    segments: tuple[internal_Segment | None, ...],
    styles: tuple[tuple[object, ...], ...],
    order: int,
) -> TextRun:
    concrete = tuple(
        segment for segment in segments[matches[0].start : matches[-1].stop] if segment is not None
    )
    padding = concrete[0].line_width * 0.5
    min_x = min(concrete[0].x0, concrete[0].x1)
    max_x = max(concrete[0].x0, concrete[0].x1)
    min_y = min(concrete[0].y0, concrete[0].y1)
    max_y = max(concrete[0].y0, concrete[0].y1)
    for segment in concrete[1:]:
        sx0 = segment.x0
        sx1 = segment.x1
        seg_min_x = min(sx1, sx0)
        seg_max_x = max(sx0, sx1)
        if seg_min_x < min_x:
            min_x = seg_min_x
        if seg_max_x > max_x:
            max_x = seg_max_x
        sy0 = segment.y0
        sy1 = segment.y1
        seg_min_y = min(sy1, sy0)
        seg_max_y = max(sy0, sy1)
        if seg_min_y < min_y:
            min_y = seg_min_y
        if seg_max_y > max_y:
            max_y = seg_max_y
    x0 = min_x - padding
    y0 = min_y - padding
    x1 = max_x + padding
    y1 = max_y + padding
    transform = matches[0].transform
    origin = matches[0].translation
    advance = matches[-1].translation + matches[-1].width * transform.matrix[0]
    angle = (
        round(
            math.degrees(math.atan2(float(transform.matrix[0, 1]), float(transform.matrix[0, 0])))
        )
        % 360
    )
    maximum_error = max(match.error for match in matches)
    return TextRun(
        text=internal_sequence_text(matches),
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        tx=float(origin[0]),
        ty=float(origin[1]),
        font_size=transform.scale * 21.0,
        space_width=transform.x_scale * RAW_SPACE_WIDTH,
        order=order,
        stream_order=matches[0].start,
        xobject_depth=0,
        font_name="KiCad Newstroke",
        rotation_angle=angle,
        visible=True,
        seqno=matches[0].start,
        fill_color=cast(tuple[float, ...] | None, styles[concrete[0].style][0]),
        advance_bbox=(x0, y0, x1, y1),
        ink_bbox=(x0, y0, x1, y1),
        baseline=(float(origin[0]), float(origin[1]), float(advance[0]), float(advance[1])),
        provenance=(
            ("unicode_source", "newstroke-template"),
            ("newstroke_max_error", maximum_error),
        ),
        confidence=max(90.0, 100.0 - maximum_error * 100.0),
    )


def decode_newstroke_drawings(drawings: tuple[CapturedDrawing, ...]) -> NewstrokeDecode:
    segments, styles, candidate_count = internal_segments(drawings)
    if candidate_count < MIN_CANDIDATE_SEGMENTS:
        return NewstrokeDecode(candidate_segments=candidate_count)
    templates = internal_templates()
    continuity = internal_continuity(segments)
    point_data = numpy.zeros((len(segments), 2, 2), dtype=numpy.float64)
    style_data = numpy.full(len(segments), -1, dtype=numpy.int16)
    for index, segment in enumerate(segments):
        if segment is not None:
            point_data[index] = ((segment.x0, segment.y0), (segment.x1, segment.y1))
            style_data[index] = segment.style
    sequences: list[tuple[internal_Match, ...]] = []
    known_transforms: dict[int, list[internal_Transform]] = {}
    position = 0
    available_start = 0
    while position < len(segments):
        first = segments[position]
        if first is None:
            position += 1
            continue
        seeds: list[internal_Match] = []
        for transform in known_transforms.get(first.style, ()):
            seed = internal_fixed_match(
                segments,
                continuity,
                templates,
                position,
                transform,
                first.style,
                point_data,
                style_data,
            )
            if seed is not None:
                seeds.append(seed)
        if not seeds:
            for template in templates.robust:
                size = len(template.segments)
                if continuity[position : position + size - 1] != template.continuity:
                    continue
                seed = internal_fit_match(segments, position, template, point_data, style_data)
                if seed is not None:
                    seeds.append(seed)
        candidates: list[tuple[tuple[int, int, float], tuple[internal_Match, ...]]] = []
        for seed in seeds:
            decoded = internal_decode_around(
                segments,
                continuity,
                templates,
                seed,
                available_start,
                point_data,
                style_data,
            )
            span = decoded[-1].stop - decoded[0].start
            candidates.append(
                ((span, len(decoded), -sum(match.error for match in decoded)), decoded)
            )
        if not candidates:
            position += 1
            continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        decoded = candidates[0][1]
        span = decoded[-1].stop - decoded[0].start
        if len(decoded) < 2 or span < 10:
            position += 1
            continue
        sequences.append(decoded)
        transforms = known_transforms.setdefault(first.style, [])
        matrix = decoded[0].transform.matrix
        if not any(
            float(numpy.max(numpy.abs(transform.matrix - matrix)))
            <= max(0.001, float(numpy.max(numpy.abs(transform.matrix))) * 0.01)
            for transform in transforms
        ):
            transforms.append(decoded[0].transform)
        position = decoded[-1].stop
        available_start = position

    matched_segments = sum(sequence[-1].stop - sequence[0].start for sequence in sequences)
    glyphs = sum(len(sequence) for sequence in sequences)
    runs = tuple(
        internal_sequence_run(sequence, segments, styles, order)
        for order, sequence in enumerate(sequences)
    )
    characters = sum(not character.isspace() for run in runs for character in run.text)
    maximum_error = max(
        (match.error for sequence in sequences for match in sequence),
        default=0.0,
    )
    return NewstrokeDecode(
        runs=runs,
        candidate_segments=candidate_count,
        matched_segments=matched_segments,
        glyphs=glyphs,
        characters=characters,
        sequences=len(sequences),
        maximum_error=maximum_error,
    )


__all__ = ("NewstrokeDecode", "decode_newstroke_drawings")
