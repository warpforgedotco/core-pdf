# SPDX-License-Identifier: AGPL-3.0-only

"""What TextState's methods share: the records of marked content, text layouts,
stroke rows, cached images and saved graphics state, and helpers that need no state."""

from __future__ import annotations

from array import array
from collections.abc import Callable
from math import copysign, hypot
from typing import TYPE_CHECKING, Any, TypeAlias

lazy import numpy

from core_pdf.impl.base.glyphs import GlyphStyle
from core_pdf.impl.base.runs import TextRun
from core_pdf.impl.base.types import (
    RecordType,
    Rectangle,
    ReplaceFields,
    ReprFields,
)
from core_pdf.impl.capture.glyphs import (
    GlyphPaint,
)
from core_pdf.impl.capture.records import (
    EMPTY_LINES,
    PATH_PAINT_DEFAULTS,
    CapturedDrawing,
    CapturedLines,
    CapturedPath,
)
from core_pdf.impl.fonts.decoder import FontDecoder
from core_pdf.impl.spec.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_content.model import GraphicsState, PdfPath
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfValueResolver
from core_pdf.impl.spec.s_08_graphics.color_rendering import (
    DEFAULT_COLOR_RENDERING,
    ColorRendering,
)
from core_pdf.impl.spec.s_08_graphics.matrix import Matrix
lazy from core_pdf.impl.graphics.images import decode_soft_mask
lazy from core_pdf.impl.spec.s_08_graphics import image_spec
lazy from core_pdf.impl.spec.s_08_graphics.image_spec import ImageSource
lazy from core_pdf_cythonized import SlotLayout, flatten_path_commands

if TYPE_CHECKING:
    pass


if TYPE_CHECKING:
    pass


CaptureMarks: TypeAlias = tuple[int, int, int, int, int, int]


NO_MARKS: CaptureMarks = (0, 0, 0, 0, 0, 0)


class MarkedContentEntry(ReprFields, ReplaceFields, metaclass=RecordType, frozen=False):
    layer: str | None = None
    actual_text: str | None = None
    mcid: int | None = None
    run: TextRun | None = None
    font_decoder: object | None = None
    effective_font_height: float = 0.0

    def add_run(
        self,
        run: TextRun,
        *,
        font_decoder: object | None = None,
        effective_font_height: float = 0.0,
    ) -> None:
        captured = self.run
        if captured is None:
            self.run = run
            self.font_decoder = font_decoder
            self.effective_font_height = effective_font_height
            return
        captured.absorb_extent(run)


MATRIX_TOLERANCE = 0.1


def same_floats(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    """Whether the floats are equal, a zero's sign included."""
    return all(
        a == b and copysign(1.0, a) == copysign(1.0, b) for a, b in zip(left, right, strict=True)
    )


def detect_rotation_from_linear(A: float, B: float, C: float, D: float) -> int:
    scale_x = hypot(A, B)
    scale_y = hypot(C, D)
    if scale_x <= 0 or scale_y <= 0:
        return 0
    na, nb, nc, nd = A / scale_x, B / scale_x, C / scale_y, D / scale_y
    tolerance = MATRIX_TOLERANCE
    if (
        abs(na) < tolerance
        and abs(nb - 1.0) < tolerance
        and abs(nc + 1.0) < tolerance
        and abs(nd) < tolerance
    ):
        return 90
    if (
        abs(na + 1.0) < tolerance
        and abs(nb) < tolerance
        and abs(nc) < tolerance
        and abs(nd + 1.0) < tolerance
    ):
        return 180
    if (
        abs(na) < tolerance
        and abs(nb + 1.0) < tolerance
        and abs(nc - 1.0) < tolerance
        and abs(nd) < tolerance
    ):
        return 270
    return 0


class TextLayout:
    __slots__ = (
        "decoder",
        "given_paint",
        "identity_text_matrix",
        "glyph_paint",
        "fill_color",
        "paints_text",
        "pushes_clip_scope",
        "font_size",
        "rise",
        "font_scale",
        "ascent",
        "descent",
        "advance_scale",
        "rotation",
        "scale_factor",
        "effective_font_size",
        "effective_font_height",
        "actual_text_span",
        "style",
        "space_width",
        "run_provenance",
        "width_source",
    )

    decoder: FontDecoder
    given_paint: GlyphPaint | None
    identity_text_matrix: bool
    glyph_paint: GlyphPaint | None
    fill_color: tuple[float, ...] | None
    paints_text: bool
    pushes_clip_scope: bool
    font_size: float
    rise: float
    font_scale: float
    ascent: float
    descent: float
    advance_scale: float
    rotation: int
    scale_factor: float
    effective_font_size: float
    effective_font_height: float
    actual_text_span: MarkedContentEntry | None
    style: GlyphStyle | None
    # What capture_glyphs reads this decoder's widths from, looked up once per layout.
    width_source: Callable[[int], float] | tuple[dict[Any, float], float]
    space_width: float
    run_provenance: tuple[tuple[str, object], ...]


def flatten_path(
    source: PdfPath,
    matrix: Matrix | None,
    lines: StrokeLineRows | None = None,
    line_width: float = 0.0,
) -> CapturedPath:
    xs, ys, spans, bbox, has_segments = flatten_path_commands(
        source.ops,
        source.coords,
        matrix,
        hypot,
        None if lines is None else lines.rows,
        line_width,
    )
    return CapturedPath.deferred_flattened(xs, ys, spans, bbox, has_segments)


class StrokeLineRows:
    __slots__ = ("rows",)

    def __init__(self) -> None:
        self.rows: array[float] = array("d")

    @property
    def count(self) -> int:
        return len(self.rows) // 5

    def since(self, mark: int) -> CapturedLines:
        if mark >= self.count:
            return EMPTY_LINES
        table = numpy.frombuffer(self.rows, dtype=numpy.float64)
        return CapturedLines.from_array(table[mark * 5 :].reshape(-1, 5).copy())


def image_source_from_stream(
    stream: PdfStream,
    resolver: PdfValueResolver,
    *,
    color_rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> ImageSource:
    return image_spec.image_source_from_stream(
        stream,
        resolver,
        semantic_context=getattr(resolver, "semantic_context", None),
        color_rendering=color_rendering,
    )


class CachedImageSource:
    """An image XObject's source, and its soft mask's mean alpha once a capture asks for it."""

    __slots__ = ("mean_alpha", "measured", "source")

    def __init__(self, source: ImageSource) -> None:
        self.source = source
        self.mean_alpha: float | None = None
        self.measured = False

    def soft_mask_alpha(self) -> float | None:
        if not self.measured:
            self.mean_alpha = soft_mask_mean_alpha(self.source)
            self.measured = True
        return self.mean_alpha


def soft_mask_mean_alpha(source: ImageSource) -> float | None:
    mask = source.soft_mask
    if mask is None:
        return None
    try:
        raster = decode_soft_mask(source, mask)
    except PdfParseError, TypeError, ValueError, ArithmeticError:
        return None
    if raster is None or not raster.array.size:
        return None
    samples = raster.array[:, :, 0]
    return int(samples.sum(dtype=numpy.uint64)) / (255.0 * samples.size)


class CaptureGraphicsSave(ReprFields, ReplaceFields, metaclass=RecordType, frozen=False):
    clip_bbox: Rectangle | None
    group_alpha: float | None
    clip_scope_emitted: bool = False


GRAPHICS_STATE_FIELDS = GraphicsState.__fields__


MASK_OVERRIDDEN_FIELDS = frozenset(
    {"ctm", "soft_mask", "fill_opacity", "stroke_opacity", "blend_mode"}
)


MASK_KEYED_FIELDS = tuple(
    name for name in GRAPHICS_STATE_FIELDS if name not in MASK_OVERRIDDEN_FIELDS
)


def state_key(value: object) -> object:
    if isinstance(value, tuple):
        return tuple(state_key(part) for part in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return (type(value), value)
    return ("identity", id(value))


CaptureColorMemo: TypeAlias = tuple[object, tuple[float, ...], object, object, tuple[float, ...]]


DRAWING_LAYOUT = SlotLayout(CapturedDrawing, CapturedDrawing.__fields__)
DRAWING_RAW_DATA = PATH_PAINT_DEFAULTS["raw_data"]
DRAWING_DICTIONARY = PATH_PAINT_DEFAULTS["dictionary"]
DRAWING_IMAGE_SOURCE = PATH_PAINT_DEFAULTS["image_source"]
DRAWING_IMAGE_CLIP = PATH_PAINT_DEFAULTS["image_clip"]
DRAWING_BBOX = PATH_PAINT_DEFAULTS["bbox"]
DRAWING_COLOR_RENDERING = PATH_PAINT_DEFAULTS["color_rendering"]
DRAWING_PAINTS = PATH_PAINT_DEFAULTS["paints"]
DRAWING_GROUP_ISOLATED = PATH_PAINT_DEFAULTS["group_isolated"]
DRAWING_GROUP_KNOCKOUT = PATH_PAINT_DEFAULTS["group_knockout"]


# color_space_paints is True for every other kind.
SPACES_THAT_MAY_NOT_PAINT = frozenset(("Indexed", "Pattern", "Separation", "DeviceN"))


GLYPH_PAINT_KEEPING_OPERATORS = frozenset(
    ("BT", "Td", "TD", "Tm", "T*", "Tj", "TJ", "'", '"', "Tc", "Tw", "Tz", "TL", "Ts", "Tf")
)


TEXT_LAYOUT_KEEPING_OPERATORS = frozenset(("Tj", "TJ", "TL", "Tw"))


LINE_MOVING_OPERATORS = frozenset(("Td", "TD", "T*", "'"))
