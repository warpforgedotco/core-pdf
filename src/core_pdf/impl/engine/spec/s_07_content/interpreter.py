# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import struct
import typing
from functools import lru_cache
from typing import TypeAlias

from core_pdf.impl.engine.spec.s_07_content.models import TextRun
from core_pdf.impl.engine.spec.s_07_content.traces import CapturedLine, DrawingTrace, GlyphTrace
from core_pdf.impl.engine.spec.s_07_objects.resolver import ObjectResolver
from core_pdf.impl.engine.spec.s_07_syntax.lexer import (
    KeywordCacheValue,
    OperationHandler,
    ParsedOperand,
    PdfLexer,
)
from core_pdf.impl.engine.spec.s_07_syntax.primitives import (
    MISSING,
    Matrix,
    PdfDictLike,
    PdfName,
    PdfObject,
    PdfReference,
    PdfStream,
    PdfString,
    matrix_multiply,
)
from core_pdf.impl.engine.spec.s_08_graphics.geometry import RectBox
from core_pdf.impl.engine.spec.s_09_fonts.decoder import FontDecoder
from core_pdf.impl.engine.spec.s_09_fonts.encoding import split_chunks
from core_pdf.impl.engine.spec.s_09_fonts.truetype import (
    tt_cmap,
    tt_gid_composite_info,
    tt_loca,
    tt_tables,
)


class TextDocument(typing.Protocol):
    @property
    def resolver(self) -> ObjectResolver: ...

    @property
    def decoder_cache(self) -> dict[tuple[int, int] | int, FontDecoder]: ...

    def resolve(self, ref: PdfObject) -> PdfObject: ...


DecoderCache: TypeAlias = dict[tuple[int, int] | int, FontDecoder]
PdfDict: TypeAlias = PdfDictLike
ResolvedResourceCache: TypeAlias = dict[str, PdfDict | None]
ResourceLookupCache: TypeAlias = dict[str, dict[str, PdfObject | None]]
ExtGStateCache: TypeAlias = dict[tuple[int, str], PdfDict | None]


GraphicsState: TypeAlias = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    tuple[float, ...] | None,
    float | None,
    tuple[float, ...] | None,
    float | None,
    float,
    int,
    int,
    float,
    tuple[list[float], float],
]


def is_pdf_value(value: object) -> typing.TypeGuard[PdfObject | None]:
    return value is None or isinstance(
        value,
        (
            int,
            float,
            str,
            bytes,
            PdfName,
            PdfReference,
            PdfString,
            PdfStream,
            list,
            tuple,
            dict,
        ),
    )


class StreamState:
    __slots__ = (
        "resources",
        "resources_id",
        "ctm",
        "text_matrix",
        "line_matrix",
        "font_size",
        "current_font",
        "current_decoder",
        "graphics_stack_len",
        "marked_content_stack_len",
        "fill_color",
        "fill_opacity",
        "pending_line_break",
        "xobject_depth",
    )

    resources: PdfDict
    resources_id: int
    ctm: Matrix
    text_matrix: Matrix
    line_matrix: Matrix
    font_size: float
    current_font: str | None
    current_decoder: FontDecoder | None
    graphics_stack_len: int
    marked_content_stack_len: int
    fill_color: tuple[float, ...] | None
    fill_opacity: float | None
    pending_line_break: bool
    xobject_depth: int

    def __init__(
        self,
        resources: PdfDict,
        resources_id: int,
        ctm: Matrix,
        text_matrix: Matrix,
        line_matrix: Matrix,
        font_size: float,
        current_font: str | None,
        current_decoder: FontDecoder | None,
        graphics_stack_len: int,
        marked_content_stack_len: int,
        fill_color: tuple[float, ...] | None,
        fill_opacity: float | None,
        pending_line_break: bool,
        xobject_depth: int,
    ) -> None:
        self.resources = resources
        self.resources_id = resources_id
        self.ctm = ctm
        self.text_matrix = text_matrix
        self.line_matrix = line_matrix
        self.font_size = font_size
        self.current_font = current_font
        self.current_decoder = current_decoder
        self.graphics_stack_len = graphics_stack_len
        self.marked_content_stack_len = marked_content_stack_len
        self.fill_color = fill_color
        self.fill_opacity = fill_opacity
        self.pending_line_break = pending_line_break
        self.xobject_depth = xobject_depth


FILL_OPS = frozenset({"f", "f*", "F"})
FILL_AND_STROKE_OPS = frozenset({"B", "b", "B*", "b*"})
PAINT_OPS = FILL_OPS | FILL_AND_STROKE_OPS


@lru_cache(maxsize=256)
def cached_encode_latin1(s: str) -> bytes:
    return s.encode("latin-1", "replace")


class TextState:
    __slots__ = (
        "document",
        "page",
        "capture_runs",
        "capture_glyphs",
        "capture_graphics",
        "runs",
        "glyphs",
        "lines",
        "drawings",
        "pending_edges",
        "pending_rects",
        "current_point",
        "subpath_start",
        "stack",
        "tm_a",
        "tm_b",
        "tm_c",
        "tm_d",
        "tm_e",
        "tm_f",
        "ca",
        "cb",
        "cc",
        "cd",
        "ce",
        "cf",
        "lm_a",
        "lm_b",
        "lm_c",
        "lm_d",
        "lm_e",
        "lm_f",
        "fill_color",
        "fill_opacity",
        "stroke_color",
        "stroke_opacity",
        "line_width",
        "line_cap",
        "line_join",
        "miter_limit",
        "dash_pattern",
        "font_size",
        "horizontal_scale",
        "char_space",
        "word_space",
        "rise",
        "leading",
        "render_mode",
        "current_font",
        "current_decoder",
        "sequence",
        "stream_order",
        "xobject_depth",
        "marked_content_stack",
        "active_streams",
        "resources",
        "resources_id",
        "hidden_layers",
        "resolved_resource_categories",
        "resource_cache",
        "extgstate_cache",
        "decoder_cache",
        "kw_cache",
        "pending_line_break",
        "pending_run",
        "invisible_text_layer",
        "op_handlers",
        "fast_op_handlers",
        "resolve",
        "resolve_dict",
        "resolve_name",
        "resolve_str",
        "combined_A",
        "combined_B",
        "combined_C",
        "combined_D",
        "cached_rotation",
        "is_garbage",
        "operands",
        "run_pool",
        "run_pool_idx",
    )

    current_font: str | None
    current_decoder: FontDecoder | None
    resources: PdfDict
    resolved_resource_categories: ResolvedResourceCache
    resource_cache: ResourceLookupCache
    extgstate_cache: ExtGStateCache
    decoder_cache: DecoderCache
    kw_cache: dict[bytes, KeywordCacheValue]
    op_handlers: dict[str, OperationHandler]
    fast_op_handlers: list[OperationHandler | None]
    operands: list[ParsedOperand]

    def __init__(
        self,
        document: TextDocument,
        page: PdfDict,
        hidden_layers: frozenset[str] = frozenset(),
        capture_runs: bool = True,
        capture_glyphs: bool = False,
        capture_graphics: bool = False,
        decoder_cache: DecoderCache | None = None,
    ) -> None:
        self.document = document
        self.page = page

        # CTM components
        self.ca, self.cb, self.cc, self.cd, self.ce, self.cf = 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
        # Text Matrix components
        self.tm_a, self.tm_b, self.tm_c, self.tm_d, self.tm_e, self.tm_f = (
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )
        # Line Matrix components
        self.lm_a, self.lm_b, self.lm_c, self.lm_d, self.lm_e, self.lm_f = (
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )

        self.fill_color: tuple[float, ...] | None = (0.0, 0.0, 0.0)
        self.fill_opacity: float | None = 1.0
        self.stroke_color: tuple[float, ...] | None = (0.0, 0.0, 0.0)
        self.stroke_opacity: float | None = 1.0
        self.line_width = 1.0
        self.line_cap = 0
        self.line_join = 0
        self.miter_limit = 10.0
        self.dash_pattern: tuple[list[float], float] = ([], 0.0)
        self.stack: list[GraphicsState] = []
        self.capture_runs = capture_runs
        self.capture_glyphs = capture_glyphs
        self.capture_graphics = capture_graphics
        self.runs: list[TextRun] = []
        self.glyphs: list[GlyphTrace] = []
        self.lines: list[CapturedLine] = []
        self.drawings: list[DrawingTrace] = []
        self.pending_edges: list[tuple[float, float, float, float]] = []
        self.pending_rects: list[RectBox] = []
        self.current_point = None
        self.subpath_start = None
        self.font_size = 12.0
        self.horizontal_scale = 100.0
        self.char_space = 0.0
        self.word_space = 0.0
        self.rise = 0.0
        self.leading = 0.0
        self.render_mode = 0
        self.current_font = None
        self.current_decoder = None
        self.sequence = 0
        self.stream_order = -1
        self.xobject_depth = 0
        self.marked_content_stack: list[str | None] = []
        self.active_streams: set[int] = set()
        self.resources: PdfDict = {}
        self.resources_id = 0
        self.hidden_layers = hidden_layers
        self.resolved_resource_categories: ResolvedResourceCache = {}
        self.resource_cache: ResourceLookupCache = {}
        self.extgstate_cache: ExtGStateCache = {}
        self.decoder_cache: DecoderCache = decoder_cache if decoder_cache is not None else {}
        self.kw_cache = getattr(self.document.resolver, "kw_cache", {})
        self.pending_line_break = False
        self.pending_run = None
        self.invisible_text_layer = False
        self.op_handlers = {op: getattr(self, method) for op, method in self.TEXT_OP.items()}
        # Use 64K list for O(1) operator dispatch
        self.fast_op_handlers = [None] * 65536
        for op, handler in self.op_handlers.items():
            if len(op) == 1:
                self.fast_op_handlers[ord(op[0]) << 8] = handler
            elif len(op) == 2:
                self.fast_op_handlers[(ord(op[0]) << 8) | ord(op[1])] = handler

        # Fused components and rotation

        self.combined_A = 1.0
        self.combined_B = 0.0
        self.combined_C = 0.0
        self.combined_D = 1.0
        self.cached_rotation = 0

        # Pre-bind dependencies for performance
        self.resolve = self.document.resolver.resolve
        self.resolve_dict = self.document.resolver.resolve_dict
        self.resolve_name = self.document.resolver.resolve_name
        self.resolve_str = self.document.resolver.resolve_str

        # Pre-bind hot-path functions
        self.is_garbage = is_garbage_text
        self.operands: list[ParsedOperand] = [None] * 16
        self.run_pool: list[TextRun] = []
        self.run_pool_idx: int = 0

    @property
    def ctm(self) -> Matrix:
        return (self.ca, self.cb, self.cc, self.cd, self.ce, self.cf)

    @ctm.setter
    def ctm(self, val: Matrix) -> None:
        self.ca, self.cb, self.cc, self.cd, self.ce, self.cf = val
        self.update_combined()

    @property
    def text_matrix(self) -> Matrix:
        return (self.tm_a, self.tm_b, self.tm_c, self.tm_d, self.tm_e, self.tm_f)

    @text_matrix.setter
    def text_matrix(self, val: Matrix) -> None:
        self.tm_a, self.tm_b, self.tm_c, self.tm_d, self.tm_e, self.tm_f = val
        self.update_combined()

    @property
    def line_matrix(self) -> Matrix:
        return (self.lm_a, self.lm_b, self.lm_c, self.lm_d, self.lm_e, self.lm_f)

    @line_matrix.setter
    def line_matrix(self, val: Matrix) -> None:
        self.lm_a, self.lm_b, self.lm_c, self.lm_d, self.lm_e, self.lm_f = val

    def update_combined(self) -> None:
        ta, tb, tc, td = self.tm_a, self.tm_b, self.tm_c, self.tm_d
        ca, cb, cc, cd = self.ca, self.cb, self.cc, self.cd

        if ta == 1.0 and tb == 0.0 and tc == 0.0 and td == 1.0:
            # TM is identity — combined is just CTM (common after BT reset)
            A, B, C, D = ca, cb, cc, cd
        else:
            A = ta * ca + tb * cc
            B = ta * cb + tb * cd
            C = tc * ca + td * cc
            D = tc * cb + td * cd

        self.combined_A = A
        self.combined_B = B
        self.combined_C = C
        self.combined_D = D

        if A == 1.0 and B == 0.0 and C == 0.0 and D == 1.0:
            self.cached_rotation = 0
        else:
            self.cached_rotation = detect_rotation_from_linear(A, B, C, D)

    TEXT_OP = {
        "BT": "op_BT",
        "ET": "op_ET",
        "T*": "op_T_star",
        "Td": "op_Td",
        "TD": "op_TD",
        "Tj": "op_Tj",
        "TJ": "op_TJ",
        "Tm": "op_Tm",
        "Tf": "op_Tf",
        "TL": "op_TL",
        "Tc": "op_Tc",
        "Tw": "op_Tw",
        "Tz": "op_Tz",
        "Tr": "op_Tr",
        "Ts": "op_Ts",
        "'": "op_quote",
        '"': "op_double_quote",
        "Do": "op_Do",
        "BDC": "op_BDC",
        "BMC": "op_BMC",
        "EMC": "op_EMC",
        "q": "op_q",
        "Q": "op_Q",
        "cm": "op_cm",
        "g": "op_g",
        "rg": "op_rg",
        "k": "op_k",
        "G": "op_G",
        "RG": "op_RG",
        "K": "op_K",
        "w": "op_w",
        "J": "op_J",
        "j": "op_j",
        "M": "op_M",
        "d": "op_d",
        "gs": "op_gs",
        "m": "op_m",
        "l": "op_l",
        "re": "op_re",
        "h": "op_h",
        "c": "op_c",
        "v": "op_v",
        "y": "op_y",
        "S": "op_paint_stroke",
        "s": "op_paint_stroke",
        "f": "op_paint_fill",
        "F": "op_paint_fill",
        "f*": "op_paint_fill",
        "B": "op_paint_fillstroke",
        "b": "op_paint_fillstroke",
        "B*": "op_paint_fillstroke",
        "b*": "op_paint_fillstroke",
        "n": "op_paint_clear",
    }

    def flush_run(self) -> None:
        if not self.capture_runs:
            self.pending_run = None
            return
        if self.pending_run:
            self.runs.append(self.pending_run)
            self.pending_run = None

    def op_BT(self, o, d):
        if self.pending_run:
            self.flush_run()
        self.tm_a = self.lm_a = 1.0
        self.tm_b = self.lm_b = 0.0
        self.tm_c = self.lm_c = 0.0
        self.tm_d = self.lm_d = 1.0
        self.tm_e = self.lm_e = 0.0
        self.tm_f = self.lm_f = 0.0
        self.update_combined()
        self.invisible_text_layer = False
        if self.current_decoder is None:
            self.current_decoder = self.get_decoder()

    def op_ET(self, o, d):
        self.flush_run()

    def op_T_star(self, o, d):
        self.flush_run()
        self.shift_line(ty=-self.leading)
        # combined translation updated in shift_line

    def op_Td(self, o, d):
        self.flush_run()
        self.shift_line(tx=o[0], ty=o[1])

    def op_TD(self, o, d):
        self.flush_run()
        self.shift_line(tx=o[0], ty=o[1], set_leading=True)

    def op_Tj(self, o, d):
        if not o:
            return
        operand = o[0]
        if type(operand) is PdfString:
            self.append_text(_data=operand.data)
        else:
            self.append_text(operand)

    def op_TJ(self, o, d):
        if not o:
            return
        self.append_tj_array(o[0])

    def op_Tm(self, o, d):
        if len(o) >= 6:
            self.flush_run()
            self.tm_a = self.lm_a = float(o[0])
            self.tm_b = self.lm_b = float(o[1])
            self.tm_c = self.lm_c = float(o[2])
            self.tm_d = self.lm_d = float(o[3])
            self.tm_e = self.lm_e = float(o[4])
            self.tm_f = self.lm_f = float(o[5])
            self.update_combined()

    def op_Tf(self, o, d):
        self.current_font = self.resolve_name(o[0]) if o else None
        self.font_size = o[1] if len(o) >= 2 else self.font_size
        self.current_decoder = self.get_decoder()

    def op_TL(self, o, d):
        self.leading = o[0]

    def op_Tc(self, o, d):
        self.char_space = o[0]

    def op_Tw(self, o, d):
        self.word_space = o[0]

    def op_Tr(self, o, d):
        self.render_mode = int(o[0])

    def op_Tz(self, o, d):
        self.horizontal_scale = o[0]

    def op_Ts(self, o, d):
        self.rise = o[0]

    def op_quote(self, o, d):
        self.shift_line(ty=-self.leading)
        self.pending_line_break = True
        if not o:
            return
        operand = o[0]
        if type(operand) is PdfString:
            self.append_text(_data=operand.data)
        else:
            self.append_text(operand)

    def op_double_quote(self, o, d):
        if len(o) < 3:
            return
        self.word_space = o[0]
        self.char_space = o[1]
        self.shift_line(ty=-self.leading)
        self.pending_line_break = True
        operand = o[2]
        if type(operand) is PdfString:
            self.append_text(_data=operand.data)
        else:
            self.append_text(operand)

    def op_Do(self, o, d):
        self.append_xobject(o[0], d)

    def op_BDC(self, o, d):
        tag = self.resolve_name(o[0]) if o else None
        layer = None
        if tag == "OC":
            layer = self.resolve_str(o[1]) if len(o) >= 2 else None
        self.marked_content_stack.append(layer)

    def op_BMC(self, o, d):
        self.marked_content_stack.append(None)

    def op_EMC(self, o, d):
        if self.marked_content_stack:
            self.marked_content_stack.pop()

    def edge_rect(self, x0: float, y0: float, x1: float, y1: float) -> RectBox:
        line_width = max(0.0, self.line_width)
        half_width = line_width * 0.5
        left = (x0 if x0 < x1 else x1) - half_width
        right = (x0 if x0 > x1 else x1) + half_width
        bottom = (y0 if y0 < y1 else y1) - half_width
        top = (y0 if y0 > y1 else y1) + half_width

        ca, cb, cc, cd, ce, cf = self.ca, self.cb, self.cc, self.cd, self.ce, self.cf
        px0 = left * ca + bottom * cc + ce
        py0 = left * cb + bottom * cd + cf
        px1 = right * ca + top * cc + ce
        py1 = right * cb + top * cd + cf

        return RectBox(
            px0 if px0 < px1 else px1,
            py0 if py0 < py1 else py1,
            px0 if px0 > px1 else px1,
            py0 if py0 > py1 else py1,
            seqno=self.sequence,
            fill=self.stroke_color,
            fill_opacity=self.stroke_opacity,
        )

    def transform_point(self, x: float, y: float) -> tuple[float, float]:
        return (x * self.ca + y * self.cc + self.ce, x * self.cb + y * self.cd + self.cf)

    def flush_drawing(self, kind: str) -> None:
        if not self.capture_graphics:
            self.pending_rects.clear()
            self.pending_edges.clear()
            return

        items: list[tuple[str, RectBox]] = []
        if self.pending_rects:
            items.extend(("re", rect.normalize()) for rect in self.pending_rects)
            self.pending_rects.clear()

        if self.pending_edges:
            lines = self.lines
            lw = self.line_width
            for x0, y0, x1, y1 in self.pending_edges:
                px0, py0 = self.transform_point(x0, y0)
                px1, py1 = self.transform_point(x1, y1)
                if abs(px1 - px0) > 0.01 or abs(py1 - py0) > 0.01:
                    lines.append(CapturedLine(x0=px0, y0=py0, x1=px1, y1=py1, line_width=lw))
                items.append(("l", self.edge_rect(x0, y0, x1, y1)))
            self.pending_edges.clear()

        if items:
            self.drawings.append(
                DrawingTrace(
                    seqno=self.sequence,
                    fill=self.fill_color,
                    fill_opacity=self.fill_opacity,
                    stroke_color=self.stroke_color,
                    stroke_opacity=self.stroke_opacity,
                    line_width=self.line_width,
                    kind=kind,
                    items=items,
                )
            )

    def op_G(self, o, d):
        if o:
            gray = max(0.0, min(1.0, float(o[0])))
            self.stroke_color = (gray,)

    def op_RG(self, o, d):
        if len(o) >= 3:
            r = max(0.0, min(1.0, float(o[0])))
            g = max(0.0, min(1.0, float(o[1])))
            b = max(0.0, min(1.0, float(o[2])))
            self.stroke_color = (r, g, b)

    def op_K(self, o, d):
        if len(o) >= 4:
            c = max(0.0, min(1.0, float(o[0])))
            m = max(0.0, min(1.0, float(o[1])))
            y = max(0.0, min(1.0, float(o[2])))
            k = max(0.0, min(1.0, float(o[3])))
            self.stroke_color = (c, m, y, k)

    def op_w(self, o, d):
        if o:
            self.line_width = max(0.0, float(o[0]))

    def op_J(self, o, d):
        if o:
            self.line_cap = int(o[0])

    def op_j(self, o, d):
        if o:
            self.line_join = int(o[0])

    def op_M(self, o, d):
        if o:
            self.miter_limit = max(1.0, float(o[0]))

    def op_d(self, o, d):
        if o and len(o) >= 2:
            array_obj = o[0]
            phase = float(o[1])
            if isinstance(array_obj, (list, tuple)):
                dash_array = [float(v) for v in array_obj]
            else:
                dash_array = []
            self.dash_pattern = (dash_array, phase)

    def op_m(self, o, d):
        if len(o) >= 2:
            self.current_point = (float(o[0]), float(o[1]))
            self.subpath_start = self.current_point

    def op_l(self, o, d):
        if len(o) >= 2 and self.current_point is not None:
            if self.capture_graphics:
                nx, ny = float(o[0]), float(o[1])
                self.pending_edges.append((self.current_point[0], self.current_point[1], nx, ny))
                self.current_point = (nx, ny)
            else:
                self.current_point = (float(o[0]), float(o[1]))

    def op_re(self, o, d):
        if len(o) >= 4:
            x, y = float(o[0]), float(o[1])
            if self.capture_graphics:
                w, h = float(o[2]), float(o[3])
                self.pending_edges.append((x, y, x + w, y))
                self.pending_edges.append((x + w, y, x + w, y + h))
                self.pending_edges.append((x + w, y + h, x, y + h))
                self.pending_edges.append((x, y + h, x, y))
                self.pending_rects.append(
                    RectBox(
                        x,
                        y,
                        x + w,
                        y + h,
                        seqno=self.sequence,
                        fill=self.fill_color,
                        fill_opacity=self.fill_opacity,
                    )
                )
            self.current_point = (x, y)
            self.subpath_start = (x, y)

    def op_h(self, o, d):
        if self.current_point is not None and self.subpath_start is not None:
            if self.capture_graphics:
                self.pending_edges.append(
                    (
                        self.current_point[0],
                        self.current_point[1],
                        self.subpath_start[0],
                        self.subpath_start[1],
                    )
                )
            self.current_point = self.subpath_start

    def op_c(self, o, d):
        if len(o) >= 6:
            self.current_point = (float(o[4]), float(o[5]))

    def op_v(self, o, d):
        if len(o) >= 4:
            self.current_point = (float(o[2]), float(o[3]))

    def op_y(self, o, d):
        if len(o) >= 4:
            self.current_point = (float(o[2]), float(o[3]))

    def op_paint_stroke(self, o, d):
        if (
            self.capture_graphics
            and d == "s"
            and self.current_point is not None
            and self.subpath_start is not None
        ):
            self.pending_edges.append(
                (
                    self.current_point[0],
                    self.current_point[1],
                    self.subpath_start[0],
                    self.subpath_start[1],
                )
            )
        self.flush_drawing("stroke")
        self.current_point = None
        self.subpath_start = None

    def op_paint_fill(self, o, d):
        self.flush_drawing("fill")
        self.current_point = None
        self.subpath_start = None

    def op_paint_fillstroke(self, o, d):
        if (
            self.capture_graphics
            and (d == "b" or d == "b*")
            and self.current_point is not None
            and self.subpath_start is not None
        ):
            self.pending_edges.append(
                (
                    self.current_point[0],
                    self.current_point[1],
                    self.subpath_start[0],
                    self.subpath_start[1],
                )
            )
        self.flush_drawing("fillstroke")
        self.current_point = None
        self.subpath_start = None

    def op_paint_clear(self, o, d):
        self.pending_edges.clear()
        self.pending_rects.clear()
        self.current_point = None
        self.subpath_start = None

    def set_fill_color(self, *components: float) -> None:
        self.fill_color = tuple(components)

    def resolve_extgstate(self, name: str) -> PdfDict | None:
        cache_key = (self.resources_id, name)
        if cache_key in self.extgstate_cache:
            return self.extgstate_cache[cache_key]
        extgstate = self.resources.get("ExtGState")
        if not isinstance(extgstate, dict):
            self.extgstate_cache[cache_key] = None
            return None
        resolved = extgstate.get(name)
        if not isinstance(resolved, dict):
            self.extgstate_cache[cache_key] = None
            return None
        self.extgstate_cache[cache_key] = resolved
        return resolved

    def op_q(self, o, d):
        self.stack.append(
            (
                self.ca,
                self.cb,
                self.cc,
                self.cd,
                self.ce,
                self.cf,
                self.fill_color,
                self.fill_opacity,
                self.stroke_color,
                self.stroke_opacity,
                self.line_width,
                self.line_cap,
                self.line_join,
                self.miter_limit,
                self.dash_pattern,
            )
        )

    def op_Q(self, o, d):
        if self.stack:
            (
                self.ca,
                self.cb,
                self.cc,
                self.cd,
                self.ce,
                self.cf,
                self.fill_color,
                self.fill_opacity,
                self.stroke_color,
                self.stroke_opacity,
                self.line_width,
                self.line_cap,
                self.line_join,
                self.miter_limit,
                self.dash_pattern,
            ) = self.stack.pop()
            self.update_combined()

    def op_cm(self, o, d):
        if not o or len(o) < 6:
            raise ValueError("invalid matrix operands")
        m_a, m_b, m_c, m_d, m_e, m_f = (
            float(o[0]),
            float(o[1]),
            float(o[2]),
            float(o[3]),
            float(o[4]),
            float(o[5]),
        )

        ca, cb, cc, cd, ce, cf = self.ca, self.cb, self.cc, self.cd, self.ce, self.cf
        self.ca = m_a * ca + m_b * cc
        self.cb = m_a * cb + m_b * cd
        self.cc = m_c * ca + m_d * cc
        self.cd = m_c * cb + m_d * cd
        self.ce = m_e * ca + m_f * cc + ce
        self.cf = m_e * cb + m_f * cd + cf
        self.update_combined()

    def op_g(self, o, d):
        if o:
            gray = max(0.0, min(1.0, float(o[0])))
            self.set_fill_color(gray)

    def op_rg(self, o, d):
        if len(o) >= 3:
            r = max(0.0, min(1.0, float(o[0])))
            g = max(0.0, min(1.0, float(o[1])))
            b = max(0.0, min(1.0, float(o[2])))
            self.set_fill_color(r, g, b)

    def op_k(self, o, d):
        if len(o) >= 4:
            c = max(0.0, min(1.0, float(o[0])))
            m = max(0.0, min(1.0, float(o[1])))
            y = max(0.0, min(1.0, float(o[2])))
            k = max(0.0, min(1.0, float(o[3])))
            self.set_fill_color(c, m, y, k)

    def op_gs(self, o, d):
        if not o:
            raise ValueError("invalid graphics state")
        name = self.resolve_name(o[0])
        if not name:
            raise ValueError("invalid graphics state")
        extgstate = self.resolve_extgstate(name)
        if not extgstate:
            raise ValueError("invalid graphics state")
        try:
            fill_opacity = extgstate.get("ca")
            if isinstance(fill_opacity, (int, float, str, bytes)):
                self.fill_opacity = max(0.0, min(1.0, float(fill_opacity)))
            stroke_opacity = extgstate.get("CA")
            if isinstance(stroke_opacity, (int, float, str, bytes)):
                self.stroke_opacity = max(0.0, min(1.0, float(stroke_opacity)))
        except TypeError, ValueError:
            raise ValueError("invalid graphics state opacity")

    def shift_line(self, tx: float = 0.0, ty: float = 0.0, *, set_leading: bool = False) -> None:
        if set_leading:
            self.leading = -ty
        self.tm_e += tx
        self.tm_f += ty
        self.lm_e = self.tm_e
        self.lm_f = self.tm_f

    def apply_operation(self, operation: tuple[str, tuple[PdfObject, ...]], depth: int) -> None:
        operator, operands = operation
        handler = self.op_handlers.get(operator)
        if handler is not None:
            handler(operands, depth)

    def capture_stream_state(self) -> StreamState:
        return StreamState(
            resources=self.resources,
            resources_id=self.resources_id,
            ctm=self.ctm,
            text_matrix=self.text_matrix,
            line_matrix=self.line_matrix,
            font_size=self.font_size,
            current_font=self.current_font,
            current_decoder=self.current_decoder,
            graphics_stack_len=len(self.stack),
            marked_content_stack_len=len(self.marked_content_stack),
            fill_color=self.fill_color,
            fill_opacity=self.fill_opacity,
            pending_line_break=self.pending_line_break,
            xobject_depth=self.xobject_depth,
        )

    def restore_stream_state(self, state: StreamState) -> None:
        self.resources = state.resources
        self.resources_id = state.resources_id
        self.ctm = state.ctm
        self.text_matrix = state.text_matrix
        self.line_matrix = state.line_matrix
        self.font_size = state.font_size
        self.current_font = state.current_font
        self.current_decoder = state.current_decoder
        del self.stack[state.graphics_stack_len :]
        del self.marked_content_stack[state.marked_content_stack_len :]
        self.fill_color = state.fill_color
        self.fill_opacity = state.fill_opacity
        self.pending_line_break = state.pending_line_break
        self.xobject_depth = state.xobject_depth

    def consume_stream(
        self,
        stream: PdfStream,
        resources: PdfDict,
        ctm: Matrix,
        depth: int,
    ) -> None:
        if id(stream) in self.active_streams or depth > 10:
            return

        self.active_streams.add(id(stream))
        old_state = self.capture_stream_state()
        self.resources = resources
        self.resources_id = id(resources)
        self.ctm = ctm
        self.xobject_depth = depth

        self.pending_line_break = False
        self.stream_order += 1

        try:
            lexer = PdfLexer(stream.data_view, kw_cache=self.kw_cache)
            lexer.dispatch_operations(
                self.op_handlers, self.fast_op_handlers, depth, operands=self.operands
            )

            self.flush_run()
        finally:
            # Restore parent stream context even if parsing fails.
            self.restore_stream_state(old_state)
            self.active_streams.remove(id(stream))

    def lookup_page_resource(
        self, category: str, name: str, parent_category: str | None = None
    ) -> PdfObject | None:
        # Optimized nested cache lookup using pre-calculated resources_id
        cat_cache = self.resource_cache.get(category)
        if cat_cache is None:
            self.resource_cache[category] = cat_cache = {}
        else:
            res = cat_cache.get(name, MISSING)
            if res is not MISSING:
                if is_pdf_value(res):
                    return res
                raise ValueError("invalid cached resource value")

        # Resolve category dictionary
        category_res = self.resolved_resource_categories.get(category, MISSING)
        if category_res is MISSING:
            raw_category = self.resources.get(category)
            category_res = self.resolve_dict(raw_category) if raw_category is not None else None
            self.resolved_resource_categories[category] = category_res

        if isinstance(category_res, dict):
            res = category_res.get(name)
            if is_pdf_value(res) and res is not None:
                resolved = self.resolve(res)
                cat_cache[name] = resolved
                return resolved

        if parent_category:
            parent_res = self.resolved_resource_categories.get(parent_category, MISSING)
            if parent_res is MISSING:
                raw_parent = self.resources.get(parent_category)
                parent_res = self.resolve_dict(raw_parent) if raw_parent is not None else None
                self.resolved_resource_categories[parent_category] = parent_res

            if isinstance(parent_res, dict):
                # Search in sub-resources (common in some complex PDFs)
                for p_val in parent_res.values():
                    if isinstance(p_val, dict):
                        sub_res_dict = p_val.get("Resources")
                        if isinstance(sub_res_dict, dict):
                            sub_cat = sub_res_dict.get(category)
                            if isinstance(sub_cat, dict):
                                found = sub_cat.get(name)
                                if is_pdf_value(found) and found is not None:
                                    resolved = self.resolve(found)
                                    cat_cache[name] = resolved
                                    return resolved

        cat_cache[name] = None
        return None

    def chunk_advance(self, code: int, decoder: "FontDecoder") -> float:
        fs = self.font_size
        scale = fs * self.horizontal_scale / 100000.0
        base = decoder.glyph_width(code)
        char_extra = self.char_space * 1000.0 / fs if fs else 0.0
        word_extra = self.word_space * 1000.0 / fs if fs and code == 32 else 0.0
        return (base + char_extra + word_extra) * scale

    def decode_operand(
        self, operand: PdfObject, decoder: FontDecoder
    ) -> tuple[str, bytes, list[bytes] | None, list[str] | None]:
        if type(operand) is PdfString:
            data = operand.data
            needs_decode = True
            text = None
        elif type(operand) is bytes:
            data = operand
            needs_decode = True
            text = None
        elif type(operand) is str:
            data = cached_encode_latin1(operand)
            needs_decode = False
            text = operand
        else:
            text = self.resolve_str(operand)
            if text is None:
                return "", b"", None, None
            data = cached_encode_latin1(text)
            needs_decode = False

        chunks: list[bytes] | None = None
        chunk_texts: list[str] | None = None
        if needs_decode:
            text = decoder.decode(data)
        if self.capture_glyphs:
            chunks = split_chunks(data, decoder.is_cid_font, decoder.to_unicode or decoder.cmap)
            chunk_texts = decoder.decode_chunks(chunks)
            if text is None:
                text = "".join(chunk_texts)
        if text is None:
            text = ""
        return text, data, chunks, chunk_texts

    def is_text_visible(self, text: str) -> bool:
        if self.is_garbage(text):
            return False

        # Inline is_invisible_text_mode logic
        is_invisible_mode = self.render_mode == 3 or self.font_size < 0.1
        if is_invisible_mode:
            if not any(r.visible for r in self.runs):
                self.invisible_text_layer = True
            if not self.invisible_text_layer:
                return False

        if self.marked_content_stack:
            for layer in self.marked_content_stack:
                if layer and layer in self.hidden_layers:
                    return False
        return True

    def alloc_run(
        self,
        text: str,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        tx: float,
        ty: float,
        font_size: float,
        space_width: float,
        order: int,
        stream_order: int,
        xobject_depth: int,
        font_name: str | None,
        is_vertical: bool,
        rotation_angle: int,
        visible: bool,
        line_break_before: bool,
        seqno: int,
        fill_color: tuple[float, ...] | None,
    ) -> TextRun:
        idx = self.run_pool_idx
        if idx < len(self.run_pool):
            existing = self.run_pool[idx]
        else:
            existing = None
        r = TextRun.reinit(
            existing,
            text,
            x0,
            y0,
            x1,
            y1,
            tx,
            ty,
            font_size,
            space_width,
            order,
            stream_order,
            xobject_depth,
            font_name,
            is_vertical,
            rotation_angle,
            visible,
            line_break_before,
            seqno,
            fill_color,
        )
        if existing is None:
            self.run_pool.append(r)
        self.run_pool_idx = idx + 1
        return r

    def update_pending_run(self, new_run: TextRun) -> None:
        if not self.pending_run:
            self.pending_run = new_run
            return

        p = self.pending_run
        merge_threshold = max(p.space_width * 0.45, 2.0)

        is_same_style = (
            p.rotation_angle == new_run.rotation_angle
            and p.visible == new_run.visible
            and not new_run.line_break_before
            and p.font_size == new_run.font_size
            and p.font_name == new_run.font_name
            and p.fill_color == new_run.fill_color
        )

        if is_same_style and p.rotation_angle == 90:
            y_gap = new_run.y0 - p.y1
            max_y_gap = max(p.space_width * 0.5, p.font_size * 0.8, 2.0)
            if abs(y_gap) > max_y_gap:
                is_same_style = False
        elif is_same_style and p.rotation_angle == 0:
            if abs(p.y0 - new_run.y0) > p.font_size * 0.5:
                is_same_style = False

        merged = False
        if is_same_style:
            if p.rotation_angle in (0, 90):
                if p.rotation_angle == 0:
                    gap = new_run.x0 - p.x1
                    if -2.0 <= gap < merge_threshold:
                        p.text += new_run.text
                        p.x1 = max(p.x1, new_run.x1)
                        merged = True
                    else:
                        gap_rtl = p.x0 - new_run.x1
                        if -2.0 <= gap_rtl < merge_threshold:
                            p.text = new_run.text + p.text
                            p.x0 = min(p.x0, new_run.x0)
                            merged = True
                else:
                    gap = new_run.y0 - p.y1
                    if -2.0 <= gap < merge_threshold:
                        p.text += new_run.text
                        p.y1 = max(p.y1, new_run.y1)
                        merged = True
            else:
                h_gap_inv = p.x0 - new_run.x1
                if -2.0 <= h_gap_inv < merge_threshold:
                    p.text += new_run.text
                    p.x0 = min(p.x0, new_run.x0)
                    merged = True
                else:
                    h_gap_inv_rtl = new_run.x0 - p.x1
                    if -2.0 <= h_gap_inv_rtl < merge_threshold:
                        p.text = new_run.text + p.text
                        p.x1 = max(p.x1, new_run.x1)
                        merged = True

        if not merged:
            self.runs.append(p)
            self.pending_run = new_run

    def record_glyph_traces(
        self,
        text: str,
        data: bytes,
        decoder: "FontDecoder",
        bbox: tuple[float, float, float, float],
        rotation_angle: int,
        visible: bool,
        chunks: list[bytes] | None = None,
        chunk_texts: list[str] | None = None,
    ) -> None:
        if not self.capture_glyphs:
            return
        if chunks is None:
            chunks = split_chunks(data, decoder.is_cid_font, decoder.to_unicode or decoder.cmap)
        if not chunks:
            return
        if chunk_texts is None:
            chunk_texts = decoder.decode_chunks(chunks)

        chunk_codes = [
            chunk[0] if len(chunk) == 1 else int.from_bytes(chunk, "big") for chunk in chunks
        ]
        advances = [self.chunk_advance(code, decoder) for code in chunk_codes]
        total_advance = sum(advances)
        if total_advance <= 0:
            return

        x0, y0, x1, y1 = bbox
        axis_is_x = rotation_angle in (0, 180) or rotation_angle == 0
        span = (x1 - x0) if axis_is_x else (y1 - y0)
        if span == 0:
            return

        offset = 0.0
        seqno = self.sequence
        fill = self.fill_color
        fill_opacity = self.fill_opacity
        append_glyph = self.glyphs.append
        cursor = 0
        for chunk, advance, chunk_text in zip(chunks, advances, chunk_texts, strict=True):
            if not chunk_text:
                chunk_text = text[cursor : cursor + 1]
            cursor += max(1, len(chunk_text))
            if not chunk_text:
                offset += advance
                continue

            start_ratio = offset / total_advance
            end_ratio = (offset + advance) / total_advance
            if axis_is_x:
                cx0 = x0 + span * start_ratio
                cx1 = x0 + span * end_ratio
                cy0, cy1 = y0, y1
            else:
                cy0 = y0 + span * start_ratio
                cy1 = y0 + span * end_ratio
                cx0, cx1 = x0, x1

            if len(chunk_text) == 1:
                append_glyph(
                    GlyphTrace(
                        rect=RectBox(
                            cx0, cy0, cx1, cy1, seqno=seqno, fill=fill, fill_opacity=fill_opacity
                        ),
                        c=chunk_text,
                        seqno=seqno,
                        fill=fill,
                        visible=visible,
                    )
                )
            else:
                char_span = (cx1 - cx0) if axis_is_x else (cy1 - cy0)
                if char_span == 0:
                    offset += advance
                    continue
                char_offset = 0.0
                per_char = char_span / len(chunk_text)
                for ch in chunk_text:
                    if axis_is_x:
                        gx0 = cx0 + char_offset
                        gx1 = gx0 + per_char
                        gy0, gy1 = cy0, cy1
                    else:
                        gy0 = cy0 + char_offset
                        gy1 = gy0 + per_char
                        gx0, gx1 = cx0, cx1
                    append_glyph(
                        GlyphTrace(
                            rect=RectBox(
                                gx0,
                                gy0,
                                gx1,
                                gy1,
                                seqno=seqno,
                                fill=fill,
                                fill_opacity=fill_opacity,
                            ),
                            c=ch,
                            seqno=seqno,
                            fill=fill,
                            visible=visible,
                        )
                    )
                    char_offset += per_char
            offset += advance

    def append_text(self, operand: PdfObject = None, *, _data: bytes | None = None) -> None:
        decoder = self.get_decoder()

        if _data is not None:
            data = _data
            text = decoder.decode(data)
            chunks = None
            chunk_texts = None
        else:
            text, data, chunks, chunk_texts = self.decode_operand(operand, decoder)
        if not text:
            return

        visible = self.is_text_visible(text)

        fs = self.font_size
        rise = self.rise

        # Inline the single-byte fast path from text_advance_vector to avoid
        # function-call overhead for ~80% of PDF text (standard encodings).
        if (
            chunks is None
            and not decoder.is_cid_font
            and decoder.to_unicode is None
            and decoder.cmap is None
        ):
            widths = decoder.fast_widths
            cs = self.char_space * 1000.0 / fs if fs else 0.0
            ws = self.word_space * 1000.0 / fs if fs else 0.0
            scale = fs * self.horizontal_scale / 100000.0
            total = 0.0
            space_count = 0
            for b in data:
                total += widths[b]
                if b == 32:
                    space_count += 1
            total += len(data) * cs + space_count * ws
            if decoder.is_vertical:
                adv_x, adv_y = 0.0, -total * scale
            else:
                adv_x, adv_y = total * scale, 0.0
        else:
            adv_x, adv_y = decoder.text_advance_vector(
                data,
                font_size=fs,
                char_space=self.char_space,
                word_space=self.word_space,
                horizontal_scale=self.horizontal_scale,
                chunks=chunks,
            )

        inv1000 = fs / 1000.0
        ascent = decoder.ascent * inv1000
        descent = decoder.descent * inv1000

        # Use cached components
        A, B, C, D = self.combined_A, self.combined_B, self.combined_C, self.combined_D
        # Current translation components of fused T * CTM
        ca, cb, cc, cd, ce, cf = self.ca, self.cb, self.cc, self.cd, self.ce, self.cf
        te, tf = self.tm_e, self.tm_f
        E = te * ca + tf * cc + ce
        F = te * cb + tf * cd + cf

        # Inline bounding box calculation (fused matrix)
        ar = ascent + rise
        dr = descent + rise

        c0_x = dr * C + E
        c0_y = dr * D + F
        c1_x = ar * C + E
        c1_y = ar * D + F

        adv_A = adv_x * A
        adv_B = adv_x * B

        c2_x = adv_A + c0_x
        c2_y = adv_B + c0_y
        c3_x = adv_A + c1_x
        c3_y = adv_B + c1_y

        x0 = c0_x if c0_x < c1_x else c1_x
        if c2_x < x0:
            x0 = c2_x
        if c3_x < x0:
            x0 = c3_x

        y0 = c0_y if c0_y < c1_y else c1_y
        if c2_y < y0:
            y0 = c2_y
        if c3_y < y0:
            y0 = c3_y

        x1 = c0_x if c0_x > c1_x else c1_x
        if c2_x > x1:
            x1 = c2_x
        if c3_x > x1:
            x1 = c3_x

        y1 = c0_y if c0_y > c1_y else c1_y
        if c2_y > y1:
            y1 = c2_y
        if c3_y > y1:
            y1 = c3_y

        rot = self.cached_rotation
        seqno = self.sequence
        ta, tb, tc, td = self.tm_a, self.tm_b, self.tm_c, self.tm_d

        if not self.capture_runs:
            if self.capture_glyphs:
                self.record_glyph_traces(
                    text,
                    data,
                    decoder,
                    (x0, y0, x1, y1),
                    rot,
                    visible,
                    chunks=chunks,
                    chunk_texts=chunk_texts,
                )
            self.sequence = seqno + 1
            self.pending_line_break = False
            self.tm_e = te + adv_x * ta + adv_y * tc
            self.tm_f = tf + adv_x * tb + adv_y * td
            return

        new_run = self.alloc_run(
            text=text,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            tx=te,
            ty=tf,
            font_size=fs,
            font_name=self.current_font,
            space_width=decoder.glyph_width(32) * inv1000,
            order=seqno,
            stream_order=self.stream_order,
            xobject_depth=self.xobject_depth,
            is_vertical=decoder.is_vertical,
            rotation_angle=rot,
            visible=visible,
            line_break_before=self.pending_line_break,
            seqno=seqno,
            fill_color=self.fill_color,
        )

        if self.capture_glyphs:
            self.record_glyph_traces(
                text,
                data,
                decoder,
                (x0, y0, x1, y1),
                rot,
                visible,
                chunks=chunks,
                chunk_texts=chunk_texts,
            )

        self.update_pending_run(new_run)

        self.sequence = seqno + 1

        # Advance text matrix components directly (Inline translation)
        self.tm_e = te + adv_x * ta + adv_y * tc
        self.tm_f = tf + adv_x * tb + adv_y * td
        self.pending_line_break = False

    def append_tj_array(self, array: PdfObject) -> None:
        if not isinstance(array, (list, tuple)):
            return
        pending_bytes = bytearray()
        fs = self.font_size
        scale = fs * self.horizontal_scale / 100000.0

        decoder = self.get_decoder()
        is_vert = decoder.is_vertical

        # Localize components for performance
        te, tf = self.tm_e, self.tm_f
        ta, tb, tc, td = self.tm_a, self.tm_b, self.tm_c, self.tm_d

        for item in array:
            if isinstance(item, PdfString):
                pending_bytes.extend(item.data)
            elif isinstance(item, bytes):
                pending_bytes.extend(item)
            elif isinstance(item, str):
                pending_bytes.extend(item.encode("latin-1"))
            elif isinstance(item, (int, float)):
                if pending_bytes:
                    self.tm_e, self.tm_f = te, tf
                    self.append_text(_data=bytes(pending_bytes))
                    te, tf = self.tm_e, self.tm_f
                    pending_bytes.clear()
                delta = -item * scale
                if is_vert:
                    te += delta * tc
                    tf += delta * td
                else:
                    te += delta * ta
                    tf += delta * tb

        if pending_bytes:
            self.tm_e, self.tm_f = te, tf
            self.append_text(_data=bytes(pending_bytes))
            te, tf = self.tm_e, self.tm_f

        self.tm_e, self.tm_f = te, tf

    def append_xobject(self, name_obj: PdfObject, depth: int) -> None:
        name = self.resolve_name(name_obj)
        if not name:
            return
        xobj = self.lookup_page_resource("XObject", name, "Font")
        if not isinstance(xobj, PdfStream):
            return
        xobj_dict = self.document.resolver.resolve_dict(xobj.dictionary)
        if xobj_dict is None:
            return
        if (
            getattr(xobj_dict.get("Type"), "value", None) == "ObjStm"
            or self.resolve_name(xobj_dict.get("Subtype")) != "Form"
        ):
            return
        resources = self.resolve_dict(xobj_dict.get("Resources")) or self.resources
        # Skip fontless XObject forms that contain no text operators
        if isinstance(resources, dict) and not resources.get("Font"):
            raw = xobj._raw_data
            if raw and b"BT" not in raw and b"Tj" not in raw and b"TJ" not in raw:
                return
        from core_pdf.impl.engine.spec.s_07_syntax.primitives import parse_matrix_operand

        nested_ctm = matrix_multiply(parse_matrix_operand(xobj_dict.get("Matrix")), self.ctm)
        self.consume_stream(xobj, resources, nested_ctm, depth + 1)

    def get_decoder(self) -> "FontDecoder":
        if self.current_decoder is not None:
            return self.current_decoder

        font_obj_ref = (
            self.lookup_page_resource("Font", self.current_font) if self.current_font else None
        )
        if font_obj_ref is None:
            return FontDecoder({})

        font_obj = self.document.resolver.resolve(font_obj_ref)
        if isinstance(font_obj, PdfStream):
            font_obj = font_obj.dictionary
        if not isinstance(font_obj, dict):
            return FontDecoder({})

        # Document-wide cache key based on font object reference (stable across pages)
        cache_key = (
            (font_obj_ref.obj_num, font_obj_ref.gen_num)
            if isinstance(font_obj_ref, PdfReference)
            else id(font_obj_ref)
        )
        doc_cache = self.document.decoder_cache
        cached_decoder = doc_cache.get(cache_key, MISSING)
        if isinstance(cached_decoder, FontDecoder):
            self.current_decoder = cached_decoder
            return self.current_decoder

        # Resolve full font dict (recursive) only when creating new decoder
        resolved_font = self.document.resolver.resolve_font_dict(font_obj)
        self.current_decoder = FontDecoder(
            resolved_font,
            ligature_overrides=detect_ligature_overrides(
                self.document, self.resources, resolved_font
            ),
        )
        doc_cache[cache_key] = self.current_decoder
        return self.current_decoder


MATRIX_TOLERANCE = 0.1


def detect_rotation_from_linear(
    A: float, B: float, C: float, D: float, tolerance: float = MATRIX_TOLERANCE
) -> int:
    from math import sqrt

    scale_x = sqrt(A * A + B * B)
    scale_y = sqrt(C * C + D * D)
    if scale_x <= 0 or scale_y <= 0:
        return 0
    na, nb, nc, nd = A / scale_x, B / scale_x, C / scale_y, D / scale_y
    if (
        abs(na - 1.0) < tolerance
        and abs(nb) < tolerance
        and abs(nc) < tolerance
        and abs(nd - 1.0) < tolerance
    ):
        return 0
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


def get_font_file(document: TextDocument, font_obj: PdfDict) -> PdfStream | None:
    descriptor = font_obj.get("FontDescriptor")
    if not isinstance(descriptor, dict):
        return None
    font_file = document.resolve(descriptor.get("FontFile2"))
    return font_file if isinstance(font_file, PdfStream) else None


def find_companion_font(
    document: TextDocument,
    resources: PdfDict,
    base_name: str,
    ligature_starters: set[str],
) -> tuple[dict[int, float], dict[str, float], bytes | None]:
    font_resources = resources.get("Font")
    if not isinstance(font_resources, dict):
        return {}, {}, None

    for fref in font_resources.values():
        fobj = document.resolve(fref)
        if not isinstance(fobj, dict):
            continue
        comp_base = str(fobj.get("BaseFont", ""))
        if "+" in comp_base:
            comp_base = comp_base.split("+", 1)[1]
        if comp_base != base_name:
            continue

        fc = fobj.get("FirstChar")
        lc = fobj.get("LastChar")
        if not (isinstance(fc, int) and isinstance(lc, int) and lc > fc + 5):
            continue

        widths_raw = fobj.get("Widths")
        if widths_raw is None:
            continue
        if not isinstance(widths_raw, (list, tuple)):
            raise ValueError("invalid font widths array")

        font_file = get_font_file(document, fobj)
        if font_file is None:
            continue

        starter_widths: dict[int, float] = {}
        starter_chars: dict[str, float] = {}
        for i, w in enumerate(widths_raw):
            if isinstance(w, (int, float)) and float(w) > 0:
                code = fc + i
                try:
                    ch = bytes([code]).decode("mac_roman")
                except UnicodeDecodeError:
                    ch = chr(code) if code < 128 else ""
                if ch in ligature_starters:
                    starter_widths[code] = float(w)
                    starter_chars[ch] = float(w)

        if starter_widths:
            return starter_widths, starter_chars, font_file.data

    return {}, {}, None


def detect_ligature_overrides(
    document: TextDocument, resources: PdfDict, font_obj: PdfDict
) -> dict[int, str]:
    """Detect Chrome ligature substitutions for small TrueType ligature subset fonts."""
    first_char = font_obj.get("FirstChar")
    last_char = font_obj.get("LastChar")
    if not (isinstance(first_char, int) and isinstance(last_char, int)):
        return {}

    font_file = get_font_file(document, font_obj)
    if font_file is None:
        return {}
    tt_data = font_file.data

    tables = tt_tables(tt_data)
    if not {"maxp", "glyf", "loca", "head"} <= tables.keys():
        return {}
    try:
        n_glyphs = struct.unpack(">H", tt_data[tables["maxp"][0] + 4 : tables["maxp"][0] + 6])[0]
        loca = tt_loca(tt_data, tables, n_glyphs)
    except struct.error, IndexError, KeyError:
        return {}
    if loca is None:
        return {}

    glyf_off = tables["glyf"][0]
    cp_to_gid = tt_cmap(tt_data, tables)

    base_font_raw = str(font_obj.get("BaseFont", ""))
    base_name = base_font_raw.split("+", 1)[1] if "+" in base_font_raw else base_font_raw
    if not base_name:
        return {}

    starter_widths, starter_chars, companion_data = find_companion_font(
        document, resources, base_name, set("ftscFTSC")
    )

    if companion_data is None or not starter_widths:
        return {}

    # Verify companion font metadata
    comp_tables = tt_tables(companion_data)
    if not {"maxp", "glyf", "loca", "head"} <= comp_tables.keys():
        return {}
    try:
        comp_n = struct.unpack(
            ">H", companion_data[comp_tables["maxp"][0] + 4 : comp_tables["maxp"][0] + 6]
        )[0]
        if not tt_loca(companion_data, comp_tables, comp_n):
            return {}
    except struct.error, IndexError, KeyError:
        return {}

    lig_widths_raw = font_obj.get("Widths")
    if lig_widths_raw is not None and not isinstance(lig_widths_raw, (list, tuple)):
        raise ValueError("invalid font widths array")
    overrides: dict[int, str] = {}

    for pdf_code in range(first_char, last_char + 1):
        try:
            cp = ord(bytes([pdf_code]).decode("mac_roman"))
        except UnicodeDecodeError:
            cp = pdf_code

        if cp in cp_to_gid:
            continue

        gid = pdf_code - first_char
        body_bbox, is_composite = tt_gid_composite_info(tt_data, glyf_off, loca, gid)
        if not (is_composite and body_bbox):
            continue

        ft_width = starter_chars.get("f", 0.0) + starter_chars.get("t", 0.0)
        if ft_width <= 0:
            continue

        lig_width = 0.0
        if lig_widths_raw is not None:
            idx = pdf_code - first_char
            if 0 <= idx < len(lig_widths_raw):
                w = lig_widths_raw[idx]
                if isinstance(w, (int, float)):
                    lig_width = float(w)

        if lig_width and 0.85 <= lig_width / ft_width <= 0.98:
            overrides[pdf_code] = "ft"

    return overrides


@lru_cache(maxsize=4096)
def is_garbage_text(text: str) -> bool:
    if not text:
        return True
    for c in text:
        o = ord(c)
        if not (o < 32 or 0xE000 <= o <= 0xF8FF):
            return False
    return True
