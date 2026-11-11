"""The page walk pdfplumber's objects are read from.

pdfplumber reports the graphics state as pdfminer.six's interpreter keeps it, which is not
the state a renderer keeps: colors are the operands as written, a line width is scaled by
the CTM when ``w`` runs, ``gs`` changes nothing, and a form XObject starts from a fresh
state. ``PlumberTextState`` keeps that state beside core's own and stamps it on the paths,
text and images the walk records.

The rules follow pdfminer.six's ``PDFPageInterpreter`` (pdfminer/pdfinterp.py), Copyright
(c) 2004-2016 Yusuke Shinyama, under the MIT License.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, NamedTuple, cast

from core_pdf import PdfPage
from core_pdf.impl.capture_glyphs import GlyphPaint, untrack_tuple
from core_pdf.impl.capture_program import CapturedProgram
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.fonts_metrics import FONT_DATA
from core_pdf.impl.types import PdfName, PdfReference, PdfString
from core_pdf_spec.s_07_content.inline_images import (
    InlineImage,
    normalize_inline_image_dictionary,
    parse_inline_image_dictionary,
)
from core_pdf_spec.s_07_content.model import PdfPath
from core_pdf_spec.s_07_content.operations import ContentOperands
from core_pdf_spec.s_07_content.streams import ContentStreamFrame
from core_pdf_spec.s_07_syntax.stream import PdfStream

from ..pdfminer._capture import (
    PdfminerContentLexer,
    PdfminerRecovery,
    PdfminerTextState,
)
from ..pdfminer._names import pdfminer_named_object

# pdfminer.six's PREDEFINED_COLORSPACE: name -> number of components.
PREDEFINED_COLOR_SPACES = {
    "DeviceGray": 1,
    "CalRGB": 3,
    "CalGray": 1,
    "Lab": 3,
    "DeviceRGB": 3,
    "DeviceCMYK": 4,
    "Separation": 1,
    "Indexed": 1,
    "Pattern": 1,
}


class PlumberContentLexer(PdfminerContentLexer):
    """Notes where the last inline image's dictionary starts, just after its BI.

    Core reads the entries after BI and expands their abbreviations at once; pdfminer.six
    keeps them as written, so the facade reads them again from there.
    """

    inline_image_start: int | None = None

    def skip_ignored(self) -> None:
        # Core reads an inline image's dictionary from just after BI, skipping first.
        position = self.pos
        if position >= 2 and self.raw_data[position - 2 : position] == b"BI":
            self.inline_image_start = position
        super().skip_ignored()


def written_inline_dictionary(lexer: object, image: InlineImage) -> dict[Any, Any] | None:
    """The inline image's dictionary as written, its abbreviations kept, as pdfminer.six
    reads it; None when it cannot be read again to the dictionary core read."""
    if not isinstance(lexer, PlumberContentLexer) or lexer.inline_image_start is None:
        return None
    reader = PdfminerContentLexer(lexer.raw_data, semantic_context=lexer.semantic_context)
    try:
        reader.pos = lexer.inline_image_start
        written = parse_inline_image_dictionary(reader)
        if normalize_inline_image_dictionary(written) != image.dictionary:
            return None
        return written
    except PdfParseError:
        return None
    finally:
        reader.close()


class MinerGraphics:
    """pdfminer.six's PDFGraphicState, limited to what pdfplumber reports."""

    __slots__ = ("dash", "linewidth", "ncolor", "ncs", "scolor", "scs")

    def __init__(self) -> None:
        self.linewidth: float = 0
        self.dash: Any = None
        self.scolor: Any = 0
        self.scs: tuple[str, int] = ("DeviceGray", 1)
        self.ncolor: Any = 0
        self.ncs: tuple[str, int] = ("DeviceGray", 1)

    def copy(self) -> MinerGraphics:
        other = MinerGraphics()
        other.linewidth = self.linewidth
        other.dash = self.dash
        other.scolor = self.scolor
        other.scs = self.scs
        other.ncolor = self.ncolor
        other.ncs = self.ncs
        return other


class MinerPath(NamedTuple):
    """A painted path as pdfminer.six's curpath holds it, operators and operands as written."""

    segments: tuple[tuple[Any, ...], ...]
    ctm: Any
    stroke: bool
    fill: bool
    evenodd: bool
    linewidth: float
    stroking_color: Any
    non_stroking_color: Any
    dash: Any
    mcid: int | None
    tag: str | None
    sequence: int


class PlumberMark(NamedTuple):
    """What pdfminer.six's graphics and text state hold when a glyph is shown."""

    glyph_index: int
    mcid: int | None
    tag: str | None
    color_space: str | None
    stroke_color: Any
    fill_color: Any
    matrix: tuple[float, float, float, float, float, float]
    rise: float
    scaling: float
    char_space: float


class PaintedImage(NamedTuple):
    name: str
    dictionary: dict[Any, Any]
    stream: Any
    ctm: Any
    inline: bool
    mcid: int | None
    tag: str | None
    sequence: int


def safe_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except OverflowError:
            return None
    return None


def color_components(values: list[Any]) -> Any:
    # pdfminer.six's _parse_color_components: gray is a float, RGB and CMYK are tuples.
    floats = [safe_float(value) for value in values]
    if len(values) not in (1, 3, 4) or any(value is None for value in floats):
        return None
    return floats[0] if len(floats) == 1 else tuple(floats)


def as_name(value: object) -> str | None:
    if isinstance(value, PdfName):
        return value.value
    return None


def literal_name(value: PdfName) -> str:
    # pdfminer.six's literal_name: a name that is not UTF-8 reads as its bytes' repr.
    raw = value.value_bytes
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return str(raw)


def as_list(value: object) -> Any:
    if isinstance(value, tuple):
        return [as_list(item) for item in value]
    if isinstance(value, list):
        return [as_list(item) for item in value]
    return value


# pdfplumber's CP936_FONTNAMES (pdfplumber/page.py), Copyright (c) 2015, Jeremy Singer-Vine,
# under the MIT License.
CP936_FONT_NAMES = {
    b"\xcb\xce\xcc\xe5": "SimSun,Regular",
    b"\xba\xda\xcc\xe5": "SimHei,Regular",
    b"\xbf\xac\xcc\xe5_GB2312": "SimKai,Regular",
    b"\xb7\xc2\xcb\xce_GB2312": "SimFang,Regular",
    b"\xc1\xa5\xca\xe9": "SimLi,Regular",
}


def fix_fontname_bytes(fontname: bytes) -> str:
    # pdfplumber's fix_fontname_bytes, for a FontName written as a string.
    if b"+" in fontname:
        split_at = fontname.index(b"+") + 1
        prefix, suffix = fontname[:split_at], fontname[split_at:]
    else:
        prefix, suffix = b"", fontname
    return str(prefix)[2:-1] + CP936_FONT_NAMES.get(suffix, str(suffix)[2:-1])


def miner_font_name(resolver: Any, spec: object, depth: int = 0) -> Any:
    """LTChar.fontname: the FontName of the descriptor pdfminer.six's font class reads."""
    spec = resolver.resolve(spec)
    if not isinstance(spec, dict):
        return "unknown"
    subtype = spec.get("Subtype")
    subtype_name = literal_name(subtype) if isinstance(subtype, PdfName) else "Type1"
    descriptor: object = None
    if subtype_name == "Type0":
        descendants = resolver.resolve(spec.get("DescendantFonts"))
        if isinstance(descendants, list) and descendants and depth < 8:
            return miner_font_name(resolver, descendants[0], depth + 1)
        return "unknown"
    if subtype_name in ("Type3", "CIDFontType0", "CIDFontType2"):
        descriptor = spec.get("FontDescriptor")
    else:
        base_font = spec.get("BaseFont")
        metrics = FONT_DATA.get(literal_name(base_font)) if isinstance(base_font, PdfName) else None
        if isinstance(metrics, dict):
            return metrics["props"]["FontName"]
        descriptor = spec.get("FontDescriptor")
    descriptor = resolver.resolve(descriptor)
    if not isinstance(descriptor, dict) or "FontName" not in descriptor:
        return "unknown"
    name = resolver.resolve(descriptor["FontName"])
    if isinstance(name, PdfName):
        return literal_name(name)
    if isinstance(name, PdfString):
        return fix_fontname_bytes(bytes(name.data))
    return name


class PlumberTextState(PdfminerTextState):
    def __init__(self, document: Any, **options: Any) -> None:
        super().__init__(document, **options)
        self.miner = MinerGraphics()
        self.miner_stack: list[MinerGraphics] = []
        self.miner_path: list[tuple[Any, ...]] = []
        self.miner_paths: list[MinerPath] = []
        self.miner_frames: dict[
            int, tuple[MinerGraphics, list[MinerGraphics], list[tuple[Any, ...]]]
        ] = {}
        self.color_space_maps: dict[int, tuple[object, dict[str, tuple[str, int]]]] = {}
        self.painted_images: list[PaintedImage] = []
        self.plumber_marks: list[PlumberMark] = []
        self.xobject_name: str | None = None
        self.xobject_reference: PdfReference | None = None
        # The lexer of each content stream being read, innermost last.
        self.stream_lexers: list[object] = []
        self.in_type3_glyph = 0
        # pdfminer.six's F paints nothing and keeps the path, unlike f.
        self.default_handlers["F"] = self.op_F

    def op_BDC(self, operands: ContentOperands, depth: int) -> None:
        super().op_BDC(operands, depth)
        if operands and isinstance(operands[0], PdfName):
            self.current_tag = literal_name(operands[0])

    def op_BMC(self, operands: ContentOperands, depth: int) -> None:
        super().op_BMC(operands, depth)
        if operands and isinstance(operands[0], PdfName):
            self.current_tag = literal_name(operands[0])

    def op_double_quote(self, operands: ContentOperands, depth: int) -> None:
        # pdfminer.six's " sets the spacing and shows the text without moving to a new line.
        if len(operands) >= 3:
            word_space, char_space, text = operands[-3:]
            self.op_Tw((word_space,), depth)
            self.op_Tc((char_space,), depth)
            self.op_Tj((text,), depth)

    # The graphics state pdfminer.six keeps.

    def color_spaces(self) -> dict[str, tuple[str, int]]:
        resources = self.resources
        known = self.color_space_maps.get(id(resources))
        if known is not None and known[0] is resources:
            return known[1]
        spaces = {name: (name, count) for name, count in PREDEFINED_COLOR_SPACES.items()}
        resolver = self.resolver
        entries = (
            resolver.resolve(resources.get("ColorSpace")) if isinstance(resources, dict) else None
        )
        for key, raw in (entries if isinstance(entries, dict) else {}).items():
            space = self.miner_color_space(resolver.resolve(raw))
            if space is not None:
                spaces[literal_name(key) if isinstance(key, PdfName) else str(key)] = space
        self.color_space_maps[id(resources)] = (resources, spaces)
        return spaces

    def miner_color_space(self, spec: object) -> tuple[str, int] | None:
        resolver = self.resolver
        if isinstance(spec, list) and spec:
            name = as_name(resolver.resolve(spec[0]))
        else:
            name = as_name(spec)
        if name is None:
            return None
        if name == "ICCBased" and isinstance(spec, list) and len(spec) >= 2:
            stream = resolver.resolve(spec[1])
            dictionary = getattr(stream, "dictionary", None)
            count = resolver.resolve(dictionary.get("N")) if isinstance(dictionary, dict) else None
            return (name, count) if isinstance(count, int) else None
        if name == "DeviceN" and isinstance(spec, list) and len(spec) >= 2:
            names = resolver.resolve(spec[1])
            return (name, len(names)) if isinstance(names, list) else None
        count = PREDEFINED_COLOR_SPACES.get(name)
        return None if count is None else (name, count)

    def op_q(self, operands: ContentOperands, depth: int) -> None:
        self.miner_stack.append(self.miner.copy())
        super().op_q(operands, depth)

    def op_Q(self, operands: ContentOperands, depth: int) -> None:
        if self.miner_stack:
            self.miner = self.miner_stack.pop()
        super().op_Q(operands, depth)

    def op_w(self, operands: ContentOperands, depth: int) -> None:
        if operands and (width := safe_float(operands[-1])) is not None:
            ctm = self.graphics.ctm
            self.miner.linewidth = width * math.hypot(ctm.a, ctm.b)
        super().op_w(operands, depth)

    def op_d(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) >= 2:
            self.miner.dash = (as_list(operands[-2]), operands[-1])
        super().op_d(operands, depth)

    def set_miner_device_color(self, operands: ContentOperands, space: str, stroke: bool) -> None:
        count = PREDEFINED_COLOR_SPACES[space]
        if len(operands) < count:
            return
        color = color_components(list(operands[-count:]))
        if color is None:
            return
        if stroke:
            self.miner.scolor, self.miner.scs = color, (space, count)
        else:
            self.miner.ncolor, self.miner.ncs = color, (space, count)

    def op_G(self, operands: ContentOperands, depth: int) -> None:
        self.set_miner_device_color(operands, "DeviceGray", True)
        super().op_G(operands, depth)

    def op_g(self, operands: ContentOperands, depth: int) -> None:
        self.set_miner_device_color(operands, "DeviceGray", False)
        super().op_g(operands, depth)

    def op_RG(self, operands: ContentOperands, depth: int) -> None:
        self.set_miner_device_color(operands, "DeviceRGB", True)
        super().op_RG(operands, depth)

    def op_rg(self, operands: ContentOperands, depth: int) -> None:
        self.set_miner_device_color(operands, "DeviceRGB", False)
        super().op_rg(operands, depth)

    def op_K(self, operands: ContentOperands, depth: int) -> None:
        self.set_miner_device_color(operands, "DeviceCMYK", True)
        super().op_K(operands, depth)

    def op_k(self, operands: ContentOperands, depth: int) -> None:
        self.set_miner_device_color(operands, "DeviceCMYK", False)
        super().op_k(operands, depth)

    def set_miner_color_space(self, operands: ContentOperands, stroke: bool) -> None:
        if not operands or not isinstance(operands[-1], PdfName):
            return
        space = self.color_spaces().get(literal_name(operands[-1]))
        if space is None:
            return
        if stroke:
            self.miner.scs = space
        else:
            self.miner.ncs = space

    def op_CS(self, operands: ContentOperands, depth: int) -> None:
        self.set_miner_color_space(operands, True)
        super().op_CS(operands, depth)

    def op_cs(self, operands: ContentOperands, depth: int) -> None:
        self.set_miner_color_space(operands, False)
        super().op_cs(operands, depth)

    def set_miner_color(self, operands: ContentOperands, stroke: bool) -> None:
        name, count = self.miner.scs if stroke else self.miner.ncs
        components = list(operands[-count:]) if count else list(operands)
        if len(components) != count:
            return
        color: Any
        if name != "Pattern":
            color = color_components(components)
        else:
            pattern = components[-1]
            if not isinstance(pattern, PdfName):
                return
            if len(components) == 1:
                color = literal_name(pattern)
            else:
                base = color_components(components[:-1])
                color = None if base is None else (base, literal_name(pattern))
        if color is None:
            return
        if stroke:
            self.miner.scolor = color
        else:
            self.miner.ncolor = color

    def op_SC(self, operands: ContentOperands, depth: int) -> None:
        self.set_miner_color(operands, True)
        super().op_SC(operands, depth)

    def op_SCN(self, operands: ContentOperands, depth: int) -> None:
        self.set_miner_color(operands, True)
        super().op_SCN(operands, depth)

    def op_sc(self, operands: ContentOperands, depth: int) -> None:
        self.set_miner_color(operands, False)
        super().op_sc(operands, depth)

    def op_scN(self, operands: ContentOperands, depth: int) -> None:
        self.set_miner_color(operands, False)
        super().op_scN(operands, depth)

    # The path pdfminer.six builds: every operator as written, `re` as m l l l h.

    def miner_segment(self, operator: str, operands: ContentOperands, count: int) -> None:
        if len(operands) < count:
            return
        values = [safe_float(value) for value in operands[len(operands) - count :]]
        if all(value is not None for value in values):
            self.miner_path.append((operator, *values))

    def op_m(self, operands: ContentOperands, depth: int) -> None:
        self.miner_segment("m", operands, 2)
        super().op_m(operands, depth)

    def op_l(self, operands: ContentOperands, depth: int) -> None:
        self.miner_segment("l", operands, 2)
        super().op_l(operands, depth)

    def op_c(self, operands: ContentOperands, depth: int) -> None:
        self.miner_segment("c", operands, 6)
        super().op_c(operands, depth)

    def op_v(self, operands: ContentOperands, depth: int) -> None:
        self.miner_segment("v", operands, 4)
        super().op_v(operands, depth)

    def op_y(self, operands: ContentOperands, depth: int) -> None:
        self.miner_segment("y", operands, 4)
        super().op_y(operands, depth)

    def op_h(self, operands: ContentOperands, depth: int) -> None:
        self.miner_path.append(("h",))
        super().op_h(operands, depth)

    def op_re(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) >= 4:
            values = [safe_float(value) for value in operands[len(operands) - 4 :]]
            if all(value is not None for value in values):
                x, y, w, h = cast("list[float]", values)
                self.miner_path += [
                    ("m", x, y),
                    ("l", x + w, y),
                    ("l", x + w, y + h),
                    ("l", x, y + h),
                    ("h",),
                ]
        super().op_re(operands, depth)

    def miner_paint(self, stroke: bool, fill: bool, evenodd: bool, close: bool = False) -> None:
        if close:
            self.miner_path.append(("h",))
        if not self.in_type3_glyph:
            miner = self.miner
            self.miner_paths.append(
                MinerPath(
                    tuple(self.miner_path),
                    self.graphics.ctm,
                    stroke,
                    fill,
                    evenodd,
                    miner.linewidth,
                    miner.scolor,
                    miner.ncolor,
                    miner.dash,
                    self.current_mcid,
                    self.current_tag,
                    self.sequence,
                )
            )
        self.miner_path = []

    def op_paint_stroke(self, operands: ContentOperands, depth: int) -> None:
        self.miner_paint(True, False, False)
        super().op_paint_stroke(operands, depth)

    def op_paint_close_stroke(self, operands: ContentOperands, depth: int) -> None:
        self.miner_paint(True, False, False, close=True)
        super().op_paint_close_stroke(operands, depth)

    def op_paint_fill(self, operands: ContentOperands, depth: int) -> None:
        self.miner_paint(False, True, False)
        super().op_paint_fill(operands, depth)

    def op_F(self, operands: ContentOperands, depth: int) -> None:
        super().op_paint_fill(operands, depth)

    def op_paint_fill_evenodd(self, operands: ContentOperands, depth: int) -> None:
        self.miner_paint(False, True, True)
        super().op_paint_fill_evenodd(operands, depth)

    def op_paint_fillstroke(self, operands: ContentOperands, depth: int) -> None:
        self.miner_paint(True, True, False)
        super().op_paint_fillstroke(operands, depth)

    def op_paint_fillstroke_evenodd(self, operands: ContentOperands, depth: int) -> None:
        self.miner_paint(True, True, True)
        super().op_paint_fillstroke_evenodd(operands, depth)

    def op_paint_close_fillstroke(self, operands: ContentOperands, depth: int) -> None:
        self.miner_paint(True, True, False, close=True)
        super().op_paint_close_fillstroke(operands, depth)

    def op_paint_close_fillstroke_evenodd(self, operands: ContentOperands, depth: int) -> None:
        self.miner_paint(True, True, True, close=True)
        super().op_paint_close_fillstroke_evenodd(operands, depth)

    def op_paint_clear(self, operands: ContentOperands, depth: int) -> None:
        self.miner_path = []
        super().op_paint_clear(operands, depth)

    def enter_stream(self, state: object, frame: ContentStreamFrame) -> None:
        # Core saves the graphics state with op_q as a stream starts and restores it
        # directly as the stream ends; pdfminer.six's stack never holds that save.
        if self.miner_stack:
            self.miner_stack.pop()
        if frame.is_form:
            # pdfminer.six runs a form in a new interpreter, from a fresh state.
            self.miner_frames[id(frame)] = (self.miner, self.miner_stack, self.miner_path)
            self.miner, self.miner_stack, self.miner_path = MinerGraphics(), [], []
        self.stream_lexers.append(frame.lexer)
        super().enter_stream(state, frame)

    def exit_stream(self, state: object, frame: ContentStreamFrame) -> None:
        if self.stream_lexers:
            self.stream_lexers.pop()
        saved = self.miner_frames.pop(id(frame), None)
        if saved is not None:
            self.miner, self.miner_stack, self.miner_path = saved
        super().exit_stream(state, frame)

    def render_type3_glyphs(self, data: bytes | memoryview, decoder: Any) -> None:
        # pdfminer.six does not run a Type 3 glyph's procedure.
        saved = (self.miner, self.miner_stack, self.miner_path)
        self.miner, self.miner_stack, self.miner_path = self.miner.copy(), [], []
        self.in_type3_glyph += 1
        try:
            super().render_type3_glyphs(data, decoder)
        finally:
            self.in_type3_glyph -= 1
            self.miner, self.miner_stack, self.miner_path = saved

    # What the walk records, stamped with that state.

    def show_text(
        self,
        state: object,
        text: str,
        data: bytes | memoryview,
        glyphs: tuple[Any, ...],
        decoder: Any,
        adv_x: float,
        adv_y: float,
        *,
        glyph_paint: GlyphPaint | None = None,
    ) -> None:
        # Named parameters rather than *args: this runs once per glyph.
        start = len(self.glyphs)
        graphics = self.graphics
        # pdfminer.six's char matrix: the text matrix at the glyph times the CTM, no rise.
        placed = self.text_matrix.multiply(graphics.ctm)
        matrix = (placed.a, placed.b, placed.c, placed.d, placed.e, placed.f)
        super().show_text(state, text, data, glyphs, decoder, adv_x, adv_y, glyph_paint=glyph_paint)
        self.append_marks(start, (matrix,))

    def show_glyph_run(
        self,
        glyphs: tuple[Any, ...],
        texts: list[str],
        origins: list[float],
        decoder: Any,
        *,
        glyph_paint: GlyphPaint | None = None,
    ) -> tuple[tuple[float, float, float, float], list[float]] | None:
        start = len(self.glyphs)
        shown = super().show_glyph_run(glyphs, texts, origins, decoder, glyph_paint=glyph_paint)
        if shown is None:
            return None
        (a, b, c, d), placed = shown
        # As show_text marks each glyph: its char matrix is the text matrix at the glyph
        # times the CTM.
        self.append_marks(
            start,
            ((a, b, c, d, placed[index], placed[index + 1]) for index in range(0, len(placed), 2)),
        )
        return shown

    def append_marks(
        self, start: int, matrices: Iterable[tuple[float, float, float, float, float, float]]
    ) -> None:
        """Mark the glyphs from start on, one per char matrix, with the current state.

        A page keeps its marks while its chars are wanted, and marks hold only numbers,
        strings and tuples of them, so the collector need not walk them.
        """
        graphics = self.graphics
        miner = self.miner
        scolor = miner.scolor
        ncolor = miner.ncolor
        stroking = untrack_tuple(scolor if isinstance(scolor, tuple) else (scolor,))
        non_stroking = untrack_tuple(ncolor if isinstance(ncolor, tuple) else (ncolor,))
        mcid = self.current_mcid
        tag = self.current_tag
        color_space = miner.ncs[0]
        rise = graphics.rise
        horizontal_scale = graphics.horizontal_scale
        scaling = horizontal_scale * 0.01
        char_space = graphics.char_space * horizontal_scale * 0.01
        append = self.plumber_marks.append
        for index, matrix in enumerate(matrices, start):
            append(
                untrack_tuple(
                    PlumberMark(
                        index,
                        mcid,
                        tag,
                        color_space,
                        stroking,
                        non_stroking,
                        matrix,
                        rise,
                        scaling,
                        char_space,
                    )
                )
            )

    def paint_path(self, state: object, source: PdfPath, kind: str, fill_rule: str) -> None:
        # The paths pdfplumber reports are the ones miner_paint records.
        super(PdfminerTextState, self).paint_path(state, source, kind, fill_rule)

    def append_xobject(self, name_obj: Any, depth: int) -> ContentStreamFrame | None:
        self.xobject_name = literal_name(name_obj) if isinstance(name_obj, PdfName) else None
        reference = (
            self.lookup_page_resource("XObject", name_obj.value)
            if isinstance(name_obj, PdfName)
            else None
        )
        self.xobject_reference = reference if isinstance(reference, PdfReference) else None
        try:
            return super().append_xobject(name_obj, depth)
        finally:
            self.xobject_name = None
            self.xobject_reference = None

    def paint_image(self, state: object, xobj: Any) -> None:
        image = xobj
        if self.xobject_reference is not None:
            # The image as pdfminer.six reads it, when one of its names reads otherwise.
            named = pdfminer_named_object(self.document, self.xobject_reference)
            if isinstance(named, PdfStream):
                image = named
        dictionary = image.dictionary
        if (
            not self.in_type3_glyph
            and self.xobject_name is not None
            and "Width" in dictionary
            and "Height" in dictionary
        ):
            self.painted_images.append(
                PaintedImage(
                    self.xobject_name,
                    dictionary,
                    image,
                    self.graphics.ctm,
                    False,
                    self.current_mcid,
                    self.current_tag,
                    self.sequence,
                )
            )
        super().paint_image(state, xobj)

    def paint_inline_image(self, state: object, image: Any) -> None:
        lexer = self.stream_lexers[-1] if self.stream_lexers else None
        written = written_inline_dictionary(lexer, image)
        # pdfminer.six places an inline image whose W and H are written abbreviated.
        dictionary, sized = (
            (image.dictionary, ("Width", "Height")) if written is None else (written, ("W", "H"))
        )
        if not self.in_type3_glyph and all(key in dictionary for key in sized):
            # pdfminer.six names an inline image by the id() of its stream.
            self.painted_images.append(
                PaintedImage(
                    str(id(image)),
                    dictionary,
                    PdfStream(raw_data=image.data, dictionary=dictionary),
                    self.graphics.ctm,
                    True,
                    self.current_mcid,
                    self.current_tag,
                    self.sequence,
                )
            )
        super().paint_inline_image(state, image)


def plumber_page_capture(
    page: PdfPage,
) -> tuple[
    CapturedProgram, tuple[MinerPath, ...], tuple[PlumberMark, ...], tuple[PaintedImage, ...]
]:
    state = PlumberTextState(page.document, page_clip=page.effective_page_clip())
    state.lexer_factory = PlumberContentLexer
    state.content_recovery = PdfminerRecovery()
    page.consume_contents(state)
    state.run_accumulator.flush()
    program = CapturedProgram(
        runs=tuple(state.runs),
        glyphs=tuple(state.glyphs),
        drawings=tuple(state.drawings),
        inline_images=tuple(state.inline_images),
        lines=state.lines.since(0),
        text_boundaries=tuple(state.text_boundaries),
        options=state.options,
    )
    return (
        program,
        tuple(state.miner_paths),
        tuple(state.plumber_marks),
        tuple(state.painted_images),
    )


__all__ = (
    "MinerPath",
    "miner_font_name",
    "PaintedImage",
    "PlumberTextState",
    "plumber_page_capture",
)
