# SPDX-License-Identifier: AGPL-3.0-only

from math import isfinite
from typing import TYPE_CHECKING, Any, Protocol, cast

from core_pdf.impl.capture.recording import RecordingMethods
from core_pdf.impl.capture.recovery import CaptureRecovery, iter_content_operations
from core_pdf.impl.capture.text_runs import RunAccumulator
from core_pdf.impl.document.recovery.lexer import PdfLexer
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.fonts.decoder import FontDecoder
from core_pdf.impl.fonts.font_program import (
    TrueTypeFontProgram,
    cached_truetype_program,
)
from core_pdf.impl.fonts.helpers import strip_subset_tag
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.runtime.scalars import parse_float_strict, parse_int_strict
from core_pdf.impl.types import Rectangle
from core_pdf_spec.s_07_content.streams import ContentStreamExecutor, ContentStreamFrame, StreamKey
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict

# Two strict parsers with the same name, deliberately: the runtime pair above is
# tolerant of the values a damaged content stream yields, the spec pair below is
# token-based and is what the ligature helpers were written against.
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    parse_float_strict as parse_spec_float_strict,
)
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    parse_int_strict as parse_spec_int_strict,
)
from core_pdf_spec.s_08_graphics.matrix import Matrix

if TYPE_CHECKING:
    from core_pdf.impl.capture.tolerant_state import RecoveringTextState


class TextState(RecordingMethods):
    def __init__(
        self,
        document: Any,
        hidden_layers: frozenset[str] = frozenset(),
        page_clip: Rectangle | None = None,
        *,
        capture_ink_bounds: bool = True,
        capture_text_runs: bool = True,
    ):
        self.document = document
        self.runs = []
        self.glyphs = []
        self.glyph_cluster_count = 0
        self.lines = []
        self.drawings = []
        self.inline_images = []
        self.hidden_layers = hidden_layers
        self.page_clip = page_clip
        self.capture_ink_bounds = capture_ink_bounds
        self.capture_text_runs = capture_text_runs
        self.clip_bbox = None
        self.layout_form_bbox = None
        self.layout_form_id = None
        self.capture_source = "native_text"
        self.stream_order = -1
        self.sequence = 0
        self.text_object_id = 0
        self.text_boundaries = []
        self.capture_text_open = False
        self.capture_text_frames = {}
        self.pending_line_break = False
        self.group_alpha = None
        self.run_accumulator = RunAccumulator(self.runs)
        self.capture_graphics_stack = []
        self.capture_marked_entries = {}
        self.capture_frames = {}
        self.capture_patterns = {}
        self.capture_image_sources = {}
        self.capture_colors = {}
        self.capture_soft_masks = {}
        self.capture_mask_resources = {}
        self.capture_active_mask_groups = set()
        self.capture_font_decoders = {}
        self.capture_font_companions = {}

        def font_provider(font: dict[str, Any], resources: dict[str, Any]) -> FontDecoder:
            return FontDecoder(
                font,
                ligature_overrides=detect_ligature_overrides(document, resources, font),
                raster_font_provider=getattr(document, "raster_font_provider", None),
                semantic_context=getattr(
                    document,
                    "font_semantic_context",
                    getattr(document.resolver, "semantic_context", None),
                ),
            )

        super().__init__(
            document.resolver,
            sink=self,
            font_provider=font_provider,
            lexer_factory=PdfLexer,
            semantic_context=getattr(document.resolver, "semantic_context", None),
        )

        self.recovery = CaptureRecovery()
        self.stream_executor = CaptureStreamExecutor(self)

        self.graphics.font_size = 12.0
        self.graphics.fill_color = (0.0, 0.0, 0.0)
        self.graphics.stroke_color = (0.0, 0.0, 0.0)

    @staticmethod
    def as_float(value: Any) -> float:
        parsed = parse_float_strict(value, "invalid numeric operand")
        if not isfinite(parsed):
            raise ValueError("invalid numeric operand")
        return parsed

    @staticmethod
    def as_int(value: Any) -> int:
        return parse_int_strict(value, "invalid numeric operand")


class FontResourceDocument(Protocol):
    def resolve(self, value: object, /) -> object: ...


def get_font_file(document: FontResourceDocument, font_obj: object) -> PdfStream | None:
    if not isinstance(font_obj, dict):
        return None
    descriptor = document.resolve(font_obj.get("FontDescriptor"))
    if not isinstance(descriptor, dict):
        return None
    font_file = document.resolve(descriptor.get("FontFile2"))
    return font_file if isinstance(font_file, PdfStream) else None


def load_ligature_font_tables(tt_data: bytes) -> TrueTypeFontProgram | None:
    try:
        return cached_truetype_program(tt_data)
    except ValueError:
        return None


def find_companion_font(
    document: FontResourceDocument,
    resources: object,
    base_name: str,
    ligature_starters: set[str],
) -> tuple[dict[int, float], dict[str, float], bytes | None]:
    if not isinstance(resources, dict):
        return {}, {}, None
    font_resources = document.resolve(resources.get("Font"))
    if not isinstance(font_resources, dict):
        return {}, {}, None

    for fref in font_resources.values():
        fobj = document.resolve(fref)
        if not isinstance(fobj, dict):
            continue
        comp_base = strip_subset_tag(recover_pdf_name(document.resolve(fobj.get("BaseFont"))) or "")
        if comp_base != base_name:
            continue

        fc = document.resolve(fobj.get("FirstChar"))
        lc = document.resolve(fobj.get("LastChar"))
        try:
            fc_int = parse_spec_int_strict(fc, "invalid font FirstChar")
            lc_int = parse_spec_int_strict(lc, "invalid font LastChar")
        except ValueError:
            continue
        if lc_int < fc_int:
            continue

        widths_raw = fobj.get("Widths")
        if widths_raw is None:
            continue
        widths_raw = document.resolve(widths_raw)
        if not isinstance(widths_raw, (list, tuple)):
            raise ValueError("invalid font widths array")

        starter_widths: dict[int, float] = {}
        starter_chars: dict[str, float] = {}
        for i, width_value in enumerate(widths_raw):
            try:
                width = parse_spec_float_strict(
                    document.resolve(width_value), "invalid font widths array"
                )
            except ValueError:
                continue
            if width <= 0:
                continue
            code = fc_int + i
            if code < 0 or code > 255:
                continue
            try:
                character = bytes([code]).decode("mac_roman")
            except UnicodeDecodeError:
                character = chr(code) if code < 128 else ""
            if character in ligature_starters:
                starter_widths[code] = width
                starter_chars[character] = width

        if starter_widths:
            font_file = get_font_file(document, fobj)
            if font_file is None:
                return starter_widths, starter_chars, None
            try:
                font_data = font_file.data
            except PdfParseError:
                font_data = None
            return starter_widths, starter_chars, font_data

    return {}, {}, None


def detect_ligature_overrides(
    document: FontResourceDocument,
    resources: object,
    font_obj: object,
) -> dict[int, str]:
    if not isinstance(font_obj, dict):
        return {}
    first_char = font_obj.get("FirstChar")
    last_char = font_obj.get("LastChar")
    try:
        first_char_int = parse_spec_int_strict(first_char, "invalid font FirstChar")
        last_char_int = parse_spec_int_strict(last_char, "invalid font LastChar")
    except ValueError:
        return {}

    base_name = strip_subset_tag(recover_pdf_name(font_obj.get("BaseFont")) or "")
    if not base_name:
        return {}

    font_file = get_font_file(document, font_obj)
    if font_file is None:
        return {}

    try:
        starter_widths, starter_chars, companion_data = find_companion_font(
            document, resources, base_name, set("ftscFTSC")
        )
    except ValueError:
        try:
            tt_data = font_file.data
        except PdfParseError:
            return {}
        if load_ligature_font_tables(tt_data) is None:
            return {}
        raise

    if companion_data is None or not starter_widths:
        return {}

    try:
        tt_data = font_file.data
    except PdfParseError:
        return {}
    parsed_primary = load_ligature_font_tables(tt_data)
    if parsed_primary is None:
        return {}

    try:
        cached_truetype_program(companion_data)
    except ValueError:
        return {}

    lig_widths_raw = font_obj.get("Widths")
    if lig_widths_raw is not None and not isinstance(lig_widths_raw, (list, tuple)):
        raise ValueError("invalid font widths array")
    overrides: dict[int, str] = {}

    for pdf_code in range(first_char_int, last_char_int + 1):
        if pdf_code < 0 or pdf_code > 255:
            continue
        try:
            codepoint = ord(bytes([pdf_code]).decode("mac_roman"))
        except UnicodeDecodeError:
            codepoint = pdf_code

        if codepoint in parsed_primary.unicode_cmap:
            continue

        glyph_id = pdf_code - first_char_int
        body_bbox, is_composite = parsed_primary.composite_body_bbox(glyph_id)
        if not (is_composite and body_bbox):
            continue

        ft_width = starter_chars.get("f", 0.0) + starter_chars.get("t", 0.0)
        if ft_width <= 0:
            continue

        lig_width = 0.0
        if lig_widths_raw is not None:
            width_index = pdf_code - first_char_int
            if 0 <= width_index < len(lig_widths_raw):
                width_value = lig_widths_raw[width_index]
                if type(width_value) in (int, float):
                    lig_width = float(cast(Any, width_value))

        if lig_width and 0.85 <= lig_width / ft_width <= 0.98:
            overrides[pdf_code] = "ft"

    return overrides


__all__ = (
    "FontResourceDocument",
    "detect_ligature_overrides",
    "find_companion_font",
    "get_font_file",
    "load_ligature_font_tables",
)


class CaptureStreamExecutor(ContentStreamExecutor):
    state: RecoveringTextState

    def queue(
        self,
        stream: PdfStream,
        resources: PdfDict,
        ctm: Matrix,
        depth: int,
        *,
        clip_bbox: Rectangle | None = None,
        form_bbox_operand: object = None,
        group_alpha: float | None = None,
        stream_key: StreamKey | None = None,
    ) -> ContentStreamFrame | None:
        if depth > 10 or (stream_key or self.execution_key(stream)) in self.active_streams:
            return None
        return super().queue(
            stream,
            resources,
            ctm,
            depth,
            clip_bbox=clip_bbox,
            form_bbox_operand=form_bbox_operand,
            group_alpha=group_alpha,
            stream_key=stream_key,
        )

    def enter(self, frame: ContentStreamFrame) -> bool:
        if (
            frame.depth > 10
            or (frame.stream_key or self.execution_key(frame.stream)) in self.active_streams
        ):
            return False
        return super().enter(frame)

    def dispatch_frame(self, frame: ContentStreamFrame) -> ContentStreamFrame | None:
        state = self.state
        assert frame.lexer is not None
        operator_names = frozenset(
            name.encode("latin-1") for name in (*state.default_handlers, *state.operator_overrides)
        )
        for name, operands in iter_content_operations(
            frame.lexer,
            recovery=state.recovery,
            is_operator=operator_names.__contains__,
        ):
            child = state.execute_operation(name, operands, frame.depth)
            if child is not None:
                return child
        return None

    def handle_parse_error(self, frame: ContentStreamFrame, error: PdfParseError) -> None:
        if not frame.is_form:
            raise error
