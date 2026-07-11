from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Iterator, TypeAlias, TypedDict, TypeGuard

from core_pdf.impl.engine.spec.s_07_content.interpreter import TextState
from core_pdf.impl.engine.spec.s_07_content.rendering import render_page_text
from core_pdf.impl.engine.spec.s_07_content.tables import TableExtractionResult, TableExtractor
from core_pdf.impl.engine.spec.s_07_content.traces import CapturedLine
from core_pdf.impl.engine.spec.s_07_document.models import AnnotationRecord, TextTraceSpan
from core_pdf.impl.engine.spec.s_07_objects.resolver import is_pdf_object
from core_pdf.impl.engine.spec.s_07_syntax.primitives import (
    MISSING,
    PdfDictLike,
    PdfObject,
    PdfStream,
    collect_inherited_values,
)
from core_pdf.impl.engine.spec.s_08_graphics.geometry import RectBox

if TYPE_CHECKING:
    from core_pdf.impl.engine.spec.s_07_content.models import TextRun
    from core_pdf.impl.engine.spec.s_07_document.document import PdfDocument
    from core_pdf.impl.engine.spec.s_07_document.models import FieldRecord
    from core_pdf.impl.engine.spec.s_07_document.redactions import (
        RedactionAnalysis,
        RedactionCandidate,
    )


INHERITED_PAGE_KEYS = (
    "MediaBox",
    "CropBox",
    "BleedBox",
    "TrimBox",
    "ArtBox",
    "Rotate",
    "Resources",
    "Annots",
)
PdfDict: TypeAlias = dict[str, PdfObject]
PageDict: TypeAlias = PdfDictLike
BoxValue: TypeAlias = tuple[float, float, float, float] | None
PageTableResult: TypeAlias = TableExtractionResult | tuple[list[list[str]], list[list[list[str]]]]


class PageDrawing(TypedDict):
    kind: str
    seqno: int
    fill: tuple[float, ...] | None
    fill_opacity: float | None
    stroke_color: tuple[float, ...] | None
    stroke_opacity: float | None
    line_width: float
    items: list[tuple[str, RectBox]]
    rect: RectBox | None


def is_box_value(value: object) -> TypeGuard[BoxValue]:
    return value is None or (
        isinstance(value, tuple)
        and len(value) == 4
        and all(isinstance(component, int | float) for component in value)
    )


def is_page_dict(value: object) -> TypeGuard[PageDict]:
    return isinstance(value, dict) and all(is_pdf_object(item) for item in value.values())


class PdfPage:
    __slots__ = (
        "document",
        "page_dict",
        "page_number",
        "_inherited_values",
        "contents",
        "_content_streams",
        "state",
        "graphics",
        "grid_lines",
        "texttrace",
        "tables",
        "_media_box",
        "_crop_box",
        "_bleed_box",
        "_trim_box",
        "_art_box",
        "_rotation",
        "_cached_resources",
        "text_cache",
    )

    document: PdfDocument
    page_dict: PageDict
    page_number: int
    _inherited_values: dict[str, PdfObject] | None
    contents: PdfObject
    _content_streams: tuple[PdfStream, ...] | None
    state: TextState | None
    graphics: TextState | None
    grid_lines: list[CapturedLine] | None
    texttrace: list[TextTraceSpan] | None
    tables: dict[tuple[str, bool, bool], PageTableResult]
    _media_box: BoxValue | object
    _crop_box: BoxValue | object
    _bleed_box: BoxValue | object
    _trim_box: BoxValue | object
    _art_box: BoxValue | object
    _rotation: int | object
    _cached_resources: PageDict | object

    def __init__(
        self,
        document: PdfDocument,
        page_dict: PageDict,
        page_number: int,
    ) -> None:
        self.document = document
        self.page_dict = page_dict
        self.page_number = page_number
        self._inherited_values = None
        self.contents = self.page_dict.get("Contents")
        self._content_streams = None
        self.state = None
        self.graphics = None
        self.grid_lines = None
        self.texttrace = None
        self.tables = {}
        self._media_box = MISSING
        self._crop_box = MISSING
        self._bleed_box = MISSING
        self._trim_box = MISSING
        self._art_box = MISSING
        self._rotation = MISSING
        self._cached_resources = MISSING
        self.text_cache: dict[bool, str] | None = None

    @property
    def inherited_values(self) -> dict[str, PdfObject]:
        if self._inherited_values is None:
            self._inherited_values = self.collect_inherited_values()
        return self._inherited_values

    @property
    def media_box(self) -> BoxValue:
        if self._media_box is MISSING:
            self._media_box = self.resolve_box("MediaBox")
        value = self._media_box
        if is_box_value(value):
            return value
        raise ValueError("page MediaBox not initialized")

    @property
    def crop_box(self) -> BoxValue:
        if self._crop_box is MISSING:
            self._crop_box = self.resolve_box("CropBox")
        value = self._crop_box
        if is_box_value(value):
            return value
        raise ValueError("page CropBox not initialized")

    @property
    def bleed_box(self) -> BoxValue:
        if self._bleed_box is MISSING:
            self._bleed_box = self.resolve_box("BleedBox")
        value = self._bleed_box
        if is_box_value(value):
            return value
        raise ValueError("page BleedBox not initialized")

    @property
    def trim_box(self) -> BoxValue:
        if self._trim_box is MISSING:
            self._trim_box = self.resolve_box("TrimBox")
        value = self._trim_box
        if is_box_value(value):
            return value
        raise ValueError("page TrimBox not initialized")

    @property
    def art_box(self) -> BoxValue:
        if self._art_box is MISSING:
            self._art_box = self.resolve_box("ArtBox")
        value = self._art_box
        if is_box_value(value):
            return value
        raise ValueError("page ArtBox not initialized")

    @property
    def rotation(self) -> int:
        if self._rotation is MISSING:
            self._rotation = self.resolve_rotation()
        value = self._rotation
        if isinstance(value, int):
            return value
        raise ValueError("page rotation not initialized")

    @property
    def cached_resources(self) -> PageDict:
        if self._cached_resources is MISSING:
            self._cached_resources = self.resolve_resources()
        value = self._cached_resources
        if is_page_dict(value):
            return value
        raise ValueError("page resources not initialized")

    @property
    def content_streams(self) -> tuple[PdfStream, ...]:
        if self._content_streams is None:
            self._content_streams = self.collect_content_streams()
        return self._content_streams

    def collect_inherited_values(self) -> dict[str, PdfObject]:
        return collect_inherited_values(
            self.page_dict,
            INHERITED_PAGE_KEYS,
            self.document.resolver.deep_resolve,
            self.document.inherited_values_cache,
        )

    def resolve_box(self, key: str) -> BoxValue:
        return self.document.resolver.resolve_box(self.inherited_values.get(key))

    def resolve_rotation(self) -> int:
        rotate_ref = self.inherited_values.get("Rotate")
        if rotate_ref is None:
            return 0
        rotate = self.document.resolver.resolve_int(rotate_ref)
        if rotate is None:
            raise ValueError("invalid page Rotate value")
        return rotate

    def resolve_resources(self) -> PageDict:
        resources = self.inherited_values.get("Resources")
        if resources is None:
            return {}
        resolved = self.document.resolver.resolve_dict(resources)
        if resolved is None:
            raise ValueError("invalid page Resources dictionary")
        return resolved

    def collect_content_streams(self) -> tuple[PdfStream, ...]:
        queue: deque[PdfObject] = deque()
        contents = self.contents
        if isinstance(contents, (list, tuple)):
            queue.extend(contents)
        elif contents is not None:
            if not isinstance(contents, (PdfStream, int)):
                raise ValueError("invalid page Contents value")
            queue.append(contents)
        streams: list[PdfStream] = []
        while queue:
            stream = self.document.resolver.resolve(queue.popleft())
            if isinstance(stream, (list, tuple)):
                queue.extendleft(reversed(stream))
                continue
            if isinstance(stream, PdfStream):
                streams.append(stream)
            elif stream is not None:
                raise ValueError("invalid page content stream")
        return tuple(streams)

    def iter_content_streams(self) -> Iterator[PdfStream]:
        yield from self.content_streams

    def consume_contents(self, state: TextState) -> None:
        if self.contents is None:
            return
        resources = self.cached_resources
        for stream in self.iter_content_streams():
            state.consume_stream(stream, resources, state.ctm, 0)

    def get_state(self) -> TextState:
        if self.state is None:
            state = TextState(
                self.document,
                self.page_dict,
                capture_graphics=False,
                hidden_layers=self.document.oc_hidden_layers(),
                decoder_cache=self.document.decoder_cache,
            )
            self.consume_contents(state)
            self.state = state
        return self.state

    def capture_texttrace_state(self) -> TextState:
        state = TextState(
            self.document,
            self.page_dict,
            capture_glyphs=True,
            capture_graphics=True,
            hidden_layers=self.document.oc_hidden_layers(),
            decoder_cache=self.document.decoder_cache,
        )
        self.consume_contents(state)
        return state

    def get_graphics(self) -> TextState:
        if self.graphics is None:
            state = TextState(
                self.document,
                self.page_dict,
                capture_graphics=True,
                hidden_layers=self.document.oc_hidden_layers(),
                decoder_cache=self.document.decoder_cache,
            )
            self.consume_contents(state)
            self.graphics = state
        return self.graphics

    def get_grid_lines(self) -> list[CapturedLine]:
        if self.grid_lines is None:
            self.grid_lines = self.get_graphics().lines
        return self.grid_lines

    @property
    def structure(self):
        structure = self.document.structure
        if structure is None:
            from core_pdf.impl.engine.spec.s_07_document.structure import PageStructure

            return PageStructure(self, [])
        return structure.page_structure(self)

    @property
    def width(self) -> float:
        mb = self.media_box
        if mb is not None:
            return mb[2] - mb[0]
        return 0.0

    @property
    def height(self) -> float:
        mb = self.media_box
        if mb is not None:
            return mb[3] - mb[1]
        return 0.0

    @property
    def chars(self) -> list[TextRun]:
        return self.get_state().runs

    @property
    def lines(self) -> list[CapturedLine]:
        return self.get_grid_lines()

    def get_drawings(self) -> list[PageDrawing]:
        graphics = self.get_graphics()
        return [
            {
                "kind": drawing.kind,
                "seqno": drawing.seqno,
                "fill": drawing.fill,
                "fill_opacity": drawing.fill_opacity,
                "stroke_color": drawing.stroke_color,
                "stroke_opacity": drawing.stroke_opacity,
                "line_width": drawing.line_width,
                "items": list(drawing.items),
                "rect": drawing.rect,
            }
            for drawing in graphics.drawings
        ]

    def get_texttrace(self) -> list[TextTraceSpan]:
        if self.texttrace is None:
            state = (
                self.state
                if self.state is not None and self.state.glyphs
                else self.capture_texttrace_state()
            )
            spans: dict[tuple[int, tuple[float, ...] | None, bool], TextTraceSpan] = {}
            for glyph in state.glyphs:
                key = (glyph.seqno, glyph.fill, glyph.visible)
                span = spans.get(key)
                if span is None:
                    span = TextTraceSpan(
                        seqno=glyph.seqno, color=glyph.fill, bbox=glyph.rect, chars=[]
                    )
                    spans[key] = span
                else:
                    rect = span["bbox"]
                    span["bbox"] = RectBox(
                        min(rect.x0, glyph.rect.x0),
                        min(rect.y0, glyph.rect.y0),
                        max(rect.x1, glyph.rect.x1),
                        max(rect.y1, glyph.rect.y1),
                        seqno=glyph.seqno,
                        fill=glyph.fill,
                        fill_opacity=None,
                    )
                span["chars"].append((ord(glyph.c), 0, 0, glyph.rect))
            self.texttrace = list(spans.values())
        return self.texttrace

    def crop(self, bbox: tuple[float, float, float, float]) -> PdfPage:
        """Return a version of the page cropped to the bounding box (x0, y0, x1, y1)."""
        x0, y0, x1, y1 = bbox
        new_page = PdfPage(self.document, self.page_dict, self.page_number)

        graphics = self.get_graphics()
        new_state = TextState(self.document, self.page_dict)
        new_state.runs = [
            r for r in graphics.runs if r.x1 > x0 and r.x0 < x1 and r.y1 > y0 and r.y0 < y1
        ]
        new_page.state = new_state

        grid_lines = self.get_grid_lines()
        new_page.grid_lines = [
            line
            for line in grid_lines
            if max(line.x0, line.x1) > x0
            and min(line.x0, line.x1) < x1
            and max(line.y0, line.y1) > y0
            and min(line.y0, line.y1) < y1
        ]
        return new_page

    def within_bbox(self, bbox: tuple[float, float, float, float]) -> PdfPage:
        """Return a page with objects entirely within the bbox (x0, y0, x1, y1)."""
        x0, y0, x1, y1 = bbox
        new_page = PdfPage(self.document, self.page_dict, self.page_number)

        graphics = self.get_graphics()
        new_state = TextState(self.document, self.page_dict)
        new_state.runs = [
            r for r in graphics.runs if r.x0 >= x0 and r.x1 <= x1 and r.y0 >= y0 and r.y1 <= y1
        ]
        new_page.state = new_state

        grid_lines = self.get_grid_lines()
        new_page.grid_lines = [
            line
            for line in grid_lines
            if min(line.x0, line.x1) >= x0
            and max(line.x0, line.x1) <= x1
            and min(line.y0, line.y1) >= y0
            and max(line.y0, line.y1) <= y1
        ]
        return new_page

    def extract_text(self, layout: bool = True) -> str:
        cache = self.text_cache
        if cache is None:
            self.text_cache = cache = {}
        if layout not in cache:
            cache[layout] = render_page_text(
                self.chars, rotate=self.rotation, media_box=self.media_box, layout=layout
            )
        return cache[layout]

    def to_markdown(self) -> str:
        """Extract page content as structured Markdown."""
        from core_pdf.impl.engine.spec.s_07_content.rendering import MarkdownRenderer

        return MarkdownRenderer.render_page(self)

    def get_annotations(self) -> list[AnnotationRecord]:
        """Return all annotations present on the page."""
        annots_raw = self.inherited_values.get("Annots")
        if annots_raw is None:
            return []
        if not isinstance(annots_raw, list):
            raise ValueError("invalid page Annots array")

        results = []
        for annot_ref in annots_raw:
            annot = self.document.resolver.resolve(annot_ref)
            if not isinstance(annot, dict):
                raise ValueError("invalid page annotation entry")

            subtype = self.document.resolver.resolve_name(annot.get("Subtype"))
            rect = self.document.resolver.resolve_box(annot.get("Rect"))
            if rect is None:
                raise ValueError("invalid page annotation rectangle")
            contents = self.document.resolver.resolve_str(annot.get("Contents")) or ""

            results.append(
                AnnotationRecord(
                    subtype=subtype,
                    rect=rect,
                    contents=contents,
                    dict_=annot,
                )
            )
        return results

    def get_fields(self) -> list[FieldRecord]:
        """Return all AcroForm fields that have a widget on this page."""
        all_fields = self.document.fields()
        page_fields = []
        # In PDF, a field's visual representation (Widget) might be on a specific page
        # or the field itself might be a Widget.
        for field in all_fields:
            if field.widget:
                if not isinstance(field.widget, dict):
                    raise ValueError("invalid field widget entry")
                pg_ref = field.widget.get("P")
                if pg_ref is not None:
                    pg_obj = self.document.resolver.resolve(pg_ref)
                    if pg_obj is self.page_dict:
                        page_fields.append(field)
            elif field.kids:
                # Check kids for widgets on this page
                if not isinstance(field.kids, list):
                    raise ValueError("invalid field kids array")
                for kid_ref in field.kids:
                    kid = self.document.resolver.resolve(kid_ref)
                    if (
                        isinstance(kid, dict)
                        and self.document.resolver.resolve_name(kid.get("Subtype")) == "Widget"
                    ):
                        pg_ref = kid.get("P")
                        if pg_ref is not None:
                            pg_obj = self.document.resolver.resolve(pg_ref)
                            if pg_obj is self.page_dict:
                                page_fields.append(field)
                                break
        return page_fields

    def find_text_near(
        self,
        target_box: tuple[float, float, float, float],
        direction: str = "left",
        distance: float = 100.0,
    ) -> list[TextRun]:
        """Find text runs near a target box in a given direction."""
        x0, y0, x1, y1 = target_box
        runs = self.chars

        candidates: list[tuple[float, TextRun]] = []
        mid_x = (x0 + x1) * 0.5
        mid_y = (y0 + y1) * 0.5

        for r in runs:
            if not r.text.strip():
                continue

            # Use mid-points for vertical/horizontal alignment checks
            rmid_x = r.mid_x
            rmid_y = r.mid_y

            dist = -1.0

            if direction == "left":
                # Must be to the left, and Y-aligned
                if r.x1 <= x0 and abs(rmid_y - mid_y) < max(r.height, y1 - y0, 10.0):
                    dist = x0 - r.x1
            elif direction == "right":
                if r.x0 >= x1 and abs(rmid_y - mid_y) < max(r.height, y1 - y0, 10.0):
                    dist = r.x0 - x1
            elif direction == "above":
                if r.y0 >= y1 and abs(rmid_x - mid_x) < max(r.x1 - r.x0, x1 - x0, 20.0):
                    dist = r.y0 - y1
            elif (
                direction == "below"
                and r.y1 <= y0
                and abs(rmid_x - mid_x) < max(r.x1 - r.x0, x1 - x0, 20.0)
            ):
                dist = y0 - r.y1

            if 0 <= dist <= distance:
                candidates.append((dist, r))

        # Sort by distance
        candidates.sort(key=lambda x: x[0])
        return [c[1] for c in candidates]

    def get_redaction_analysis(self) -> RedactionAnalysis:
        """Perform redaction analysis on the page."""
        from core_pdf.impl.engine.spec.s_07_document.redactions import RedactionAnalyzer

        return RedactionAnalyzer().analyze(self)

    def iter_redaction_candidates(self) -> Iterator["RedactionCandidate"]:
        """Yield redaction candidates identified on the page."""
        return iter(self.get_redaction_analysis().candidates)

    def extract_tables(
        self,
        flavor: str = "lattice",
        detect_header: bool = False,
        include_span_info: bool = False,
    ) -> PageTableResult:
        """Extract tables from the page."""
        cache_key = (flavor, detect_header, include_span_info)
        if cache_key in self.tables:
            return self.tables[cache_key]

        visible_runs = [r for r in self.chars if r.visible]
        if not visible_runs:
            result: PageTableResult = ([], []) if (include_span_info or detect_header) else []
            self.tables[cache_key] = result
            return result

        if flavor == "lattice":
            grid_lines = self.lines
            grid = TableExtractor.detect_grid(grid_lines) if grid_lines else None
        else:
            grid = TableExtractor.detect_stream_grid(visible_runs)

        if grid is not None and grid.is_valid():
            result = TableExtractor.extract_grid(visible_runs, grid, include_span_info)
        else:
            result = TableExtractor.extract_heuristic(
                visible_runs, detect_header, include_span_info
            )

        self.tables[cache_key] = result
        return result
