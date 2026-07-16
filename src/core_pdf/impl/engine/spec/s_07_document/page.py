# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, cast

from core_pdf.impl.engine.extraction.cache import ExtractionCache
from core_pdf.impl.engine.extraction.common.page_content import (
    PageContentHost,
    PageContentMixin,
)
from core_pdf.impl.engine.extraction.tables.types import (
    TableCacheKey,
    TableExtractionResult,
)
from core_pdf.impl.engine.spec.s_07_content import TextState
from core_pdf.impl.engine.spec.s_07_content.capture import CapturedLine
from core_pdf.impl.engine.spec.s_07_document.page_boxes import rotate_page_runs
from core_pdf.impl.engine.spec.s_07_document.page_interactions import (
    PageInteractionsMixin,
)
from core_pdf.impl.engine.spec.s_07_document.page_state import PageStateMixin
from core_pdf.impl.engine.spec.s_07_objects.object_cache import (
    CachedPdfObject,
    InheritedValueMap,
)
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import (
    collect_inherited_values,
    lookup_dict_key,
)
from core_pdf.impl.engine.spec.s_14_structure.tree import PageStructure
from core_pdf.impl.objects import (
    MISSING,
    MissingObject,
    PdfStream,
)
from core_pdf.impl.types import PdfDict, Rectangle

if TYPE_CHECKING:
    from core_pdf.impl.engine.layout.models import LayoutLine, TextRun
    from core_pdf.impl.engine.spec.s_07_document.document import PdfDocument
    from core_pdf.impl.models import LinkRecord, TextSpan


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

PageBoxCacheValue = Rectangle | None | MissingObject


class PdfPage(PageInteractionsMixin, PageStateMixin):
    __slots__ = (
        "document",
        "page_dict",
        "page_number",
        "inherited_values_cache",
        "contents",
        "content_streams_cache",
        "state",
        "graphics",
        "grid_lines",
        "text_spans",
        "links",
        "text_lines",
        "tables",
        "media_box_cache",
        "crop_box_cache",
        "bleed_box_cache",
        "trim_box_cache",
        "art_box_cache",
        "rotation_cache",
        "resources_cache",
        "extraction_cache",
    )

    document: PdfDocument
    page_dict: PdfDict
    page_number: int
    inherited_values_cache: InheritedValueMap | None
    contents: CachedPdfObject | None
    content_streams_cache: tuple[PdfStream, ...] | None
    state: TextState | None
    graphics: TextState | None
    grid_lines: list[CapturedLine] | None
    text_lines: list[LayoutLine] | None
    text_spans: list[TextSpan] | None
    links: list[LinkRecord] | MissingObject
    tables: dict[TableCacheKey, TableExtractionResult]
    media_box_cache: PageBoxCacheValue
    crop_box_cache: PageBoxCacheValue
    bleed_box_cache: PageBoxCacheValue
    trim_box_cache: PageBoxCacheValue
    art_box_cache: PageBoxCacheValue
    rotation_cache: int | MissingObject
    resources_cache: PdfDict | MissingObject
    extraction_cache: ExtractionCache | None

    def __init__(
        self,
        document: PdfDocument,
        page_dict: PdfDict,
        page_number: int,
    ) -> None:
        self.document = document
        self.page_dict = page_dict
        self.page_number = page_number
        self.inherited_values_cache = None
        self.contents = cast(CachedPdfObject | None, lookup_dict_key(self.page_dict, "Contents"))
        self.content_streams_cache = None
        self.state = None
        self.graphics = None
        self.grid_lines = None
        self.text_spans = None
        self.links = MISSING
        self.text_lines = None
        self.tables = {}
        self.media_box_cache = MISSING
        self.crop_box_cache = MISSING
        self.bleed_box_cache = MISSING
        self.trim_box_cache = MISSING
        self.art_box_cache = MISSING
        self.rotation_cache = MISSING
        self.resources_cache = MISSING
        page_caches = getattr(document, "page_extraction_caches", None)
        if page_caches is None:
            page_caches = {}
            with contextlib.suppress(AttributeError):
                document.page_extraction_caches = page_caches
        self.extraction_cache = page_caches.setdefault(page_number, ExtractionCache())

    @property
    def inherited_values(self) -> InheritedValueMap:
        if self.inherited_values_cache is None:
            self.inherited_values_cache = self.collect_inherited_values()
        return self.inherited_values_cache

    @property
    def media_box(self) -> tuple[float, float, float, float] | None:
        if self.media_box_cache is MISSING:
            self.media_box_cache = self.resolve_box("MediaBox")
        return cast(tuple[float, float, float, float] | None, self.media_box_cache)

    @property
    def crop_box(self) -> tuple[float, float, float, float] | None:
        if self.crop_box_cache is MISSING:
            self.crop_box_cache = self.resolve_box("CropBox")
        return cast(tuple[float, float, float, float] | None, self.crop_box_cache)

    @property
    def bleed_box(self) -> tuple[float, float, float, float] | None:
        if self.bleed_box_cache is MISSING:
            self.bleed_box_cache = self.resolve_box("BleedBox")
        return cast(tuple[float, float, float, float] | None, self.bleed_box_cache)

    @property
    def trim_box(self) -> tuple[float, float, float, float] | None:
        if self.trim_box_cache is MISSING:
            self.trim_box_cache = self.resolve_box("TrimBox")
        return cast(tuple[float, float, float, float] | None, self.trim_box_cache)

    def find_text_near(
        self,
        target_box: tuple[float, float, float, float],
        direction: str = "left",
        distance: float = 100.0,
    ) -> list[TextRun]:
        return PageContentMixin.find_text_near(
            cast(PageContentHost, self),
            target_box,
            direction,
            distance,
        )

    @property
    def art_box(self) -> tuple[float, float, float, float] | None:
        if self.art_box_cache is MISSING:
            self.art_box_cache = self.resolve_box("ArtBox")
        return cast(tuple[float, float, float, float] | None, self.art_box_cache)

    @property
    def rotation(self) -> int:
        if self.rotation_cache is MISSING:
            self.rotation_cache = self.resolve_rotation()
        return cast(int, self.rotation_cache)

    @property
    def label(self) -> str | None:
        return self.document.page_label(self.page_number - 1)

    @property
    def cached_resources(self) -> PdfDict:
        if self.resources_cache is MISSING:
            self.resources_cache = self.resolve_resources()
        return cast(PdfDict, self.resources_cache)

    @property
    def resources(self) -> PdfDict:
        return self.cached_resources

    @property
    def content_streams(self) -> tuple[PdfStream, ...]:
        if self.content_streams_cache is None:
            self.content_streams_cache = self.collect_content_streams()
        return self.content_streams_cache

    def collect_inherited_values(self) -> InheritedValueMap:
        return collect_inherited_values(
            self.page_dict,
            INHERITED_PAGE_KEYS,
            self.document.resolver.resolve,
            self.document.inherited_values_cache,
        )

    def resolve_box(self, key: str) -> tuple[float, float, float, float] | None:
        try:
            return self.document.resolver.resolve_box(lookup_dict_key(self.inherited_values, key))
        except ValueError:
            return None

    def resolve_rotation(self) -> int:
        rotate_ref = lookup_dict_key(self.inherited_values, "Rotate")
        if rotate_ref is None:
            return 0
        rotate = self.document.resolver.resolve_int(rotate_ref)
        if rotate is None:
            raise ValueError("invalid page Rotate value")
        rotate %= 360
        if rotate not in (0, 90, 180, 270):
            raise ValueError("invalid page Rotate value")
        return rotate

    def resolve_resources(self) -> PdfDict:
        resources = lookup_dict_key(self.inherited_values, "Resources")
        if resources is None:
            return {}
        if type(resources) is dict:
            return cast(PdfDict, resources)
        resolved = self.document.resolver.resolve_dict(resources)
        if resolved is None:
            return {}
        return resolved

    def resolve_transparency_group_alpha(self) -> float | None:
        group = self.document.resolver.resolve(lookup_dict_key(self.page_dict, "Group"))
        if not isinstance(group, dict):
            return None
        if self.document.resolver.resolve_name(lookup_dict_key(group, "S")) != "Transparency":
            return None
        ca = self.document.resolver.resolve_float(lookup_dict_key(group, "ca"), default=None)
        if ca is None:
            return None
        return max(0.0, min(1.0, ca))

    @property
    def structure(self):
        structure = self.document.structure
        if structure is None:
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
    def display_chars(self) -> list[TextRun]:
        """Text runs transformed into the page's displayed rotation frame."""
        return rotate_page_runs(
            self.chars,
            rotate=self.rotation,
            page_width=self.width,
            page_height=self.height,
        )

    @property
    def lines(self) -> list[CapturedLine]:
        return self.get_grid_lines()

    def crop(self, bbox: tuple[float, float, float, float]) -> PdfPage:
        x0, y0, x1, y1 = bbox
        new_page = self.__class__(self.document, self.page_dict, self.page_number)

        graphics = self.get_graphics()
        new_state = TextState(cast(Any, self.document), self.page_dict)
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
        x0, y0, x1, y1 = bbox
        new_page = self.__class__(self.document, self.page_dict, self.page_number)

        graphics = self.get_graphics()
        new_state = TextState(cast(Any, self.document), self.page_dict)
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
