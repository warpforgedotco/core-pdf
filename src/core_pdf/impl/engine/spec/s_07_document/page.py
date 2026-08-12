# SPDX-License-Identifier: AGPL-3.0-only
"""Spec-level page object: boxes, resources, annotations, and content streams."""

from __future__ import annotations

import contextlib
import threading
from collections import deque
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, cast

from core_pdf.impl.engine.cache import ExtractionCache
from core_pdf.impl.engine.spec.s_07_content.capture import CapturedLine
from core_pdf.impl.engine.spec.s_07_content.page_program import PageProducts, PageProgram
from core_pdf.impl.engine.spec.s_07_content.state import TextState
from core_pdf.impl.engine.spec.s_07_document.annotation_appearance import (
    consume_annotation_appearances,
)
from core_pdf.impl.engine.spec.s_07_document.document_lock import (
    document_cache_lock,
    document_recovery_enabled,
)
from core_pdf.impl.engine.spec.s_07_document.document_pages import PAGE_INHERITED_KEYS
from core_pdf.impl.engine.spec.s_07_document.page_boxes import rotate_page_runs
from core_pdf.impl.engine.spec.s_07_document.page_links import (
    link_target_direct,
    link_target_resolved,
    pdf_box_direct,
    pdf_name_direct,
    resolve_annotation_dict,
)
from core_pdf.impl.engine.spec.s_07_objects.object_cache import (
    CachedPdfObject,
    InheritedValueMap,
)
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import (
    collect_inherited_values,
    lookup_dict_key,
)
from core_pdf.impl.engine.spec.s_14_structure.tree import PageStructure
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.models import RawAnnotation, RawLink
from core_pdf.impl.objects import (
    MISSING,
    MissingObject,
    PdfReference,
    PdfStream,
)
from core_pdf.impl.types import PdfDict, PdfObject, Rectangle

if TYPE_CHECKING:
    from core_pdf.impl.engine.layout.models import LayoutLine, TextRun
    from core_pdf.impl.engine.spec.s_07_document.document import PdfDocument
    from core_pdf.impl.models import RawFormField, RawTextSpan

PageBoxCacheValue = Rectangle | None | MissingObject


class PdfPage:
    document: PdfDocument
    page_dict: PdfDict
    page_number: int
    inherited_values_cache: InheritedValueMap | None
    contents: CachedPdfObject | None
    content_streams_cache: tuple[PdfStream, ...] | None
    page_program_cache: PageProgram | None
    grid_lines: list[CapturedLine] | None
    text_lines: list[LayoutLine] | None
    text_spans: list[RawTextSpan] | None
    links: list[RawLink] | MissingObject
    page_box_cache: dict[str, PageBoxCacheValue]
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
        page_lock = getattr(document, "page_lock", None)
        self.internal_page_lock = (
            page_lock(page_number) if callable(page_lock) else threading.RLock()
        )
        self.inherited_values_cache = None
        self.contents = cast(CachedPdfObject | None, lookup_dict_key(self.page_dict, "Contents"))
        self.content_streams_cache = None
        self.page_program_cache = None
        self.grid_lines = None
        self.text_spans = None
        self.links = MISSING
        self.text_lines = None
        self.page_box_cache = {}
        self.rotation_cache = MISSING
        self.resources_cache = MISSING
        cache_lock = document_cache_lock(document)
        if cache_lock is None:
            page_caches = getattr(document, "page_extraction_caches", None)
            if page_caches is None:
                page_caches = {}
                with contextlib.suppress(AttributeError):
                    document.page_extraction_caches = page_caches
            self.extraction_cache = page_caches.setdefault(page_number, ExtractionCache())
        else:
            with cache_lock:
                page_caches = getattr(document, "page_extraction_caches", None)
                if page_caches is None:
                    page_caches = {}
                    document.page_extraction_caches = page_caches
                self.extraction_cache = page_caches.setdefault(page_number, ExtractionCache())

    @property
    def inherited_values(self) -> InheritedValueMap:
        if self.inherited_values_cache is None:
            self.inherited_values_cache = self.collect_inherited_values()
        return self.inherited_values_cache

    @property
    def media_box(self) -> tuple[float, float, float, float] | None:
        return self._cached_page_box("MediaBox")

    @property
    def crop_box(self) -> tuple[float, float, float, float] | None:
        return self._cached_page_box("CropBox")

    @property
    def bleed_box(self) -> tuple[float, float, float, float] | None:
        return self._cached_page_box("BleedBox")

    @property
    def trim_box(self) -> tuple[float, float, float, float] | None:
        return self._cached_page_box("TrimBox")

    def find_text_near(
        self,
        target_box: tuple[float, float, float, float],
        direction: str = "left",
        distance: float = 100.0,
    ) -> list[TextRun]:
        x0, y0, x1, y1 = target_box
        mid_x = (x0 + x1) * 0.5
        mid_y = (y0 + y1) * 0.5
        candidates: list[tuple[float, TextRun]] = []
        for run in self.chars:
            if not run.text.strip():
                continue
            delta = -1.0
            if direction == "left" and run.x1 <= x0:
                if abs(run.mid_y - mid_y) < max(run.height, y1 - y0, 10.0):
                    delta = x0 - run.x1
            elif direction == "right" and run.x0 >= x1:
                if abs(run.mid_y - mid_y) < max(run.height, y1 - y0, 10.0):
                    delta = run.x0 - x1
            elif direction == "above" and run.y0 >= y1:
                if abs(run.mid_x - mid_x) < max(run.x1 - run.x0, x1 - x0, 20.0):
                    delta = run.y0 - y1
            elif (
                direction == "below"
                and run.y1 <= y0
                and abs(run.mid_x - mid_x) < max(run.x1 - run.x0, x1 - x0, 20.0)
            ):
                delta = y0 - run.y1
            if 0.0 <= delta <= distance:
                candidates.append((delta, run))
        candidates.sort(key=lambda candidate: candidate[0])
        return [run for _, run in candidates]

    def has_annotation_subtype(self, subtype_name: str) -> bool:
        for annot in self.annotation_dicts():
            subtype = self.document.resolver.resolve_name(lookup_dict_key(annot, "Subtype"))
            if subtype == subtype_name:
                return True
        return False

    def annotation_dicts(self) -> list[PdfDict]:
        return self._annotation_dicts(strict=False)

    def _annotation_dicts(self, *, strict: bool) -> list[PdfDict]:
        recover_annotations = document_recovery_enabled(self.document)
        raw_annots = self.document.resolver.resolve(
            lookup_dict_key(self.inherited_values, "Annots")
        )
        if raw_annots is None:
            return []
        if not isinstance(raw_annots, list):
            if strict and not recover_annotations:
                raise ValueError("invalid page Annots array")
            if strict:
                return []
            annots = [raw_annots]
        else:
            annots = raw_annots
        resolved_annots: list[PdfDict] = []
        for annot_ref in annots:
            annot = resolve_annotation_dict(self.document.resolver, annot_ref)
            if annot is not None:
                resolved_annots.append(annot)
            elif strict and not recover_annotations:
                raise ValueError("invalid page annotation entry")
        return resolved_annots

    def has_destination_annotation(self) -> bool:
        for annot in self.annotation_dicts():
            if lookup_dict_key(annot, "Dest") is not None:
                return True
            action = lookup_dict_key(annot, "A")
            if isinstance(action, PdfReference):
                action = self.document.resolver.resolve(action)
            if not isinstance(action, dict):
                continue
            if self.document.resolver.resolve_name(lookup_dict_key(action, "S")) != "GoTo":
                continue
            if lookup_dict_key(action, "D") is not None:
                return True
        return False

    def get_annotations(self) -> list[RawAnnotation]:
        recover_annotations = document_recovery_enabled(self.document)
        results = []
        for annot in self._annotation_dicts(strict=True):
            subtype = self.document.resolver.resolve_name(lookup_dict_key(annot, "Subtype"))
            rect = self.document.resolver.resolve_box(lookup_dict_key(annot, "Rect"))
            if rect is None:
                if recover_annotations:
                    continue
                raise ValueError("invalid page annotation rectangle")
            contents = self.document.resolver.resolve_str(lookup_dict_key(annot, "Contents")) or ""
            dest = lookup_dict_key(annot, "Dest")
            action = lookup_dict_key(annot, "A")
            if isinstance(action, PdfReference):
                action = self.document.resolver.resolve(action)
            if (
                dest is None
                and isinstance(action, dict)
                and self.document.resolver.resolve_name(lookup_dict_key(action, "S")) == "GoTo"
            ):
                dest = lookup_dict_key(action, "D")

            results.append(
                RawAnnotation(
                    subtype=subtype,
                    rect=rect,
                    contents=contents,
                    dict_=annot,
                    dest=cast(PdfObject | None, dest),
                    action=cast(PdfDict, action) if isinstance(action, dict) else None,
                )
            )
        return results

    def get_links(self) -> list[RawLink]:
        if self.links is not MISSING:
            return cast(list[RawLink], self.links)

        annots = self._annotation_dicts(strict=False)
        if not annots:
            self.links = []
            return []

        resolver = self.document.resolver
        resolve = self.document.resolve
        records: list[RawLink] = []
        for annot in annots:
            subtype = pdf_name_direct(lookup_dict_key(annot, "Subtype"))
            if subtype is None:
                subtype = resolver.resolve_name(lookup_dict_key(annot, "Subtype"))
            if subtype != "Link":
                continue

            rect = pdf_box_direct(lookup_dict_key(annot, "Rect"))
            if rect is None:
                rect = resolver.resolve_box(lookup_dict_key(annot, "Rect"))
            if rect is None:
                continue

            action = lookup_dict_key(annot, "A")
            if isinstance(action, PdfReference):
                action = resolve(action)
            link_type = None
            url = None
            if isinstance(action, dict):
                action = cast(PdfDict, action)
                raw_type = lookup_dict_key(action, "S")
                link_type = pdf_name_direct(raw_type) or resolver.resolve_name(raw_type)
                url = link_target_direct(action, link_type)
                if url is None:
                    url = link_target_resolved(resolver, action, link_type)

            records.append(
                RawLink(
                    bbox=rect,
                    url=url,
                    link_type=link_type,
                    page_number=self.page_number,
                    dict_=annot,
                )
            )

        self.links = records
        return records

    def get_fields(self) -> list[RawFormField]:
        all_fields = self.document.fields()
        page_fields = []
        page_annot_ids = {id(annot) for annot in self.annotation_dicts()}

        for field in all_fields:
            if field.widget:
                if not isinstance(field.widget, dict):
                    raise ValueError("invalid field widget entry")
                pg_ref = lookup_dict_key(field.widget, "P")
                if pg_ref is not None:
                    pg_obj = self.document.resolver.resolve(pg_ref)
                    if (
                        isinstance(pg_obj, dict)
                        and self.document.page_index_for(pg_obj) == self.page_number - 1
                    ):
                        page_fields.append(field)
                elif id(field.widget) in page_annot_ids:
                    page_fields.append(field)
            elif field.kids:
                if not isinstance(field.kids, list):
                    raise ValueError("invalid field kids array")
                for kid_ref in field.kids:
                    kid = self.document.resolver.resolve(kid_ref)
                    if (
                        isinstance(kid, dict)
                        and self.document.resolver.resolve_name(lookup_dict_key(kid, "Subtype"))
                        == "Widget"
                    ):
                        pg_ref = lookup_dict_key(kid, "P")
                        if pg_ref is not None:
                            pg_obj = self.document.resolver.resolve(pg_ref)
                            if (
                                isinstance(pg_obj, dict)
                                and self.document.page_index_for(pg_obj) == self.page_number - 1
                            ):
                                page_fields.append(field)
                                break
                        elif id(kid) in page_annot_ids:
                            page_fields.append(field)
                            break
        return page_fields

    @property
    def art_box(self) -> tuple[float, float, float, float] | None:
        return self._cached_page_box("ArtBox")

    def _cached_page_box(self, key: str) -> tuple[float, float, float, float] | None:
        value = self.page_box_cache.get(key, MISSING)
        if value is MISSING:
            value = self.resolve_box(key)
            self.page_box_cache[key] = value
        return cast(tuple[float, float, float, float] | None, value)

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
    def content_streams(self) -> tuple[PdfStream, ...]:
        if self.content_streams_cache is None:
            self.content_streams_cache = self.collect_content_streams()
        return self.content_streams_cache

    def collect_content_streams(self) -> tuple[PdfStream, ...]:
        queue: deque[object] = deque()
        try:
            contents = self.document.resolver.resolve(self.contents)
        except PdfParseError:
            return ()
        if isinstance(contents, (list, tuple)):
            queue.extend(contents)
        elif contents is not None:
            queue.append(contents)
        streams: list[PdfStream] = []
        while queue:
            try:
                stream = self.document.resolver.resolve(queue.popleft())
            except PdfParseError:
                continue
            if isinstance(stream, (list, tuple)):
                queue.extendleft(reversed(stream))
                continue
            if isinstance(stream, PdfStream):
                streams.append(self.document.resolver.resolve_stream(stream))
        return tuple(streams)

    def iter_content_streams(self) -> Iterator[PdfStream]:
        yield from self.content_streams

    def consume_contents(self, state: TextState) -> None:
        if self.contents is None:
            return
        resources = self.cached_resources
        content_streams = self.content_streams
        try:
            contents_obj = self.document.resolver.resolve(self.contents)
        except PdfParseError:
            contents_obj = None
        can_skip_bad_stream = (
            len(content_streams) > 1
            or isinstance(contents_obj, (list, tuple))
            or document_recovery_enabled(self.document)
        )
        if len(content_streams) > 1:
            try:
                data = b"\n".join(stream.data for stream in content_streams)
                state.consume_stream(
                    PdfStream(raw_data=data, decoded_data=data), resources, state.ctm, 0
                )
                return
            except PdfParseError:
                if not can_skip_bad_stream:
                    raise

        for stream in content_streams:
            try:
                state.consume_stream(stream, resources, state.ctm, 0)
            except PdfParseError:
                if can_skip_bad_stream:
                    continue
                raise

    def get_page_program(self) -> PageProgram:
        """Interpret the page once and return its canonical program."""
        with self.internal_page_lock:
            program = self.page_program_cache
            if program is not None:
                return program
            state = TextState(
                cast(Any, self.document),
                self.page_dict,
                hidden_layers=self.document.oc_hidden_layers(),
                decoder_cache=self.document.decoder_cache,
                page_clip=self.effective_page_clip(),
            )
            self.consume_contents(state)
            consume_annotation_appearances(self, state)
            program = PageProgram.from_state(state)
            self.page_program_cache = program
            return program

    def get_grid_lines(self) -> list[CapturedLine]:
        if self.grid_lines is None:
            self.grid_lines = list(self.get_page_program().products.lines)
        return self.grid_lines

    def collect_inherited_values(self) -> InheritedValueMap:
        with document_cache_lock(self.document):
            return collect_inherited_values(
                self.page_dict,
                PAGE_INHERITED_KEYS,
                self.document.resolver.resolve,
                self.document.inherited_values_cache,
            )

    def resolve_box(self, key: str) -> tuple[float, float, float, float] | None:
        try:
            return self.document.resolver.resolve_box(lookup_dict_key(self.inherited_values, key))
        except ValueError:
            return None

    def effective_page_clip(self) -> tuple[float, float, float, float] | None:
        """Return the region the page actually displays.

        7.7.3.3, Table 30 defines CropBox as the visible region of user space,
        whose contents "shall be clipped (cropped) to this rectangle", and
        defaults it to MediaBox. 14.11.2.1 adds that a crop box extending past
        the media box "[is] effectively reduced to [its] intersection with the
        media box", so the displayed region is the intersection of the two.

        A page missing both boxes is malformed; leave it unclipped rather than
        discard all of its content.
        """
        media = self.resolve_box("MediaBox")
        crop = self.resolve_box("CropBox")
        if crop is None:
            return media
        if media is None:
            return crop
        x0 = max(min(crop[0], crop[2]), min(media[0], media[2]))
        y0 = max(min(crop[1], crop[3]), min(media[1], media[3]))
        x1 = min(max(crop[0], crop[2]), max(media[0], media[2]))
        y1 = min(max(crop[1], crop[3]), max(media[1], media[3]))
        if x0 >= x1 or y0 >= y1:
            return media
        return (x0, y0, x1, y1)

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
    def structure(self) -> PageStructure:
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
        return list(self.get_page_program().products.runs)

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
        return self.internal_derive_page(
            bbox,
            run_predicate=lambda r: r.x1 > x0 and r.x0 < x1 and r.y1 > y0 and r.y0 < y1,
            line_predicate=lambda line: (
                max(line.x0, line.x1) > x0
                and min(line.x0, line.x1) < x1
                and max(line.y0, line.y1) > y0
                and min(line.y0, line.y1) < y1
            ),
        )

    def within_bbox(self, bbox: tuple[float, float, float, float]) -> PdfPage:
        x0, y0, x1, y1 = bbox
        return self.internal_derive_page(
            bbox,
            run_predicate=lambda r: r.x0 >= x0 and r.x1 <= x1 and r.y0 >= y0 and r.y1 <= y1,
            line_predicate=lambda line: (
                min(line.x0, line.x1) >= x0
                and max(line.x0, line.x1) <= x1
                and min(line.y0, line.y1) >= y0
                and max(line.y0, line.y1) <= y1
            ),
        )

    def internal_derive_page(
        self,
        bbox: tuple[float, float, float, float],
        *,
        run_predicate: Callable[[Any], bool],
        line_predicate: Callable[[Any], bool],
    ) -> PdfPage:
        x0, y0, x1, y1 = bbox
        new_page = self.__class__(self.document, self.page_dict, self.page_number)

        graphics = self.get_page_program().products
        products = graphics
        runs = tuple(r for r in graphics.runs if run_predicate(r))
        new_page.page_program_cache = self.internal_filtered_products(
            products, runs, x0, y0, x1, y1
        )

        grid_lines = self.get_grid_lines()
        new_page.grid_lines = [line for line in grid_lines if line_predicate(line)]
        return new_page

    @staticmethod
    def internal_filtered_products(
        products: PageProducts,
        runs: tuple[Any, ...],
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> Any:
        from core_pdf.impl.engine.spec.s_07_content.page_program import (
            PageProgram,
        )

        filtered = PageProducts(
            runs,
            products.glyphs,
            products.drawings,
            products.inline_images,
            products.lines,
        )
        return PageProgram(filtered)
