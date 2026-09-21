# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

from core_pdf.impl._impl.capture.interpreter import TextState
from core_pdf.impl._impl.capture.page import (
    capture_page_program,
)
from core_pdf.impl._impl.capture.program import PageProgram
from core_pdf.impl._impl.document.page_links import (
    link_target_direct,
    link_target_resolved,
    resolve_annotation_dict,
)
from core_pdf.impl._impl.document.records import RawAnnotation, RawLink
from core_pdf.impl._impl.document.recovery.resources import resolve_resource_dict
from core_pdf.impl._impl.document.structure import PageStructure
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.types import PdfReference
from core_pdf_spec.s_07_document.page import page_clip, page_rotation, page_user_unit
from core_pdf_spec.s_07_syntax.inherited_values import collect_inherited_values
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import (
    CachedPdfObject,
    InheritedValueMap,
    PdfDict,
    PdfObject,
)
from core_pdf_spec.s_07_syntax_primitives.coercion import parse_box

PAGE_INHERITED_KEYS = (
    "MediaBox",
    "CropBox",
    "BleedBox",
    "TrimBox",
    "ArtBox",
    "Rotate",
    "Resources",
    "Annots",
)


if TYPE_CHECKING:
    from core_pdf.impl._impl.document.document import PdfDocument
    from core_pdf.impl._impl.document.records import RawFormField
    from core_pdf.impl._impl.model.runs import TextRun


class PdfPage:
    document: PdfDocument
    page_dict: PdfDict
    page_number: int
    contents: CachedPdfObject | None
    inherited_values: InheritedValueMap

    def __init__(
        self,
        document: PdfDocument,
        page_dict: PdfDict,
        page_number: int,
        *,
        inherited_values: InheritedValueMap | None = None,
    ) -> None:
        self.document = document
        self.page_dict = page_dict
        self.page_number = page_number
        self.contents = cast(CachedPdfObject | None, self.page_dict.get("Contents"))
        self.inherited_values = (
            self.collect_inherited_values() if inherited_values is None else dict(inherited_values)
        )

    @property
    def media_box(self) -> tuple[float, float, float, float] | None:
        return self.resolve_box("MediaBox")

    @property
    def crop_box(self) -> tuple[float, float, float, float] | None:
        return self.resolve_box("CropBox")

    @property
    def bleed_box(self) -> tuple[float, float, float, float] | None:
        return self.resolve_box("BleedBox")

    @property
    def trim_box(self) -> tuple[float, float, float, float] | None:
        return self.resolve_box("TrimBox")

    def annotation_dicts(self) -> list[PdfDict]:
        return self._annotation_dicts(strict=False)

    def _annotation_dicts(self, *, strict: bool) -> list[PdfDict]:
        recover_annotations = self.document.recovery_enabled
        raw_annots = self.document.resolver.resolve(self.inherited_values.get("Annots"))
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

    def get_annotations(self) -> list[RawAnnotation]:
        recover_annotations = self.document.recovery_enabled
        results = []
        for annot in self._annotation_dicts(strict=True):
            subtype = self.document.resolver.resolve_name(annot.get("Subtype"))
            try:
                rect = self.document.resolver.resolve_box(annot.get("Rect"))
            except ValueError:
                if recover_annotations:
                    continue
                raise ValueError("invalid page annotation rectangle") from None
            if rect is None:
                if recover_annotations:
                    continue
                raise ValueError("invalid page annotation rectangle")
            contents = self.document.resolver.resolve_str(annot.get("Contents")) or ""
            dest = annot.get("Dest")
            action: object = annot.get("A")
            if isinstance(action, PdfReference):
                action = self.document.resolver.resolve(action)
            if (
                dest is None
                and isinstance(action, dict)
                and self.document.resolver.resolve_name(action.get("S")) == "GoTo"
            ):
                dest = action.get("D")

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

    def get_links(self, annots: Iterable[PdfDict] | None = None) -> list[RawLink]:
        annots = self._annotation_dicts(strict=False) if annots is None else annots
        if not annots:
            return []

        resolver = self.document.resolver
        resolve = self.document.resolve
        records: list[RawLink] = []
        for annot in annots:
            subtype = resolver.resolve_name(annot.get("Subtype"))
            if subtype != "Link":
                continue

            rect = parse_box(annot.get("Rect"))
            if rect is None:
                rect = resolver.resolve_box(annot.get("Rect"))
            if rect is None:
                continue

            action: object = annot.get("A")
            if isinstance(action, PdfReference):
                action = resolve(action)
            link_type = None
            url = None
            if isinstance(action, dict):
                action = cast(PdfDict, action)
                raw_type = action.get("S")
                link_type = resolver.resolve_name(raw_type)
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

        return records

    def get_fields(self) -> list[RawFormField]:
        return list(self.document.fields_by_page().get(self.page_number - 1, ()))

    @property
    def art_box(self) -> tuple[float, float, float, float] | None:
        return self.resolve_box("ArtBox")

    @property
    def rotation(self) -> int:
        rotate_ref = self.inherited_values.get("Rotate")
        if rotate_ref is None:
            return 0
        rotate = self.document.resolver.resolve_int(rotate_ref)
        if rotate is None:
            raise ValueError("invalid page Rotate value")
        return page_rotation(rotate)

    @property
    def user_unit(self) -> float:
        try:
            return page_user_unit(self.document.resolver.resolve(self.page_dict.get("UserUnit")))
        except ValueError, PdfParseError:
            if not self.document.recovery_enabled:
                raise
            return 1.0

    @property
    def label(self) -> str | None:
        return self.document.page_label(self.page_number - 1)

    @property
    def resources(self) -> PdfDict:
        return (
            resolve_resource_dict(self.inherited_values.get("Resources"), self.document.resolver)
            or {}
        )

    @property
    def content_streams(self) -> tuple[PdfStream, ...]:
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

    def consume_contents(self, state: TextState) -> None:
        if self.contents is None:
            return
        resources = self.resources
        content_streams = self.content_streams
        try:
            contents_obj = self.document.resolver.resolve(self.contents)
        except PdfParseError:
            contents_obj = None
        can_skip_bad_stream = (
            len(content_streams) > 1
            or (isinstance(contents_obj, (list, tuple)) and len(contents_obj) > 1)
            or self.document.recovery_enabled
        )
        if len(content_streams) > 1:
            try:
                data = b"\n".join(stream.data for stream in content_streams)
                state.stream_executor.consume(
                    PdfStream(raw_data=data, spec=None), resources, state.graphics.ctm, 0
                )
                return
            except PdfParseError:
                if not can_skip_bad_stream:
                    raise

        for stream in content_streams:
            try:
                state.stream_executor.consume(stream, resources, state.graphics.ctm, 0)
            except PdfParseError:
                if can_skip_bad_stream:
                    continue
                raise

    def get_page_program(
        self,
        *,
        hidden_layers: frozenset[str] | None = None,
        fields: Iterable[RawFormField] | None = None,
        annotations: Iterable[RawAnnotation] | None = None,
    ) -> PageProgram:
        return capture_page_program(
            self, hidden_layers=hidden_layers, fields=fields, annotations=annotations
        )

    def collect_inherited_values(self) -> InheritedValueMap:
        return collect_inherited_values(
            self.page_dict,
            PAGE_INHERITED_KEYS,
            self.document.resolver.resolve,
            stop_at_malformed_parent=True,
        )

    def resolve_box(self, key: str) -> tuple[float, float, float, float] | None:
        try:
            return self.document.resolver.resolve_box(self.inherited_values.get(key))
        except ValueError:
            return None

    def effective_page_clip(self) -> tuple[float, float, float, float] | None:
        media = self.resolve_box("MediaBox")
        crop = self.resolve_box("CropBox")
        if crop is None:
            return media
        if media is None:
            return crop
        clip = page_clip(media, crop)
        return media if clip[0] >= clip[2] or clip[1] >= clip[3] else clip

    def resolve_transparency_group_alpha(self) -> float | None:
        group = self.document.resolver.resolve(self.page_dict.get("Group"))
        if not isinstance(group, dict):
            return None
        if self.document.resolver.resolve_name(group.get("S")) != "Transparency":
            return None
        ca = self.document.resolver.resolve_float(group.get("ca"), default=None)
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
    def width_points(self) -> float:
        return self.width * self.user_unit

    @property
    def height_points(self) -> float:
        return self.height * self.user_unit

    @property
    def chars(self) -> list[TextRun]:
        return list(self.get_page_program().runs)
