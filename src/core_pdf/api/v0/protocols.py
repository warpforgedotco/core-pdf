"""Structural protocols for extending core-pdf locally."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from os import PathLike
from pathlib import Path
from typing import Any, Protocol, TypeAlias

from .errors import OperationCancelled
from .models import (
    AccessibilityInventory,
    AccessibilityRepairVerification,
    ActionInventory,
    AnnotationInventory,
    AnnotationRecord,
    ArchivalManifest,
    AttachmentInfo,
    ChunkRecord,
    ContentDependencyRecord,
    ContentEvent,
    DocumentAnalysisSnapshot,
    DocumentFingerprint,
    DocumentInventory,
    Drawing,
    EmbeddedResourceRecord,
    EvidenceGraph,
    FormFieldRecord,
    FormInventory,
    GeometryIssue,
    GeometrySummary,
    ImageInfo,
    IncrementalAnalysisPlan,
    LinkRecord,
    NativeFeatureInventory,
    ObjectGraphReport,
    ObjectInspection,
    ObjectRoundTripManifest,
    ObjectRoundTripVerification,
    OptionalContentLayerRecord,
    OutlineItem,
    PageContentSummary,
    PageInfo,
    PageResourceInventory,
    PreservationManifest,
    Raster,
    ReadingOrderItem,
    Rect,
    RedactionVerification,
    ResourceDependencyGraph,
    ResourceDiagnostic,
    RevisionInventory,
    RevisionObjectRecord,
    SanitizationVerification,
    SearchHit,
    StructureElementRecord,
    TableRecord,
    TextBlock,
    TextCharacter,
    TextDiagnosticRun,
    TextLine,
    TextSpan,
    TextWord,
)
from .structured import Document as StructuredDocument
from .structured import Page as StructuredPage

PageSelection: TypeAlias = int | str | range | Iterable[int]


class ReadableSource(Protocol):
    def read(self, size: int = -1, /) -> bytes | bytearray | memoryview: ...


PdfInput: TypeAlias = str | PathLike[str] | bytes | bytearray | memoryview | ReadableSource


class CancellationToken(Protocol):
    @property
    def cancelled(self) -> bool: ...

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise OperationCancelled


class ExecutionContext(Protocol):
    cancellation: CancellationToken


class PdfObjectResolverProtocol(Protocol):
    def resolve(self, value: object) -> object: ...


class PdfPageProtocol(Protocol):
    @property
    def structured_view(self) -> StructuredPage: ...

    @property
    def info(self) -> PageInfo: ...

    def text(self) -> str: ...

    def text_lines(self) -> Iterable[TextLine]: ...

    def text_blocks(self) -> Iterable[TextBlock]: ...

    def content_events(self) -> Iterable[ContentEvent]: ...

    def text_spans(self) -> Iterable[TextSpan]: ...

    def text_characters(self) -> Iterable[TextCharacter]: ...

    def text_diagnostics(
        self, *, include_invisible: bool = True
    ) -> Iterable[TextDiagnosticRun]: ...

    def words(self) -> Iterable[TextWord]: ...

    def drawings(self) -> Iterable[Drawing]: ...

    def tables(self) -> Iterable[TableRecord]: ...

    def geometry_issues(self) -> Iterable[GeometryIssue]: ...

    def geometry_summary(self) -> GeometrySummary: ...

    def annotations(self) -> Iterable[AnnotationRecord]: ...

    def links(self) -> Iterable[LinkRecord]: ...

    def form_fields(self) -> Iterable[FormFieldRecord]: ...

    def images(self) -> Iterable[ImageInfo]: ...

    def render(
        self,
        *,
        dpi: float = 72.0,
        crop: tuple[float, float, float, float] | None = None,
    ) -> Raster: ...

    def to_markdown(self) -> str: ...

    def to_json(self, *, indent: int | None = 2, sort_keys: bool = True) -> str: ...

    def to_html(self) -> str: ...


class PdfDocumentProtocol(AbstractContextManager["PdfDocumentProtocol"], Protocol):
    @property
    def structured_document(self) -> StructuredDocument: ...

    @property
    def structured_pages(self) -> Sequence[StructuredPage]: ...

    @property
    def closed(self) -> bool: ...

    def close(self) -> None: ...

    @property
    def page_count(self) -> int: ...

    @property
    def resolver(self) -> PdfObjectResolverProtocol: ...

    def page(self, index: int) -> PdfPageProtocol: ...

    def pages(self, selection: PageSelection | None = None) -> Iterable[PdfPageProtocol]: ...

    def get_metadata(self) -> Mapping[str, object]: ...

    def inventory(self) -> DocumentInventory: ...

    def analysis_snapshot(self) -> DocumentAnalysisSnapshot: ...

    def native_features(self) -> NativeFeatureInventory: ...

    def action_inventory(self) -> ActionInventory: ...

    def optional_content_layers(self) -> Iterable[OptionalContentLayerRecord]: ...

    def revisions(self) -> RevisionInventory: ...

    def revision_objects(self) -> Iterable[RevisionObjectRecord]: ...

    def fingerprint(self) -> DocumentFingerprint: ...

    def incremental_plan(self, baseline: DocumentFingerprint) -> IncrementalAnalysisPlan: ...

    def archival_manifest(self) -> ArchivalManifest: ...

    def object_graph(self) -> ObjectGraphReport: ...

    def inspect_object(self, object_number: int) -> ObjectInspection: ...

    def verify_object_roundtrip(self, object_number: int) -> ObjectRoundTripVerification: ...

    def verify_object_roundtrips(
        self, object_numbers: Iterable[int]
    ) -> ObjectRoundTripManifest: ...

    def text(self, *, pages: PageSelection | None = None) -> str: ...

    def words(self, *, pages: PageSelection | None = None) -> Iterable[TextWord]: ...

    def text_spans(self, *, pages: PageSelection | None = None) -> Iterable[TextSpan]: ...

    def text_characters(self, *, pages: PageSelection | None = None) -> Iterable[TextCharacter]: ...

    def text_lines(self, *, pages: PageSelection | None = None) -> Iterable[TextLine]: ...

    def reading_order(
        self, *, pages: PageSelection | None = None
    ) -> Iterable[ReadingOrderItem]: ...

    def text_blocks(self, *, pages: PageSelection | None = None) -> Iterable[TextBlock]: ...

    def text_diagnostics(
        self, *, include_invisible: bool = True, pages: PageSelection | None = None
    ) -> Iterable[TextDiagnosticRun]: ...

    def search(
        self,
        query: str,
        *,
        mode: str = "exact",
        threshold: float = 0.8,
        region: Rect | None = None,
        pages: PageSelection | None = None,
    ) -> Iterable[SearchHit]: ...

    def elements(self, *, pages: PageSelection | None = None) -> Iterable[Mapping[str, object]]: ...

    def chunks(
        self, *, max_characters: int = 2000, pages: PageSelection | None = None
    ) -> Iterable[ChunkRecord]: ...

    def images(self, *, pages: PageSelection | None = None) -> Iterable[ImageInfo]: ...

    def resource_inventory(
        self, *, pages: PageSelection | None = None
    ) -> Iterable[PageResourceInventory]: ...

    def content_dependencies(
        self, *, pages: PageSelection | None = None
    ) -> Iterable[ContentDependencyRecord]: ...

    def resource_dependency_graph(self) -> ResourceDependencyGraph: ...

    def evidence_graph(self, *, pages: PageSelection | None = None) -> EvidenceGraph: ...

    def resource_diagnostics(self) -> Iterable[ResourceDiagnostic]: ...

    def content_summaries(
        self, *, pages: PageSelection | None = None
    ) -> Iterable[PageContentSummary]: ...

    def form_inventory(self, *, pages: PageSelection | None = None) -> FormInventory: ...

    def annotation_inventory(
        self, *, pages: PageSelection | None = None
    ) -> AnnotationInventory: ...

    def annotations(self, *, pages: PageSelection | None = None) -> Iterable[AnnotationRecord]: ...

    def links(self, *, pages: PageSelection | None = None) -> Iterable[LinkRecord]: ...

    def form_fields(self, *, pages: PageSelection | None = None) -> Iterable[FormFieldRecord]: ...

    def tables(self, *, pages: PageSelection | None = None) -> Iterable[TableRecord]: ...

    def drawings(self, *, pages: PageSelection | None = None) -> Iterable[Drawing]: ...

    def geometry_issues(self, *, pages: PageSelection | None = None) -> Iterable[GeometryIssue]: ...

    def geometry_summaries(
        self, *, pages: PageSelection | None = None
    ) -> Iterable[GeometrySummary]: ...

    def structure_elements(self) -> Iterable[StructureElementRecord]: ...

    def accessibility_inventory(
        self, *, pages: PageSelection | None = None
    ) -> AccessibilityInventory: ...

    def to_json(self, *, indent: int | None = 2, sort_keys: bool = True) -> str: ...

    def to_structured_json(
        self,
        *,
        pages: PageSelection | None = None,
        indent: int | None = 2,
        sort_keys: bool = True,
    ) -> str: ...

    def to_html(self) -> str: ...

    def to_markdown(self) -> str: ...

    def to_csv(self, *, pages: PageSelection | None = None) -> str: ...

    def to_tei(self, *, pages: PageSelection | None = None) -> str: ...

    @property
    def outlines(self) -> Iterable[OutlineItem]: ...

    @property
    def attachments(self) -> Iterable[AttachmentInfo]: ...

    def embedded_resources(self) -> Iterable[EmbeddedResourceRecord]: ...

    def edit(self) -> "PdfEditorProtocol": ...


class PdfEditorProtocol(Protocol):
    def encrypt(
        self, user_password: str, *, owner_password: str | None = None
    ) -> "PdfEditorProtocol": ...

    def sign(
        self, provider: "SignatureProvider", *, contents_length: int = 8192
    ) -> "PdfEditorProtocol": ...

    def set_metadata(self, values: Mapping[str, object]) -> "PdfEditorProtocol": ...

    def set_page_geometry(
        self,
        page_number: int,
        *,
        rotation: int | None = None,
        cropbox: tuple[float, float, float, float] | None = None,
    ) -> "PdfEditorProtocol": ...

    def replace_page(self, page_number: int, page: StructuredPage) -> "PdfEditorProtocol": ...

    def insert_page(
        self, position: int, width: float = 595.0, height: float = 842.0
    ) -> "PdfEditorProtocol": ...

    def insert_structured_page(
        self, position: int, page: StructuredPage
    ) -> "PdfEditorProtocol": ...

    def update_form_field(self, name: str, value: str) -> "PdfEditorProtocol": ...

    def remove_form_fields(self, names: Iterable[str]) -> "PdfEditorProtocol": ...

    def apply_redactions(
        self, redactions: Mapping[int, Iterable[tuple[float, float, float, float]]]
    ) -> "PdfEditorProtocol": ...

    def remove_annotations(
        self, page_number: int, indices: Iterable[int] | None = None
    ) -> "PdfEditorProtocol": ...

    def remove_links(
        self, page_number: int, indices: Iterable[int] | None = None
    ) -> "PdfEditorProtocol": ...

    def delete_pages(self, selection: PageSelection) -> "PdfEditorProtocol": ...

    def set_attachments(self, values: Mapping[str, bytes]) -> "PdfEditorProtocol": ...

    def set_outlines(self, values: Iterable[Iterable[object]]) -> "PdfEditorProtocol": ...

    def replace_pages(self, pages: Iterable[StructuredPage]) -> "PdfEditorProtocol": ...

    def add_annotation(
        self,
        page_number: int,
        subtype: str,
        bbox: tuple[float, float, float, float] | None = None,
        *,
        contents: str = "",
        destination: object = None,
    ) -> "PdfEditorProtocol": ...

    def add_link(
        self,
        page_number: int,
        bbox: tuple[float, float, float, float] | None = None,
        *,
        url: str | None = None,
        link_type: str | None = None,
        text: str = "",
    ) -> "PdfEditorProtocol": ...

    def commit(self, target: str | Path | Any) -> bytes: ...

    def commit_verified(
        self, target: str | Path | Any, *, expected_unchanged_pages: tuple[int, ...] = ()
    ) -> PreservationManifest: ...

    def commit_redactions_verified(
        self,
        target: str | Path | Any,
        redactions: Mapping[int, Iterable[tuple[float, float, float, float]]],
        *,
        queries: Iterable[str] = (),
    ) -> RedactionVerification: ...

    def commit_sanitized_verified(
        self,
        target: str | Path | Any,
        *,
        metadata: bool = True,
        annotations: bool = True,
        links: bool = True,
        forms: bool = True,
        attachments: bool = True,
        outlines: bool = True,
        actions: bool = True,
    ) -> SanitizationVerification: ...

    def commit_accessibility_repair_verified(
        self,
        target: str | Path | Any,
        *,
        title: str | None = None,
        language: str | None = None,
    ) -> AccessibilityRepairVerification: ...

    def commit_document(self) -> StructuredDocument: ...

    def rollback(self) -> None: ...


class SignatureProvider(Protocol):
    def sign(self, data: bytes) -> bytes: ...


__all__ = (
    "CancellationToken",
    "ExecutionContext",
    "PageSelection",
    "PdfDocumentProtocol",
    "PdfEditorProtocol",
    "PdfInput",
    "PdfObjectResolverProtocol",
    "PdfPageProtocol",
    "ReadableSource",
    "SignatureProvider",
)
