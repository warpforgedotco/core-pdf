"""Immutable data exchanged by the public core-pdf protocols."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, TypeAlias, TypeVar, cast

Point: TypeAlias = tuple[float, float]
Color: TypeAlias = tuple[float, ...]

_T = TypeVar("_T")


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class SourceRef:
    """A stable pointer from an output record to source PDF evidence."""

    page_index: int | None = None
    page_number: int | None = None
    page_label: str | None = None
    object_number: int | None = None
    generation: int | None = None
    sequence: int | None = None
    stage: str | None = None


def freeze_mappings(*names: str) -> Callable[[type[_T]], type[_T]]:
    """Class decorator wrapping the named mapping fields in ``MappingProxyType``.

    Apply below ``@dataclass`` so the generated ``__init__`` calls the injected
    ``__post_init__``.
    """

    def wrap(cls: type[_T]) -> type[_T]:
        previous = getattr(cls, "__post_init__", None)

        def __post_init__(self: Any) -> None:
            for name in names:
                object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
            if previous is not None:
                previous(self)

        cast(Any, cls).__post_init__ = __post_init__
        return cls

    return wrap


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, SourceRef):
        return {
            spec.name: item
            for spec in fields(SourceRef)
            if (item := getattr(value, spec.name)) is not None
        }
    if isinstance(value, Rect):
        return {"x0": value.x0, "y0": value.y0, "x1": value.x1, "y1": value.y1}
    if is_dataclass(value) and not isinstance(value, type):
        return {spec.name: _to_jsonable(getattr(value, spec.name)) for spec in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


class JsonSerializable:
    """Fields-driven ``to_dict``/``to_json`` for frozen dataclass records."""

    __slots__ = ()

    def to_dict(self) -> dict[str, object]:
        specs = fields(cast(Any, self))
        return {spec.name: _to_jsonable(getattr(self, spec.name)) for spec in specs}

    def to_json(self, *, indent: int | None = 2, sort_keys: bool = True) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=sort_keys)


class CoordinateOrigin(StrEnum):
    TOP_LEFT = "top-left"
    BOTTOM_LEFT = "bottom-left"


@dataclass(frozen=True, slots=True)
class CoordinateSpace:
    """Describes the coordinate system attached to geometry records."""

    name: str
    origin: CoordinateOrigin
    width: float | None = None
    height: float | None = None
    unit: str = "point"


@dataclass(frozen=True, slots=True)
class Rect:
    """An axis-aligned rectangle with an explicit coordinate space."""

    x0: float
    y0: float
    x1: float
    y1: float
    space: CoordinateSpace

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        """Return the non-negative area of this rectangle."""
        return max(0.0, self.width) * max(0.0, self.height)

    def intersection(self, other: Rect) -> Rect | None:
        """Return the positive-area intersection with a rectangle in the same space."""
        if self.space != other.space:
            raise ValueError("rectangles use different coordinate spaces")
        x0 = max(self.x0, other.x0)
        y0 = max(self.y0, other.y0)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return Rect(x0, y0, x1, y1, self.space)

    def overlap_ratio_min(self, other: Rect) -> float:
        """Return intersection area relative to the smaller positive rectangle."""
        intersection = self.intersection(other)
        smaller_area = min(self.area, other.area)
        return (
            intersection.area / smaller_area if intersection is not None and smaller_area else 0.0
        )

    def to_origin(self, origin: CoordinateOrigin) -> Rect:
        """Return this rectangle expressed with the requested vertical origin."""
        if origin is self.space.origin:
            return self
        if self.space.height is None:
            raise ValueError("coordinate space has no height for origin conversion")
        height = self.space.height
        return Rect(
            self.x0,
            height - self.y1,
            self.x1,
            height - self.y0,
            replace(self.space, origin=origin),
        )


@dataclass(frozen=True, slots=True)
class GeometryIssue:
    code: str
    severity: Severity
    subject: str
    bbox: Rect | None = None
    message: str = ""
    details: Mapping[str, object] = field(default_factory=dict)
    repairable: bool = False


@dataclass(frozen=True, slots=True)
class GeometrySummary:
    issue_count: int
    error_count: int
    warning_count: int
    repairable_count: int
    text_run_count: int
    line_count: int
    issue_codes: tuple[tuple[str, int], ...]
    suspicion_score: float

    @property
    def has_issues(self) -> bool:
        return self.issue_count > 0

    @property
    def has_repairable_issues(self) -> bool:
        return self.repairable_count > 0


class EvidenceLayer(StrEnum):
    PDF_OBJECT = "pdf-object"
    CONTENT_EVENT = "content-event"
    NATIVE_TEXT = "native-text"
    OCR = "ocr"
    STRUCTURED = "structured"
    RASTER = "raster"


@dataclass(frozen=True, slots=True)
@freeze_mappings("attributes")
class EvidenceRecord:
    layer: EvidenceLayer
    value: str
    source: SourceRef = field(default_factory=SourceRef)
    bbox: Rect | None = None
    confidence: float | None = None
    attributes: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvidenceNode:
    """One deterministic node in the document evidence graph."""

    node_id: str
    layer: EvidenceLayer
    value: str
    source: SourceRef = field(default_factory=SourceRef)
    bbox: Rect | None = None


@dataclass(frozen=True, slots=True)
class EvidenceEdge:
    """A typed relationship between two evidence nodes."""

    source_id: str
    target_id: str
    relation: str


@dataclass(frozen=True, slots=True)
class EvidenceGraph(JsonSerializable):
    """Immutable evidence graph joining page and content-layer observations."""

    nodes: tuple[EvidenceNode, ...] = ()
    edges: tuple[EvidenceEdge, ...] = ()


@dataclass(frozen=True, slots=True)
class RemediationAction:
    """A concrete review or transformation step derived from local evidence."""

    code: str
    priority: str
    message: str
    page_number: int | None = None
    source: SourceRef = field(default_factory=SourceRef)


@dataclass(frozen=True, slots=True)
class AnalysisFinding:
    code: str
    severity: Severity
    message: str
    evidence: tuple[EvidenceRecord, ...] = ()
    page_number: int | None = None
    bbox: Rect | None = None
    confidence: float | None = None
    remediation: str | None = None


@dataclass(frozen=True, slots=True)
@freeze_mappings("metrics")
class AnalysisReport:
    analyzer_id: str
    version: str
    findings: tuple[AnalysisFinding, ...] = ()
    metrics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CompliancePreflightSummary:
    """Stable typed roll-up of a local PDF/A or PDF/UA preflight run."""

    profile: str
    passed: bool
    error_count: int = 0
    warning_count: int = 0
    finding_codes: tuple[str, ...] = ()
    has_resource_errors: bool = False
    has_font_errors: bool = False
    has_color_warnings: bool = False
    has_transparency_warnings: bool = False


@dataclass(frozen=True, slots=True)
class DocumentInventory:
    """Deterministic structural and security inventory of a PDF document."""

    byte_count: int
    object_count: int
    page_count: int
    encrypted: bool
    xref_recovered: bool
    page_tree_recovered: bool
    has_attachments: bool
    has_outlines: bool
    has_javascript: bool
    has_open_action: bool
    trailer_keys: tuple[str, ...] = ()
    object_types: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class NativeFeatureInventory:
    """PDF-native feature markers useful for preservation and security review."""

    has_optional_content: bool = False
    has_acroform: bool = False
    has_xfa: bool = False
    has_metadata_stream: bool = False
    has_names: bool = False
    has_collection: bool = False
    has_embedded_files: bool = False
    has_incremental_revision: bool = False
    is_linearized: bool = False


@dataclass(frozen=True, slots=True)
@freeze_mappings("action_type_counts")
class ActionInventory:
    """Resolved PDF action markers found during local object inspection."""

    action_count: int = 0
    javascript_count: int = 0
    open_action_count: int = 0
    additional_action_count: int = 0
    launch_count: int = 0
    action_type_counts: Mapping[str, int] = field(default_factory=dict)
    source_objects: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocumentAnalysisSnapshot(JsonSerializable):
    """Single local snapshot joining the principal deterministic inventories."""

    inventory: DocumentInventory
    native_features: NativeFeatureInventory
    actions: ActionInventory
    accessibility: AccessibilityInventory
    forms: FormInventory
    annotations: AnnotationInventory
    revisions: RevisionInventory
    revision_objects: tuple[RevisionObjectRecord, ...] = ()
    embedded_resources: tuple[EmbeddedResourceRecord, ...] = ()
    content_summaries: tuple[PageContentSummary, ...] = ()
    resource_diagnostics: tuple[ResourceDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class OptionalContentLayerRecord:
    """Configured optional-content group state."""

    name: str
    enabled: bool


@dataclass(frozen=True, slots=True)
class RevisionInventory:
    """Raw revision markers discovered in a PDF byte stream."""

    revision_count: int
    startxref_offsets: tuple[int, ...] = ()
    eof_offsets: tuple[int, ...] = ()
    revision_ranges: tuple[tuple[int, int], ...] = ()
    revision_sha256: tuple[str, ...] = ()
    has_incremental_updates: bool = False


@dataclass(frozen=True, slots=True)
class RevisionObjectRecord:
    """Raw object provenance within one incremental revision range."""

    revision_index: int
    object_number: int
    generation: int
    offset: int
    byte_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class RevisionObjectDiff:
    """Object-level changes between two revision snapshots."""

    added: tuple[tuple[int, int], ...] = ()
    removed: tuple[tuple[int, int], ...] = ()
    changed: tuple[tuple[int, int], ...] = ()


def _set_diff(
    before: Iterable[Any], after: Iterable[Any], *, key: Callable[[Any], Any] | None = None
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Return (added, removed) items between two before/after collections."""
    before_set = set(before)
    after_set = set(after)
    return (
        tuple(sorted(after_set - before_set, key=key)),
        tuple(sorted(before_set - after_set, key=key)),
    )


def _diff_keys(
    before: Mapping[Any, Any], after: Mapping[Any, Any]
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    """Return (added, removed, changed) keys between two before/after mappings."""
    added, removed = _set_diff(before, after)
    changed = tuple(
        sorted(key for key in before.keys() & after.keys() if before[key] != after[key])
    )
    return added, removed, changed


def _union_sorted(*groups: Iterable[Any]) -> tuple[Any, ...]:
    """Return the sorted union of several key collections."""
    merged: set[Any] = set()
    for group in groups:
        merged.update(group)
    return tuple(sorted(merged))


def compare_revision_objects(
    before: Iterable[RevisionObjectRecord], after: Iterable[RevisionObjectRecord]
) -> RevisionObjectDiff:
    before_map = {(record.object_number, record.generation): record.sha256 for record in before}
    after_map = {(record.object_number, record.generation): record.sha256 for record in after}
    added, removed, changed = _diff_keys(before_map, after_map)
    return RevisionObjectDiff(added=added, removed=removed, changed=changed)


@dataclass(frozen=True, slots=True)
class DocumentFingerprint:
    """Stable content fingerprints for incremental local analysis."""

    document_sha256: str
    page_sha256: tuple[tuple[int, str], ...] = ()
    object_sha256: tuple[tuple[int, int, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ArchivalManifest(JsonSerializable):
    """Serializable local evidence summary for preservation workflows."""

    document_sha256: str
    byte_count: int
    page_count: int
    object_count: int
    reachable_object_count: int
    unreachable_object_count: int
    encrypted: bool
    has_javascript: bool
    has_attachments: bool
    title: str | None = None


@dataclass(frozen=True, slots=True)
class ArchivalManifestDiff:
    """Field-level changes between two archival manifests."""

    changed_fields: tuple[str, ...] = ()
    document_changed: bool = False


def compare_archival_manifests(
    before: ArchivalManifest, after: ArchivalManifest
) -> ArchivalManifestDiff:
    before_values = before.to_dict()
    after_values = after.to_dict()
    changed = tuple(key for key in before_values if before_values[key] != after_values[key])
    return ArchivalManifestDiff(
        changed_fields=changed,
        document_changed=before.document_sha256 != after.document_sha256,
    )


@dataclass(frozen=True, slots=True)
class FingerprintDiff:
    """Changed units between two document fingerprints."""

    document_changed: bool
    changed_pages: tuple[int, ...] = ()
    added_objects: tuple[tuple[int, int], ...] = ()
    removed_objects: tuple[tuple[int, int], ...] = ()
    changed_objects: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class IncrementalAnalysisPlan:
    """Work selection derived from a fingerprint diff."""

    affected_pages: tuple[int, ...] = ()
    affected_objects: tuple[tuple[int, int], ...] = ()
    revision_objects: tuple[tuple[int, int], ...] = ()
    full_document_scan: bool = False


@dataclass(frozen=True, slots=True)
class PreservationManifest:
    """Post-write verification of content expected to remain unchanged."""

    expected_unchanged_pages: tuple[int, ...] = ()
    changed_pages: tuple[int, ...] = ()
    changed_objects: tuple[tuple[int, int], ...] = ()
    violated_pages: tuple[int, ...] = ()
    passed: bool = True


@dataclass(frozen=True, slots=True)
class RedactionVerification:
    """Proof that requested redaction queries are absent after reopening output."""

    requested_queries: tuple[str, ...] = ()
    remaining_queries: tuple[str, ...] = ()
    remaining_raw_queries: tuple[str, ...] = ()
    changed_pages: tuple[int, ...] = ()
    became_unreachable_objects: tuple[int, ...] = ()
    removed_objects: tuple[int, ...] = ()
    passed: bool = True


@dataclass(frozen=True, slots=True)
class SanitizationVerification:
    """Post-write report for requested local PDF sanitization steps."""

    removed_annotations: bool = False
    removed_links: bool = False
    removed_forms: bool = False
    removed_attachments: bool = False
    removed_outlines: bool = False
    removed_actions: bool = False
    remaining_annotations: int = 0
    remaining_links: int = 0
    remaining_forms: int = 0
    remaining_attachments: int = 0
    remaining_outlines: bool = False
    remaining_actions: int = 0
    became_unreachable_objects: tuple[int, ...] = ()
    removed_objects: tuple[int, ...] = ()
    passed: bool = True


@dataclass(frozen=True, slots=True)
class AccessibilityRepairVerification:
    """Verification of metadata-only accessibility repairs."""

    requested_title: str | None = None
    requested_language: str | None = None
    has_title: bool = False
    language: str | None = None
    passed: bool = True


def verify_preservation(
    before: DocumentFingerprint,
    after: DocumentFingerprint,
    *,
    expected_unchanged_pages: tuple[int, ...] = (),
) -> PreservationManifest:
    """Verify that explicitly protected pages did not change after a rewrite."""
    diff = compare_fingerprints(before, after)
    expected = tuple(sorted(set(expected_unchanged_pages)))
    changed = set(diff.changed_pages)
    violated = tuple(page for page in expected if page in changed)
    return PreservationManifest(
        expected_unchanged_pages=expected,
        changed_pages=diff.changed_pages,
        changed_objects=diff.changed_objects,
        violated_pages=violated,
        passed=not violated,
    )


def plan_incremental_analysis(diff: FingerprintDiff) -> IncrementalAnalysisPlan:
    """Select the smallest safe local analysis scope for a fingerprint diff."""
    affected_objects = _union_sorted(diff.added_objects, diff.removed_objects, diff.changed_objects)
    return IncrementalAnalysisPlan(
        affected_pages=diff.changed_pages,
        affected_objects=affected_objects,
        full_document_scan=diff.document_changed
        and not diff.changed_pages
        and not affected_objects,
    )


def plan_revision_analysis(diff: RevisionObjectDiff) -> IncrementalAnalysisPlan:
    """Select changed object units directly from an incremental revision diff."""
    objects = _union_sorted(diff.added, diff.removed, diff.changed)
    return IncrementalAnalysisPlan(
        affected_objects=objects,
        revision_objects=objects,
        full_document_scan=not objects,
    )


def compare_fingerprints(
    before: DocumentFingerprint, after: DocumentFingerprint
) -> FingerprintDiff:
    """Compare fingerprints without opening either source document."""
    before_pages = dict(before.page_sha256)
    after_pages = dict(after.page_sha256)
    changed_pages = tuple(
        page_number
        for page_number in sorted(before_pages.keys() | after_pages.keys())
        if before_pages.get(page_number) != after_pages.get(page_number)
    )
    before_objects = {
        (number, generation): digest for number, generation, digest in before.object_sha256
    }
    after_objects = {
        (number, generation): digest for number, generation, digest in after.object_sha256
    }
    added_objects, removed_objects, changed_objects = _diff_keys(before_objects, after_objects)
    return FingerprintDiff(
        document_changed=before.document_sha256 != after.document_sha256,
        changed_pages=changed_pages,
        added_objects=added_objects,
        removed_objects=removed_objects,
        changed_objects=changed_objects,
    )


@dataclass(frozen=True, slots=True)
class ObjectGraphNode:
    object_number: int
    generation: int
    object_type: str
    reachable: bool


@dataclass(frozen=True, slots=True)
class ObjectInspection(JsonSerializable):
    """JSON-safe summary of one resolved PDF object."""

    object_number: int
    generation: int
    object_type: str
    dictionary_keys: tuple[str, ...] = ()
    value_repr: str = ""
    is_stream: bool = False
    raw_offset: int | None = None
    raw_length: int | None = None
    raw_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ObjectRoundTripVerification:
    """Proof that an object's original source bytes survive a local reopen."""

    object_number: int
    source_sha256: str | None
    reopened_sha256: str | None
    passed: bool


@dataclass(frozen=True, slots=True)
class ObjectRoundTripManifest:
    """Verification results for a deterministic set of PDF objects."""

    objects: tuple[ObjectRoundTripVerification, ...] = ()

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.objects)


@dataclass(frozen=True, slots=True)
class ObjectGraphEdge:
    source_object: int
    target_object: int
    key: str | None = None


@dataclass(frozen=True, slots=True)
class ObjectGraphReport(JsonSerializable):
    nodes: tuple[ObjectGraphNode, ...]
    edges: tuple[ObjectGraphEdge, ...]
    root_objects: tuple[int, ...] = ()

    @property
    def unreachable_objects(self) -> tuple[int, ...]:
        return tuple(node.object_number for node in self.nodes if not node.reachable)

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        payload["unreachable_objects"] = list(self.unreachable_objects)
        return payload


@dataclass(frozen=True, slots=True)
class ObjectReachabilityDiff:
    """Deterministic before/after reachability and edge changes."""

    became_reachable: tuple[int, ...] = ()
    became_unreachable: tuple[int, ...] = ()
    added_objects: tuple[int, ...] = ()
    removed_objects: tuple[int, ...] = ()
    added_edges: tuple[ObjectGraphEdge, ...] = ()
    removed_edges: tuple[ObjectGraphEdge, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(
            self.became_reachable
            or self.became_unreachable
            or self.added_objects
            or self.removed_objects
            or self.added_edges
            or self.removed_edges
        )


def compare_object_graphs(
    before: ObjectGraphReport, after: ObjectGraphReport
) -> ObjectReachabilityDiff:
    before_reachable = {node.object_number for node in before.nodes if node.reachable}
    after_reachable = {node.object_number for node in after.nodes if node.reachable}
    before_objects = {node.object_number for node in before.nodes}
    after_objects = {node.object_number for node in after.nodes}
    edge_key: Callable[[ObjectGraphEdge], Any] = lambda edge: (  # noqa: E731
        edge.source_object,
        edge.target_object,
        edge.key or "",
    )
    became_reachable, became_unreachable = _set_diff(before_reachable, after_reachable)
    added_objects, removed_objects = _set_diff(before_objects, after_objects)
    added_edges, removed_edges = _set_diff(before.edges, after.edges, key=edge_key)
    return ObjectReachabilityDiff(
        became_reachable=became_reachable,
        became_unreachable=became_unreachable,
        added_objects=added_objects,
        removed_objects=removed_objects,
        added_edges=added_edges,
        removed_edges=removed_edges,
    )


@dataclass(frozen=True, slots=True)
@freeze_mappings("metadata")
class ChunkRecord:
    """A bounded retrieval unit retaining its structural provenance."""

    text: str
    page_numbers: tuple[int, ...] = ()
    element_ids: tuple[str, ...] = ()
    element_types: tuple[str, ...] = ()
    section_path: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    sources: tuple[SourceRef, ...] = ()
    element_bboxes: tuple[Rect, ...] = ()


@dataclass(frozen=True, slots=True)
class PageInfo:
    index: int
    number: int
    label: str | None
    width: float
    height: float
    rotation: int
    space: CoordinateSpace


@dataclass(frozen=True, slots=True)
class TextCharacter:
    text: str
    bbox: Rect
    font_name: str | None = None
    font_size: float | None = None
    color: Color | None = None
    sequence: int | None = None
    visible: bool = True
    source: SourceRef = field(default_factory=SourceRef)
    rotation_angle: int = 0


@dataclass(frozen=True, slots=True)
class TextSpan:
    text: str
    bbox: Rect
    characters: tuple[TextCharacter, ...] = ()
    font_name: str | None = None
    font_size: float | None = None
    color: Color | None = None
    sequence: int | None = None
    source: SourceRef = field(default_factory=SourceRef)


@dataclass(frozen=True, slots=True)
class TextDiagnosticRun:
    text: str
    bbox: Rect
    font_name: str | None
    font_size: float
    is_vertical: bool
    visible: bool
    rotation: int
    sequence: int
    geometry_issues: tuple[GeometryIssue, ...] = ()


@dataclass(frozen=True, slots=True)
class TextLine:
    text: str
    bbox: Rect | None = None
    characters: tuple[TextCharacter, ...] = ()
    source: SourceRef = field(default_factory=SourceRef)


@dataclass(frozen=True, slots=True)
class ReadingOrderItem:
    """One text line in deterministic visual reading order."""

    page_number: int
    order: int
    text: str
    bbox: Rect | None = None
    source: SourceRef = field(default_factory=SourceRef)


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str
    bbox: Rect | None = None
    lines: tuple[TextLine, ...] = ()
    source: SourceRef = field(default_factory=SourceRef)


@dataclass(frozen=True, slots=True)
class TextWord:
    text: str
    bbox: Rect | None = None
    block_index: int | None = None
    line_index: int | None = None
    word_index: int | None = None
    source: SourceRef = field(default_factory=SourceRef)


@dataclass(frozen=True, slots=True)
class SearchHit:
    page_number: int
    bbox: Rect
    text: str = ""
    score: float = 1.0
    mode: str = "exact"
    source: SourceRef = field(default_factory=SourceRef)


@dataclass(frozen=True, slots=True)
class TableCell:
    text: str
    bbox: Rect | None = None
    row: int = 0
    column: int = 0


@dataclass(frozen=True, slots=True)
class TableRecord:
    bbox: Rect | None
    cells: tuple[TableCell, ...] = ()
    rows: int = 0
    columns: int = 0
    source: SourceRef = field(default_factory=SourceRef)


@dataclass(frozen=True, slots=True)
class AnnotationRecord:
    subtype: str
    bbox: Rect | None = None
    contents: str = ""
    destination: str | None = None
    source: SourceRef = field(default_factory=SourceRef)


@dataclass(frozen=True, slots=True)
class LinkRecord:
    bbox: Rect | None
    url: str | None = None
    link_type: str = "uri"
    source: SourceRef = field(default_factory=SourceRef)


@dataclass(frozen=True, slots=True)
class FormFieldRecord:
    name: str
    field_type: str
    value: str = ""
    bbox: Rect | None = None
    field_index: int | None = None
    source: SourceRef = field(default_factory=SourceRef)
    required: bool = False
    read_only: bool = False
    no_export: bool = False
    options: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
@freeze_mappings("field_type_counts")
class FormInventory:
    """Document-level inventory of locally parsed interactive form fields."""

    field_count: int = 0
    populated_count: int = 0
    empty_count: int = 0
    page_count: int = 0
    field_type_counts: Mapping[str, int] = field(default_factory=dict)
    duplicate_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
@freeze_mappings("subtype_counts")
class AnnotationInventory:
    """Document-level inventory of annotations and links."""

    annotation_count: int = 0
    link_count: int = 0
    page_count: int = 0
    subtype_counts: Mapping[str, int] = field(default_factory=dict)
    external_link_count: int = 0


@dataclass(frozen=True, slots=True)
class OutlineItem:
    title: str
    page_number: int | None = None
    level: int = 1
    uri: str | None = None


@dataclass(frozen=True, slots=True)
class AttachmentInfo:
    name: str
    size: int
    media_type: str | None = None
    data: bytes | memoryview | None = None


@dataclass(frozen=True, slots=True)
class EmbeddedResourceRecord:
    """Stable inventory record for one embedded file payload."""

    name: str
    filename: str
    byte_count: int
    sha256: str
    media_type: str | None = None


@dataclass(frozen=True, slots=True)
@freeze_mappings("attributes")
class StructureElementRecord:
    """A typed projection of one tagged-PDF logical structure element."""

    role: str
    depth: int
    page_number: int | None = None
    title: str | None = None
    language: str | None = None
    alternate_description: str | None = None
    actual_text: str | None = None
    attributes: Mapping[str, object] = field(default_factory=dict)
    source: SourceRef = field(default_factory=SourceRef)


@dataclass(frozen=True, slots=True)
class AccessibilityInventory:
    """Document-level accessibility coverage signals."""

    tagged_element_count: int = 0
    language_element_count: int = 0
    alternate_text_count: int = 0
    figure_count: int = 0
    image_count: int = 0
    table_count: int = 0
    missing_alternate_text_count: int = 0
    document_language: str | None = None
    has_title: bool = False


@dataclass(frozen=True, slots=True)
@freeze_mappings("data")
class DrawingItem:
    kind: str
    bbox: Rect | None = None
    data: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
@freeze_mappings("data")
class Drawing:
    kind: str
    bbox: Rect | None
    sequence: int
    items: tuple[DrawingItem, ...] = ()
    fill: Color | None = None
    stroke: Color | None = None
    fill_opacity: float | None = None
    stroke_opacity: float | None = None
    source: SourceRef = field(default_factory=SourceRef)
    data: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ImageInfo:
    bbox: Rect | None
    width: int
    height: int
    channels: int
    color_model: str
    alpha: bool
    sequence: int | None = None
    source: SourceRef = field(default_factory=SourceRef)
    data: bytes | memoryview | None = None


@dataclass(frozen=True, slots=True)
class PageResourceInventory:
    """High-level resources observed while parsing one page."""

    page_number: int
    image_count: int = 0
    font_names: tuple[str, ...] = ()
    has_images: bool = False
    has_fonts: bool = False


@dataclass(frozen=True, slots=True)
class Raster:
    data: bytes | memoryview
    width: int
    height: int
    channels: int
    pixel_format: str
    space: CoordinateSpace
    dpi: float | None = None


class ContentEventKind(StrEnum):
    TEXT = "text"
    DRAWING = "drawing"
    IMAGE = "image"
    STATE = "state"
    CLIP = "clip"
    MARKED_CONTENT = "marked-content"


@dataclass(frozen=True, slots=True)
@freeze_mappings("data")
class ContentEvent:
    kind: ContentEventKind
    sequence: int
    bbox: Rect | None = None
    text: TextSpan | None = None
    drawing: Drawing | None = None
    image: ImageInfo | None = None
    source: SourceRef = field(default_factory=SourceRef)
    data: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContentDependencyRecord:
    """One content event's page/resource dependency summary."""

    page_number: int
    sequence: int
    kind: ContentEventKind
    resource_names: tuple[str, ...] = ()
    source: SourceRef = field(default_factory=SourceRef)


@dataclass(frozen=True, slots=True)
class ResourceDependencyGraph:
    """Page-to-resource dependencies observed in content events."""

    page_resources: tuple[tuple[int, tuple[str, ...]], ...] = ()

    @property
    def resources(self) -> tuple[str, ...]:
        return tuple(
            sorted({resource for _, resources in self.page_resources for resource in resources})
        )


@dataclass(frozen=True, slots=True)
class ResourceDiagnostic:
    """A decode, recovery, or availability issue affecting a PDF resource."""

    code: str
    message: str
    severity: Severity = Severity.WARNING
    page_number: int | None = None
    resource_name: str | None = None


@dataclass(frozen=True, slots=True)
@freeze_mappings("operator_counts")
class PageContentSummary:
    """Preflight summary of one page's content streams and operators."""

    page_number: int
    operator_counts: Mapping[str, int] = field(default_factory=dict)
    stream_filters: tuple[str, ...] = ()
    raw_stream_bytes: int = 0
    decoded_stream_bytes: int = 0
    malformed_operator_count: int = 0
    classification: str | None = None


__all__ = (
    "Color",
    "ContentEvent",
    "ContentDependencyRecord",
    "ResourceDependencyGraph",
    "ResourceDiagnostic",
    "PageContentSummary",
    "ContentEventKind",
    "CoordinateOrigin",
    "CoordinateSpace",
    "Drawing",
    "DrawingItem",
    "ImageInfo",
    "PageResourceInventory",
    "PageInfo",
    "Point",
    "Raster",
    "Rect",
    "SourceRef",
    "TextCharacter",
    "TextLine",
    "ReadingOrderItem",
    "TextBlock",
    "TextWord",
    "SearchHit",
    "TextSpan",
    "TableCell",
    "TableRecord",
    "AnnotationRecord",
    "AnnotationInventory",
    "LinkRecord",
    "FormFieldRecord",
    "FormInventory",
    "OutlineItem",
    "AttachmentInfo",
    "EmbeddedResourceRecord",
    "AnalysisFinding",
    "AnalysisReport",
    "CompliancePreflightSummary",
    "EvidenceLayer",
    "EvidenceRecord",
    "EvidenceEdge",
    "EvidenceGraph",
    "EvidenceNode",
    "RemediationAction",
    "DocumentInventory",
    "NativeFeatureInventory",
    "ActionInventory",
    "DocumentAnalysisSnapshot",
    "OptionalContentLayerRecord",
    "RevisionInventory",
    "RevisionObjectRecord",
    "RevisionObjectDiff",
    "compare_revision_objects",
    "DocumentFingerprint",
    "ArchivalManifest",
    "ArchivalManifestDiff",
    "compare_archival_manifests",
    "FingerprintDiff",
    "compare_fingerprints",
    "IncrementalAnalysisPlan",
    "plan_incremental_analysis",
    "plan_revision_analysis",
    "PreservationManifest",
    "RedactionVerification",
    "SanitizationVerification",
    "AccessibilityRepairVerification",
    "verify_preservation",
    "ObjectGraphEdge",
    "ObjectGraphNode",
    "ObjectInspection",
    "ObjectRoundTripVerification",
    "ObjectRoundTripManifest",
    "ObjectGraphReport",
    "ObjectReachabilityDiff",
    "compare_object_graphs",
    "StructureElementRecord",
    "AccessibilityInventory",
    "ChunkRecord",
    "GeometryIssue",
    "GeometrySummary",
    "Severity",
    "TextDiagnosticRun",
)
