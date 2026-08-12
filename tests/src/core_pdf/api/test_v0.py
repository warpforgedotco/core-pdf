from __future__ import annotations

from collections.abc import Callable
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf import PdfDocument
from core_pdf.api import (
    AnalysisCache,
    DocumentClosed,
    LocalExecutionContext,
)
from core_pdf.api import (
    PdfDocument as V0PdfDocument,
)
from core_pdf.api.document import PdfPage
from core_pdf.api.models import (
    AccessibilityInventory,
    AccessibilityRepairVerification,
    ActionInventory,
    AnnotationInventory,
    ArchivalManifest,
    ArchivalManifestDiff,
    ChunkRecord,
    CompliancePreflightSummary,
    ContentDependencyRecord,
    ContentEvent,
    CoordinateOrigin,
    CoordinateSpace,
    DocumentAnalysisSnapshot,
    DocumentFingerprint,
    Drawing,
    DrawingItem,
    EmbeddedResourceRecord,
    EvidenceGraph,
    FormInventory,
    IncrementalAnalysisPlan,
    NativeFeatureInventory,
    ObjectInspection,
    OptionalContentLayerRecord,
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
    SanitizationVerification,
    SourceRef,
    TextSpan,
    compare_archival_manifests,
    compare_fingerprints,
    compare_object_graphs,
    compare_revision_objects,
    plan_incremental_analysis,
    plan_revision_analysis,
    verify_preservation,
)
from core_pdf.api.compat import (
    LAParams,
    LTChar,
    extract_pages,
    extract_text,
    extract_text_to_fp,
    inspect_xray,
)
from core_pdf.api.v0.operations import (
    AccessibilityValidationOperation,
    AnnotationValidationOperation,
    AttachmentValidationOperation,
    BadRedactionOperation,
    CitationAnalysisOperation,
    CompliancePreflightOperation,
    FigureCaptionAnalysisOperation,
    FontValidationOperation,
    ForensicAnalysisOperation,
    FormValidationOperation,
    GeometryValidationOperation,
    IdentifierAnalysisOperation,
    ImageValidationOperation,
    LayerConsistencyOperation,
    LayoutAnalysisOperation,
    LinkValidationOperation,
    QualityPreflightOperation,
    ReferenceEntryAnalysisOperation,
    SectionHierarchyAnalysisOperation,
    StructureAnalysisOperation,
    normalize_metadata,
    plan_accessibility_remediation,
)
from core_pdf.impl.engine.structured import (
    Block,
    BlockKind,
    Document,
    FormField,
    Page,
    Table,
    TableCell,
    TextLine,
)
from core_pdf.impl.engine.writing import serialize_pdf_file
from core_pdf.impl.objects import PdfName, PdfReference, PdfStream

# Migration helpers for assertions that deliberately exercise an already-open engine document.
# New application code enters through ``core_pdf.api.PdfDocument.open`` instead.
PdfDocumentAdapter = V0PdfDocument
PdfPageAdapter = PdfPage
adapt_document = V0PdfDocument.internal_from_engine
adapt_structured = V0PdfDocument.from_structured


def simple_pdf(*, annotation: bool = False, unreachable: bool = False) -> bytes:
    objects = {
        1: {PdfName.of("Type"): PdfName.of("Catalog"), PdfName.of("Pages"): PdfReference(2)},
        2: {
            PdfName.of("Type"): PdfName.of("Pages"),
            PdfName.of("Kids"): [PdfReference(3)],
            PdfName.of("Count"): 1,
        },
        3: {
            PdfName.of("Type"): PdfName.of("Page"),
            PdfName.of("Parent"): PdfReference(2),
            PdfName.of("MediaBox"): [0, 0, 10, 10],
            PdfName.of("Contents"): PdfReference(4),
            **({PdfName.of("Annots"): [PdfReference(5)]} if annotation else {}),
        },
        4: PdfStream({}, b"q\nQ\n"),
    }
    if annotation:
        objects[5] = {
            PdfName.of("Type"): PdfName.of("Annot"),
            PdfName.of("Subtype"): PdfName.of("Text"),
            PdfName.of("Rect"): [1, 1, 3, 3],
            PdfName.of("Contents"): b"note",
        }
    if unreachable:
        objects[6] = {PdfName.of("Type"): PdfName.of("Metadata"), PdfName.of("Note"): b"unused"}
    return serialize_pdf_file(objects, trailer={PdfName.of("Root"): PdfReference(1)})


def malformed_icc_pdf() -> bytes:
    objects = {
        1: {PdfName.of("Type"): PdfName.of("Catalog"), PdfName.of("Pages"): PdfReference(2)},
        2: {
            PdfName.of("Type"): PdfName.of("Pages"),
            PdfName.of("Kids"): [PdfReference(3)],
            PdfName.of("Count"): 1,
        },
        3: {
            PdfName.of("Type"): PdfName.of("Page"),
            PdfName.of("Parent"): PdfReference(2),
            PdfName.of("MediaBox"): [0, 0, 10, 10],
            PdfName.of("Resources"): {
                PdfName.of("ColorSpace"): {
                    PdfName.of("CS0"): [PdfName.of("ICCBased"), PdfReference(4)]
                }
            },
            PdfName.of("Contents"): PdfReference(5),
        },
        4: PdfStream({}, b""),
        5: PdfStream({}, b"q\nQ\n"),
    }
    return serialize_pdf_file(objects, trailer={PdfName.of("Root"): PdfReference(1)})


def transparency_pdf() -> bytes:
    objects = {
        1: {PdfName.of("Type"): PdfName.of("Catalog"), PdfName.of("Pages"): PdfReference(2)},
        2: {
            PdfName.of("Type"): PdfName.of("Pages"),
            PdfName.of("Kids"): [PdfReference(3)],
            PdfName.of("Count"): 1,
        },
        3: {
            PdfName.of("Type"): PdfName.of("Page"),
            PdfName.of("Parent"): PdfReference(2),
            PdfName.of("MediaBox"): [0, 0, 10, 10],
            PdfName.of("Resources"): {
                PdfName.of("ExtGState"): {PdfName.of("GS0"): PdfReference(4)}
            },
            PdfName.of("Contents"): PdfReference(5),
        },
        4: {PdfName.of("Type"): PdfName.of("ExtGState"), PdfName.of("ca"): 0.5},
        5: PdfStream({}, b"/GS0 gs\nq\nQ\n"),
    }
    return serialize_pdf_file(objects, trailer={PdfName.of("Root"): PdfReference(1)})


def malformed_font_pdf() -> bytes:
    objects = {
        1: {PdfName.of("Type"): PdfName.of("Catalog"), PdfName.of("Pages"): PdfReference(2)},
        2: {
            PdfName.of("Type"): PdfName.of("Pages"),
            PdfName.of("Kids"): [PdfReference(3)],
            PdfName.of("Count"): 1,
        },
        3: {
            PdfName.of("Type"): PdfName.of("Page"),
            PdfName.of("Parent"): PdfReference(2),
            PdfName.of("MediaBox"): [0, 0, 10, 10],
            PdfName.of("Resources"): {PdfName.of("Font"): {PdfName.of("F0"): PdfReference(4)}},
            PdfName.of("Contents"): PdfReference(5),
        },
        4: {PdfName.of("Type"): PdfName.of("Font"), PdfName.of("Subtype"): PdfName.of("Type1")},
        5: PdfStream({}, b"BT /F0 12 Tf 1 1 Td (Broken) Tj ET\n"),
    }
    return serialize_pdf_file(objects, trailer={PdfName.of("Root"): PdfReference(1)})


def test_geometry_records_carry_coordinate_space() -> None:
    space = CoordinateSpace("page", CoordinateOrigin.BOTTOM_LEFT, width=10, height=20)
    rect = Rect(1, 2, 4, 8, space)

    assert rect.width == 3
    assert rect.height == 6
    assert rect.space.origin is CoordinateOrigin.BOTTOM_LEFT


def test_rect_geometry_operations_preserve_space_and_exact_overlap() -> None:
    space = CoordinateSpace("page", CoordinateOrigin.TOP_LEFT, width=10.0, height=10.0)
    left = Rect(0.0, 0.0, 0.5, 0.5, space)
    right = Rect(0.25, 0.0, 1.0, 0.5, space)

    intersection = left.intersection(right)

    assert intersection == Rect(0.25, 0.0, 0.5, 0.5, space)
    assert left.area == 0.25
    assert left.overlap_ratio_min(right) == 0.5


def test_rect_intersection_rejects_mixed_coordinate_spaces() -> None:
    top = CoordinateSpace("page", CoordinateOrigin.TOP_LEFT, height=10.0)
    bottom = CoordinateSpace("page", CoordinateOrigin.BOTTOM_LEFT, height=10.0)

    with pytest.raises(ValueError, match="different coordinate spaces"):
        Rect(0.0, 0.0, 1.0, 1.0, top).intersection(Rect(0.0, 0.0, 1.0, 1.0, bottom))


def test_current_document_conforms_through_v0_adapter() -> None:
    with PdfDocument.open(simple_pdf()) as document:
        adapted = adapt_document(document)

        assert isinstance(adapted, PdfDocumentAdapter)
        assert adapted.page_count == 1
        page = adapted.page(0)
        assert page.info.number == 1
        assert page.info.width == 10
        assert tuple(page.content_events()) == ()
        assert page.text() == page.structured_view.text
        assert tuple(page.text_lines()) == ()
        assert tuple(page.text_blocks()) == ()
        assert tuple(page.annotations()) == ()
        assert tuple(page.links()) == ()
        assert tuple(page.form_fields()) == ()
        assert tuple(adapted.structure_elements()) == ()
        assert page.structured_view.page_number == 1
        assert adapted.structured.pages[0].page_number == 1
        assert adapted.structured.text == adapted.text()
        assert tuple(adapted.structured.lines) == tuple(adapted.text_lines())
        assert tuple(adapted.structured.blocks) == tuple(adapted.text_blocks())
        assert tuple(adapted.structured.words) == tuple(adapted.words())
        inventory = adapted.inventory()
        assert inventory.byte_count == len(simple_pdf())
        assert inventory.object_count >= 4
        assert inventory.page_count == 1
        assert not inventory.encrypted
        assert not inventory.has_javascript


def test_layer_consistency_operation_returns_evidence_report() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        report = LayerConsistencyOperation().run(adapt_document(source))

    assert report.analyzer_id == "analysis.layer-consistency"
    assert report.version == "1.0"
    assert report.findings == ()
    assert report.metrics["invisible_text_runs"] == 0


def test_v0_adapter_exposes_reachable_pdf_object_graph() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        graph = adapt_document(source).object_graph()

    assert graph.root_objects == (1,)
    assert graph.unreachable_objects == ()
    assert any(node.object_type == "Catalog" and node.reachable for node in graph.nodes)
    assert any(edge.source_object == 1 and edge.target_object == 2 for edge in graph.edges)
    assert graph.to_dict()["unreachable_objects"] == []
    assert '"root_objects": [1]' in graph.to_json(indent=None)


def test_object_graph_comparison_reports_new_unreachable_objects() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        before = adapt_document(source).object_graph()
    with PdfDocument.open(simple_pdf(unreachable=True)) as source:
        after = adapt_document(source).object_graph()

    diff = compare_object_graphs(before, after)
    assert diff.changed
    assert diff.became_unreachable == ()
    assert diff.added_objects == (6,)


def test_v0_adapter_exposes_deterministic_document_fingerprints() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        fingerprints = adapt_document(source).fingerprint()

    assert len(fingerprints.document_sha256) == 64
    assert fingerprints.page_sha256[0][0] == 1
    assert fingerprints.object_sha256
    with PdfDocument.open(simple_pdf()) as source:
        assert not adapt_document(source).incremental_plan(fingerprints).full_document_scan


def test_v0_object_inspection_serializes_json_safely() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        inspection = adapt_document(source).inspect_object(1)

    assert inspection.to_dict()["object_type"] == "Catalog"
    assert inspection.raw_offset is not None
    assert inspection.raw_length
    assert inspection.raw_sha256 is not None
    assert '"object_number": 1' in inspection.to_json(indent=None)


def test_v0_object_roundtrip_verification_preserves_source_hash() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        verification = adapt_document(source).verify_object_roundtrip(1)

    assert verification.passed
    assert verification.source_sha256 == verification.reopened_sha256


def test_v0_object_roundtrip_manifest_is_deterministic() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        manifest = adapt_document(source).verify_object_roundtrips((3, 1, 3))

    assert tuple(item.object_number for item in manifest.objects) == (1, 3)
    assert manifest.passed


def test_v0_revision_objects_expose_raw_spans() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        records = tuple(adapt_document(source).revision_objects())

    assert records
    assert all(record.revision_index == 0 for record in records)
    assert all(record.byte_count > 0 and len(record.sha256) == 64 for record in records)


def test_revision_object_comparison_is_stable() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        records = tuple(adapt_document(source).revision_objects())

    diff = compare_revision_objects(records, records)
    assert diff.added == ()
    assert diff.removed == ()
    assert diff.changed == ()
    assert plan_revision_analysis(diff).full_document_scan


def test_v0_adapter_exposes_archival_manifest() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        manifest = adapt_document(source).archival_manifest()

    assert isinstance(manifest, ArchivalManifest)
    assert manifest.page_count == 1
    assert manifest.object_count >= 4
    assert manifest.reachable_object_count == manifest.object_count
    assert not manifest.has_javascript


def test_v0_adapter_exposes_native_feature_inventory() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        features = adapt_document(source).native_features()

    assert isinstance(features, NativeFeatureInventory)
    assert not features.has_optional_content
    assert not features.has_incremental_revision
    assert not features.is_linearized


def test_v0_adapter_exposes_embedded_resource_inventory() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        resources = tuple(adapt_document(source).embedded_resources())

    assert resources == ()
    assert not any(isinstance(resource, EmbeddedResourceRecord) for resource in resources)


def test_v0_adapter_exposes_optional_content_layers() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        layers = tuple(adapt_document(source).optional_content_layers())

    assert layers == ()
    assert not any(isinstance(layer, OptionalContentLayerRecord) for layer in layers)


def test_v0_adapter_exposes_page_resource_inventory() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        resources = tuple(adapt_document(source).resource_inventory())

    assert resources == (PageResourceInventory(page_number=1),)


def test_v0_adapter_exposes_content_dependencies() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        dependencies = tuple(adapt_document(source).content_dependencies())

    assert dependencies == ()
    assert not any(isinstance(item, ContentDependencyRecord) for item in dependencies)


def test_v0_adapter_exposes_resource_dependency_graph() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        graph = adapt_document(source).resource_dependency_graph()

    assert isinstance(graph, ResourceDependencyGraph)
    assert graph.page_resources == ()
    assert graph.resources == ()


def test_v0_adapter_exposes_resource_diagnostics() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        diagnostics = tuple(adapt_document(source).resource_diagnostics())

    assert diagnostics == ()
    assert not any(isinstance(item, ResourceDiagnostic) for item in diagnostics)


def test_v0_adapter_exposes_page_content_summaries() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        summaries = tuple(adapt_document(source).content_summaries())

    assert len(summaries) == 1
    assert isinstance(summaries[0], PageContentSummary)
    assert summaries[0].page_number == 1
    assert summaries[0].raw_stream_bytes >= summaries[0].decoded_stream_bytes


def test_v0_adapter_exposes_form_inventory() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        inventory = adapt_document(source).form_inventory()

    assert isinstance(inventory, FormInventory)
    assert inventory.field_count == 0
    assert inventory.empty_count == 0


def test_v0_adapter_exports_csv_and_tei() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        document = adapt_document(source)
        csv_output = document.to_csv()
        tei_output = document.to_tei()

    assert csv_output.startswith("page_number,line_index,text,x0,y0,x1,y1\n")
    assert tei_output.startswith("<TEI><text><body>")
    assert '<pb n="1" />' in tei_output


def test_v0_adapter_exposes_annotation_inventory() -> None:
    with PdfDocument.open(simple_pdf(annotation=True)) as source:
        inventory = adapt_document(source).annotation_inventory()

    assert isinstance(inventory, AnnotationInventory)
    assert inventory.annotation_count == 1
    assert inventory.page_count == 1
    assert inventory.subtype_counts == {"text": 1}


def test_v0_search_rejects_unknown_modes() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        document = adapt_document(source)
        with pytest.raises(ValueError, match="unsupported search mode"):
            tuple(document.search("text", mode="semantic"))


def test_v0_search_accepts_coordinate_region() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        space = tuple(adapt_document(source).pages())[0].info.space
        with PdfDocument.open(simple_pdf()) as second:
            hits = tuple(adapt_document(second).search("missing", region=Rect(0, 0, 1, 1, space)))

    assert hits == ()


def test_v0_adapter_exposes_evidence_graph() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        graph = adapt_document(source).evidence_graph()

    assert isinstance(graph, EvidenceGraph)
    assert [node.node_id for node in graph.nodes] == ["page:1"]
    assert graph.to_dict()["nodes"]
    assert '"edges": []' in graph.to_json(indent=None)


def test_v0_evidence_graph_includes_annotations() -> None:
    with PdfDocument.open(simple_pdf(annotation=True)) as source:
        graph = adapt_document(source).evidence_graph()

    assert any(node.node_id == "page:1:annotation:0" for node in graph.nodes)
    assert any(edge.relation == "contains" for edge in graph.edges)


def test_v0_analysis_snapshot_includes_embedded_resource_inventory() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        snapshot = adapt_document(source).analysis_snapshot()

    assert snapshot.embedded_resources == ()
    assert snapshot.to_dict()["embedded_resources"] == []
    assert snapshot.revision_objects


def test_v0_action_inventory_counts_action_types_not_container_keys() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        inventory = adapt_document(source).action_inventory()

    assert inventory.action_count == sum(inventory.action_type_counts.values())


def test_v0_editor_redaction_verification_reports_requested_queries() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        editor = adapt_document(source).edit()
        result = editor.commit_redactions_verified(BytesIO(), {}, queries=("secret",))

    assert isinstance(result, RedactionVerification)
    assert result.requested_queries == ("secret",)
    assert result.remaining_queries == ()
    assert result.remaining_raw_queries == ()
    assert result.passed
    assert isinstance(result.removed_objects, tuple)


def test_v0_editor_sanitization_verification() -> None:
    with PdfDocument.open(simple_pdf(annotation=True)) as source:
        result = adapt_document(source).edit().commit_sanitized_verified(BytesIO())

    assert isinstance(result, SanitizationVerification)
    assert result.remaining_annotations == 0
    assert result.remaining_actions == 0
    assert isinstance(result.became_unreachable_objects, tuple)
    assert result.passed


def test_v0_editor_accessibility_metadata_repair_verification() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        result = (
            adapt_document(source)
            .edit()
            .commit_accessibility_repair_verified(BytesIO(), title="Example", language="en-US")
        )

    assert isinstance(result, AccessibilityRepairVerification)
    assert result.has_title
    assert result.language == "en-US"
    assert result.passed


def test_v0_adapter_exposes_accessibility_inventory() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        inventory = adapt_document(source).accessibility_inventory()

    assert isinstance(inventory, AccessibilityInventory)
    assert inventory.tagged_element_count == 0
    assert inventory.image_count == 0
    assert inventory.document_language is None
    assert not inventory.has_title


def test_v0_adapter_exposes_reading_order() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        items = tuple(adapt_document(source).reading_order())

    assert all(isinstance(item, ReadingOrderItem) for item in items)
    assert [item.order for item in items] == list(range(len(items)))


def test_v0_adapter_exposes_action_inventory() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        inventory = adapt_document(source).action_inventory()

    assert isinstance(inventory, ActionInventory)
    assert inventory.action_count == 0
    assert inventory.source_objects == ()


def test_v0_adapter_exposes_analysis_snapshot() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        snapshot = adapt_document(source).analysis_snapshot()

    assert isinstance(snapshot, DocumentAnalysisSnapshot)
    assert snapshot.inventory.page_count == 1
    assert snapshot.forms.field_count == 0
    assert snapshot.annotations.annotation_count == 0
    manifest = snapshot.to_dict()
    inventory = cast(dict[str, object], manifest["inventory"])
    assert inventory["page_count"] == 1
    assert '"page_count": 1' in snapshot.to_json()


def test_v0_adapter_inspects_a_resolved_object() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        inspection = adapt_document(source).inspect_object(1)

    assert isinstance(inspection, ObjectInspection)
    assert inspection.object_number == 1
    assert "Type" in inspection.dictionary_keys


def test_v0_adapter_exposes_revision_markers() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        revisions = adapt_document(source).revisions()

    assert isinstance(revisions, RevisionInventory)
    assert revisions.revision_count == 1
    assert len(revisions.startxref_offsets) == 1
    assert not revisions.has_incremental_updates


def test_archival_manifest_serializes_and_compares() -> None:
    before = ArchivalManifest("a", 1, 1, 1, 1, 0, False, False, False, "Old")
    after = ArchivalManifest("b", 2, 1, 1, 1, 0, False, False, False, "New")

    diff = compare_archival_manifests(before, after)

    assert isinstance(diff, ArchivalManifestDiff)
    assert diff.document_changed
    assert diff.changed_fields == ("document_sha256", "byte_count", "title")
    assert '"title": "Old"' in before.to_json()


def test_fingerprint_diff_reports_changed_pages_and_objects() -> None:
    before = DocumentFingerprint(
        "before", page_sha256=((1, "a"), (2, "b")), object_sha256=((1, 0, "a"), (2, 0, "b"))
    )
    after = DocumentFingerprint(
        "after",
        page_sha256=((1, "a"), (2, "c"), (3, "d")),
        object_sha256=((1, 0, "z"), (3, 0, "d")),
    )

    diff = compare_fingerprints(before, after)

    assert diff.document_changed
    assert diff.changed_pages == (2, 3)
    assert diff.changed_objects == ((1, 0),)
    assert diff.added_objects == ((3, 0),)
    assert diff.removed_objects == ((2, 0),)


def test_incremental_analysis_plan_selects_changed_units() -> None:
    before = DocumentFingerprint("before", page_sha256=((1, "a"),))
    after = DocumentFingerprint("after", page_sha256=((1, "b"),))

    plan = plan_incremental_analysis(compare_fingerprints(before, after))

    assert isinstance(plan, IncrementalAnalysisPlan)
    assert plan.affected_pages == (1,)
    assert plan.affected_objects == ()
    assert not plan.full_document_scan


def test_preservation_verification_flags_changed_protected_pages() -> None:
    before = DocumentFingerprint("before", page_sha256=((1, "a"), (2, "b")))
    after = DocumentFingerprint("after", page_sha256=((1, "c"), (2, "b")))

    manifest = verify_preservation(before, after, expected_unchanged_pages=(1, 2))

    assert isinstance(manifest, PreservationManifest)
    assert manifest.changed_pages == (1,)
    assert manifest.violated_pages == (1,)
    assert not manifest.passed


def test_v0_adapter_exposes_typed_provenance_preserving_chunks() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        chunks = tuple(adapt_document(source).chunks())

    assert chunks == ()
    assert all(isinstance(chunk, ChunkRecord) for chunk in chunks)


def test_forensic_analysis_reports_unreachable_objects() -> None:
    with PdfDocument.open(simple_pdf(unreachable=True)) as source:
        report = ForensicAnalysisOperation().run(adapt_document(source))

    finding = next(item for item in report.findings if item.code == "pdf.unreachable-object")
    assert finding.evidence[0].source.object_number == 6
    assert report.metrics["unreachable_object_count"] == 1


def test_layout_analysis_is_deterministic_for_empty_page_content() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        report = LayoutAnalysisOperation().run(adapt_document(source))

    assert report.findings == ()
    assert report.metrics == {
        "page_count": 1,
        "column_page_count": 0,
        "repeated_edge_text_count": 0,
    }


def test_structure_analysis_is_empty_for_an_empty_structured_page() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        report = StructureAnalysisOperation().run(adapt_document(source))

    assert report.findings == ()
    assert report.metrics == {"element_counts": {}, "element_count": 0}


def test_citation_analysis_is_empty_for_an_empty_structured_page() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        report = CitationAnalysisOperation().run(adapt_document(source))

    assert report.findings == ()
    assert report.metrics == {
        "numeric_citation_count": 0,
        "author_year_citation_count": 0,
        "reference_section_count": 0,
    }


def test_figure_caption_analysis_is_empty_for_an_empty_page() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        report = FigureCaptionAnalysisOperation().run(adapt_document(source))

    assert report.findings == ()
    assert report.metrics == {
        "figure_count": 0,
        "associated_figure_count": 0,
        "uncaptioned_figure_count": 0,
    }


def test_section_hierarchy_analysis_is_empty_for_an_empty_page() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        report = SectionHierarchyAnalysisOperation().run(adapt_document(source))

    assert report.findings == ()
    assert report.metrics == {
        "heading_count": 0,
        "list_count": 0,
        "max_heading_depth": 0,
        "heading_depth_gap_count": 0,
    }


def test_identifier_analysis_is_empty_for_an_empty_page() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        report = IdentifierAnalysisOperation().run(adapt_document(source))

    assert report.findings == ()
    assert report.metrics == {
        "doi": 0,
        "isbn": 0,
        "url": 0,
        "email": 0,
        "date": 0,
        "identifier_count": 0,
    }


def test_compliance_preflight_leaves_title_checks_to_accessibility_validation() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        report = CompliancePreflightOperation().run(adapt_document(source))

    assert not any(finding.code == "metadata.title-missing" for finding in report.findings)
    assert report.metrics["profile"] == "pdf/a-2u"
    summary = cast(CompliancePreflightSummary, report.metrics["summary"])
    assert summary.profile == "pdf/a-2u"
    assert summary.passed
    assert summary.warning_count == 0
    assert not summary.has_font_errors
    assert not summary.has_color_warnings


def test_analysis_cache_reuses_same_fingerprint_report() -> None:
    cache = AnalysisCache()
    with PdfDocument.open(simple_pdf()) as source:
        adapted = adapt_document(source)
        context = cast(Any, LocalExecutionContext())
        first = cache.run(CompliancePreflightOperation(), adapted, context)
        second = cache.run(CompliancePreflightOperation(), adapted, context)

    assert first is second
    assert cache.misses == 1
    assert cache.hits == 1
    assert cache.invalidate(IncrementalAnalysisPlan(affected_pages=(2,))) == 1


def test_compliance_preflight_reports_resource_diagnostics() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        report = CompliancePreflightOperation().run(
            adapt_document(source), options={"profile": "pdf/a-2u"}
        )

    assert not any(finding.code.startswith("pdfa.resource.") for finding in report.findings)


def test_resource_diagnostics_have_stable_empty_page_baseline() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        diagnostics = tuple(adapt_document(source).resource_diagnostics())

    assert all(
        diagnostic.page_number is None or diagnostic.page_number >= 1 for diagnostic in diagnostics
    )


def test_resource_diagnostics_do_not_claim_font_embedding_without_evidence() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        diagnostics = tuple(adapt_document(source).resource_diagnostics())

    assert not any(diagnostic.code == "font.not-embedded" for diagnostic in diagnostics)


def test_resource_diagnostics_do_not_invent_color_profiles() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        diagnostics = tuple(adapt_document(source).resource_diagnostics())

    assert all(diagnostic.code != "color.icc-missing" for diagnostic in diagnostics)


def test_resource_diagnostics_do_not_report_icc_errors_without_icc_spaces() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        diagnostics = tuple(adapt_document(source).resource_diagnostics())

    assert not any(diagnostic.code == "color.icc-profile-invalid" for diagnostic in diagnostics)


def test_malformed_icc_fixture_reaches_typed_preflight_summary() -> None:
    with PdfDocument.open(malformed_icc_pdf()) as source:
        report = CompliancePreflightOperation().run(
            adapt_document(source), options={"profile": "pdf/a-2u"}
        )

    summary = cast(CompliancePreflightSummary, report.metrics["summary"])
    assert "pdfa.resource.color.icc-profile-invalid" in summary.finding_codes
    assert summary.has_color_warnings


def test_transparency_fixture_reaches_typed_preflight_summary() -> None:
    with PdfDocument.open(transparency_pdf()) as source:
        report = CompliancePreflightOperation().run(
            adapt_document(source), options={"profile": "pdf/a-2u"}
        )

    summary = cast(CompliancePreflightSummary, report.metrics["summary"])
    assert summary.has_transparency_warnings


def test_malformed_font_fixture_reaches_typed_preflight_summary() -> None:
    with PdfDocument.open(malformed_font_pdf()) as source:
        report = CompliancePreflightOperation().run(
            adapt_document(source), options={"profile": "pdf/a-2u"}
        )

    summary = cast(CompliancePreflightSummary, report.metrics["summary"])
    assert "pdfa.resource.font.resource-missing" in summary.finding_codes
    assert summary.has_font_errors


def test_metadata_normalization_canonicalizes_common_fields_and_preserves_unknowns() -> None:
    normalized = normalize_metadata({"/title": "  A   title ", "Custom": "  keep  "})

    assert normalized == {"Title": "A title", "Custom": "keep"}


def test_metadata_normalization_collapses_whitespace_in_string_values() -> None:
    assert normalize_metadata({"Author": "  Ada   Lovelace  "}) == {"Author": "Ada Lovelace"}


def test_accessibility_remediation_plan_reports_title_and_structure_actions() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        report = AccessibilityValidationOperation().run(adapt_document(source))

    actions = plan_accessibility_remediation(report)
    assert {action.code for action in actions} == {"metadata.title", "structure.tagged-content"}
    assert all(action.priority == "high" for action in actions)
    assert all(action.source.stage == "accessibility-remediation" for action in actions)


def test_accessibility_validation_reports_missing_document_requirements() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        report = AccessibilityValidationOperation().run(adapt_document(source))

    assert {finding.code for finding in report.findings} >= {
        "ua.document-title",
        "ua.document-language",
        "ua.tagged-structure",
    }


def test_form_validation_reports_empty_document_without_findings() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        report = FormValidationOperation().run(adapt_document(source))

    assert report.findings == ()
    assert report.metrics["field_count"] == 0


def test_form_validation_reports_invalid_choice_and_unknown_type() -> None:
    page = Page(
        page_number=1,
        width=100,
        height=100,
        form_fields=(
            FormField("choice", "choice", "missing", (10, 10, 40, 20), options=("one",)),
            FormField("custom", "unknown", "", (10, 30, 40, 40)),
        ),
    )
    document = adapt_structured(Document(pages=(page,)))

    report = FormValidationOperation().run(document)

    assert {finding.code for finding in report.findings} >= {
        "form.value-not-in-options",
        "form.type-unknown",
    }


def test_link_validation_reports_empty_document_without_findings() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        report = LinkValidationOperation().run(adapt_document(source))

    assert report.findings == ()
    assert report.metrics["link_count"] == 0


def test_attachment_validation_reports_empty_document_without_findings() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        report = AttachmentValidationOperation().run(adapt_document(source))

    assert report.findings == ()
    assert report.metrics["attachment_count"] == 0


def test_font_validation_reports_simple_document_without_findings() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        report = FontValidationOperation().run(adapt_document(source))

    assert report.findings == ()
    assert report.metrics["page_count"] == 1


def test_image_validation_reports_simple_document_without_findings() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        report = ImageValidationOperation().run(adapt_document(source))

    assert report.findings == ()
    assert report.metrics["image_count"] == 0


def test_geometry_validation_reports_simple_document_without_findings() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        report = GeometryValidationOperation().run(adapt_document(source))

    assert report.findings == ()
    assert report.metrics["issue_count"] == 0


def test_annotation_validation_reports_simple_document_without_findings() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        report = AnnotationValidationOperation().run(adapt_document(source))

    assert report.findings == ()
    assert report.metrics["annotation_count"] == 0


def test_quality_preflight_combines_local_validators() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        report = QualityPreflightOperation().run(adapt_document(source))

    assert report.analyzer_id == "analysis.quality-preflight"
    assert report.metrics["operation_count"] == 10
    assert report.metrics["finding_count"] == len(report.findings)
    reports = cast(dict[str, object], report.metrics["reports"])
    assert "analysis.accessibility" in reports


def test_reference_entry_analysis_is_empty_for_an_empty_page() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        report = ReferenceEntryAnalysisOperation().run(adapt_document(source))

    assert report.findings == ()
    assert report.metrics == {
        "reference_entry_count": 0,
        "numbered_reference_count": 0,
        "doi_reference_count": 0,
    }


def test_v0_editor_can_encrypt_a_committed_document() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        data = adapt_document(source).edit().encrypt("secret").commit(BytesIO())

    with PdfDocument.open(BytesIO(data), password="secret") as reopened:
        assert reopened.page_count() == 1


def test_v0_noop_editor_preserves_original_pdf_bytes() -> None:
    original = simple_pdf()
    with PdfDocument.open(original) as source:
        data = adapt_document(source).edit().commit(BytesIO())

    assert data == original


def test_v0_editor_round_trips_page_geometry() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        data = (
            adapt_document(source)
            .edit()
            .set_page_geometry(page_number=1, rotation=90, cropbox=(1, 2, 9, 8))
            .commit(BytesIO())
        )

    with PdfDocument.open(BytesIO(data)) as reopened:
        page = reopened.pages[0]
        assert page.rotation == 90
        assert page.crop_box == (1, 2, 9, 8)
    assert b"q\nQ\n" in data


def test_v0_editor_commit_verified_returns_preservation_manifest() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        manifest = (
            adapt_document(source)
            .edit()
            .set_metadata({"Title": "verified"})
            .commit_verified(BytesIO(), expected_unchanged_pages=(1,))
        )

    assert manifest.passed
    assert manifest.expected_unchanged_pages == (1,)


def test_v0_annotation_removal_preserves_original_content_stream() -> None:
    with PdfDocument.open(simple_pdf(annotation=True)) as source:
        cleaned = adapt_document(source).edit().remove_annotations(1).commit(BytesIO())

    with PdfDocument.open(BytesIO(cleaned)) as reopened:
        assert tuple(reopened.pages[0].get_annotations()) == ()
    assert b"q\nQ\n" in cleaned


def test_adapter_exposes_engine_owned_high_level_records() -> None:
    with PdfDocument.open(simple_pdf()) as document:
        adapted = adapt_document(document)

        assert adapted.metadata == document.get_metadata()
        assert tuple(adapted.images()) == ()
        assert tuple(adapted.annotations()) == ()
        assert tuple(adapted.links()) == ()
        assert tuple(adapted.form_fields()) == ()
        assert tuple(adapted.outlines) == ()
        assert tuple(adapted.attachments) == ()


def test_adapter_exposes_document_text_elements_and_chunks() -> None:
    with PdfDocument.open(simple_pdf()) as document:
        adapted = adapt_document(document)
        assert adapted.text() == document.structured_document.text
        assert tuple(adapted.words()) == tuple(adapted.page(0).words())
        assert tuple(adapted.text_spans()) == tuple(adapted.page(0).text_spans())
        assert tuple(adapted.text_characters()) == tuple(adapted.page(0).text_characters())
        assert tuple(adapted.text_lines()) == tuple(adapted.page(0).text_lines())
        assert tuple(adapted.text_blocks()) == tuple(adapted.page(0).text_blocks())
        assert tuple(adapted.annotations()) == ()
        assert tuple(adapted.links()) == ()
        assert tuple(adapted.form_fields()) == ()
        assert tuple(adapted.images()) == ()
        hits = tuple(adapted.search("missing"))
        assert hits == ()
        assert tuple(adapted.elements()) == ()
        assert tuple(adapted.chunks(max_characters=20)) == ()


def test_structured_conveniences_live_on_v0_adapter_only() -> None:
    with PdfDocument.open(simple_pdf()) as document:
        adapted = adapt_document(document)
        assert not hasattr(document, "to_markdown")
        assert not hasattr(document, "extract_elements")
        assert adapted.to_markdown()


def test_engine_document_editor_commits_transactionally() -> None:
    output = BytesIO()
    with PdfDocument.open(simple_pdf()) as document:
        document.edit().set_metadata({"Title": "edited"}).insert_page(1, 20, 30).commit(output)

    with PdfDocument.open(output.getvalue()) as reopened:
        assert reopened.page_count() == 2
        assert reopened.get_metadata()["info"]["Title"] == "edited"


def test_v0_adapter_returns_typed_editor_facade() -> None:
    output = BytesIO()
    with PdfDocument.open(simple_pdf()) as document:
        adapted = adapt_document(document)
        adapted.edit().set_metadata({"Title": "adapted"}).commit(output)

    with PdfDocument.open(output.getvalue()) as reopened:
        assert reopened.get_metadata()["info"]["Title"] == "adapted"


def test_engine_document_editor_exposes_structured_commit_and_rollback() -> None:
    with PdfDocument.open(simple_pdf()) as document:
        editor = document.edit().set_metadata({"Title": "structured"})
        committed = editor.commit_document()
        assert committed.metadata["Title"] == "structured"

    with PdfDocument.open(simple_pdf()) as document:
        editor = document.edit()
        editor.rollback()
        with pytest.raises(RuntimeError, match="closed"):
            editor.commit_document()


def test_engine_document_editor_commits_navigation_and_attachments() -> None:
    output = BytesIO()
    with PdfDocument.open(simple_pdf()) as document:
        (
            document.edit()
            .set_outlines(((1, "Start", 1),))
            .set_attachments({"note.txt": b"hello"})
            .commit(output)
        )

    with PdfDocument.open(output.getvalue()) as reopened:
        assert reopened.outlines[0].title == "Start"
        assert reopened.attachments[0].filename == "note.txt"


def test_engine_document_editor_adds_page_annotations_and_links() -> None:
    output = BytesIO()
    with PdfDocument.open(simple_pdf()) as document:
        (
            document.edit()
            .add_annotation(1, "Text", (1, 1, 10, 10), contents="note")
            .add_link(1, (10, 10, 20, 20), url="https://example.test")
            .commit(output)
        )

    with PdfDocument.open(output.getvalue()) as reopened:
        assert reopened.extract_annotations()[0].record.contents == "note"
        assert reopened.extract_links()[0].record.url == "https://example.test"


def test_v0_editor_removes_annotations_and_links() -> None:
    output = BytesIO()
    with PdfDocument.open(simple_pdf()) as document:
        (
            adapt_document(document)
            .edit()
            .add_annotation(1, "Text", (1, 1, 10, 10), contents="note")
            .add_link(1, (1, 1, 10, 10), url="https://example.test")
            .remove_annotations(1)
            .remove_links(1)
            .commit(output)
        )

    with PdfDocument.open(output.getvalue()) as reopened:
        assert reopened.extract_annotations() == ()
        assert reopened.extract_links() == ()


def test_engine_document_can_open_a_structured_snapshot() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        snapshot = source.structured_document
    with PdfDocument.from_structured(snapshot) as document:
        assert document.page_count() == 1
        assert document.structured_document.pages[0].width == 10


def test_v0_can_adapt_a_structured_snapshot() -> None:
    with PdfDocument.open(simple_pdf()) as source:
        snapshot = source.structured_document
    with adapt_structured(snapshot) as adapted:
        assert adapted.page_count == 1


def test_adapter_rejects_access_after_close() -> None:
    document = PdfDocument.open(simple_pdf())
    adapted = adapt_document(document)
    document.close()

    calls: tuple[tuple[str, Callable[[], object]], ...] = (
        ("page", lambda: adapted.page(0)),
        ("pages", lambda: tuple(adapted.pages())),
        ("metadata", lambda: adapted.metadata),
        ("inventory", adapted.inventory),
        ("fingerprint", adapted.fingerprint),
        ("to_json", adapted.to_json),
        ("to_csv", adapted.to_csv),
        ("elements", lambda: tuple(adapted.elements())),
        ("images", lambda: tuple(adapted.images())),
        ("words", lambda: tuple(adapted.words())),
        ("chunks", lambda: tuple(adapted.chunks())),
        ("search", lambda: tuple(adapted.search("x"))),
        ("resource_inventory", lambda: tuple(adapted.resource_inventory())),
        ("evidence_graph", adapted.evidence_graph),
        ("resource_diagnostics", lambda: tuple(adapted.resource_diagnostics())),
        ("content_summaries", lambda: tuple(adapted.content_summaries())),
        ("structure_elements", lambda: tuple(adapted.structure_elements())),
        ("annotation_inventory", adapted.annotation_inventory),
        ("accessibility_inventory", adapted.accessibility_inventory),
        ("edit", adapted.edit),
        ("outlines", lambda: tuple(adapted.outlines)),
        ("attachments", lambda: tuple(adapted.attachments)),
        ("embedded_resources", lambda: tuple(adapted.embedded_resources())),
    )
    for name, call in calls:
        with pytest.raises(DocumentClosed):
            call()
        del name


def test_star_import_surface_matches_all() -> None:
    from core_pdf import api

    missing = [name for name in api.__all__ if not hasattr(api, name)]

    assert missing == []


def test_concrete_v0_document_is_the_public_entrypoint() -> None:
    from core_pdf import api

    with api.PdfDocument.open(simple_pdf()) as document:
        assert document.page_count == 1
        assert document.page(0) is document.page(0)
        assert document.structured.pages[0].page_number == 1
        assert document.metadata["info"] == {}

    removed = (
        "adapt_document",
        "adapt_structured",
        "PdfDocumentAdapter",
        "PdfPageAdapter",
        "PdfEditorAdapter",
        "PdfDocumentProtocol",
        "PdfPageProtocol",
        "PdfEditorProtocol",
        "BadRedactionOperation",
        "QualityPreflightOperation",
    )
    assert all(not hasattr(api, name) for name in removed)


def test_public_api_accepts_path_inputs_in_type_surface(tmp_path: Path) -> None:
    path = tmp_path / "simple.pdf"
    path.write_bytes(simple_pdf())
    with PdfDocument.open(path) as document:
        assert adapt_document(document).page_count == 1


def test_adapter_exposes_native_character_geometry() -> None:
    fixture = Path(__file__).parents[4] / "tests/fixtures/x-ray/tests/assets/rectangles_yes_2.pdf"
    if not fixture.exists():
        pytest.skip()
    with PdfDocument.open(fixture) as document:
        page = adapt_document(document).page(0)
        characters = tuple(page.text_characters())

    assert characters
    assert "".join(character.text for character in characters) == "abcdefghi "


def test_adapter_exposes_canonical_words_and_search() -> None:
    fixture = Path("tests/fixtures/pdfminer.six/samples/simple1.pdf")
    with PdfDocument.open(fixture) as document:
        page = adapt_document(document).page(0)
        words = tuple(page.words())
        assert words
        assert words[0].text == "Hello"
        assert page.to_markdown()
        assert '"page_number"' in page.to_json()
        assert page.to_html()
        diagnostics = tuple(page.text_diagnostics())
        assert diagnostics
        assert diagnostics[0].bbox.space.name == "pdf-page"
        assert tuple(adapt_document(document).text_diagnostics())
        page_hits = tuple(adapt_document(document).search("Hello"))
        assert page_hits
        assert all(hit.bbox.space.name == "pdf-page" for hit in page_hits)
        assert isinstance(tuple(page.geometry_issues()), tuple)
        summary = page.geometry_summary()
        assert summary.text_run_count >= 0
        assert tuple(adapt_document(document).geometry_summaries())


def test_adapter_preserves_form_field_indexes() -> None:
    page = Page(
        page_number=1,
        width=100,
        height=100,
        form_fields=(
            FormField(
                "name",
                "text",
                "Alice",
                (1, 2, 20, 10),
                field_index=3,
                required=True,
                read_only=True,
                options=("Alice", "Bob"),
            ),
        ),
    )
    document = adapt_structured(Document(pages=(page,)))

    fields = tuple(document.page(0).form_fields())
    assert fields[0].name == "name"
    assert fields[0].field_index == 0
    assert fields[0].required
    assert fields[0].read_only
    assert fields[0].options == ("Alice", "Bob")

    committed = document.edit().remove_form_fields(("name",)).commit_document()
    assert committed.pages[0].form_fields == ()


def test_v0_editor_applies_structured_redactions() -> None:
    page = Page(
        page_number=1,
        width=100,
        height=100,
        blocks=(
            Block(
                order=0,
                kind=BlockKind.PARAGRAPH,
                lines=(
                    TextLine("secret", bbox=(10, 10, 20, 20)),
                    TextLine("keep", bbox=(50, 50, 60, 60)),
                ),
                bbox=(10, 10, 60, 60),
            ),
        ),
    )
    document = adapt_structured(Document(pages=(page,)))

    committed = document.edit().apply_redactions({1: ((0, 0, 50, 30),)}).commit_document()
    assert committed.pages[0].text == "keep"


def test_v0_editor_partially_redacts_a_text_line() -> None:
    page = Page(
        page_number=1,
        width=100,
        height=100,
        blocks=(
            Block(
                order=0,
                kind=BlockKind.PARAGRAPH,
                lines=(TextLine("secret", bbox=(10, 10, 70, 20)),),
                bbox=(10, 10, 70, 20),
            ),
        ),
    )
    document = adapt_structured(Document(pages=(page,)))

    committed = document.edit().apply_redactions({1: ((10, 10, 15, 20),)}).commit_document()
    assert committed.pages[0].text == "cret"


def test_writer_emits_form_default_and_normal_appearance() -> None:
    document = Document(
        pages=(
            Page(
                page_number=1,
                width=100,
                height=100,
                form_fields=(FormField("name", "text", "Alice", (10, 10, 90, 30)),),
            ),
        )
    )
    from core_pdf.impl.engine.writing.semantic import serialize_document_to_pdf

    data = serialize_document_to_pdf(document)
    assert b"/AP" in data
    assert b"/DA" in data
    assert b"/NeedAppearances false" in data
    assert b"Alice" not in data


def test_writer_emits_button_appearance_states() -> None:
    document = Document(
        pages=(
            Page(
                page_number=1,
                width=100,
                height=100,
                form_fields=(FormField("agree", "button", "on", (10, 10, 30, 30)),),
            ),
        )
    )
    from core_pdf.impl.engine.writing.semantic import serialize_document_to_pdf

    data = serialize_document_to_pdf(document)
    assert b"/AS /Yes" in data
    assert b"/Off" in data


def test_adapter_exposes_structured_serialization() -> None:
    with PdfDocument.open(Path("tests/fixtures/pdfminer.six/samples/simple1.pdf")) as document:
        adapted = adapt_document(document)
        assert '"pages"' in adapted.to_json()
        assert '"pages"' in adapted.to_structured_json(pages=1)
        assert "<article" in adapted.to_html()
        assert adapted.to_markdown()

        adapted.close()
        with pytest.raises(DocumentClosed):
            adapted.to_json()


def test_typed_text_capabilities_use_real_page_content() -> None:
    fixture = Path("tests/fixtures/pdfminer.six/samples/simple1.pdf")
    with PdfDocument.open(fixture) as document:
        adapted = adapt_document(document)
        page = adapted.page(0)
        assert "Hello" in page.text()
        assert tuple(page.text_lines())
        assert tuple(page.text_blocks())
        assert any(word.text == "Hello" for word in page.words())
        assert tuple(adapted.search("Hello"))


def test_adapter_exposes_canonical_tables() -> None:
    structured = Document(
        pages=(
            Page(
                page_number=1,
                width=100,
                height=100,
                tables=(
                    Table(
                        order=0,
                        rows=((TableCell(row=0, column=0, text="cell"),),),
                    ),
                ),
            ),
        )
    )
    adapter = PdfPageAdapter(
        cast(Any, SimpleNamespace(width=100, height=100, structured_view=structured.pages[0]))
    )
    tables = tuple(adapter.tables())

    assert len(tables) == 1
    assert tables[0].rows == 1
    assert tables[0].cells[0].text == "cell"

    engine_page = SimpleNamespace(
        width=100,
        height=100,
        structured_view=structured.pages[0],
        get_drawings=lambda: (),
    )
    engine_document = SimpleNamespace(
        closed=False,
        pages=(engine_page,),
        page_count=lambda: 1,
        selected_page_indexes=lambda selection: (0,),
    )
    document_adapter = PdfDocumentAdapter(cast(Any, engine_document))
    assert tuple(document_adapter.tables()) == tables
    assert tuple(document_adapter.drawings()) == ()


def test_xray_compatibility_facade_returns_xray_shape() -> None:
    assert inspect_xray(simple_pdf()) == {}


def test_pdfminer_compatibility_facade_exposes_text_and_layout() -> None:
    fixture = Path(__file__).parents[4] / "tests/fixtures/x-ray/tests/assets/rectangles_yes_2.pdf"
    if not fixture.exists():
        pytest.skip()

    text = extract_text(fixture, laparams=LAParams(char_margin=1.0))
    page = next(extract_pages(fixture))

    assert "defghi" in text
    assert page.pageid == 1
    assert page.bbox == (0.0, 0.0, 612.0, 792.0)
    assert any("defghi" in cast(Any, item).get_text() for item in page)
    character = next(
        cast(LTChar, child)
        for item in page
        for line in cast(Any, item)
        for child in line
        if isinstance(child, LTChar)
    )
    assert character.fontname == character.font_name
    assert character.adv == character.width
    assert len(character.matrix) == 6

    output = StringIO()
    extract_text_to_fp(fixture, output)
    assert "defghi" in output.getvalue()

    xml = StringIO()
    extract_text_to_fp(fixture, xml, output_type="xml")
    assert "<pages>" in xml.getvalue()

    html = StringIO()
    extract_text_to_fp(fixture, html, output_type="html")
    assert "<!doctype html>" in html.getvalue()


def test_pdfminer_facade_preserves_vertical_text_and_page_breaks() -> None:
    fixture = Path(__file__).parents[4] / "tests/fixtures/pdfminer.six/samples/simple3.pdf"
    if not fixture.exists():
        pytest.skip()

    text = extract_text(fixture)
    assert "あ\nい\nう\nえ\nお" in text
    assert text.endswith("\f")


def test_bad_redaction_operation_finds_text_under_later_opaque_rectangle() -> None:
    space = CoordinateSpace("page", CoordinateOrigin.BOTTOM_LEFT, width=100, height=100)
    rectangle = Rect(10, 10, 90, 30, space)
    text = TextSpan("secret", Rect(20, 12, 50, 28, space), sequence=1)
    drawing = Drawing(
        kind="fill",
        bbox=rectangle,
        sequence=2,
        items=(DrawingItem("re", rectangle),),
        fill=(0.0, 0.0, 0.0),
        fill_opacity=1.0,
        source=SourceRef(page_number=1, sequence=2),
    )

    class Page:
        info = PageInfo(0, 1, None, 100, 100, 0, space)

        def drawings(self) -> tuple[Drawing, ...]:
            return (drawing,)

        def text_spans(self) -> tuple[TextSpan, ...]:
            return (text,)

        def content_events(self) -> tuple[ContentEvent, ...]:
            return ()

        def images(self) -> tuple[object, ...]:
            return ()

        def render(
            self,
            *,
            dpi: float = 72.0,
            crop: tuple[float, float, float, float] | None = None,
        ) -> Raster:
            return Raster(bytes([0, 0, 0, 255]) * 4, 2, 2, 4, "rgba8", space, dpi)

    class Document:
        def pages(self, selection: object = None) -> tuple[Page, ...]:
            return (Page(),)

    report = BadRedactionOperation().run(cast(Any, Document()))

    assert report.analyzer_id == "analysis.bad-redactions"
    assert len(report.findings) == 1
    finding = report.findings[0]
    assert finding.code == "redaction.text-under-fill"
    assert finding.severity == "error"
    assert finding.page_number == 1
    assert finding.bbox == rectangle
    assert finding.evidence[0].value == "secret"
    assert finding.evidence[0].attributes["rectangle_sequence"] == 2
    assert finding.evidence[0].attributes["text_sequences"] == (1,)
    assert finding.evidence[0].source == drawing.source


def test_bad_redaction_operation_ignores_transparent_rectangles() -> None:
    space = CoordinateSpace("page", CoordinateOrigin.BOTTOM_LEFT, width=100, height=100)
    rectangle = Rect(10, 10, 90, 30, space)
    drawing = Drawing(
        kind="fill",
        bbox=rectangle,
        sequence=2,
        items=(DrawingItem("re", rectangle),),
        fill=(0.0, 0.0, 0.0),
        fill_opacity=0.5,
    )

    class Document:
        def pages(self, selection: object = None) -> tuple[object, ...]:
            class Page:
                info = PageInfo(0, 1, None, 100, 100, 0, space)

                def drawings(self) -> tuple[Drawing, ...]:
                    return (drawing,)

                def text_spans(self) -> tuple[TextSpan, ...]:
                    return ()

            return (Page(),)

    report = BadRedactionOperation().run(cast(Any, Document()))

    assert report.findings == ()
