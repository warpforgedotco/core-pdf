"""Inventory and forensic analysis algorithms behind the v0 document surface.

Functions here either consume the ENGINE document (``document: Any`` holding a
``core_pdf.impl`` document) or the protocol-level adapter surface
(``document: PdfDocumentProtocol``) when the algorithm needs converted records.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from contextlib import suppress
from hashlib import sha256
from re import finditer
from typing import Any, cast

from core_pdf.impl.engine.parse.pipeline import page_extraction
from core_pdf.impl.objects import PdfReference, PdfStream

from .errors import InvalidRequest
from .models import (
    AccessibilityInventory,
    ActionInventory,
    AnnotationInventory,
    ArchivalManifest,
    ContentDependencyRecord,
    ContentEventKind,
    DocumentAnalysisSnapshot,
    DocumentFingerprint,
    DocumentInventory,
    EmbeddedResourceRecord,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceLayer,
    EvidenceNode,
    FormInventory,
    IncrementalAnalysisPlan,
    NativeFeatureInventory,
    ObjectGraphEdge,
    ObjectGraphNode,
    ObjectGraphReport,
    ObjectInspection,
    ObjectRoundTripManifest,
    ObjectRoundTripVerification,
    OptionalContentLayerRecord,
    PageContentSummary,
    PageResourceInventory,
    ResourceDependencyGraph,
    ResourceDiagnostic,
    RevisionInventory,
    RevisionObjectRecord,
    Severity,
    SourceRef,
    StructureElementRecord,
    compare_fingerprints,
    plan_incremental_analysis,
)
from .protocols import PageSelection, PdfDocumentProtocol


def document_inventory(document: Any) -> DocumentInventory:
    """Summarize the engine document's objects, markers, and recovery state."""
    catalog = document.catalog()
    markers = {
        "JavaScript": False,
        "JS": False,
        "OpenAction": False,
        "AA": False,
        "Launch": False,
    }
    seen: set[int] = set()

    def walk(value: object) -> None:
        resolved = document.resolver.resolve(value)
        identity = id(resolved)
        if identity in seen:
            return
        if isinstance(resolved, dict):
            seen.add(identity)
            for key, child in resolved.items():
                name = str(key).lstrip("/")
                if name in markers:
                    markers[name] = True
                walk(child)
        elif isinstance(resolved, list):
            for child in resolved:
                walk(child)

    walk(catalog)
    type_counts: dict[str, int] = {}
    for key, entry in document.xref.items():
        if not entry.in_use:
            continue
        try:
            value = document.resolver.resolve(PdfReference(key >> 16, key & 0xFFFF))
        except (ValueError, KeyError):
            continue
        type_name = "Untyped"
        if isinstance(value, dict):
            type_key = next((item for item in value if str(item).lstrip("/") == "Type"), None)
            if type_key is not None:
                type_name = str(value[type_key]).lstrip("/")
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
    return DocumentInventory(
        byte_count=len(document.raw_data),
        object_count=sum(1 for entry in document.xref.values() if entry.in_use),
        page_count=int(document.page_count()),
        encrypted=document.decipher is not None,
        xref_recovered=bool(document.xref_was_recovered),
        page_tree_recovered=bool(document.page_tree_was_recovered),
        has_attachments=bool(document.embedded_files()),
        has_outlines=bool(document.iter_outlines()),
        has_javascript=markers["JavaScript"] or markers["JS"],
        has_open_action=markers["OpenAction"] or markers["AA"] or markers["Launch"],
        trailer_keys=tuple(sorted(str(key).lstrip("/") for key in document.trailer_dict)),
        object_types=tuple(sorted(type_counts.items())),
    )


def native_features(document: Any) -> NativeFeatureInventory:
    """Report which native PDF features the engine document declares."""
    catalog = document.catalog()
    names = {str(key).lstrip("/") for key in catalog}
    return NativeFeatureInventory(
        has_optional_content="OCProperties" in names,
        has_acroform="AcroForm" in names,
        has_xfa="XFA" in names,
        has_metadata_stream="Metadata" in names,
        has_names="Names" in names,
        has_collection="Collection" in names,
        has_embedded_files=bool(document.embedded_files()),
        has_incremental_revision=any(
            str(key).lstrip("/") == "Prev" for key in document.trailer_dict
        ),
        is_linearized=b"/Linearized" in bytes(document.raw_data[:1024]),
    )


def action_inventory(document: Any) -> ActionInventory:
    """Count action attachment points and resolved action types."""
    counts: dict[str, int] = {}
    typed_counts: dict[str, int] = {}
    source_objects: set[int] = set()
    seen: set[tuple[int, str]] = set()

    def visit(value: object, object_number: int) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                name = str(key).lstrip("/")
                if name in {"A", "AA", "OpenAction", "Launch"}:
                    marker = (object_number, name)
                    if marker not in seen:
                        seen.add(marker)
                        counts[name] = counts.get(name, 0) + 1
                        source_objects.add(object_number)
                if name == "S":
                    action_type = str(child).lstrip("/")
                    if action_type in {"JavaScript", "URI", "GoTo", "Launch", "SubmitForm"}:
                        typed_counts[action_type] = typed_counts.get(action_type, 0) + 1
                        source_objects.add(object_number)
                visit(child, object_number)
        elif isinstance(value, list):
            for child in value:
                visit(child, object_number)

    for key, entry in document.xref.items():
        if not entry.in_use:
            continue
        object_number = key >> 16
        try:
            value = document.resolver.resolve(PdfReference(object_number, key & 0xFFFF))
        except (ValueError, KeyError):
            continue
        visit(value, object_number)
    # Container keys such as /OpenAction and /AA identify attachment points,
    # not additional action objects.  Count the resolved /S entries as the
    # canonical action total; otherwise one /OpenAction /S /GoTo is reported
    # twice (once for the container and once for the action type).
    action_types = typed_counts
    return ActionInventory(
        action_count=sum(action_types.values()),
        javascript_count=typed_counts.get("JavaScript", 0),
        open_action_count=counts.get("OpenAction", 0),
        additional_action_count=counts.get("AA", 0),
        launch_count=typed_counts.get("Launch", 0),
        action_type_counts=action_types,
        source_objects=tuple(sorted(source_objects)),
    )


def optional_content_layers(document: Any) -> tuple[OptionalContentLayerRecord, ...]:
    """List the optional-content layers declared by the engine document."""
    document.load_oc_layers()
    return tuple(
        OptionalContentLayerRecord(name=name, enabled=enabled)
        for name, enabled in sorted((document.oc_layers or {}).items())
    )


def revisions(document: Any) -> RevisionInventory:
    """Locate incremental-update revisions in the raw document bytes."""
    data = bytes(document.raw_data)
    offsets = tuple(int(match.group(1)) for match in finditer(rb"startxref\s+(\d+)", data))
    eof_offsets = tuple(match.start() for match in re.finditer(rb"%%EOF", data))
    eof_ends = tuple(offset + len(b"%%EOF") for offset in eof_offsets)
    starts = (0,) + eof_ends[:-1]
    ends = eof_ends[:-1] + (len(data),)
    revision_ranges = tuple(zip(starts, ends, strict=True))
    revision_sha256 = tuple(sha256(data[start:end]).hexdigest() for start, end in revision_ranges)
    revision_count = max(1, data.count(b"%%EOF"))
    return RevisionInventory(
        revision_count=revision_count,
        startxref_offsets=offsets,
        eof_offsets=eof_offsets,
        revision_ranges=revision_ranges,
        revision_sha256=revision_sha256,
        has_incremental_updates=revision_count > 1 or len(offsets) > 1,
    )


def revision_objects(document: Any) -> tuple[RevisionObjectRecord, ...]:
    """Attribute in-use objects to the revision that wrote them."""
    data = bytes(document.raw_data)
    ranges = revisions(document).revision_ranges
    entries = sorted(
        (entry.offset, key >> 16, entry.generation)
        for key, entry in document.xref.items()
        if entry.in_use and entry.offset > 0
    )
    records: list[RevisionObjectRecord] = []
    for offset, object_number, generation in entries:
        revision_index = next(
            (index for index, (start, end) in enumerate(ranges) if start <= offset < end),
            len(ranges) - 1,
        )
        revision_end = ranges[revision_index][1]
        next_offset = next(
            (candidate[0] for candidate in entries if candidate[0] > offset), revision_end
        )
        end = min(next_offset, revision_end)
        raw = data[offset:end]
        records.append(
            RevisionObjectRecord(
                revision_index=revision_index,
                object_number=object_number,
                generation=generation,
                offset=offset,
                byte_count=len(raw),
                sha256=sha256(raw).hexdigest(),
            )
        )
    return tuple(records)


def document_fingerprint(document: Any) -> DocumentFingerprint:
    """Hash the document bytes, page content, and every resolvable object."""
    page_hashes: list[tuple[int, str]] = []
    for page in document.structured_pages:
        elements = tuple(
            (
                type(element).__name__,
                getattr(element, "order", None),
                getattr(element, "text", ""),
                getattr(element, "bbox", None),
                getattr(getattr(element, "kind", None), "value", None),
                getattr(element, "level", None),
            )
            for element in page.elements
        )
        payload = repr((page.page_number, page.width, page.height, elements)).encode()
        page_hashes.append((page.page_number, sha256(payload).hexdigest()))
    object_hashes: list[tuple[int, int, str]] = []
    for key, entry in sorted(document.xref.items()):
        if not entry.in_use:
            continue
        object_number = key >> 16
        reference = PdfReference(object_number, entry.generation)
        with suppress(KeyError, ValueError):
            value = document.resolver.resolve(reference)
            digest = sha256(repr(value).encode()).hexdigest()
            object_hashes.append((object_number, entry.generation, digest))
    return DocumentFingerprint(
        document_sha256=sha256(document.raw_data).hexdigest(),
        page_sha256=tuple(page_hashes),
        object_sha256=tuple(object_hashes),
    )


def incremental_plan(document: Any, baseline: DocumentFingerprint) -> IncrementalAnalysisPlan:
    """Plan which analyses must rerun after the baseline fingerprint."""
    return plan_incremental_analysis(compare_fingerprints(baseline, document_fingerprint(document)))


def object_graph(document: Any) -> ObjectGraphReport:
    """Trace object reachability from the trailer root."""
    reachable: set[int] = set()
    edges: set[tuple[int, int, str | None]] = set()
    visiting: set[int] = set()

    def visit_reference(reference: PdfReference, parent: int | None, key: str | None) -> None:
        target = reference.object_number
        if parent is not None:
            edges.add((parent, target, key))
        if target in visiting or target in reachable:
            return
        visiting.add(target)
        reachable.add(target)
        with suppress(KeyError, ValueError):
            visit_value(document.resolver.resolve(reference), target, None)
        visiting.remove(target)

    def visit_value(value: object, parent: int | None, key: str | None) -> None:
        match value:
            case PdfReference():
                visit_reference(value, parent, key)
            case PdfStream():
                visit_value(value.dictionary, parent, key)
            case dict():
                for item_key, child in value.items():
                    visit_value(child, parent, str(item_key).lstrip("/"))
            case list() | tuple():
                for child in value:
                    visit_value(child, parent, key)

    root = document.trailer_dict.get("Root")
    root_objects: tuple[int, ...] = ()
    if isinstance(root, PdfReference):
        root_objects = (root.object_number,)
        visit_reference(root, None, "Root")
    nodes: list[ObjectGraphNode] = []
    for key, entry in sorted(document.xref.items()):
        if not entry.in_use:
            continue
        object_number = key >> 16
        try:
            value = document.resolver.resolve(PdfReference(object_number, entry.generation))
        except (KeyError, ValueError):
            value = None
        object_type = type(value).__name__
        if isinstance(value, dict):
            type_key = next((item for item in value if str(item).lstrip("/") == "Type"), None)
            if type_key is not None:
                object_type = str(value[type_key]).lstrip("/")
        nodes.append(
            ObjectGraphNode(
                object_number, entry.generation, object_type, object_number in reachable
            )
        )
    return ObjectGraphReport(
        nodes=tuple(nodes),
        edges=tuple(ObjectGraphEdge(*edge) for edge in sorted(edges)),
        root_objects=root_objects,
    )


def inspect_object(document: Any, object_number: int) -> ObjectInspection:
    """Describe one indirect object, including its raw byte span."""
    key = next(
        (
            key
            for key, entry in document.xref.items()
            if entry.in_use and key >> 16 == object_number
        ),
        None,
    )
    if key is None:
        raise InvalidRequest(f"unknown object number: {object_number}")
    entry = document.xref[key]
    raw_data = bytes(document.raw_data)
    next_offsets = sorted(
        other.offset
        for other in document.xref.values()
        if other.in_use and other.offset > entry.offset
    )
    raw_end = next_offsets[0] if next_offsets else len(raw_data)
    raw_bytes = raw_data[entry.offset : raw_end]
    value = document.resolver.resolve(PdfReference(object_number, entry.generation))
    is_stream = isinstance(value, PdfStream)
    dictionary = value.dictionary if is_stream else value
    keys = (
        tuple(sorted(str(item).lstrip("/") for item in dictionary))
        if isinstance(dictionary, dict)
        else ()
    )
    object_type = type(value).__name__
    if isinstance(dictionary, dict):
        type_key = next((item for item in dictionary if str(item).lstrip("/") == "Type"), None)
        if type_key is not None:
            object_type = str(dictionary[type_key]).lstrip("/")
    return ObjectInspection(
        object_number=object_number,
        generation=entry.generation,
        object_type=object_type,
        dictionary_keys=keys,
        value_repr=repr(value),
        is_stream=is_stream,
        raw_offset=entry.offset,
        raw_length=len(raw_bytes),
        raw_sha256=sha256(raw_bytes).hexdigest(),
    )


def verify_object_roundtrip(document: Any, object_number: int) -> ObjectRoundTripVerification:
    """Reopen the raw bytes and compare one object's raw hash."""
    from core_pdf import PdfDocument

    source_hash = inspect_object(document, object_number).raw_sha256
    with PdfDocument.open(bytes(document.raw_data)) as reopened:
        reopened_hash = inspect_object(reopened, object_number).raw_sha256
    return ObjectRoundTripVerification(
        object_number=object_number,
        source_sha256=source_hash,
        reopened_sha256=reopened_hash,
        passed=source_hash is not None and source_hash == reopened_hash,
    )


def verify_object_roundtrips(
    document: Any, object_numbers: Iterable[int]
) -> ObjectRoundTripManifest:
    """Round-trip several objects and collect the verdicts."""
    results = tuple(
        verify_object_roundtrip(document, object_number)
        for object_number in sorted(set(object_numbers))
    )
    return ObjectRoundTripManifest(objects=results)


def embedded_resources(document: Any) -> tuple[EmbeddedResourceRecord, ...]:
    """List embedded files with sizes and content hashes."""
    return tuple(
        EmbeddedResourceRecord(
            name=str(item.name),
            filename=str(item.filename),
            byte_count=len(item.data),
            sha256=sha256(item.data).hexdigest(),
            media_type=(
                str(item.stream.dictionary.get("Subtype"))
                if "Subtype" in item.stream.dictionary
                else None
            ),
        )
        for item in document.embedded_files()
    )


def structure_elements(document: Any) -> Iterator[StructureElementRecord]:
    """Walk the tagged-structure tree into flat records."""
    structure = getattr(document, "structure", None)
    if structure is None:
        return

    def walk(items: Iterable[object], depth: int) -> Iterator[StructureElementRecord]:
        for item in items:
            if item.__class__.__name__ != "StructureElement":
                continue
            page = getattr(item, "page", None)
            page_number = getattr(page, "page_number", None)
            yield StructureElementRecord(
                role=str(getattr(item, "role", "")),
                depth=depth,
                page_number=page_number,
                title=getattr(item, "title", None),
                language=getattr(item, "language", None),
                alternate_description=getattr(item, "alternate_description", None),
                actual_text=getattr(item, "actual_text", None),
                attributes=getattr(item, "attributes", None) or {},
                source=SourceRef(
                    page_index=page_number - 1 if isinstance(page_number, int) else None,
                    page_number=page_number,
                    stage="tagged-structure",
                ),
            )
            yield from walk(cast(Iterable[object], item), depth + 1)

    yield from walk(structure, 1)


def content_summaries(
    document: Any, *, pages: PageSelection | None = None
) -> Iterator[PageContentSummary]:
    """Summarize each page's content-stream operators and filters."""
    for index in document.selected_page_indexes(pages):
        engine_page = document.pages[index]
        preflight = page_extraction(engine_page).preflight()
        features = preflight.features
        counts = features.operator_counts
        operator_counts = {
            "text": counts.text,
            "image": counts.image,
            "vector_path": counts.vector_path,
            "vector_paint": counts.vector_paint,
            "graphics_state": counts.graphics_state,
            "unknown": counts.unknown,
            "malformed": counts.malformed,
        }
        yield PageContentSummary(
            page_number=engine_page.page_number,
            operator_counts=operator_counts,
            stream_filters=tuple(features.stream_filters),
            raw_stream_bytes=features.raw_stream_bytes,
            decoded_stream_bytes=features.decoded_stream_bytes,
            malformed_operator_count=counts.malformed,
            classification=preflight.recommendation.page_class.value,
        )


def archival_manifest(document: PdfDocumentProtocol) -> ArchivalManifest:
    """Combine inventory, reachability, and metadata into one manifest."""
    inventory = document.inventory()
    graph = document.object_graph()
    metadata = document.get_metadata()
    info = metadata.get("info", metadata)
    title = info.get("Title") if isinstance(info, Mapping) else None
    return ArchivalManifest(
        document_sha256=document.fingerprint().document_sha256,
        byte_count=inventory.byte_count,
        page_count=inventory.page_count,
        object_count=inventory.object_count,
        reachable_object_count=sum(1 for node in graph.nodes if node.reachable),
        unreachable_object_count=len(graph.unreachable_objects),
        encrypted=inventory.encrypted,
        has_javascript=inventory.has_javascript,
        has_attachments=inventory.has_attachments,
        title=str(title) if title else None,
    )


def resource_inventory(
    document: PdfDocumentProtocol, *, pages: PageSelection | None = None
) -> Iterator[PageResourceInventory]:
    """Inventory fonts and images referenced by each page."""
    for page in document.pages(pages):
        spans = tuple(page.text_spans())
        font_names = tuple(sorted({span.font_name for span in spans if span.font_name is not None}))
        image_count = sum(1 for _ in page.images())
        yield PageResourceInventory(
            page_number=page.info.number,
            image_count=image_count,
            font_names=font_names,
            has_images=image_count > 0,
            has_fonts=bool(font_names),
        )


def content_dependencies(
    document: PdfDocumentProtocol, *, pages: PageSelection | None = None
) -> Iterator[ContentDependencyRecord]:
    """Relate each content event to the resources it depends on."""
    for page in document.pages(pages):
        for event in page.content_events():
            names: set[str] = set()
            if event.text is not None and event.text.font_name:
                names.add(event.text.font_name)
            if event.image is not None:
                names.add("image")
            names.update(
                str(value)
                for key, value in event.data.items()
                if key.casefold() in {"font", "xobject", "resource", "name"}
                and isinstance(value, str)
            )
            yield ContentDependencyRecord(
                page_number=page.info.number,
                sequence=event.sequence,
                kind=event.kind,
                resource_names=tuple(sorted(names)),
                source=event.source,
            )


def resource_dependency_graph(document: PdfDocumentProtocol) -> ResourceDependencyGraph:
    """Aggregate content dependencies into a page-to-resource graph."""
    dependencies: dict[int, set[str]] = {}
    for dependency in content_dependencies(document):
        dependencies.setdefault(dependency.page_number, set()).update(dependency.resource_names)
    return ResourceDependencyGraph(
        page_resources=tuple(
            (page_number, tuple(sorted(resources)))
            for page_number, resources in sorted(dependencies.items())
        )
    )


def evidence_graph(
    document: PdfDocumentProtocol, *, pages: PageSelection | None = None
) -> EvidenceGraph:
    """Build the provenance graph tying pages, events, and structure together."""
    nodes: list[EvidenceNode] = []
    edges: list[EvidenceEdge] = []
    selected: set[int] = set()
    for page in document.pages(pages):
        selected.add(page.info.number)
        page_id = f"page:{page.info.number}"
        page_source = SourceRef(
            page_index=page.info.index, page_number=page.info.number, stage="page"
        )
        nodes.append(EvidenceNode(page_id, EvidenceLayer.PDF_OBJECT, "page", page_source))
        for event in page.content_events():
            node_id = f"page:{page.info.number}:event:{event.sequence}"
            if event.kind is ContentEventKind.TEXT and event.text is not None:
                value = event.text.text
                layer = EvidenceLayer.NATIVE_TEXT
            elif event.kind is ContentEventKind.IMAGE:
                value = "image"
                layer = EvidenceLayer.CONTENT_EVENT
            else:
                value = event.kind.value
                layer = EvidenceLayer.CONTENT_EVENT
            nodes.append(EvidenceNode(node_id, layer, value, event.source, event.bbox))
            edges.append(EvidenceEdge(page_id, node_id, "contains"))
        for index, annotation in enumerate(page.annotations()):
            node_id = f"page:{page.info.number}:annotation:{index}"
            nodes.append(
                EvidenceNode(
                    node_id,
                    EvidenceLayer.STRUCTURED,
                    annotation.subtype,
                    annotation.source,
                    annotation.bbox,
                )
            )
            edges.append(EvidenceEdge(page_id, node_id, "contains"))
        for index, field in enumerate(page.form_fields()):
            node_id = f"page:{page.info.number}:form:{index}"
            nodes.append(
                EvidenceNode(
                    node_id,
                    EvidenceLayer.STRUCTURED,
                    field.name,
                    field.source,
                    field.bbox,
                )
            )
            edges.append(EvidenceEdge(page_id, node_id, "contains"))
    hierarchy: list[tuple[int, int]] = []
    for index, element in enumerate(document.structure_elements()):
        if element.page_number is not None and element.page_number not in selected:
            continue
        node_id = f"structure:{index}"
        value = element.role
        if element.alternate_description:
            value = f"{value}: {element.alternate_description}"
        node = EvidenceNode(node_id, EvidenceLayer.STRUCTURED, value, element.source)
        nodes.append(node)
        while hierarchy and hierarchy[-1][1] >= element.depth:
            hierarchy.pop()
        if hierarchy:
            edges.append(EvidenceEdge(f"structure:{hierarchy[-1][0]}", node_id, "parent"))
        hierarchy.append((index, element.depth))
        if element.page_number is not None:
            edges.append(EvidenceEdge(f"page:{element.page_number}", node_id, "contains-structure"))
    return EvidenceGraph(tuple(nodes), tuple(edges))


def resource_diagnostics(document: Any) -> tuple[ResourceDiagnostic, ...]:
    """Diagnose missing fonts, undecodable images, and risky graphics state.

    ``document`` is the v0 document adapter: the checks need both converted
    records (spans, images) and raw engine surfaces (page dictionaries, the
    object resolver).
    """
    engine = document.document
    diagnostics: list[ResourceDiagnostic] = []
    if engine.xref_was_recovered:
        diagnostics.append(
            ResourceDiagnostic(
                code="xref.recovered",
                message="Cross-reference data was recovered during parsing.",
                severity=Severity.WARNING,
            )
        )
    for page in document.pages():
        text_spans = tuple(page.text_spans())
        if text_spans and not any(span.font_name for span in text_spans):
            diagnostics.append(
                ResourceDiagnostic(
                    code="font.resource-missing",
                    message="Text was extracted without a resolved page font resource.",
                    severity=Severity.ERROR,
                    page_number=page.info.number,
                    resource_name="font",
                )
            )
        for image in page.images():
            if image.data is None:
                diagnostics.append(
                    ResourceDiagnostic(
                        code="image.decode-unavailable",
                        message="Image metadata was parsed but decoded payload is unavailable.",
                        page_number=page.info.number,
                        resource_name="image",
                    )
                )
        page_dict = getattr(page.page, "page_dict", {})
        contents = next((value for key, value in page_dict.items() if str(key) == "Contents"), None)
        content_stream = engine.resolver.resolve(contents)
        content_data = getattr(content_stream, "data", b"")
        if isinstance(content_data, (bytes, bytearray, memoryview)) and re.search(
            rb"(?:^|\s)gs(?:\s|$)", bytes(content_data)
        ):
            diagnostics.append(
                ResourceDiagnostic(
                    code="graphics.transparency-operator",
                    message=("The page invokes a graphics state that may introduce transparency."),
                    severity=Severity.WARNING,
                    page_number=page.info.number,
                    resource_name="content-stream",
                )
            )
        resources = engine.resolver.resolve(
            next((value for key, value in page_dict.items() if str(key) == "Resources"), None)
        )
        if isinstance(resources, dict):
            fonts = next((value for key, value in resources.items() if str(key) == "Font"), None)
            resolved_fonts = engine.resolver.resolve(fonts)
            if isinstance(resolved_fonts, dict):
                for name, value in resolved_fonts.items():
                    font = engine.resolver.resolve(value)
                    has_base_font = isinstance(font, dict) and any(
                        str(key) == "BaseFont" for key in font
                    )
                    if not has_base_font:
                        diagnostics.append(
                            ResourceDiagnostic(
                                code="font.resource-missing",
                                message="A page font resource has no usable BaseFont entry.",
                                severity=Severity.ERROR,
                                page_number=page.info.number,
                                resource_name=str(name),
                            )
                        )
            color_spaces = next(
                (value for key, value in resources.items() if str(key) == "ColorSpace"), None
            )
            resolved_spaces = engine.resolver.resolve(color_spaces)
            if isinstance(resolved_spaces, dict):
                for name, value in resolved_spaces.items():
                    resolved = engine.resolver.resolve(value)
                    color_name = (
                        str(resolved[0]).lstrip("/")
                        if isinstance(resolved, list) and resolved
                        else str(resolved).lstrip("/")
                    )
                    if color_name == "ICCBased":
                        profile = resolved[1] if len(resolved) > 1 else None
                        profile = engine.resolver.resolve(profile)
                        profile_data = getattr(profile, "data", None)
                        if (
                            not isinstance(profile_data, (bytes, bytearray, memoryview))
                            or not profile_data
                        ):
                            diagnostics.append(
                                ResourceDiagnostic(
                                    code="color.icc-profile-invalid",
                                    message=(
                                        "An ICCBased color space has no readable profile stream."
                                    ),
                                    severity=Severity.ERROR,
                                    page_number=page.info.number,
                                    resource_name=str(name),
                                )
                            )
                    elif color_name in {"DeviceGray", "DeviceRGB", "DeviceCMYK"}:
                        diagnostics.append(
                            ResourceDiagnostic(
                                code="color.device-space",
                                message=(
                                    f"Color space {name} uses {color_name} without an "
                                    "explicit profile."
                                ),
                                severity=Severity.WARNING,
                                page_number=page.info.number,
                                resource_name=str(name),
                            )
                        )
    for summary in content_summaries(engine):
        if summary.malformed_operator_count:
            diagnostics.append(
                ResourceDiagnostic(
                    code="content.malformed-operators",
                    message=(
                        f"Page {summary.page_number} contains "
                        f"{summary.malformed_operator_count} malformed content operators."
                    ),
                    severity=Severity.ERROR,
                    page_number=summary.page_number,
                    resource_name="content-stream",
                )
            )
        if any(operator in summary.operator_counts for operator in ("gs", "CA", "ca")):
            diagnostics.append(
                ResourceDiagnostic(
                    code="graphics.transparency-operator",
                    message=(
                        "The page uses graphics-state operators that may introduce transparency."
                    ),
                    severity=Severity.WARNING,
                    page_number=summary.page_number,
                    resource_name="content-stream",
                )
            )
    return tuple(diagnostics)


def form_inventory(
    document: PdfDocumentProtocol, *, pages: PageSelection | None = None
) -> FormInventory:
    """Summarize form fields, their types, values, and duplicate names."""
    field_pages = tuple(
        (page.info.number, field) for page in document.pages(pages) for field in page.form_fields()
    )
    fields = tuple(field for _, field in field_pages)
    names: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    page_numbers: set[int] = set()
    populated = 0
    for page_number, field in field_pages:
        names[field.name] = names.get(field.name, 0) + 1
        type_counts[field.field_type] = type_counts.get(field.field_type, 0) + 1
        if field.value:
            populated += 1
        page_numbers.add(page_number)
    return FormInventory(
        field_count=len(fields),
        populated_count=populated,
        empty_count=len(fields) - populated,
        page_count=len(page_numbers),
        field_type_counts=type_counts,
        duplicate_names=tuple(sorted(name for name, count in names.items() if count > 1)),
    )


def annotation_inventory(
    document: PdfDocumentProtocol, *, pages: PageSelection | None = None
) -> AnnotationInventory:
    """Count annotations and links, grouped by subtype and page."""
    annotation_pages: set[int] = set()
    link_pages: set[int] = set()
    subtype_counts: dict[str, int] = {}
    annotation_count = 0
    link_count = 0
    external_link_count = 0
    for page in document.pages(pages):
        annotations = tuple(page.annotations())
        links = tuple(page.links())
        annotation_count += len(annotations)
        link_count += len(links)
        if annotations:
            annotation_pages.add(page.info.number)
        if links:
            link_pages.add(page.info.number)
        for annotation in annotations:
            subtype = annotation.subtype.casefold() or "unknown"
            subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
        external_link_count += sum(1 for link in links if link.url)
    return AnnotationInventory(
        annotation_count=annotation_count,
        link_count=link_count,
        page_count=len(annotation_pages | link_pages),
        subtype_counts=subtype_counts,
        external_link_count=external_link_count,
    )


def accessibility_inventory(
    document: PdfDocumentProtocol, *, pages: PageSelection | None = None
) -> AccessibilityInventory:
    """Measure tagged structure, alternate text, and document language."""
    selected: set[int] = set()
    images = 0
    tables = 0
    for page in document.pages(pages):
        selected.add(page.info.number)
        images += sum(1 for _ in page.images())
        tables += sum(1 for _ in page.tables())
    elements = tuple(
        element
        for element in document.structure_elements()
        if element.page_number is None or element.page_number in selected
    )
    figures = sum(element.role.casefold() in {"figure", "fig"} for element in elements)
    alternate = sum(bool(element.alternate_description) for element in elements)
    metadata = document.get_metadata()
    info = metadata.get("info", metadata)
    language = info.get("Lang") if isinstance(info, Mapping) else None
    title = info.get("Title") if isinstance(info, Mapping) else None
    return AccessibilityInventory(
        tagged_element_count=len(elements),
        language_element_count=sum(bool(element.language) for element in elements),
        alternate_text_count=alternate,
        figure_count=figures,
        image_count=images,
        table_count=tables,
        missing_alternate_text_count=max(0, images - alternate),
        document_language=str(language) if language else None,
        has_title=bool(title),
    )


def analysis_snapshot(document: PdfDocumentProtocol) -> DocumentAnalysisSnapshot:
    """Capture the standard document-level analysis reports in one record."""
    return DocumentAnalysisSnapshot(
        inventory=document.inventory(),
        native_features=document.native_features(),
        actions=document.action_inventory(),
        accessibility=document.accessibility_inventory(),
        forms=document.form_inventory(),
        annotations=document.annotation_inventory(),
        revisions=document.revisions(),
        revision_objects=tuple(document.revision_objects()),
        embedded_resources=tuple(document.embedded_resources()),
        content_summaries=tuple(document.content_summaries()),
        resource_diagnostics=tuple(document.resource_diagnostics()),
    )


__all__ = (
    "accessibility_inventory",
    "action_inventory",
    "analysis_snapshot",
    "annotation_inventory",
    "archival_manifest",
    "content_dependencies",
    "content_summaries",
    "document_fingerprint",
    "document_inventory",
    "embedded_resources",
    "evidence_graph",
    "form_inventory",
    "incremental_plan",
    "inspect_object",
    "native_features",
    "object_graph",
    "optional_content_layers",
    "resource_dependency_graph",
    "resource_diagnostics",
    "revision_objects",
    "revisions",
    "structure_elements",
    "verify_object_roundtrip",
    "verify_object_roundtrips",
)
