# Public API and extension contracts

The extension surface is `core_pdf.api`. It is the high-level contract for the
all-in-one local PDF engine: parsing, extraction, search, rendering, analysis, validation,
transformation, writing, inspection, and compatibility projections all operate through the
same document and page capabilities. It is independent of the implementation modules under
`core_pdf.impl`.

```python
from core_pdf.api import PdfDocument

with PdfDocument.open("document.pdf") as document:
    for page in document.pages():
        for event in page.content_events():
            print(event.kind, event.bbox)
```

The contract uses concrete `PdfDocument`, `PdfPage`, and `PdfEditor` capability objects.
They cover content events, text, drawings, images, rendering, analysis operations, and
editing without exposing the engine object model. Record methods use plain names —
`annotations()`, `links()`, `images()`, `tables()`, `form_fields()`, `chunks()` — and
return typed records only.

Geometry records carry an explicit `CoordinateSpace` (and `Rect.to_origin()` converts
between coordinate origins); source provenance is represented by `SourceRef`. Analysis
findings carry a `Severity` value — a `StrEnum`, so comparisons against `"error"` and
friends keep working.

Both document and page capabilities expose structured output through `to_json()`,
`to_html()`, and `to_markdown()`. Documents additionally support page-selected
structured JSON through `to_structured_json(pages=...)`; page and document records
also expose typed geometry and text diagnostics.
Documents additionally provide deterministic `to_csv(pages=...)` text/layout rows and
`to_tei(pages=...)` TEI-like XML with page-boundary markers for downstream tooling.

Local cancellation and operation contexts are available from
`core_pdf.api.execution` (`LocalExecutionContext`, `LocalCancellationToken`,
`AnalysisCache`). The structured IR types behind `structured_document` and
`structured_view` — `Document`, `Page`, `Block`, `Table`, and friends — are re-exported
at `core_pdf.api.structured` so callers can name them without importing internals.

The root `PdfDocument` and `PdfPage` objects retain engine primitives such as `extract()` and
`render()`. Structured serializers, element/chunk projections, and compatibility-shaped
conveniences live on the capability objects so the same behavior is not maintained twice. Code
importing `core_pdf.impl.*` is using internal APIs and should not be used as an extension
contract.

## Native operations

Every operation subclasses one template-method base: `run(document, context=None,
options=None)` returns an `AnalysisReport` of typed `AnalysisFinding` records, and all
operations honour a `pages` option. The local bad-redaction analyzer, for example:

```python
from core_pdf.api import PdfDocument
from core_pdf.api.v0.operations import BadRedactionOperation

with PdfDocument.open("document.pdf") as document:
    report = BadRedactionOperation().run(document, options={"inspect_raster": True})
    for finding in report.findings:
        print(finding.page_number, finding.bbox, finding.message)
```

Operations are local, typed, and return provenance-bearing reports rather than adding
feature-specific methods to the document object. Invalid option values raise
`InvalidRequest`.

### Aggregate quality preflight

Applications that want the standard local validation set run the aggregate operation:

```python
from core_pdf.api import PdfDocument, Severity
from core_pdf.api.v0.operations import QualityPreflightOperation

with PdfDocument.open("document.pdf") as document:
    report = QualityPreflightOperation().run(document)
    if report.metrics["error_count"]:
        for finding in report.findings:
            if finding.severity is Severity.ERROR:
                print(finding.code, finding.message)
```

`QualityPreflightOperation` composes the current high-level validators for document
integrity, accessibility, forms, links, attachments, fonts, images, geometry, annotations,
and the existing compliance preflight. The component operations remain independently
available: `AccessibilityValidationOperation`, `FormValidationOperation`,
`LinkValidationOperation`, `AttachmentValidationOperation`, `FontValidationOperation`,
`ImageValidationOperation`, `GeometryValidationOperation`,
`AnnotationValidationOperation`, and `DocumentIntegrityOperation`. All findings retain
severity, page/object provenance where available, and remediation guidance. The aggregate
operation is local and deterministic; it does not invoke a server, LLM, or VLLM.

The public document also exposes a deterministic structural inventory:

```python
inventory = document.inventory()
print(inventory.object_count, inventory.encrypted, inventory.has_javascript)
```

The inventory reports object and byte counts, recovery flags, security/action markers,
attachments, outlines, trailer keys, and PDF object types without exposing parser internals.

For incremental local analysis, `document.fingerprint()` returns SHA-256 fingerprints for
the complete input, each structured page, and each resolvable in-use PDF object. The
page/object records can be compared across runs to skip unchanged analysis work.

`compare_fingerprints(before, after)` returns changed page numbers and added, removed,
or modified object references without reopening either source document.

`plan_incremental_analysis(diff)` converts that result into an `IncrementalAnalysisPlan`
with the affected page/object selections and a conservative full-scan flag for changes
that cannot be localized.

`verify_preservation(before, after, expected_unchanged_pages=...)` produces a
`PreservationManifest` for post-write verification. It records changed pages and objects
and fails explicitly when a protected page changed.

The editor facade also exposes `commit_verified(...)`, which commits, reopens the emitted
bytes locally, fingerprints the result, and returns the manifest in one operation.

`document.archival_manifest()` combines the document hash, byte/page/object counts,
reachability, encryption, JavaScript, attachments, and title into one preservation record
that can be serialized alongside an archival or transformation output.

`document.native_features()` exposes local PDF-native markers for optional content,
AcroForms/XFA, metadata streams, names, collections, embedded files, incremental
revisions, and linearization.

`document.embedded_resources()` inventories embedded payload names, filenames, sizes,
media types, and SHA-256 hashes without requiring callers to load the bytes into memory
again.

`document.optional_content_layers()` exposes optional-content group names and their
default enabled state, making hidden-layer review explicit and deterministic.

`document.resource_inventory()` provides per-page image counts and observed font names,
with explicit presence flags for high-level resource analysis.

`document.content_dependencies()` exposes event sequence, content kind, observed resource
names, and source provenance for per-page content-stream dependency tracing.

`document.resource_dependency_graph()` compacts those events into page-to-resource edges
and exposes the distinct resource set for incremental invalidation.

`document.resource_diagnostics()` reports recovered cross-reference data and resources
whose decoded payload is unavailable, making malformed or partial extraction explicit.

`document.content_summaries()` exposes parser-owned per-page operator counts, stream
filters, raw/decoded byte sizes, malformed-operator counts, and preflight classification.
`document.form_inventory()` provides a document-level count of parsed form fields,
population status, field types, duplicate names, and page coverage.
`document.annotation_inventory()` provides annotation/link counts, subtype counts,
page coverage, and external-link counts for deterministic review.
`document.search(query, mode=..., threshold=..., region=..., pages=...)` is the one
search entry point: exact, normalized, regex, and fuzzy matching, including multi-word
line matches and coordinate-region filtering.
Scores and matched text are returned in each `SearchHit`.
Search hits also preserve the originating text `SourceRef`.
`document.evidence_graph()` joins page nodes to content-event, native-text, drawing,
image, annotation, and form-field nodes with typed provenance edges.
Tagged structure elements are included as structured nodes with page relationships and
alternate-text-bearing values when available.
Table records likewise carry page/stage provenance through `TableRecord.source`.
The returned graph supports deterministic `to_dict()` and `to_json()` evidence-manifest
serialization.
`editor.commit_redactions_verified(...)` applies source-level structured redactions,
reopens the emitted PDF locally, and reports any requested query strings that remain.
`editor.commit_sanitized_verified(...)` composes local removal of metadata, annotations,
links, forms, attachments, and outlines, then verifies the reopened inventories.
Action objects are included in the verification by default.
`document.accessibility_inventory()` reports tagged-structure, language, alternate-text,
figure, image, and table coverage with an explicit missing-alternate-text count.
It also reports document language and title presence for PDF/UA-style review.
`editor.commit_accessibility_repair_verified(...)` writes requested title/language
metadata and verifies those signals after reopening the output.
`document.reading_order()` returns a deterministic visual line sequence with page numbers,
geometry, and source provenance for accessibility and structured exports.
`ChunkRecord.sources` retains page-level retrieval provenance for every chunk.
`ChunkRecord.element_bboxes` retains typed page-coordinate geometry for contributing
elements when the parser recovered it.
`document.action_inventory()` reports resolved action types, JavaScript/open/additional/
launch counts, and source object numbers for security review.
`document.inspect_object(number)` returns a JSON-safe object summary with type, generation,
dictionary keys, stream status, and a diagnostic value representation.
`document.revisions()` also exposes deterministic `%%EOF` offsets for revision-boundary
inspection.
`RevisionInventory.revision_ranges` provides the corresponding byte intervals for
incremental-section hashing and comparison.
`RevisionInventory.revision_sha256` provides stable digests for those intervals.
`document.analysis_snapshot()` aggregates the principal local inventories and per-page
content summaries into one typed, deterministic analysis result.
Snapshots support deterministic `to_dict()` and `to_json()` manifest serialization.

`document.revisions()` reports raw revision markers, `startxref` offsets, and incremental
update presence. It is intentionally an evidence-level history indicator; historical
object snapshots can be added later without overstating what the current parser knows.

Manifests provide deterministic `to_dict()` and `to_json()` methods. Compare two records
with `compare_archival_manifests(before, after)` to obtain changed fields and document
identity changes.

For low-level PDF forensics, the same adapter exposes a typed object graph. It records
references from the trailer root through dictionaries, arrays, and streams, while also
listing in-use but unreachable objects:

```python
graph = document.object_graph()
for node in graph.nodes:
    print(node.object_number, node.object_type, node.reachable)
print(graph.unreachable_objects)
```

The built-in forensic operation turns those signals into an analysis report with
object-level provenance:

```python
report = ForensicAnalysisOperation().run(document)
```

It reports recovery, encryption, executable actions, embedded files, and unreachable
objects without invoking a remote service or a generative model.

Retrieval callers can use `document.chunks()` for typed, bounded records. Each
`ChunkRecord` retains page numbers, source element IDs, and element kinds alongside
the text, plus the active `section_path`, so downstream indexing does not discard
document structure.

`LayoutAnalysisOperation` provides deterministic page-structure signals for retrieval:
repeated edge text is reported as probable header/footer furniture, and separated block
clusters are reported as columns. Every finding retains the originating structured block
and page geometry.

`StructureAnalysisOperation` summarizes parser-owned classical element kinds—headings,
lists, tables, figures, captions, quotes, and code where available—and attaches a
representative structured evidence record to each result. It does not infer semantics
with an LLM or rewrite the source document.

`CitationAnalysisOperation` recognizes common numeric (`[12]`) and author-year
(`Smith, 2024`) citation forms, plus reference-section headings. These are local
signals with source blocks and page geometry; they are not claims that a citation has
been fully resolved against a bibliography.

`FigureCaptionAnalysisOperation` associates figures with the nearest horizontally
compatible caption using structured geometry and reports uncaptioned figures for review.
The association is deterministic and remains local to the parsed PDF.

`SectionHierarchyAnalysisOperation` exposes heading depth, list counts, maximum nesting,
and skipped heading levels. It emits page-backed heading evidence and warns when the
parser's hierarchy contains a depth jump.

`IdentifierAnalysisOperation` detects DOI, ISBN, URL, email, and date-like identifiers
using local regular expressions. Matches retain the exact extracted spelling and page
evidence so normalization can remain reversible.

`ReferenceEntryAnalysisOperation` recognizes common numbered bibliography entries after
a reference heading and extracts optional DOI and author-year signals while preserving
the complete original entry as evidence.

`CompliancePreflightOperation` provides conservative, profile-labeled PDF/A-style and
PDF/UA-style checks for metadata, encryption, JavaScript, and tagged structure. Document
title and figure alternate-text coverage are owned by
`AccessibilityValidationOperation` and deliberately not duplicated here (the aggregate
`QualityPreflightOperation` runs both). It is a deterministic preflight layer, not a
claim of complete external standards certification.

`normalize_metadata(values)` provides a reversible local preparation step for editor
workflows: common keys are canonicalized and whitespace is normalized, while unknown
metadata fields remain present for callers to review.

`plan_accessibility_remediation(report)` converts an `AccessibilityValidationOperation`
report's missing title, tagged-structure, and figure alternate-text findings into
explicit `RemediationAction` records. It produces a review plan only; it does not invent
semantic descriptions or mutate the source PDF.

Transactional editors also support structured redaction application:

```python
with PdfDocument.open("document.pdf") as document:
    document.edit().apply_redactions({1: ((72, 72, 180, 100),)}).commit("sanitized.pdf")
```

Covered text lines are removed from the structured output while unrelated content is
preserved. This is a sanitized structured-document operation; it does not claim to
preserve arbitrary source-PDF object semantics.

Pages expose both `text_spans()` and native `text_characters()` geometry. When
available, each span also carries its `TextCharacter` records for precise
occlusion, search, and layout analysis.

## x-ray compatibility

The upstream FreelawProject x-ray repository is tracked as the `tests/fixtures/x-ray`
submodule for fixture and behavior reference. Core-pdf provides a local facade
with the same result shape:

```python
from core_pdf.api.compat import inspect_xray

findings = inspect_xray("document.pdf")
```

The facade uses core-pdf’s local analyzer and never invokes a server or the
vendored project at runtime.

Run the vendored x-ray suite against the facade with:

```sh
PYTHONPATH=src uv run --with requests --with PyMuPDF \
  --with numpy --with tesserocr --with imagecodecs \
  python scripts/run_xray_compat_tests.py -q
```

The runner replaces only the public `xray.inspect` function. Tests that call
x-ray’s private PyMuPDF helpers directly remain upstream implementation tests,
not compatibility tests; the runner deselects one such test
(`test_bad_redactions_on_single_page`), which exercises upstream x-ray against
the installed PyMuPDF and fails with current PyMuPDF releases even without
core-pdf involved.

## pdfminer.six compatibility

The pdfminer.six reference source is tracked at `tests/fixtures/pdfminer.six`. The
local facade exposes the common high-level entry points and LT-shaped layout
objects:

```python
from core_pdf.api.compat import LAParams, extract_pages, extract_text

text = extract_text("document.pdf", laparams=LAParams())
for page in extract_pages("document.pdf"):
    for element in page:
        print(element.bbox, element.get_text())
```

These functions use core-pdf’s parser and native character geometry locally;
pdfminer.six is retained as a compatibility reference and test corpus.

## Other high-level compatibility facades

The remaining local, high-level facades follow the same namespace layout:

```python
from core_pdf.api.compat.pdfplumber import open as open_pdf
from core_pdf.api.compat.pymupdf import open as open_fitz
from core_pdf.api.compat.pikepdf import Pdf
from core_pdf.api.compat.unstructured import partition_pdf
from core_pdf.api.compat.llamaindex import load_data
```

They are projections over core-pdf’s local parser and document model. They do not
start a server, call a remote service, or require the reference projects at runtime.
They share one kernel (`api/compat/_common.py`) for document opening, byte writing,
lifecycle, and geometry coercion. Use `core_pdf.api` directly for typed records,
structured JSON/HTML/Markdown output, geometry diagnostics, text diagnostics, and
editing.

Tagged PDFs expose their logical structure through the same contract:

```python
for element in document.structure_elements():
    print(element.depth, element.role, element.actual_text or element.title)
```

The stream is empty for untagged documents. Each record preserves the mapped role,
hierarchy depth, accessibility text, page provenance, and structure attributes.

Editors also expose local document security primitives:

```python
document.edit().encrypt("user-password", owner_password="owner-password").commit("locked.pdf")
document.edit().sign(cms_provider).commit("signed.pdf")
```

`cms_provider` implements `SignatureProvider.sign(data) -> bytes` and produces detached
CMS/PKCS#7 bytes. Encryption and signing cannot be combined in one output.

An editor with no changes preserves the original PDF bytes exactly, including content
streams, images, fonts, and metadata that are outside the structured projection.

Page geometry can be changed through the same editor:

```python
document.edit().set_page_geometry(1, rotation=90, cropbox=(0, 0, 500, 700))
```

Geometry-only edits use an incremental update and retain the original page content.

Removing annotations or links likewise uses an incremental page update when no other
structured changes are pending, preserving the original page content and resources.
