# All-in-one local PDF parsing engine roadmap

Core-pdf is intended to be the all-in-one, local PDF engine: one composable system for
opening, parsing, extracting, searching, rendering, analyzing, validating, transforming,
writing, and inspecting PDF documents.

## 1. Source-level secure transformations

- Remove redacted content from compressed streams, object streams, and every prior
  incremental revision, including encoded string representations.
- Add source-level scrubbing for hidden layers, metadata streams, attachments, and
  unreachable sensitive objects with explicit byte-level postconditions.
- Add flattening for annotations, forms, optional content, and transparency where the
  result is provably equivalent in the selected rendering profile.
- Add adversarial redaction and sanitization corpus coverage, including compressed and
  incrementally updated PDFs.

### 2. Classical document understanding

- Detect and represent lists, footnotes, equations, appendices, indexes, and repeated
  document templates as typed structures.
- Improve deterministic entity, date, clause, definition, and domain-rule extraction
  with configurable grammars and alternatives.
- Add confidence and competing-evidence records to all remaining heuristic analyses.
- Add reading-order validation and repair for multi-column, rotated, and mixed-direction
  pages.

### 3. Retrieval and export completeness

- Complete TEI-like XML, evidence-manifest, and lossless structured export formats.
- Add query/result provenance graphs that explain every normalized or fuzzy match.
- Add incremental query invalidation and cache tests across page, object, and revision
  dependencies.

### 4. Compliance and accessibility depth

- Validate PDF/UA table headers, row/column semantics, reading order, artifacts, and
  annotation appearance states beyond the current high-level checks.
- Add deterministic tagged-structure repair for table semantics, reading order, and
  artifact marking, followed by reopen-and-validate verification.
- Complete PDF/A conformance for embedded fonts, ICC output intents, XMP metadata,
  transparency, associated files, and profile-specific restrictions.
- Add profile-aware validation for PDF/A-1, PDF/A-2, PDF/A-3, and PDF/UA variants.

### 5. PDF-native inspection and editing

- Add lossless editing of arbitrary existing PDF objects while preserving untouched raw
  bytes and indirect-reference identity where safe.
- Add resource deduplication, linearization analysis/repair, portfolios, associated files,
  optional-content editing, and PDF 2.0 feature inspection.
- Add form calculation scripts, validation actions, choice hierarchies, and complete
  appearance regeneration for all supported widget types.
- Add annotation appearance repair and content-operator tracing with source offsets.

### 6. Compatibility and corpus maturity

- Continue high-level API coverage for `core_pdf.api.compat.pdfminer`, `pdfplumber`,
  `pymupdf`, `pypdf`, `pikepdf`, `unstructured`, `llamaindex`, and `xray` only where it
  maps cleanly to the shared capability model.
- Run compatibility behavior matrices against each vendored/reference test corpus.
- Expand corpus, golden rendering, malformed-input, security, and performance coverage
  for every completed capability.

## Dependency policy

The core remains Python 3.13+, local, deterministic, and based on the existing parser,
OCR, NumPy, and image-codec stack. No LLM, VLLM, hosted API, vector database, or server
is required. Future non-generative local model adapters may be optional extensions, but
they must not become dependencies of the core contracts.

### Explicit non-goals

- Full command-line compatibility with every third-party PDF library.
