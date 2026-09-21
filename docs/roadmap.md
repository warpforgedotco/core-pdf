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
- Compare redaction and sanitization behavior with reference implementations on compressed
  and incrementally updated PDFs.

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
- Add incremental query invalidation across page, object, and revision dependencies.

### 4. Compliance and accessibility depth

The version/extension foundation and optional veraPDF companion are implemented.
Recognizing a version or profile does not establish full support.
Historical lexical rules, namespace-aware transitive structure roles, 16-bit image
processing, and version-sensitive ColorDodge/ColorBurn now use that foundation.
Page UserUnit now reaches physical rendering and OCR sizing while retaining raw native
geometry. Historical PDFDocEncoding/WinAnsi assignments select verified earlier rules;
the font audit records where the sources do not justify a version branch. Rendering
intent and PDF 2.0 black-point controls reach the current color-conversion pipeline.
Real veraPDF positive and negative fixtures cover all 15 advertised targets.

- Validate PDF/UA table headers, row/column semantics, reading order, artifacts, and
  annotation appearance states beyond the current high-level checks.
- Add deterministic tagged-structure repair for table semantics, reading order, and
  artifact marking, followed by reopen-and-validate verification.
- Connect external validation findings to inspection and repair workflows while
  retaining the original bytes as validation evidence.
- Continue the remaining chapter work, including NChannel mixing hints and separation-aware overprinting,
  halftones, transparency-group blending spaces, and Type 3 glyph programs.
- Add a verified PDF/X backend when an executable and usable license are available;
  this integration is currently deferred.

### 5. PDF-native inspection and editing

- Add lossless editing of arbitrary existing PDF objects while preserving untouched raw
  bytes and indirect-reference identity where safe.
- Add resource deduplication, linearization analysis/repair, portfolios, associated files,
  optional-content editing, and PDF 2.0 feature inspection.
- Add form calculation scripts, validation actions, choice hierarchies, and complete
  appearance regeneration for all supported widget types.
- Add annotation appearance repair and content-operator tracing with source offsets.

### 6. Compatibility and corpus maturity

- Continue high-level API compatibility for `core_pdf.api.compat.pdfminer`, `pdfplumber`,
  `pypdf`, `pikepdf`, `unstructured`, `llamaindex`, and `xray` only where it
  maps cleanly to the shared capability model.
- Maintain differential comparisons in `tests/src/core_pdf/api/compat/differential` against
  each reference corpus using `uv run --locked --extra unstructured --group test --group vendor-test pytest -n auto`.
- Expand differential comparisons over malformed, security-sensitive, and real-world PDFs.
  Set `CORE_PDF_COMPAT_DIFFERENTIAL_FULL=1` to run the exhaustive facade/fixture matrix.

## Dependency policy

The core remains Python 3.14+, local, deterministic, and based on the existing parser,
NumPy, and image-codec stack. OCR lives in the separate `core-pdf-ocr` package, which depends
on core-pdf; the core never imports or discovers it. No LLM, VLLM, hosted API, vector database, or server
is required. Future non-generative local model adapters may be optional extensions, but
they must not become dependencies of the core contracts.

### Explicit non-goals

- Full command-line compatibility with every third-party PDF library.
