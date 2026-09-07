# core-pdf architecture

An orientation to core-pdf's public boundary, extraction pipeline, source layout, and enforced
dependency direction.

---

The root package, `src/core_pdf`, owns PDF parsing, native extraction, rendering, and structured
output. `packages/core-pdf-ocr/src/core_pdf_ocr` owns OCR and vector text recognition. The two
distributions share a uv workspace and release version; the companion pins that exact core
version because it reuses internal extraction stages.

Core's `impl/_impl/extract/` initializer exposes only `extract_page` and `extract_document`;
stage internals are imported from their owning modules. Native extraction runs in this order:

```text
bytes → capture_page → native observations → extract_tables → layout → assemble_page
```

Native observations validate column shapes, dtypes, and selection bounds before applying
allocation-saving fast paths. Table detection owns candidate acceptance. Final projection
removes duplicate text using complete content and local geometry; it does not reject accepted
tables based on cell vocabulary or column count. Layout reconstructs text from positioned runs,
and emission preserves its character spacing, word boundaries, and literal line-end hyphens.
Table cells retain decoded letters, digits, and punctuation; cleanup does not join text based
on spelling or remove punctuation that resembles a decorative leader.

Output lines own their canonical text. Words are reconciled with that text, and inline spans
are retained only when they reproduce it, keeping plain text, Markdown, HTML, and JSON
consistent. JSON payloads are interned by object identity within each page; repeated ordered
nodes may reference one payload, while distinct objects with equal content remain distinct.

The companion enriches captured PDF evidence, selects recognition work, and combines native
and recovered text before using core's generic layout and output stages:

```text
        ┌── capture ──┐
bytes → │ capture_page│ → plan_page ─────────────────┐
        └─────────────┘    (evidence)       (decision)       │
                                                       ▼
                              ┌───────────────── WorkPlan ──────────────────┐
                              │                                             │
                    native text is trusted                        text is missing/untrusted
                              │                                             │
                              │                    recognize_page → RecognitionResult
                              │                                             │
                              └──────────► fuse_observations ◄──────────────┘
                                                  │
                                    extract_tables (complete table stage)
                                                  │
                                layout_blocks_with_evidence (layout)
                                                  │
                                    assemble_page → Page (extract_page)
```

## Source layout

```text
src/core_pdf/
  __init__.py            lazy public export table
  cli.py, __main__.py    command-line entry point
  api/document.py        PdfDocument, PdfPage, and their shared operation lifecycle
  api/compat/            independent third-party compatibility facades
  _vendor/               vendored third-party source and data
  impl/
    exceptions.py        error hierarchy
    types.py             PDF primitives, capture records, buffers, protocols, geometry aliases
    spec/                PDF specification implementation (see below)
    _impl/
      document/          source/lifecycle, recovery, page operations, document projections
      capture/           interpreter event recording, runs, glyphs, and page programs
      fonts/             Unicode recovery, substitute fonts, raster/backend adapters
      graphics/          output color, prepared images/shading, tolerant codec adapters
      extract/           native extraction, block layout, and tables
      layout/            text-line records, reconstruction, diagnostics, and word rules
      model/             shared geometry/text models, text rules, and page selections
      output/            structured document/page models, views, and serialization
      render/            display lists, raster kernels, targets, and page composition
      runtime/           engine-independent caching, arrays, and execution support
```

```text
packages/core-pdf-ocr/
  pyproject.toml         independently installable companion distribution
  src/core_pdf_ocr/
    api/document.py     PdfDocument/PdfPage subclasses with recognition extraction
    cli.py, __main__.py  core-pdf-ocr and python -m core_pdf_ocr
    _vendor/            Newstroke templates with their original license notices
    impl/extract/       routing, fusion, learned text, recognition-specific output policies
      ocr/              Tesseract, raster tasks, rescue passes, and vector recognition
```

### The `spec/s_NN_*` scheme

`spec/` contains PDF-defined semantics and algorithms from referenced standards. A choice
permitted to a PDF reader still belongs under `_impl/`: selected substitute fonts and ICC
profiles, Unicode recovery, output raster formats, clipping approximations, and malformed-input
recovery are reader policy. Public behavior is assembled in `_impl/document/`.

The content interpreter accepts a spec-owned `ContentSink` and font service. It emits exact
PDF state, paths, decoded character evidence, and scope events. `_impl/capture/` turns these
into extraction observations and rendering records, including path flattening and compatibility
geometry. Text advancement remains in the interpreter regardless of whether capture accepts a
character. `ImageSource` holds passive PDF image inputs; `_impl/graphics/images.py` prepares and
decodes them for a selected raster output.

`PdfStream` accepts a spec-owned `StreamDecoder`. Its default uses strict filter semantics;
the document reader supplies the codec/recovery adapter. Replacement and indirect resolution
preserve an explicitly supplied decoder. The strict syntax layer shares object parsing,
cross-reference row decoding, revision precedence, and tree traversal with recovery adapters;
resynchronization, guessed offsets, and skipped malformed entries live in `_impl/document/recovery/`.

Subpackages under `spec/` mirror chapters of the PDF specification:

| Package | PDF chapter |
| --- | --- |
| `s_07_syntax_primitives` | 7 — tokens, scanning, coercion |
| `s_07_syntax` | 7 — lexer, streams, xref, object model, resolver, text strings |
| `s_07_filters` | 7 — stream filters (Flate, LZW, CCITT, JBIG2, …) |
| `s_07_content` | 7 — content streams, operators, text state |
| `s_07_document` | 7 — catalog, page tree, metadata |
| `s_07_security` | 7 — encryption handlers |
| `s_08_graphics` | 8 — color-space/image semantics, functions, shading geometry, matrices |
| `s_09_fonts` | 9 — font formats, CMaps, encodings, widths, and semantic font services |
| `s_14_structure` | 14 — logical structure tree |

---

## Dependency direction

`core_pdf_ocr` depends on `core_pdf`; core never imports, discovers, or registers the companion.
Installing OCR therefore cannot change native extraction or compatibility-facade behavior.
Both implementation packages stay below their public APIs. Runtime cancellation, PDF capture,
ordinary rendering, text geometry, native tables, and structured output remain core-owned.

Recognition-specific routing, evidence, rasterization, artifact cleanup, and learned text live
in the companion. Core accepts prepared text and output products through internal generic stage
boundaries. Source/provenance labels in structured records remain ordinary data, so the companion
can preserve `ocr` and `hybrid` output without recognition branches in core. Word reconstruction
for text already embedded in PDFs, including hidden text layers, stays in core.

Import-linter contracts in the root `pyproject.toml` enforce runtime package direction. Spec
modules cannot import document recovery, capture, font policy, graphics output, or
derived-processing packages. Type-only imports follow the same organization rule; neutral
geometry and runtime array utilities are allowed. Runtime stays independent of PDF semantics.

Font and graphics adapters stay below capture and document composition. Capture can use source
recovery and passive document records, while page/document orchestration remains above it.
The extraction/render/layout/output layering remains enforced separately.

## Workspace validation

The authored test suite contains differential comparisons in
`tests/src/core_pdf/api/compat/differential`. Each case runs a compatibility facade and
its reference implementation against the same PDF. Reference corpora remain under
`tests/fixtures`.

Initialize the corpora and install the workspace's development dependencies:

```sh
git submodule update --init --recursive
uv sync --all-packages --all-groups
```

Run the differential suite and source checks with:

```sh
uv run --locked --group test --group vendor-test pytest -n auto
uv run --all-packages --group lint ruff check .
uv run --all-packages --group lint mypy
uv run --all-packages --group lint --group test --group vendor-test ty check
uv run --all-packages --group lint lint-imports
```

The default differential matrix uses each facade's own upstream corpus, plus selected
cross-corpus redaction cases for x-ray. To run every facade against every PDF fixture:

```sh
CORE_PDF_COMPAT_DIFFERENTIAL_FULL=1 \
  uv run --locked --group test --group vendor-test pytest -n auto
```

## Rendering constraints

Content frames own their graphics saves and clipping markers. A nested stream cannot pop a
parent frame's state, and leaving a stream unwinds its local clipping even after malformed
content. Resource lookup resolves dictionaries on demand while retaining indirect references,
so equivalent direct and indirect resource dictionaries preserve Form identity.

Raster paint state separates object opacity from isolated-group compositing opacity. A group's
opacity is applied once when it is composited into its parent. Stroke dash phase advances across
the segments of one subpath, and a zero-width stroke retains its device-pixel hairline semantics.

Images, soft masks, and stencils sample through the original image placement. Page crops and
captured Form clips restrict the destination without stretching the source; masks retain their
own resolution while sharing the color image's coordinates. Decoder inputs resolve indirect
dimensions and decode arrays without changing the source stream dictionary.

### Device colour

PDF leaves DeviceCMYK conversion undefined. `_impl/graphics/device_profiles.py` converts it through
the press profile in `_vendor/icc/` and uses the uncalibrated ink formula only when the profile is
unavailable. The ICC implementation documents its rendering intent, black-point compensation, and
optimized byte path alongside the code.
