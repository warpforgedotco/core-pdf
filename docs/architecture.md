# core-pdf architecture

An orientation guide for the all-in-one local PDF parsing engine. It covers how a PDF becomes
structured evidence, how parsing, rendering, analysis, compatibility, and writing share one
document model, how the source tree is organized, the naming conventions in use.

Roughly 87k lines of non-vendor Python live under `src/core_pdf/`, plus a vendored copy of
fontTools in `src/core_pdf/_vendor/` that is excluded from linting, typing, and formatting.

---

## 1. Public surface

`core_pdf/__init__.py` exports lazily. Rather than importing submodules at package import
time, it maps each public name to its defining module in an `internal_EXPORTS` table and
resolves it on first attribute access (`install_lazy_module_exports`). This keeps `import
core_pdf` cheap; it also means the public API is exactly the contents of that table, not
"whatever is importable".

Everything under `core_pdf.impl.*` is internal and may change without notice.

The two central objects:

- **`PdfDocument`** (`impl/engine/document.py`) — `open`, page access, canonical structured
  extraction (`extract`), and engine-owned PDF capabilities. It owns caches that must be
  shared across pages, notably the image cache. Structured serializers are kept on the
  structured IR instead of being duplicated here.
  The CLI drives it through `process_pdf` in `cli.py`.
- **`PdfPage`** (`impl/engine/page.py`, subclassing the spec-level page in
  `impl/engine/spec/s_07_document/page.py`) — per-page extraction and rendering.

The canonical public surface is the lazy export table in `core_pdf.__init__`. Document, page,
structured records, writers, runtime controls, and errors remain owned by their engine modules.

Compatibility facades under `core_pdf.api.compat.*` import those engine owners directly. There is
no shared compatibility state or conversion kernel. Each facade owns only the projection needed
for its target interface, and `compat.__init__` resolves convenience exports lazily so importing
one facade does not initialize all of them.

**Known defects.** The capture/render pipeline places some XObject-drawn vector text at
vertically mirrored y coordinates. Because neither the content-stream sequence nor raster
inspection can then see that the text paints above a fill, bad-redaction analysis reports
four documented false positives relative to real x-ray on text-over-rectangle forms; the
pinning tests are marked `xfail` in
`tests/src/core_pdf/compat/test_facade_characterization.py`.

---

## 2. The extraction pipeline

The pipeline lives in the `parse/` package, one module per stage. The stages run in roughly
this order:

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
                                  ParsedPage + ParseReport (parse_page)
                                                  │
                                    assemble_page → Page (extract_page)
```

| Module | Role |
| --- | --- |
| `parse/model.py` | Shared stage contracts: `ObservationBatch`, `PageEvidence`, typed route/fusion policies, `WorkPlan`, `RecognitionResult`, `RecognitionReport`, `ParsedPage`, and `ParseReport`. |
| `parse/capture.py` | Runs the canonical page program once and produces one cached `CapturedPage`: glyphs, drawings, images, observations, and routing evidence. |
| `parse/route.py` | Decides *how* to extract this page — native text, OCR, or both — producing a typed `WorkPlan`. |
| `parse/fusion.py` | Merges observations from multiple sources (native + OCR) into one coherent set. |
| `parse/tables.py` | Owns the complete table stage: grid, stream, and chart detection; cell merging; ordering; and nearby title/caption association. |
| `parse/ocr.py` | Tesseract integration: rasterization, region selection, hOCR parsing, adaptive rescue passes, and stroked-vector-text recovery. It returns observations and a `RecognitionReport`; caches hold reusable raster artifacts, not diagnostic side channels. |
| `parse/layout.py` | Groups runs into lines and blocks; column detection and reading order. |
| `parse/emit.py` | Text normalization, artifact removal, table/block reconciliation, and direct assembly of the canonical structured `Page`. |
| `parse/pipeline.py` | Lazy orchestration, single-flight locking, and product caching: `parse_page`, `extract_page`, and `page_extraction`. It assembles the typed `ParseReport` once per parsed page. |

`parse/__init__.py` exposes only pipeline entry points and shared stage models. Stage-specific
helpers are imported from their owner module. `internal_PageExtraction` memoizes each materialized
stage, so asking for text after tables (or tables after text) reuses capture, recognition, fusion,
layout, and emission products rather than running the content stream or OCR again.

### Table projections

Table extraction produces one canonical view. `parse/tables.py` attaches derived row and column
bands plus nearby `TableAssociatedText` records (title or caption) without rewriting cell
topology. Emission reconciles those tables with text blocks, removes duplicate or rejected
candidates, and assigns page-wide element order. `Page.tables`, `Document.table_view`, JSON, and
HTML/Markdown all consume that same final tuple. Structured JSON schema 4.0 therefore has one
`tables` field and no fallback table projection.

---

## 3. Source layout

```text
src/core_pdf/
  __init__.py            lazy public export table
  cli.py, __main__.py    CLI entry point (also carries Nuitka build directives)
  api/
    compat/              independent third-party facades over engine owners
  impl/
    runtime/             engine-independent infrastructure beneath the whole engine:
                         array_views (zero-copy numpy/memoryview), cache,
                         image_cache (byte-budgeted LRU, single-flight decoding),
                         execution (bounded thread runtime, budgets, runtime config)
    models.py            public extraction records (DrawingRecord, ImageRecord)
    objects.py           PdfStream and its lazy decoded-data cache
    exceptions.py        the PdfError hierarchy (incl. PdfDocumentClosedError)
    primitives.py        PdfName and friends (interned, precomputed hash)
    text.py              shared text kernel: collapse_ws, search_key,
                         collapse_character_spaced
    pages.py             PageSelection and its single normalization implementation
    engine/
      parse/             the extraction pipeline, one module per stage (see §2)
      render/            display lists, raster kernels, targets, and page composition
      page.py            PdfPage
      document.py        PdfDocument
      model/             the capture data model: geometry kernel, TextRun, glyph
                         records, columnar glyph storage. Beneath spec/ and layout/.
      layout/            heuristics only: line grouping, spatial index, geometry
                         quality, word frequencies
      structured/        document IR → markdown/HTML/JSON/CSV/TEI
      writing/           PDF output: objects, fonts, encryption, signatures
      spec/              PDF specification implementation (see below); document-local
                         Raw* records live in s_07_document/records.py
```

### Dependency direction

Three packages exist to be depended *upon* and must not depend upward:

| Package | May import | Status |
| --- | --- | --- |
| `impl/runtime/` | nothing internal | zero internal imports |
| `impl/engine/model/` | `impl/` base modules | one documented exception, below |
| `spec/s_07_syntax` | `impl/` base modules | clean |

`model/runs.py` is the exception: `TextRun` carries two private memo slots for results
the layout heuristics compute, and their annotations name types from `layout/`. Both
imports are under `TYPE_CHECKING`, so the runtime graph stays acyclic; an import
contract should list those two explicitly rather than relaxing the rule.

`layout/` is heuristics only and re-exports nothing — import from the owning module.
`LayoutLine` lives in `layout/lines.py` because it is what line grouping *produces*;
`TextRun` lives in `model/runs.py` because it is what capture *emits*. Likewise
`model/glyphs.py` owns the glyph records (including `GlyphSegment` and
`internal_materialize`) and `model/glyph_table.py` owns only the columnar storage that
consumes them, so the two no longer import each other.

### The `spec/s_NN_*` scheme

Subpackages under `spec/` mirror **chapters of the PDF specification**:

| Package | PDF chapter |
| --- | --- |
| `s_07_syntax` | 7 — lexer, tokens, xref, object model, resolver, coercion, text strings |
| `s_07_filters` | 7 — stream filters (Flate, LZW, CCITT, JBIG2, …) |
| `s_07_content` | 7 — content streams, operators, text state |
| `s_07_document` | 7 — catalog, page tree, metadata |
| `s_07_security` | 7 — encryption handlers |
| `s_08_graphics` | 8 — color spaces, ICC, images, matrices |
| `s_09_fonts` | 9 — font programs, CMaps, glyph decoding |
| `s_14_structure` | 14 — logical structure tree |

`s_07_syntax` is the COS layer — lexing, the object model, xref, and resolution are one
cohesive unit and were merged into a single package. Splitting them across `s_07_syntax` and a
separate `s_07_objects` produced a package-level import cycle (six edges each way) even though the
modules themselves form an acyclic graph. The package now depends only on `impl/exceptions`,
`impl/objects`, `impl/primitives`, and `impl/types`; keep it that way — nothing in `s_07_syntax`
may import from another `spec/` subpackage or from `impl/engine/`.

Operator and filter metadata each have one declarative owner. Content-operator categories,
dispatch names, text-only scan tables, Type 3 replay membership, and cached lexer keywords derive
from `s_07_syntax/content_operators.py`; stream-filter aliases, decoders, predictor support, and
decode-cache policy derive from `s_07_filters/registry.py`. Add metadata there instead of creating
another parallel table.

The spec-level `s_07_document/document.py` is the concrete owner of catalog, page-tree, labels,
navigation, forms, attachments, and optional-content behavior. `document_xref.py` remains a
separate cohesive base for xref scanning/recovery, while `document_pages.py` contains only the
read-only lazy page sequence and shared inherited-page constants.

### Device colour and the default CMYK profile

PDF 32000-1 leaves all three Device spaces device-dependent. Treating
DeviceGray and DeviceRGB as sRGB is renderer behaviour rather than anything the
spec defines, but it is what viewers do and it costs nothing: the components
map straight onto the components of the output space. DeviceCMYK has no such
correspondence, and the spec gives no conversion to RGB at all. The
uncalibrated `255*(1-ink)*(1-black)` formula that fills the gap is visibly
wrong -- it renders the process inks as saturated screen primaries and 100% K as
pure black -- so `s_08_graphics/device_profiles.py` runs DeviceCMYK through a
real press profile vendored in `_vendor/icc/`, and keeps the ink formula only as
a fallback for an install where that file is missing.

Two details in `s_08_graphics/icc_profiles.py` are load-bearing and easy to get
wrong. LUT tags are selected by **relative colorimetric** intent
(`select_icc_lut_tag` prefers the `1` suffix) because PDF 8.6.5.8 makes that the
default rendering intent, and because a perceptual table already black-points
its output, which silently disables the next step. That step is **black point
compensation**: relative colorimetric alone reproduces press black as the dark
grey it measures on paper, so a CMYK page renders washed out on a screen that
can show real black. `internal_detect_black_point` finds the darkest colour the
profile can actually reach -- a rich black mixing all four inks, not 100% K --
by asking the profile's `B2A` table which inks it would use for L\* = 0, and
`internal_compensate_black_point` scales the connection space so that lands on
zero. Together these reproduce lcms2 to within 0.21 of 255 mean.

Anything reaching an ICC LUT from an image should call `apply_uint8` rather than
`apply`: it collapses the input curves into a byte-indexed gather and
deduplicates repeated colours, which is worth roughly an order of magnitude on
photographic CMYK.

### Golden rasters

Rendering uses direct module owners rather than a barrel module: display-list records and options
live in `render/display.py`, pure raster kernels in `render/kernels.py`, the mutable paint target in
`render/target.py`, page composition in `render/page.py`, and the raster value object in
`render/raster_image.py`.

`tests/test_rendering_golden.py` pins the RGBA output of the corpus. The versioned
manifest in `tests/snapshots/raster/first_page_scale1.json` has two contracts:
ordinary pages retain exact SHA-256 digests, while pages that paint irreversible
JPEG 2000 images retain a lossless canonical PNG plus measured RGB error limits.
OpenJPEG implements that lossy 9/7 transform with different floating-point paths
on x86 and ARM, so requiring an exact digest there would characterize a wheel's
CPU implementation rather than the renderer. Tolerant entries still require
identical dimensions and alpha, and separately bound the maximum channel error,
changed RGB samples, and total RGB error. Their PNG digest is also recorded for
an exact fast path and fixture-integrity check.

The always-on layer renders 24 documents chosen by greedy line-cover — together
they exercise the rendering modules across the full 224-document reach — plus
every irreversible-JPX page. `CORE_PDF_RASTER_GOLDEN_FULL=1` sweeps the complete
corpus as one independently schedulable test per document. CI sets that variable
and uses two pytest workers, so the whole corpus gates a merge without serializing
all raster work; a focused macOS ARM job exercises the portable JPX contract as
well. Reaching every *line* of `engine/render/` is not the same as pinning every
*pixel*: four digests once
sat in the snapshot that no commit could produce, unnoticed because only the
subset ran in CI.

A refactor that preserves behavior must leave the manifest and PNG references
untouched. Regenerate them with
`uv run python scripts/update_raster_golden.py` only when an output change is
intentional and review the complete diff. Reference regeneration is restricted
to the pinned Ubuntu x86_64 codec environment; `--allow-noncanonical-write` is
an explicitly destructive bootstrap escape hatch, not the normal workflow. It
records noncanonical provenance, so the resulting tree intentionally fails its
provenance test until regenerated canonically. The updater records its
environment in the manifest, renders documents in two isolated
worker processes, classifies the page's JPEG 2000 codestreams, preserves
calibrated limits, and refuses to invent a tolerance for a newly encountered
irreversible stream. Recompute the
covering subset with `scripts/raster_cover.py` after large structural changes.
