# core-pdf architecture

An orientation guide for the all-in-one local PDF parsing engine. It covers how a PDF becomes
structured evidence, how parsing, rendering, analysis, compatibility, and writing share one
document model, how the source tree is organized, the naming conventions in use, and — most
importantly — **which code is deliberately optimized and must not be "cleaned up"**.

Roughly 68k lines of non-vendor Python live under `src/core_pdf/`, plus a vendored copy of
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
  shared across pages, notably the image cache. Structured serializers and element/chunk
  projections are kept on the v0 adapter and structured IR instead of being duplicated here.
  The CLI drives it through `process_pdf` in `cli.py`.
- **`PdfPage`** (`impl/engine/page.py`, subclassing the spec-level page in
  `impl/engine/spec/s_07_document/page.py`) — per-page extraction and rendering.

The canonical local capability surface is `core_pdf.api.v0`. Its concrete document and page
objects project engine-owned data into stable typed records. The engine owns structured
extraction, high-level records (images, annotations, links, forms, outlines, and attachments),
and the transactional `PdfDocumentEditor`. Compatibility facades under `core_pdf.api.v0.compat.*`
are intentionally high-level projections; they may retain a structured snapshot for synthetic
documents, but opened-document reads and page mutations should delegate to the engine surface.

The v0 package is organized by role:

- `document.py` — concrete `PdfDocument` / `PdfPage` / `PdfEditor` capabilities. Record
  methods use plain names (`annotations()`, `links()`,
  `images()`, `tables()`, `form_fields()`, `chunks()`, `words()`) and return the typed
  records from `models.py` exclusively — there are no parallel untyped variants. One
  `search(query, mode=..., threshold=..., region=..., pages=...)` method covers exact,
  normalized, regex, and fuzzy matching.
- `convert.py` — the single IR→api conversion point (`page_space`, `to_rect`,
  `to_*_record`); adapter methods delegate here rather than converting inline.
- `models.py` — the frozen typed records, plus `Severity` (a `StrEnum`, so string
  comparisons keep working) and `SourceRef` provenance.
- `types.py` — input aliases and the small cancellation/signature callback contracts.
- `operations/` — the analysis operations as a package (`base.py` holds the
  template-method `AnalysisOperation`; `checks.py`, `validation.py`, `content.py`,
  `forensics.py`, `preflight.py`, `transform.py` hold the concrete operations).
- `inspection.py` — the inventory/forensics algorithms (`document_inventory`,
  `object_graph`, `evidence_graph`, `resource_diagnostics`, …) as free functions the
  adapter delegates to.
- `verification.py` — the `commit_*_verified` bodies behind the editor facade.
- `structured.py` — public promotion of the engine's structured IR (see below).
- `errors.py` — the stable error surface: `ApiError`, `InvalidRequest`,
  `OperationCancelled`, `DocumentClosed`.

`PdfDocument.structured` and `PdfPage.structured_view` return the engine IR types themselves —
`core_pdf.api.v0.structured` re-exports `Document`, `Page`, `Block`, `Table`, and friends at
an api-sanctioned path so compat code can name them without importing `core_pdf.impl`. New
compatibility facades use the private engine projection hook; application code opens the
concrete v0 document directly. Facades should not recreate document/page ownership locally.

Compat facades share a small utility kernel in `api/v0/compat/_common.py` and one state owner
in `api/v0/compat/state.py`. The state owner handles both opened and synthetic snapshots;
`OpenedState` and `SyntheticState` are compatibility markers, not parallel implementations.
It also caches its v0 document/page projections so facades do not rebuild adapters per call.
Facade packages import engine types from the kernel (or from `api.v0.structured`) instead of
drilling into `impl` directly.

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
bytes → │ capture_page│ → preflight_page → plan_page ──┐
        └─────────────┘     (classify)     (decide)    │
                                                       ▼
                              ┌───────────────── WorkPlan ──────────────────┐
                              │                                             │
                    native text is trusted                        text is missing/untrusted
                              │                                             │
                              │                                    recognize_page  (ocr)
                              │                                             │
                              └──────────► fuse_observations ◄──────────────┘
                                                  │
                                    extract_tables (tables)
                                                  │
                                     layout_blocks (layout)
                                                  │
                                     assemble_page (emit)
                                                  │
                                     parse_page / extract_page (pipeline)
```

| Module | Role |
| --- | --- |
| `parse/model.py` | The dataclasses everything else speaks in: `ObservationBatch`, `PageEvidence`, `WorkPlan`, `ParsedPage`, `PagePreflight`, plus the OCR candidate record and its factory. Pure leaf — imports nothing from the other stages. |
| `parse/capture.py` | Runs the content stream and turns it into evidence: glyphs, drawings, images. Also classifies the page (`preflight_page`). |
| `parse/route.py` | Decides *how* to extract this page — native text, OCR, or both — producing a `WorkPlan`. |
| `parse/fusion.py` | Merges observations from multiple sources (native + OCR) into one coherent set. |
| `parse/tables.py` | Grid and stream table detection, cell merging, deduplication. |
| `parse/ocr.py` | Tesseract integration: rasterization, region selection, hOCR parsing, adaptive rescue passes, and stroked-vector-text recovery. The largest stage by far. |
| `parse/layout.py` | Groups runs into lines and blocks; column detection and reading order. |
| `parse/emit.py` | Text normalization and artifact removal; assembles the final structured `Page`. |
| `parse/pipeline.py` | Orchestration and caching: `parse_page`, `extract_page`, `page_extraction`. |

The modules form a strict layering in that order — each imports only from those above it, so
there are no import cycles.
act_text()` after `extract_tables()` does not re-run the pipeline.

### Table projections

Table extraction produces one merged view. During assembly
(`internal_merge_structured_tables` in `parse/emit.py`), each layout-projected table is
geometry-matched to its annotated structured counterpart; the merged table carries the
page-wide layout `order` together with the structured annotations — derived `row_bands`
and `column_bands`, and nearby `TableAssociatedText` records associated as `title` or
`caption` (annotations never rewrite cell topology). `Page.tables`,
`Page.structured_tables`, and `Document.table_view` all expose this merged tuple, and
JSON and HTML/Markdown rendering consume it directly. The one asymmetry: when emission
removes every projected table as duplicate text, `Page.tables` is empty while
`Page.structured_tables` retains the raw annotated parse-time tuple, which rendering
falls back to.

---

## 3. Source layout

```text
src/core_pdf/
  __init__.py            lazy public export table
  cli.py, __main__.py    CLI entry point (also carries Nuitka build directives)
  api/
    v0/                  the versioned capability surface (see §1):
                         adapters, convert, models, protocols, operations/,
                         inspection, verification, structured, errors, execution
      compat/            third-party facades + the shared _common kernel
  impl/
    models.py            engine record types (DrawingRecord, ImageRecord, Raw* records)
    exceptions.py        the PdfError hierarchy (incl. PdfDocumentClosedError)
    primitives.py        PdfName and friends (interned, precomputed hash)
    text.py              shared text kernel: collapse_ws, search_key,
                         collapse_character_spaced
    pages.py             the single page-selection resolver (resolve_page_selection)
    engine/
      parse/             the extraction pipeline, one module per stage (see §2)
      rendering.py       rasterization
      page.py            PdfPage
      document.py        PdfDocument
      execution.py       worker pools, shared memory, runtime config
      array_views.py     zero-copy numpy/memoryview helpers
      image_cache.py     byte-budgeted LRU with single-flight decoding
      layout/            lines, spatial index, geometry kernel, word frequencies
      structured/        document IR → markdown/HTML/JSON/CSV/TEI
      writing/           PDF output: objects, fonts, encryption, signatures
      spec/              PDF specification implementation (see below)
```

### The `spec/s_NN_*` scheme

Subpackages under `spec/` mirror **chapters of the PDF specification**:

| Package | PDF chapter |
| --- | --- |
| `s_07_syntax` | 7 — lexer, tokens, xref |
| `s_07_objects` | 7 — object model, resolver, coercion |
| `s_07_filters` | 7 — stream filters (Flate, LZW, CCITT, JBIG2, …) |
| `s_07_content` | 7 — content streams, operators, text state |
| `s_07_document` | 7 — catalog, page tree, metadata |
| `s_07_security` | 7 — encryption handlers |
| `s_08_graphics` | 8 — color spaces, ICC, images, matrices |
| `s_09_fonts` | 9 — font programs, CMaps, glyph decoding |
| `s_14_structure` | 14 — logical structure tree |

### Golden rasters

`tests/test_rendering_golden.py` hashes the RGBA output of the corpus. The
always-on layer renders 24 documents chosen by greedy line-cover — together they
execute every line of `rendering.py` the full 224 reach — and
`CORE_PDF_RASTER_GOLDEN_FULL=1` sweeps all of them. A refactor that is meant to
preserve behavior must leave `tests/snapshots/raster/first_page_scale1.json`
untouched; regenerate it with `CORE_PDF_UPDATE_RASTER_GOLDEN=1` only when an
output change is intended, and review the diff. Recompute the covering subset
with `scripts/raster_cover.py` after large structural changes.

### Verifying a performance-sensitive change

```sh
uv run --group benchmark pytest --benchmark-only --benchmark-save=baseline   # before
uv run --group benchmark pytest --benchmark-only --benchmark-compare=baseline # after
```

The benchmark suite asserts **invariants as well as timings** — that a page is extracted in a
single content-stream pass, that an image is decoded exactly once, that Type3 glyph caching
hits more than it misses, and that tiled affine blitting stays under a 1 MiB scratch budget.
A refactor that introduces a redundant pass will fail there rather than merely running slower.
Those assertions are the more trustworthy signal.

**Timings are noisy — do not gate on them naively.** Measured on an unloaded dev machine, two
consecutive runs of the *same* source produced these spreads:

| Benchmark | Run-to-run spread |
| --- | --- |
| `test_page_program_memory_profile_benchmark[native]` | 30.2% |
| `test_cold_page_program_construction_benchmark[native]` | 9.3% |
| `test_width_lookup_benchmark` | 5.6% |
| `test_cmap_construction_benchmark` | 5.4% |
| `test_end_to_end_page_extraction_benchmark[ocr]` | 5.3% |

Five of thirty benchmarks vary by more than 5% on identical code, so
`--benchmark-compare-fail=mean:5%` produces false alarms. The heavy page-program, OCR, and
rasterization benchmarks run few rounds and are dominated by scheduling noise; the micro
benchmarks (cmap, tokenizer, tounicode, color) are comparatively stable.

For a change to anything in the table above, do what the existing perf commits did rather than
trusting a single benchmark run: cProfile before and after, compare corpus wall time across
repeated runs, and confirm SCORE-Bench accuracy is unchanged across all 224 cases.

For changes to anything in the table above, also profile with cProfile and confirm
SCORE-Bench accuracy is unchanged across all 224 cases — that is the bar the existing perf
commits held themselves to.
