# core-pdf architecture

An orientation to core-pdf's public boundary, extraction pipeline, source layout, and enforced
dependency direction.

---

## 1. Public surface

`core_pdf/__init__.py` maps every public name to its defining module in `internal_EXPORTS` and
resolves it on first access with `install_lazy_module_exports`. This keeps `import core_pdf` cheap
and makes the export table the authoritative public surface. Everything under `core_pdf.impl.*` is
internal and may change without notice.

The two central objects share one public owner:

- **`PdfDocument`** (`impl/document.py`) — opens documents, provides page access and structured
  extraction, and owns caches shared across pages. The CLI drives it through `process_pdf` in
  `cli.py`.
- **`PdfPage`** (`impl/document.py`, extending the spec-level page in
  `impl/spec/s_07_document/page.py`) — provides per-page extraction and rendering.

Compatibility facades under `core_pdf.api.compat.*` project the engine's public objects into
third-party interfaces without sharing facade state. See [api.md](api.md) for the supported API,
structured JSON format, and compatibility behavior.

---

## 2. The extraction pipeline

The `extract/` package owns extraction. Its initializer exposes only `extract_page`,
`extract_document`, and the lazy OCR prewarmer; stage internals are imported from their owning
modules. The stages run in roughly this order:

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
                                  ParsedPage + ParseReport (parsed_page)
                                                  │
                                    assemble_page → Page (extract_page)
```

| Module | Role |
| --- | --- |
| `extract/contracts.py` | Shared evidence, plan, observation, result, and report records. |
| `extract/capture.py` | Runs the page program once and produces a cached `CapturedPage`. |
| `extract/observations.py` | Chooses native extraction, OCR, or both, then fuses observations. |
| `extract/ocr/` | Owns recognition orchestration, candidates, rasters, regions, backends, and stroked text. |
| `extract/tables.py` | Detects chart, ruled-grid, and whitespace-inferred tables. |
| `extract/grids.py` | Owns shared ruling geometry, ruled tables, and OCR grid tasks. |
| `extract/table_cleanup.py` | Normalizes, filters, merges, annotates, and bands table candidates. |
| `extract/table_reconcile.py` | Reconciles detected tables with emitted page elements. |
| `extract/block_layout.py` | Groups observations into blocks and repairs their reading order. |
| `extract/regions.py` | Partitions pages with projection gaps, obstacles, and XY cuts. |
| `layout/reconstruction.py` | Builds line text from positioned glyph runs. |
| `layout/text_rules.py` | Owns pure spacing, script, formula, and text-repair decisions. |
| `extract/emit.py` | Normalizes text and assembles the canonical output `Page`. |
| `extract/pipeline.py` | Owns the locked page-local stage lifecycle and product cache. |
| `extract/selection.py` | Owns cross-page enrichment, scheduling, and document extraction. |

OCR is one feature namespace: `ocr/pipeline.py` orchestrates recognition and rescue; `pass_tasks.py`
owns adaptive preflight, shared raster resources, and scope-specific task materialization;
`execution.py` owns pass admission, completion, diagnostics, and candidate-selection transitions;
`candidates.py` verifies, merges, and ranks candidate batches; `extract/grids.py` owns ruled-grid
cell recognition; `raster.py`, `regions.py`, `tesseract.py`, `vector.py`, `strokes.py`, and
`newstroke.py` own their respective mechanisms; and `types.py` holds their shared records.
`internal_PageExtraction` remains the locked owner of page-local capture, recognition, fusion,
layout, report, and assembly products.

Document extraction creates an immutable enrichment snapshot for the selected pages. Learned font
and stroked-glyph mappings apply to selection-local captures without mutating page caches or font
decoders, so direct page extraction does not depend on earlier document extraction.

### Table projections

Table extraction produces one canonical view. The table stage adds row and column bands and nearby
titles or captions; emission reconciles those tables with text blocks and assigns page-wide order.
`Page.tables`, `Document.table_view`, and the serializers consume the resulting tuple
without rerunning extraction heuristics.

---

## 3. Source layout

```text
src/core_pdf/
  __init__.py            lazy public export table
  cli.py, __main__.py    command-line entry point
  api/compat/            independent third-party compatibility facades
  _vendor/               vendored third-party source and data
  impl/
    runtime/             engine-independent caching, arrays, and execution support
    exceptions.py        error hierarchy
    records.py           public extraction records
    output.py            immutable document/page output records and views
    serialize.py         markdown/HTML/JSON/CSV/TEI serialization
    pages.py             page-selection normalization
    primitives.py        PDF primitives
    text.py              shared text normalization
    types.py             buffers, protocols, and geometry aliases
    model/               capture geometry, spatial indexes, text runs, and glyph storage
    layout/              text-line records, reconstruction, diagnostics, and word rules
    spec/                PDF specification implementation (see below)
    extract/             extraction, block layout, tables, and OCR (see section 2)
    render/              display lists, raster kernels, targets, and page composition
    document.py          PdfDocument, PdfPage, and their shared operation lifecycle
```

Rendering uses direct module owners rather than a barrel module: `render/model.py` owns display
records, render options, plans, and the raster value object. `blend.py`, `paths.py`, and
`kernels.py` own pure compositing, path, image-resampling, and coordinate kernels. Stateful image
painting is divided between `image_affine_target.py` and `image_axis_target.py`; stateful path
painting is divided among `path_shape_target.py`, `path_fill_target.py`, and
`path_stroke_target.py`. `patterns.py` owns pattern behavior and `clipping.py` owns clip-mask
operations. `target.py` composes those behaviors around the mutable buffer state and owns common
blend and typed-path dispatch, while `page.py` owns page composition.

Extraction layout follows the same ownership rule. `extract/block_layout.py` owns line grouping,
block classification, ordering, and its evidence; `extract/regions.py` owns geometric partitioning;
and `extract/grids.py` owns vector and raster ruled-grid processing. The remaining `layout/`
modules own text-line reconstruction: `lines.py` holds line records and geometry diagnostics,
`reconstruction.py` builds positioned text, and `text_rules.py` owns pure reconstruction decisions
and word-frequency data. Generic spatial indexing lives with the geometry primitives in
`model/geometry.py`.

### Dependency direction

Dependency direction is enforced at stable boundaries by the import-linter contracts in
`pyproject.toml`. The broad acyclic processing spine is:

```text
document → extract → render → output → model
```

This is deliberately not a total order over every implementation module. `extract/` consumes the
text-line records and reconstruction primitives in `layout/`, while methods on `output.py` call
`serialize.py` through function-local imports and serializers consume the output records. The
former per-stage extraction and OCR layer contracts were retired when those feature namespaces
were consolidated; the stable floors and specification boundaries remain enforced.

Three packages form dependency floors:

| Package | May import internally |
| --- | --- |
| `impl/runtime/` | nothing |
| `impl/model/` | base modules directly under `impl/` |
| `impl/spec/s_07_syntax_primitives/` | `impl/primitives.py` |

`impl/spec/` is a sibling of the derived-processing packages. Derived consumers may depend on it;
within the derived layers, the spec may depend only on the low-level capture model. Base modules
under `impl/` never depend on the spec or derived packages.

Public extraction records belong in `impl/records.py`; immutable document/page output belongs in
`impl/output.py`, and format conversion belongs in `impl/serialize.py`. The `model/` package owns
internal capture records, generic spatial indexing, and low-level records shared with derived
processing, while `layout/` owns text-line reconstruction and its diagnostics. Both packages avoid
convenience re-exports: import a symbol from the module that owns it. Document-scoped caches and
page locks live on the spec-level document, with no process-global fallback that could couple
unrelated documents.

Two relationships sit outside the simple package ordering:

- `s_14_structure/tree.py` uses type-only references to the spec-level document and page, while
  `s_07_document` imports the structure tree at runtime. Import-linter excludes those
  `TYPE_CHECKING` imports because it enforces the runtime graph.
- Output-record methods call serializers through function-local imports. This keeps serializer
  entry points on the records without creating a module-initialization cycle.

### The `spec/s_NN_*` scheme

Subpackages under `spec/` mirror chapters of the PDF specification:

| Package | PDF chapter |
| --- | --- |
| `s_07_syntax_primitives` | 7 — tokens, scanning, coercion |
| `s_07_syntax` | 7 — lexer, streams, xref, object model, resolver, text strings |
| `s_07_filters` | 7 — stream filters (Flate, LZW, CCITT, JBIG2, …) |
| `s_07_content` | 7 — content streams, operators, text state |
| `s_07_document` | 7 — catalog, page tree, metadata |
| `s_07_security` | 7 — encryption handlers |
| `s_08_graphics` | 8 — color spaces, ICC, images, matrices |
| `s_09_fonts` | 9 — font programs, CMaps, glyph decoding |
| `s_14_structure` | 14 — logical structure tree |

Chapter numbers do not determine dependency order. `s_07_syntax_primitives` contains only kernels
shared by filters and the upper COS layer; `s_07_syntax` owns the mutually dependent lexer,
streams, object model, xref, and resolution machinery.

Declarative metadata has one owner: content-operator behavior belongs in
`s_07_content/operator_tables.py`, and stream-filter behavior belongs in
`s_07_filters/registry.py`. Extend those registries instead of creating parallel tables.

---

## 4. Rendering constraints

### Device colour

PDF leaves DeviceCMYK conversion undefined. `s_08_graphics/device_profiles.py` converts it through
the press profile in `_vendor/icc/` and uses the uncalibrated ink formula only when the profile is
unavailable. The ICC implementation documents its rendering intent, black-point compensation, and
optimized byte path alongside the code.

### Golden rasters

`tests/test_rendering_golden.py` pins corpus output. Ordinary pages use exact RGBA digests; pages
containing irreversible JPEG 2000 images use canonical PNG references with bounded RGB differences
because OpenJPEG output can vary across CPU implementations. CI runs the complete corpus, while the
default local test uses a covering subset.

A behavior-preserving refactor must leave the manifest and reference images unchanged. After an
intentional output change, regenerate them with:

```sh
uv run python scripts/update_raster_golden.py
```

Review the complete artifact diff. Canonical references must be generated in the pinned Ubuntu
x86_64 codec environment; the noncanonical override is only a bootstrap mechanism and deliberately
leaves provenance validation failing. The test and updater module documentation describes the
snapshot format and regeneration mechanics. Recompute the local covering subset with
`scripts/raster_cover.py` after substantial renderer restructuring.
