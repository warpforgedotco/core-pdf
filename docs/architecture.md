# core-pdf architecture

An orientation to core-pdf's public boundary, extraction pipeline, source layout, and enforced
dependency direction.

---

## 1. Public surface

`core_pdf/__init__.py` maps every public name to its defining module in `internal_EXPORTS` and
resolves it on first access with `install_lazy_module_exports`. This keeps `import core_pdf` cheap
and makes the export table the authoritative public surface. Everything under `core_pdf.impl.*` is
internal and may change without notice.

The two central objects are:

- **`PdfDocument`** (`impl/document.py`) — opens documents, provides page access and structured
  extraction, and owns caches shared across pages. The CLI drives it through `process_pdf` in
  `cli.py`.
- **`PdfPage`** (`impl/page.py`, extending the spec-level page in
  `impl/spec/s_07_document/page.py`) — provides per-page extraction and rendering.

Compatibility facades under `core_pdf.api.compat.*` project the engine's public objects into
third-party interfaces without sharing facade state. See [api.md](api.md) for the supported API,
structured JSON format, and compatibility behavior.

---

## 2. The extraction pipeline

The `parse/` package owns extraction, with one module per stage. The stages run in roughly this
order:

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
| `parse/model.py` | Shared contracts for evidence, plans, observations, results, and reports. |
| `parse/capture.py` | Runs the page program once and produces a cached `CapturedPage`. |
| `parse/route.py` | Chooses native extraction, OCR, or both and returns a `WorkPlan`. |
| `parse/fusion.py` | Merges native and recognized observations. |
| `parse/tables.py` | Detects and reconciles grids, stream tables, charts, cells, titles, and captions. |
| `parse/ocr*.py` | Recognizes raster and stroked-vector text; OCR backends remain isolated here. |
| `parse/grid_geometry.py` | Supplies ruled-grid geometry to table and OCR stages. |
| `parse/layout.py` | Groups runs into lines and blocks and determines reading order. |
| `parse/emit.py` | Normalizes text and assembles the canonical structured `Page`. |
| `parse/pipeline.py` | Orchestrates stages, locking, and product caching. |

`parse/__init__.py` exports only pipeline entry points and shared stage models. Import stage helpers
from their owning modules. `internal_PageExtraction` is the locked owner of page-local capture,
recognition, fusion, layout, report, and assembly products.

Document extraction creates an immutable enrichment snapshot for the selected pages. Learned font
and stroked-glyph mappings apply to selection-local captures without mutating page caches or font
decoders, so direct page extraction does not depend on earlier document extraction.

### Table projections

Table extraction produces one canonical view. The table stage adds row and column bands and nearby
titles or captions; emission reconciles those tables with text blocks and assigns page-wide order.
`Page.tables`, `Document.table_view`, and the structured serializers consume the resulting tuple
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
    models.py            public extraction records
    pages.py             page-selection normalization
    primitives.py        PDF primitives
    text.py              shared text normalization
    types.py             buffers, protocols, and geometry aliases
    model/               capture geometry, text runs, and glyph storage
    layout/              layout heuristics and spatial analysis
    spec/                PDF specification implementation (see below)
    parse/               extraction pipeline (see section 2)
    render/              display lists, raster kernels, targets, and page composition
    structured/          document IR and markdown/HTML/JSON/CSV/TEI serialization
    page.py              PdfPage
    document.py          PdfDocument
```

Rendering uses direct module owners rather than a barrel module: `render/display.py` owns display
records and options, `render/kernels.py` owns pure raster kernels, `render/target.py` owns the
mutable paint target, `render/page.py` owns page composition, and `render/raster_image.py` owns the
raster value object.

### Dependency direction

There are no runtime import cycles between packages. Dependency direction is enforced by the
import-linter contracts in `pyproject.toml`, which are the source of truth when this overview and
the code disagree. Upper layers may depend on lower layers; lower layers must not import upward.
The principal derived-processing order is:

```text
document → page → parse → render → structured → layout → model
```

Three packages form dependency floors:

| Package | May import internally |
| --- | --- |
| `impl/runtime/` | nothing |
| `impl/model/` | base modules directly under `impl/` |
| `impl/spec/s_07_syntax_primitives/` | `impl/primitives.py` |

`impl/spec/` is a sibling of the derived-processing packages. Derived consumers may depend on it;
within the derived layers, the spec may depend only on the low-level capture model. Base modules
under `impl/` never depend on the spec or derived packages.

Public extraction records belong in `impl/models.py`. The `model/` package owns internal capture
records and low-level records shared with layout, while `layout/` owns layout heuristics and their
stage-specific results. Both packages avoid convenience re-exports: import a symbol from the module
that owns it. Document-scoped caches and page locks live on the spec-level document, with no
process-global fallback that could couple unrelated documents.

Two relationships sit outside the simple package ordering:

- `s_14_structure/tree.py` uses type-only references to the spec-level document and page, while
  `s_07_document` imports the structure tree at runtime. Import-linter excludes those
  `TYPE_CHECKING` imports because it enforces the runtime graph.
- Structured IR methods call serializers through function-local imports. This keeps serializer
  entry points on the IR without creating a module-initialization cycle.

### The `spec/s_NN_*` scheme

Subpackages under `spec/` mirror chapters of the PDF specification:

| Package | PDF chapter |
| --- | --- |
| `s_07_syntax_primitives` | 7 — tokens, scanning, coercion, dictionary lookup, operators |
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
`s_07_syntax_primitives/content_operators.py`, and stream-filter behavior belongs in
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
