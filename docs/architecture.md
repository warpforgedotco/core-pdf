# core-pdf architecture

An orientation to core-pdf's public boundary, extraction pipeline, source layout, and enforced
dependency direction.

---

The root package, `src/core_pdf`, owns PDF parsing, native extraction, rendering, and structured
output. `packages/core-pdf-ocr/src/core_pdf_ocr` owns OCR and vector text recognition. The two
distributions share a uv workspace and release version; the companion pins that exact core
version because it reuses internal extraction stages.

Core's `extract/` initializer exposes only `extract_page` and `extract_document`; stage internals
are imported from their owning modules. Native extraction runs in this order:

```text
bytes → capture_page → native observations → extract_tables → layout → assemble_page
```

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
    runtime/             engine-independent caching, arrays, and execution support
    exceptions.py        error hierarchy
    records.py           public extraction records
    output/              structured document/page models, views, and serialization
    primitives.py        PDF primitives
    types.py             buffers, protocols, and geometry aliases
    model/               shared geometry/text models, text rules, and page selections
    layout/              text-line records, reconstruction, diagnostics, and word rules
    spec/                PDF specification implementation (see below)
    extract/             native extraction, block layout, and tables
    render/              display lists, raster kernels, targets, and page composition
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
  tests/                OCR unit, integration, and benchmark tests
```

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

Import-linter contracts in the root `pyproject.toml` enforce package direction and the existing
core layer boundaries. Type-only imports are excluded from runtime cycle checks.

## Workspace validation

Run `uv sync --all-packages --all-groups` to install both distributions and the development tools.
The default pytest test paths include `tests/` and `packages/core-pdf-ocr/tests/`:

```sh
uv run --all-packages pytest tests/ packages/core-pdf-ocr/tests/ -n auto
uv run --all-packages --group lint ruff check .
uv run --all-packages --group lint mypy
uv run --all-packages --group lint --group test --group benchmark ty check
uv run --all-packages --group lint lint-imports
```

CI runs the native suite in a core-only environment without Tesseract and the companion suite
with English tessdata. Shared root test guards reject compiled extensions shadowing sources in
either package; tessdata discovery is confined to the companion test configuration. Coverage
measures both source roots together and retains the existing ratchet.

## Rendering constraints

### Device colour

PDF leaves DeviceCMYK conversion undefined. `s_08_graphics/device_profiles.py` converts it through
the press profile in `_vendor/icc/` and uses the uncalibrated ink formula only when the profile is
unavailable. The ICC implementation documents its rendering intent, black-point compensation, and
optimized byte path alongside the code.

### Golden rasters

`tests/test_rendering_golden.py` pins corpus output. Ordinary pages use exact RGBA digests; pages
containing irreversible JPEG 2000 images use lossless PNG references with sparse, per-sample RGB
envelopes because OpenJPEG output can vary across CPU implementations. Only values observed on
every supported CI platform are admitted; shape, alpha, and every unlisted RGB sample remain exact.
CI runs the complete corpus, while the default local test uses a covering subset.

A behavior-preserving refactor must leave the manifest and reference images unchanged. After an
intentional output change, regenerate them with:

```sh
uv run python scripts/update_raster_golden.py collect \
  --platform-id macos-arm64 --output /tmp/raster-observation
```

Collection never changes tracked files. To update the checked-in baseline, run the manual
**Update raster goldens** workflow on the target branch. It collects the same revision on pinned
Linux/x86_64 and macOS/ARM64 runners and publishes a binary patch for review; apply it from the
repository root with `git apply --binary raster-golden.patch`. A single host cannot redefine the
portable envelope. Recompute the local covering subset with `scripts/raster_cover.py` after
substantial renderer restructuring.
