# core-pdf architecture

An orientation to core-pdf's public boundary, extraction pipeline, source layout, and enforced
dependency direction.

---

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
    extract/             extraction, block layout, tables, and OCR (see section 2)
    render/              display lists, raster kernels, targets, and page composition
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
