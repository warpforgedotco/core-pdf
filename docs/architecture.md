# core-pdf architecture

An orientation to core-pdf's public boundary, extraction pipeline, source layout, and enforced
dependency direction.

---

The root package, `src/core_pdf`, owns PDF parsing, native extraction, rendering, and structured
output. `packages/core-pdf-ocr/src/core_pdf_ocr` owns OCR and vector text recognition. The companion pins the exact core release because it reuses internal extraction stages.
`packages/core-pdf-spec/src/core_pdf_spec` owns strict PDF and referenced-standard algorithms.
`packages/core-pdf-validate` owns optional external-validator execution and report normalization.
All four distributions share the uv workspace. Spec releases independently: core currently
accepts `core-pdf-spec>=0.4.1,<0.5.0`. Spec never imports core or either companion, including for typing.

The `core-pdf[unstructured]` extra supplies spaCy and the pinned English model required by the
Unstructured facade. That facade loads its model at import time and fails if it cannot load;
it never substitutes lexical approximations or installs a model at runtime. Compatibility
imports remain lazy, so core, spec, OCR, and other facades do not load or require that pipeline.

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
    types.py             capture/source records and re-exported spec primitive identities
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

```text
packages/core-pdf-spec/
  pyproject.toml          independently versioned distribution, currently 0.4.1
  src/core_pdf_spec/
    types.py             shared PDF object identities and byte/geometry types
    standards.py         neutral versions, extensions, profile identities, and provenance
    exceptions.py        base, parse, unsupported, and decryption errors
    s_NN_*/              strict chapter algorithms and semantic service protocols
    _vendor/font_data/   attributed inert standard font tables; no font backend
  tests/                 standalone strict semantics and resource tests
```

`packages/core-pdf-validate/src/core_pdf_validate` provides `validate`, immutable reports,
the public backend protocol, and a veraPDF adapter. It depends on public core APIs only
for optional declaration discovery; explicit validation operates on original bytes without
opening a core document. Its tests include recorded engine reports and process-failure cases.

Core never imports or discovers the validation companion. `PdfDocument.standards` identifies
versions and claims without invoking a validator. Shared chapter algorithms receive immutable
semantic context where versions differ; recovery stays in core. See
[PDF versions and standards validation](standards.md) for selection rules and coverage.

### The `core_pdf_spec.s_NN_*` scheme

`core_pdf_spec` contains PDF-defined semantics and algorithms from referenced standards. A choice
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
preserve an explicitly supplied decoder. The strict syntax layer exposes object/token parsing, cross-reference rows, revision
precedence, tree node/entry decoding, and strict traversal. Core recovery adapters compose
these granular operations into tolerant walks and parsing loops. Resynchronization, guessed
offsets, skipped malformed entries, and partial results live in `_impl/document/recovery/`.
Content retry/skip control and stream limits likewise live in core capture. Strict parsing
propagates failures; specification-defined defaults and prescribed fallback rules stay in spec.

Interpreter color state retains PDF components, color spaces, rendering intent, and black-point
compensation. Core capture applies the settings at paint time when creating its records;
deferred image, shading, and pattern inputs retain an immutable `ColorRendering` value.
Profile and converter choices stay in core. Vendored fontTools parsing, outline backends,
repairs, and rasterization stay in core; spec provides handwritten CFF/Type 2 algorithms,
audited Type 1 byte helpers, CMaps, widths, standard tables, and font-service protocols.

Subpackages under `core_pdf_spec` mirror chapters of the PDF specification:

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
| `s_11_transparency` | 11 — version-sensitive blend equations, group backdrop removal, image Matte semantics |
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
derived-processing packages. Type-only imports follow the same organization rule and are checked by the boundary tests.
Spec has local geometric/byte-view support rather than importing core model/runtime modules.
Core model and base modules may share spec-owned neutral type identities; they cannot depend
on chapter algorithms. Runtime stays independent of PDF semantics.

The supported low-level interfaces are explicit module `__all__` exports and documented public
parser extension methods. Cross-distribution imports of `internal_` symbols are prohibited.
Breaking spec API changes require a new minor release during `0.x`; core widens its supported
range only after compatibility validation. Existing public core exception names alias spec's
base errors so catching `core_pdf.PdfError` continues to cover spec errors.

Font and graphics adapters stay below capture and document composition. Capture can use source
recovery and passive document records, while page/document orchestration remains above it.
The extraction/render/layout/output layering remains enforced separately.

## Workspace validation

The authored test suite contains strict standalone tests in `packages/core-pdf-spec/tests`,
core/spec integration tests in `tests/src/core_pdf/spec_boundary`, and differential comparisons
in `tests/src/core_pdf/api/compat/differential`. Each case runs a compatibility facade and
its reference implementation against the same PDF. Reference corpora remain under
`tests/fixtures`.

Initialize the corpora and install the workspace's development dependencies:

```sh
git submodule update --init --recursive
uv sync --all-packages --all-groups --extra unstructured
```

Run the differential suite and source checks with:

```sh
uv run --locked --all-packages --extra unstructured --group test --group vendor-test pytest -n auto
uv run --all-packages --group lint ruff check .
uv run --all-packages --group lint mypy
uv run --all-packages --group lint --group test --group vendor-test ty check
uv run --all-packages --group lint lint-imports
```

The default differential matrix uses each facade's own upstream corpus, plus selected
cross-corpus redaction cases for x-ray. To run every facade against every PDF fixture:

```sh
CORE_PDF_COMPAT_DIFFERENTIAL_FULL=1 \
  uv run --locked --all-packages --extra unstructured --group test --group vendor-test pytest -n auto
```

## Rendering constraints

Content frames own their graphics saves and clipping markers. A nested stream cannot pop a
parent frame's state, and leaving a stream unwinds its local clipping even after malformed
content. Resource lookup resolves dictionaries on demand while retaining indirect references,
so equivalent direct and indirect resource dictionaries preserve Form identity.

Raster paint state separates object opacity from group compositing opacity. Form group isolation
survives interpretation and capture; an omitted `/I` means non-isolated. Non-isolated groups
start with their parent's current RGB backdrop and track their own paint alpha separately.
Backdrop removal precedes outer blending, opacity, and the existing scalar soft-mask factor,
so the background and group opacity each contribute once. Nested groups contribute only their
completed output alpha to the enclosing group. Stroke dash phase advances across
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

`ri` and ExtGState `RI` select rendering intent; `UseBlackPtComp` controls compensation for
supported CIE-based conversions. Image `Intent` overrides only that image's intent; stencil
images ignore it. Saved and nested graphics state preserve these settings, and conversion
caches include the settings that affect their output. Absolute colorimetric conversion ignores
compensation without changing the stored graphics-state value.

Core preserves its existing default converters: ICC and selected DeviceCMYK profiles use
relative colorimetric conversion with compensation, while calibrated spaces retain their
existing XYZ output path for the initial RelativeColorimetric/Default combination. Explicit
calibrated controls use the CMS path and the dictionary's diffuse black endpoint. Spec owns
the neutral state and mathematical helpers; core owns the chosen sRGB destination and CMS.

DeviceN attributes retain a typed process-space mapping alongside the original dictionary.
For NChannel spaces containing only process colors, core maps the components into that
process space before conversion: RGB values retain their additive meaning, and missing CMYK
components contribute no ink. ICC profiles, rendering intent, and compensation follow the
same conversion path as directly selected process spaces. This applies to scalar paint colors
and image samples, including Indexed bases. Ordinary DeviceN, mixed spot/process NChannel,
and malformed attribute dictionaries have a global alternate-space tint transform available.
Indexed palettes with Separation or DeviceN bases now use recursive conversion for both scalar
colors and every supported image sample depth; lookup, Decode, tint evaluation, and alternate
conversion retain that order.

For mixed or all-spot NChannel, core provides a selected screen approximation: convert the
process components and each spot's Separation appearance to sRGB, then multiply the appearances
over the process result (or white when there are no process components). Component conversion
retains ICC profiles and rendering settings; mixing does not apply object opacity. The policy
requires empty/absent `MixingHints` and spot zero-tint endpoints that convert to white at the
8-bit output precision. Nonempty hints, other paper endpoints, or unusable individual functions
use the whole global tint transform. This approximation does not model spectral inks,
separation-aware overprinting, or plate interactions with a transparent backdrop.

`s_08_graphics.color.color_space_paints` identifies Separation `None` and all-`None` DeviceN,
including Indexed and uncolored Pattern bases. Capture retains their extraction records while
explicit paint flags suppress raster output and unnecessary image/function decoding. Text
advances and clipping remain active, and fill/stroke suppression is independent. A DeviceN
space containing both `None` and real colorants still passes every component to its tint function.
The early image/shading probe reads only names; general color-space selection still parses
alternate ICC profiles eagerly, including an unused alternate in a malformed no-paint space.

Tiling patterns render all tiles into one group, then apply the outer opacity, scalar soft
mask, and blend mode once. Non-Normal contents see the initial backdrop through a non-isolated
group. Patterns whose captured contents use only Normal blending retain the isolated
optimization permitted by ISO 32000-2 §11.6.7. Group alpha follows the same effective coverage
and mask samples as image, path, text, and shading paint, including vectorized raster paths.

Transparency groups composite in the renderer's RGB space with its existing supported blend
modes and byte quantization. Knockout groups, arbitrary group `/CS` blending spaces, and full
transparency soft-mask groups require additional work.
Form `/BBox` currently constrains capture metadata and image placement; enforcing that bound
as a raster clip for paths and shading remains a separate geometry limitation.
