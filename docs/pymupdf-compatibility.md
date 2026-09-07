# Native PyMuPDF compatibility

The target is full public-API compatibility with PyMuPDF through
`core_pdf.api.compat.pymupdf`. This must remain a native implementation backed by core-pdf;
PyMuPDF is a test reference in the `vendor-test` dependency group, never a runtime backend.
The initial reference version is the locked PyMuPDF 1.28.2.

## Current coverage

The facade has moved out of `_unsupported` and participates in repository lint, formatting,
both type checkers, and architecture checks. This is the start of support work, not a claim
of complete compatibility. Existing methods outside the tested surface retain their previous
limitations.

The differential tests use `small-table.pdf`, `test_2957_1.pdf`, `test_4043.pdf`, and
`test-3143.pdf` from the unmodified PyMuPDF corpus. Matrix tests compare page counts and
page rectangles, then exercise matrix
construction, copying, sequence access, arithmetic, concatenation, inversion, rotation,
scaling, shearing, and translation using the same page geometry. Singular inversion and
invalid assignment indices are also checked. Matrix composition uses single-precision
arithmetic at the same boundary as the reference; pure Python matrix operations retain
double precision. Inversion comparisons allow a small floating-point tolerance.

Native `Point`, `Rect`, `IRect`, and `Quad` types now provide coordinate access, arithmetic,
affine transformations, rectangle regions, integer pixel bounds, and fixed-point morphs.
Page boxes return independent `Rect` values. Geometry tests compare these operations on normal and rotated
pages, including empty and invalid regions, half-open containment, singular shapes, and
integer rounding near pixel boundaries. This coverage does not yet prove complete parity
for every constructor/error case, integer-rectangle method, or geometry integration.
Integer-rectangle tests now also compare returning a new rectangle versus mutating the receiver,
coordinate types, unsupported `abs(IRect)`, and the mutate-then-raise behavior of direct
`intersect` and `transform` in the pinned reference release. `Rect` and `IRect` share private
coordinate operations instead of inheriting one public type from the other.

Page geometry comparisons cover `mediabox`, `cropbox`, `bleedbox`, `trimbox`, `artbox`, `rect`,
`bound()`, `cropbox_position`, `transformation_matrix`, `rotation_matrix`, and
`derotation_matrix`. In-memory variants of an upstream PDF exercise offset media/crop boxes,
all four orthogonal rotations, and zero, positive, and negative `UserUnit` values. Both engines
read the same variant; the reference fixture files are unchanged. Tests also compare detached
box values and errors after a page's document closes. Editing, rendering, and serialization
must still adopt these coordinate policies consistently.

Document tests compare filename and keyword-stream constructors (bytes, bytearrays, and
`BytesIO`), metadata, page counts, iteration, ranges, reverse slices, negative indexes,
membership, and chapter/page addresses. They also cover empty-document creation, blank-page
insertion using fixture dimensions, page-handle invalidation, and selected invalid-argument
and closed-document errors. `Document` owns its lifecycle directly rather than inheriting
pypdf's constructor and page-access policies. The original native source remains owned and
is closed even after a structured editing snapshot replaces the current document state.

Text, words, and text-block extraction now project native glyph captures directly, with
reader-specific font metrics, character clipping, spacing, and line/block grouping. Loading
a page no longer builds the whole document's structured layout. Differential coverage adds
`mupdf-title.pdf` and shared in-memory variants for all 14 standard fonts, word delimiters,
partial character clipping, orthogonal text rotations, hidden text, and spacing flags.
Symbol and ZapfDingbats substitute advances follow raw character codes even with explicit
Latin encodings. These cases do not establish parity for arbitrary fonts or layouts.

Ligature flags preserve source glyphs by default and expand them when requested. Expansion
assigns the glyph advance to its first letter and zero advance to subsequent letters;
clipping and word delimiters use these character geometries. Native source-cluster IDs
rejoin captured fragments without merging separate letters. Shared variants of
`2201.00069.pdf` exercise these policies against an embedded font. Unicode mapping variants
also compare whitespace preservation, word boundaries for Unicode spaces, and rejection of
invalid control mappings in favor of the font encoding. Unresolved character IDs remain
intact when no font mapping is available; `test_2791_content.pdf` and `test_3376.pdf`
provide regression coverage for that distinction.

Run the focused tests with:

```sh
uv run --locked --group test --group vendor-test pytest \
  tests/src/core_pdf/api/compat/differential/test_pymupdf.py \
  tests/src/core_pdf/api/compat/differential/test_pymupdf_document.py \
  tests/src/core_pdf/api/compat/differential/test_pymupdf_geometry.py \
  tests/src/core_pdf/api/compat/differential/test_pymupdf_text.py
```

These tests use eight explicit fixtures and shared in-memory variants. They do not yet run the complete PyMuPDF corpus
or expand with `CORE_PDF_COMPAT_DIFFERENTIAL_FULL`.

A diagnostic first-page audit of all 179 PDFs in the pinned PyMuPDF corpus completed
174 comparisons: 127 had exactly equal plain text and 53 had exactly equal word records
(including coordinates). Five reached the audit's ten-second per-file time budget:
`dotted-gridlines.pdf`, `test_3186.pdf`, `test_3362.pdf`, `test_3806.pdf`, and
`test_3887.pdf`. Timeout outcomes vary with concurrent validation workloads. Empty-text pages count in these results;
this is not an all-page compatibility measure or part of the regression suite. Differences
include spacing around mathematical symbols (`2201.00069.pdf`), bidirectional text
(`test-E+A.pdf`), clipping (`test_2957_1.pdf`), and CJK spacing (`chinese-tables.pdf`). These provide
concrete cases for the next text increments.

## Remaining implementation

An initial class-level inventory against 1.28.2 found 128 public `Document` members and
90 public `Page` members absent from the old facade. Counts are a starting inventory, not
coverage percentages: instance attributes and the correctness of existing methods require
separate verification.

Work must cover these areas before claiming complete compatibility:

1. Geometry and public value types: finish constructor and remaining edge-case parity;
   provide geometry types consistently across annotations and searches; implement
   colorspaces and constants; and integrate full affine transforms into rendering.
2. Document lifecycle: complete constructor and metadata semantics, authentication, all
   ownership/invalidation paths, closed-document errors across all methods, navigation,
   and damaged PDFs. Expand the currently covered creation and page-access operations
   across the full fixture corpus.
3. Text: glyph geometry, line and block grouping, all extraction formats, flags, clipping,
   sorting, searching, fonts, and text-page lifecycle.
4. Rendering and images: transformed and clipped pixmaps, colorspaces, alpha, image metadata,
   image extraction, and pixel comparisons.
5. Editing and persistence: page operations, drawing, annotations, widgets, links, attachments,
   redaction, object updates, serialization, and save/reopen comparisons.
6. Remaining public facilities, including tables, structured content, optional integrations,
   and supported input formats, with explicit behavior and dependency contracts.

Known remaining examples: extraction formats other than text/words/blocks still use the old
structured layout; image blocks, sorted-text reconstruction, several text flags, and text-page
reuse need implementation. `Document.tobytes()` is absent. Existing edit helpers change structured snapshots
without providing the full persistent editing behavior of PyMuPDF.

Each implementation increment needs differential tests over the same PDFs and explicit
comparisons of return values, geometry, errors, and persisted results as applicable. Passing
tests on a limited fixture set must not be presented as proof of complete compatibility.
