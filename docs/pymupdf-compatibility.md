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

Native object inspection now provides `xref_length()`, `pdf_catalog()`, `pdf_trailer()`,
`xref_object()`, `xref_get_keys()`, `xref_get_key()`, raw and decoded stream access, and
stream/font/image/form type checks. Tests compare every object and stream in three existing
fixtures, plus shared variants covering object syntax, float formatting, escaped names,
Unicode/binary strings, ASCII formatting policy, nested indirect dictionary paths, empty
catalogs, invalid references, and closed-document errors. Text capture preserves the objects
seen by these accessors. Native object edits now support allocation, object replacement, direct dictionary paths,
and stream replacement with optional compression. `page_xref()` and `Page.xref` expose page
references. Default `tobytes()` and `save()` serialize native objects and streams, preserve
metadata, and manage document IDs; tests reopen the results with the reference engine.
Blank-page insertion also uses native objects and preserves inherited page settings.
Retained pages reflect low-level edits while captured text pages remain frozen.

Structured editing snapshots still raise `NotImplementedError` for object access and saving
until their changes have a native object representation. Non-default save policies,
incremental output, encryption, indirect dictionary writes, complex page-label trees,
free-object semantics, generations, and damaged-file recovery need further work.

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

`Page.get_textpage()` now captures reusable native text/word/block snapshots. Tests cover
its default flags, frozen clipping and ligature policy, independent return values, page-handle
ownership, extraction after source release or closure, and stability after page edits.
Affine matrices transform glyph geometry before grouping and clipping. Comparisons include
scale, translation, rotation, shear, nonuniform and singular matrices, crop bounds, and empty blocks when
device clipping is disabled. Reused word extraction applies the reference's half-word-area
clip threshold; text and block extraction retain the snapshot's clip.

Dictionary, raw-character dictionary, JSON, and raw JSON output now use these native
snapshots too. Differential cases cover span font metrics and flags, RGB paint and opacity,
character origins and synthetic-space markers, sorting, affine transforms, independent
returned dictionaries, and extraction after document closure. JSON includes base64 image
payloads and follows the reference serialization format.

Image blocks retain their position among text blocks, image geometry, PNG/JPEG bytes, and
separate soft masks. Tests compare exact payloads on the title fixture and shared grayscale
and RGB image variants, plus partial image clipping. This is limited image coverage: other
codecs, image resolutions, color profiles, and image-block colorspace descriptions need
broader comparisons. DeviceCMYK span colors use core-pdf's native press profile, which
differs from MuPDF's and does not yet produce identical colors. Embedded-font flags,
superscripts, and bidirectional span metadata also need further work.

HTML, XHTML, and XML still use the earlier page-backed implementation and do not yet provide
this snapshot lifecycle or complete reference output shapes.

Rectangle extraction and selection now use captured character geometry.
`Page.get_textbox()` and `TextPage.extractTextbox()` include characters with positive-area
rectangle intersections and omit the final line terminator. `TextPage.extractSelection()`
orders endpoints in the captured text flow and selects at character carets, including reversed
endpoints and orthogonal text rotations. These snapshots remain usable after document closure.

`Page.search_for()` and `TextPage.search()` now return match rectangles or oriented `Quad`
objects instead of containing block boxes. Tests compare substring and multiline matches,
adjacent repeated hits, normalized whitespace, ASCII-only case folding, empty queries,
snapshot reuse, and page ownership. Vertical cursor advances round in PDF coordinates before
the page transform, preserving reference geometry on rotated text. Broader layouts,
bidirectional text, dehyphenation, nonorthogonal selection geometry, and search flags still
need corpus-wide verification. Selection/search over structured edits inherit the existing
editing snapshot's approximate glyph geometry.

Character and word spacing now advance the reader cursor independently of glyph boxes.
The cursor rounds in PDF coordinates before crop translation, and native capture identifies
explicit text-matrix resets so relative line moves retain their accumulated rounding.
Differential variants cover negative, fractional, and large spacing in all four orthogonal
rotations, ordinary and large synthetic-space metadata, overlapping search quads, and
relative positioning with repeated matrix resets. Searches over the unchanged
`test-E+A.pdf` add an embedded-font regression for spacing and crop translation. Other
queries and layouts in that fixture still require broader comparison.

Horizontal text scaling now contributes to glyph advances and the determinant-based span
font size. Tests combine compressed, expanded, mirrored, and zero horizontal scale with
character spacing and all four rotations. Additional affine cases cover skew, arbitrary
rotation, nonuniform scaling, reflection, and singular transforms, including composition
with a `TextPage` matrix. Zero-width glyphs retain the reference's distinct text, word,
block, and dictionary output behavior. These cases extend transform coverage without
establishing parity for every font, clipping boundary, or ill-conditioned matrix.

Run the focused tests with:

```sh
uv run --locked --group test --group vendor-test pytest \
  tests/src/core_pdf/api/compat/differential/test_pymupdf.py \
  tests/src/core_pdf/api/compat/differential/test_pymupdf_document.py \
  tests/src/core_pdf/api/compat/differential/test_pymupdf_geometry.py \
  tests/src/core_pdf/api/compat/differential/test_pymupdf_text.py \
  tests/src/core_pdf/api/compat/differential/test_pymupdf_textpage.py \
  tests/src/core_pdf/api/compat/differential/test_pymupdf_structured_text.py \
  tests/src/core_pdf/api/compat/differential/test_pymupdf_selection.py \
  tests/src/core_pdf/api/compat/differential/test_pymupdf_spacing.py \
  tests/src/core_pdf/api/compat/differential/test_pymupdf_text_transform.py \
  tests/src/core_pdf/api/compat/differential/test_pymupdf_objects.py \
  tests/src/core_pdf/api/compat/differential/test_pymupdf_write.py
```

These tests use nine explicit fixtures and shared in-memory variants. They do not yet run the complete PyMuPDF corpus
or expand with `CORE_PDF_COMPAT_DIFFERENTIAL_FULL`.

A diagnostic first-page audit of all 179 PDFs in the pinned PyMuPDF corpus completed
177 comparisons: 129 had exactly equal plain text and 54 had exactly equal word records
(including coordinates). Each comparison ran in a separate process with a ten-second budget;
`test_4182.pdf` and `test_4699.pdf` timed out. Empty-text pages count in these results;
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

Known remaining examples: HTML/XHTML/XML still use the old structured layout; sorted-text
reconstruction, several text flags, complete image/color behavior, and snapshot support for
the remaining text formats need implementation. High-level edit helpers still change structured snapshots
without providing the full persistent editing behavior of PyMuPDF.

Each implementation increment needs differential tests over the same PDFs and explicit
comparisons of return values, geometry, errors, and persisted results as applicable. Passing
tests on a limited fixture set must not be presented as proof of complete compatibility.
