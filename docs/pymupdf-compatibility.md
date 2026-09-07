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

The first differential tests use `small-table.pdf` and `test_2957_1.pdf` from the unmodified
PyMuPDF corpus. They compare page counts and page rectangles, then exercise matrix
construction, copying, sequence access, arithmetic, concatenation, inversion, rotation,
scaling, shearing, and translation using the same page geometry. Singular inversion and
invalid assignment indices are also checked. Matrix composition uses single-precision
arithmetic at the same boundary as the reference; pure Python matrix operations retain
double precision. Inversion comparisons allow a small floating-point tolerance.

Run the initial tests with:

```sh
uv run --locked --group test --group vendor-test pytest \
  tests/src/core_pdf/api/compat/differential/test_pymupdf.py
```

These initial tests use two explicit fixtures. They do not yet run the complete PyMuPDF corpus
or expand with `CORE_PDF_COMPAT_DIFFERENTIAL_FULL`.

## Remaining implementation

An initial class-level inventory against 1.28.2 found 128 public `Document` members and
90 public `Page` members absent from the old facade. Counts are a starting inventory, not
coverage percentages: instance attributes and the correctness of existing methods require
separate verification.

Work must cover these areas before claiming complete compatibility:

1. Geometry and public value types: `Rect`, `IRect`, `Point`, `Quad`, colorspaces, constants,
   operator behavior, errors, and integration of full affine transforms into rendering.
2. Document lifecycle: constructor overloads, empty documents, authentication, page iteration,
   indexing, ownership, closed-document errors, metadata, navigation, and damaged PDFs.
3. Text: glyph geometry, line and block grouping, all extraction formats, flags, clipping,
   sorting, searching, fonts, and text-page lifecycle.
4. Rendering and images: transformed and clipped pixmaps, colorspaces, alpha, image metadata,
   image extraction, and pixel comparisons.
5. Editing and persistence: page operations, drawing, annotations, widgets, links, attachments,
   redaction, object updates, serialization, and save/reopen comparisons.
6. Remaining public facilities, including tables, structured content, optional integrations,
   and supported input formats, with explicit behavior and dependency contracts.

Known examples from the initial `small-table.pdf` comparison: plain text combines table cells
that the reference separates into lines; word boxes have different vertical metrics;
`Document.pages` is a tuple instead of a callable iterator; `open()` cannot create an empty
document; and `Document.tobytes()` is absent. Existing edit helpers change structured snapshots
without providing the full persistent editing behavior of PyMuPDF.

Each implementation increment needs differential tests over the same PDFs and explicit
comparisons of return values, geometry, errors, and persisted results as applicable. Passing
tests on a limited fixture set must not be presented as proof of complete compatibility.
