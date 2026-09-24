# Public API and compatibility facades

`core_pdf` exports the public API: `PdfDocument`, `PdfPage`, the structured output
records, runtime controls, and errors. Every recognized PDF version goes through the
same document API.

```python
from core_pdf import PdfDocument

with PdfDocument.open("document.pdf") as document:
    print(document.structured_document.text)
```

`PdfDocument.extract()` returns a structured `Document`; `PdfPage.extract()` returns a
structured `Page`. `structured_document` and `structured_view` are property forms of
the same calls.

## Page coordinates and physical size

Page boxes, extracted geometry, annotations, fields, and raster crop arguments use the
PDF's default user space. `PdfPage.width` and `height` are those raw units;
`user_unit` (default 1, read from the page) scales them: one raw unit is
`user_unit / 72` inches. `width_points` and `height_points` give the unrotated physical
size in points. Structured `Page` records carry the same three values, and JSON output
includes the scale so exported geometry stays interpretable.

```python
with PdfDocument("document.pdf") as document:
    page = document.pages[0]
    print(page.width, page.height, page.user_unit)  # raw units and scale
    print(page.width_points, page.height_points)  # physical points
    raster = page.render().rasterize(scale=2)  # 144 pixels per physical inch
```

Rendering works in physical points: `user_unit` is applied once, including crops and
annotations, before page rotation. Raster-size limits are unchanged. Compatibility
facades follow their reference library's conventions: pypdf exposes raw boxes and
`user_unit`, pdfplumber keeps its reference's raw geometry and raster sizing, and x-ray
uses MuPDF-style physical coordinates.

## Page programs and capture options

`PdfPage.get_page_program(options=CaptureOptions(...))` returns the captured page
program that extraction and rendering both consume. `CaptureOptions` selects what the
capture records beyond the text: `ink_bounds` (glyph ink boxes), `text_runs` (layout
runs and clusters), and `render_details` (glyph transforms and bitmap requests). All
three default to on. A program captured with `render_details=False` describes the text
correctly but cannot be drawn, and rendering it raises `ValueError` rather than
producing a page without glyphs.

## Extraction adapters

`DocumentAdapter` describes one method, `apply(document: Document) -> Document`.
Adapters run in the given order after extraction and transform the structured result.
Any object with that method works; inheriting from the protocol is optional.

```python
from dataclasses import replace

from core_pdf import Document, PdfDocument


class AddTag:
    def apply(self, document: Document) -> Document:
        return replace(document, metadata={**document.metadata, "tag": "review"})


with PdfDocument.open("document.pdf") as document:
    result = document.extract(adapters=(AddTag(),))
```

## OCR extraction

`core_pdf.PdfDocument` and the `core-pdf-compat` facades extract PDF-native text only,
including existing hidden text layers. Image-only or flattened vector text needs the
separate `core-pdf-ocr` package. Installing it does not change core's behavior.

```python
from core_pdf_ocr import PdfDocument

with PdfDocument("scanned.pdf") as document:
    print(document.extract(pages=[1, 2]).to_markdown())
```

`core_pdf_ocr.PdfDocument` and `PdfPage` subclass the core classes, keep their
constructor, selection, adapter, rendering, and close/cancel behavior, and return the
same structured types. Shared records and errors stay exported from `core_pdf`. The
companion pins its exact core release; its internal stage modules are not an extension
API. The `core-pdf-ocr` command accepts the same options as `core-pdf`.

## Structured JSON

`Document.to_json_dict()` and `to_json()` emit schema 5.0: a normalized graph where
`pages` reference ordered `nodes`, each node references one block, table, or figure
payload, and blocks reference canonical lines. IDs have the form
`p{page_number}:{kind}:{zero_based_index}`. There is no encoder for the older schema 4.

Metadata must already consist of JSON scalars, mappings, lists, or tuples. Anything
else raises a `TypeError` naming the offending path rather than being stringified.

## Compatibility facades

`core_pdf_compat` holds local projections of common third-party PDF interfaces:
`pdfminer`, `pdfplumber`, `pypdf`, `pikepdf`, `unstructured`, `llamaindex`, and x-ray.
Each facade imports independently; the package root resolves its convenience exports
lazily. They reproduce useful high-level behavior, not every upstream implementation
detail or private API. The facades ship in the separately installable `core-pdf-compat`
package, which pins the exact core release because they use internal core stages; core
never imports or discovers them.

```python
from core_pdf_compat import inspect_xray
from core_pdf_compat.pdfminer import extract_text
from core_pdf_compat.pdfplumber import open as open_pdf

print(extract_text("document.pdf"))
with open_pdf("document.pdf") as pdf:
    print(pdf.pages[0].extract_words())
findings = inspect_xray("document.pdf")  # redaction inspection from engine evidence
```

### pdfplumber geometry and image display

`merge_edges` snaps parallel edges and joins collinear intervals using independent
`snap_x_tolerance`, `snap_y_tolerance`, `join_x_tolerance`, and `join_y_tolerance`
arguments (each defaults to 3). It accepts iterables, leaves input dictionaries unchanged,
and rejects unknown options. Table discovery and debugging share this normalization;
`edge_min_length_prefilter` applies before merging and `edge_min_length` afterward.

`PageImage.show()` displays the current annotated page through Pillow's platform viewer.
It uses the same PNG pixels as `save()` and `_repr_png_()`; viewer exceptions propagate.
Pillow ships in the optional `pdfplumber` extra and is imported only when display is
requested; without it `show()` raises `ImportError` naming `core-pdf-compat[pdfplumber]`. Rendering,
drawing, `save()`, and `_repr_png_()` do not require Pillow. Display integration tests
intercept the viewer call and verify image contents without opening windows.

Method-level differential tests cover geometry, cropping, shared object caches, attribute
selection, basic table grids, image sizing, annotations, merging, and display routing.
PDFMiner's stale-reference and malformed-xref recovery remains facade policy, including
its distinct page selection and lazy iterator/resource lifetime.

### Unstructured NLP requirements

The Unstructured facade needs the `unstructured` extra with spaCy and the pinned
`en_core_web_sm` 3.8.0 model; the [README](../README.md) has the install commands.
Importing `core_pdf_compat.unstructured`, any of its element classes, or the parent
package's `partition_pdf` loads that pipeline once and reuses it. A missing dependency
or model raises an `ImportError` with the original cause. There is no lexical fallback
and no runtime model download. Importing `core_pdf`, `core_pdf_compat`, or another
facade never loads the model.

### Compatibility validation

Differential tests in `packages/core-pdf-compat/tests/differential` compare each facade
with its reference library. By default each facade runs on its own upstream fixture
corpus, plus selected cross-corpus redaction cases for x-ray:

```sh
git submodule update --init --recursive
uv run --locked --all-packages --extra unstructured --group test --group vendor-test pytest -n auto
```

Set `CORE_PDF_COMPAT_DIFFERENTIAL_FULL=1` to run every facade against every fixture.

## Low-level specification library

`core-pdf-spec` is installable and versioned on its own as `core_pdf_spec`, with one
module group per PDF chapter (`s_07_syntax`, `s_08_graphics`, `s_09_fonts`, and so on).
It provides strict algorithms, PDF primitives, standard data, and semantic service
protocols, with no document facade or command line. Its supported exports are the names
in each module's `__all__`; `internal_` names are private. Parser extension methods are
documented next to their strict implementations. Spec reports malformed input rather
than repairing it, except where a referenced standard prescribes a fallback, and it is
not a complete conformance validator.

Core requires `core-pdf-spec>=0.5.0,<0.6.0` and `core-pdf-cythonized>=0.1.0,<0.2.0`;
the latter carries compiled kernels with no pure-Python fallback, so core needs a
wheel for the target platform or a C compiler at install time. Referenced-standard
kernels (the PostScript calculator, JBIG2, ciphers and PDF MAC, and the Adobe font
formats with their CMap data) are separately versioned floor packages that spec
depends on and core may import directly: `core-postscript`, `core-jbig2`,
`core-pdf-crypto`, and `core-adobe-fonts`. The PNG and TIFF predictor kernels are
small enough that they live in spec itself, as
`core_pdf_spec.s_07_filters.predictors`. Core's public exceptions keep their names
and share spec's error identities. Reader recovery, fontTools backends, Unicode guesses,
device color conversion, capture, extraction, and rendering stay in core.

## PDF versions and conformance claims

`PdfDocument.standards` returns an immutable `core_pdf_spec.standards.DocumentStandards`
with the header, catalog, and effective versions, declared extensions, unverified
profile claims, and declaration diagnostics. Its `.context` drives version-sensitive
low-level semantics.

Structure elements keep their original tag in `type`; `role` follows transitive root
and namespace role mappings, with `role_namespace`, `role_resolution` (path and terminal
status, including cycles), and `role_error` describing how it was resolved. These
projections do not certify logical hierarchy or accessibility conformance.

The separate `core_pdf_validate.validate(source, *, profiles, backend=None)` returns a
`ValidationReport` with one result per requested profile ID, or for
`profiles="declared"`. It runs a configured local validator on the original bytes and
does not need core to parse the file. See [standards interfaces and coverage](standards.md)
and the [validation companion](../packages/core-pdf-validate/README.md).
