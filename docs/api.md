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

A few records carry what the PDF says about the content beside the content itself. A
`TextSpan` showing an abbreviation has the `expansion` its `Span` marked content or
structure element gives as `/E` (ISO 32000-2 14.9.5); JSON spans include `expansion` only
when there is one, and HTML and Markdown wrap such a span in `<abbr title="...">`.
`StructureElement.expansion` reads the same entry. A text or choice `FormField` has the
`alignment` (`left`, `center` or `right`) its `/Q` gives, inherited from parent fields and
the AcroForm dictionary (12.7.4.3), and `None` for other fields. An `Annotation` records
the annotation it replies to (`in_reply_to`, `/IRT`) and how (`reply_type`, `R` or `Group`
from `/RT`), a markup annotation's `popup`, and a pop-up's `popup_parent` (12.5.6.2,
12.5.6.14): indexes in `Page.annotations`, which JSON gives as annotation ids.

## Associated and embedded files

`Document.associated_files` lists the files that `AF` arrays (ISO 32000-2 14.13, PDF 2.0,
and PDF/A-3 in PDF 1.7 files) attach to the catalog, to the extracted pages, and to the
annotations, structure elements, XObjects, and marked-content property lists on those pages.
Hybrid invoices such as ZUGFeRD and Factur-X carry their XML this way. Each `AssociatedFile`
names its `owner` (`document`, `page`, `annotation`, `structure-element`, `xobject`, or
`marked-content`) and where it is: `page_number`, the annotation's index in
`Page.annotations`, the structure element's type, `/ID`, and object reference, or the
resource name. Its `file` is an `EmbeddedFile` with the `F` and `UF` names, `name` (the first
of them set), `description`, `relationship` (`AFRelationship`, `Unspecified` when absent),
`mime_type`, and the `Params` entries `size`, `creation_date`, `modification_date` (PDF date
strings), and `checksum` (hex). Structure elements are found through the selected pages'
marked content, so an element with no content on those pages is not listed. Structure
elements are read only where they may hold `AF`: from PDF 2.0, in files claiming PDF/A-3 or
PDF/A-4, or once an `AF` has turned up elsewhere in the document; other tagged files skip
the walk up their structure trees.

No record holds file bytes. `PdfDocument.embedded_file_data(file)` reads them, decrypted and
decoded through the stream's filters. `PdfDocument.associated_files(pages=...)` returns the
same records without extracting text; `embedded_files` lists the catalog's `EmbeddedFiles`
name tree, with each key in `tree_key`; and `file_attachments(pages=...)` returns the files
of `FileAttachment` annotations as `PageScoped` records.

```python
with PdfDocument.open("invoice.pdf") as document:
    for associated in document.associated_files():
        if associated.relationship in {"Data", "Alternative"}:
            xml = document.embedded_file_data(associated.file)
```

JSON output carries a top-level `associated_files` list only when the document has
associated files, so other documents serialize as before:

```json
"associated_files": [
  {
    "id": "af:0", "owner": "document", "page_id": null, "annotation_id": null,
    "structure_type": null, "structure_id": null, "structure_ref": null,
    "resource_name": null, "relationship": "Data", "name": "factur-x.xml",
    "file_name": "factur-x.xml", "unicode_file_name": "factur-x.xml",
    "description": "Factur-X invoice", "mime_type": "text/xml", "size": 1566,
    "creation_date": "D:20240102030405+01'00'", "modification_date": "D:20240203040506Z",
    "checksum": "56e9d732896717cab89ac9800a125675", "embedded": true, "stream_ref": "1 0 R"
  },
  {
    "id": "af:6", "owner": "structure-element", "page_id": "p1", "annotation_id": null,
    "structure_type": "Formula", "structure_id": "formula-1", "structure_ref": "26 0 R",
    "resource_name": null, "relationship": "Source", "name": "formula.tex", "...": "..."
  }
]
```

## Compatibility facades

`core_pdf_compat` holds local projections of common third-party PDF interfaces:
`pdfminer`, `pdfplumber`, `pypdf`, `unstructured`, `llamaindex`, and x-ray.
Each facade imports independently; the package root resolves its convenience exports
lazily. They reproduce useful high-level behavior, not every upstream implementation
detail or private API. The facades ship in the separately installable `core-pdf-compat`
package, which pins the exact core release because they use internal core stages; core
never imports or discovers them.

```python
import asyncio

from core_pdf_compat import inspect_xray
from core_pdf_compat.pdfminer import extract_text
from core_pdf_compat.pdfplumber import open as open_pdf

print(extract_text("document.pdf"))
with open_pdf("document.pdf") as pdf:
    print(pdf.pages[0].extract_words())
findings = asyncio.run(inspect_xray("document.pdf"))  # async, as x-ray 0.4.0 is
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
drawing, `save()`, and `_repr_png_()` do not require Pillow.

PDFMiner's stale-reference and malformed-xref recovery remains facade policy, including
its distinct page selection and lazy iterator/resource lifetime.

### Unstructured NLP requirements

The Unstructured facade needs the `unstructured` extra with spaCy, langdetect,
python-iso639 and the pinned `en_core_web_sm` 3.8.0 model; the [README](../README.md) has
the install commands.
Importing `core_pdf_compat.unstructured`, any of its element classes, or the parent
package's `partition_pdf` loads that pipeline once and reuses it. A missing dependency
or model raises an `ImportError` with the original cause. There is no lexical fallback
and no runtime model download. Importing `core_pdf`, `core_pdf_compat`, or another
facade never loads the model.

## Low-level specification library

`core_pdf.impl.spec` ships inside `core-pdf`, with one
module group per PDF chapter (`s_07_syntax`, `s_08_graphics`, `s_09_fonts`, and so on).
It provides strict algorithms, PDF primitives, standard data, and semantic service
protocols, with no document facade or command line. Its supported exports are the names
in each module's `__all__`; other names, such as `INTERNAL_` constants, are private. Parser extension methods are
documented next to their strict implementations. Spec reports malformed input rather
than repairing it, except where a referenced standard prescribes a fallback, and it is
not a complete conformance validator.

Core requires `core-pdf-cythonized>=0.1.0,<0.2.0`, which carries compiled kernels with
no pure-Python fallback, so core needs a wheel for the target platform or a C compiler at
install time. The referenced-standard kernels for the PostScript calculator, JBIG2,
ciphers and PDF MAC, and the Adobe font formats with their CMap data ship inside
`core-pdf` as `core_pdf.impl.postscript`, `core_pdf.impl.jbig2`, `core_pdf.impl.crypto`,
and `core_pdf.impl.adobe_fonts`, with `core-records`, the value-class mixins every
workspace package's records share, beneath them. The JBIG2 MQ
arithmetic decoder lives in `core-pdf-cythonized`. The PNG and TIFF predictor kernels are
small enough that they live in spec itself, as
`core_pdf.impl.spec.s_07_filters.predictors`. Core's public exceptions keep their names
and share spec's error identities. Reader recovery, font program readers, Unicode guesses,
device color conversion, capture, extraction, and rendering stay in core.

## PDF versions and conformance claims

`PdfDocument.standards` returns an immutable `core_pdf.impl.spec.standards.DocumentStandards`
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
