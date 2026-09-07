# Public API and compatibility facades

The canonical public API is exported from `core_pdf`. `PdfDocument`, `PdfPage`, structured
records, runtime controls, and errors are lazy exports backed by their engine owners.
There is no parallel capability or versioned API layer.

```python
from core_pdf import PdfDocument

with PdfDocument.open("document.pdf") as document:
    print(document.structured_document.text)
```

`PdfPage.extract()` and `PdfPage.structured_view` return a structured `Page`;
`PdfDocument.extract()` and `PdfDocument.structured_document` return a structured `Document`.

## Extraction adapters

`DocumentAdapter`, exported from `core_pdf`, describes an `apply(document: Document) -> Document`
method. Adapters run in the supplied order after extraction releases its document operation.
They transform the structured result, and do not need to inherit from the protocol:

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

Starting with 0.0.6, `core_pdf.PdfDocument` and the compatibility facades extract only PDF-native
text. Existing hidden text layers remain PDF-native text; image-only or flattened vector text
requires the separate `core-pdf-ocr` package. Installing it never changes core's behavior.

```python
from core_pdf_ocr import PdfDocument

with PdfDocument("scanned.pdf") as document:
    result = document.extract(pages=[1, 2])
    print(result.to_markdown())
```

`core_pdf_ocr.PdfDocument` and `PdfPage` subclass the core public classes and preserve their
constructor, selection, adapters, rendering, and close/cancellation behavior. Their extraction
includes native, OCR, and hybrid routes and returns the same structured model types. Shared
records and errors remain exported from `core_pdf`. The companion pins its exact matching core
release; its internal stage imports are not a public extension API.

Both packages inherit the same extraction lifecycle. The companion overrides the extraction
hooks and page factory to select recognition without copying operation or adapter handling.

Use `core-pdf-ocr document.pdf --print` or `python -m core_pdf_ocr document.pdf --print` to
select recognition from the command line. The options match `core-pdf`.

## Structured JSON

`Document.to_json_dict()` and `Document.to_json()` emit schema 5.0. The document is a normalized
graph: `pages` reference ordered `nodes`, nodes reference one canonical block/table/figure payload,
and blocks reference canonical lines. Stable IDs use
`p{page_number}:{kind}:{zero_based_index}`. Schema 4's duplicated page payloads and document-wide
line/table reference snapshots are not emitted, and there is no schema-4 compatibility encoder.

Metadata must already be composed of JSON scalar, mapping, list, or tuple values. Unsupported
objects raise a path-specific `TypeError` instead of being silently converted to strings.

## Compatibility facades

`core_pdf.api.compat` provides local, dependency-free projections for common third-party PDF
interfaces. Each facade imports engine owners directly and can be imported independently; the
package root resolves convenience exports lazily.

```python
from core_pdf.api.compat.pdfminer import extract_text
from core_pdf.api.compat.pdfplumber import open as open_pdf
from core_pdf.api.compat.pypdf import PdfReader

print(extract_text("document.pdf"))
with open_pdf("document.pdf") as pdf:
    print(pdf.pages[0].extract_words())
```

Additional local facades are available for `pikepdf`, `unstructured`, `llamaindex`, and x-ray.
They reproduce useful high-level behavior, not every implementation detail or private API of the
upstream libraries.

The native PyMuPDF facade is available as `core_pdf.api.compat.pymupdf`. Its target is full
public-API compatibility with PyMuPDF, without importing PyMuPDF at runtime. Implementation
is in progress: the public import path does **not** yet imply full behavioral compatibility.
Differential coverage checks geometry, document lifecycle, native text extraction, and reusable
text/word/block snapshots using upstream PDFs.
See [PyMuPDF compatibility](pymupdf-compatibility.md) for current coverage and outstanding work.

Differential tests in `tests/src/core_pdf/api/compat/differential` compare the `pdfplumber`, `pymupdf`,
`pypdf`, `pikepdf`, `unstructured`, `llamaindex`, and x-ray facades against their reference
libraries. By default, each facade uses its own upstream fixture corpus, plus selected
cross-corpus redaction cases for x-ray:

```sh
git submodule update --init --recursive
uv run --locked --group test --group vendor-test pytest -n auto
```

Run the exhaustive every-facade/every-fixture matrix explicitly when compatibility work
calls for it:

```sh
CORE_PDF_COMPAT_DIFFERENTIAL_FULL=1 \
  uv run --locked --group test --group vendor-test pytest -n auto
```

The x-ray facade performs its redaction inspection directly from engine drawing, glyph, and
raster evidence:

```python
from core_pdf.api.compat import inspect_xray

findings = inspect_xray("document.pdf")
```
