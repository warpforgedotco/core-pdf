# Public API and compatibility facades

The canonical public API is exported from `core_pdf`. `PdfDocument`, `PdfPage`, structured
records, runtime controls, and errors are lazy exports backed by their engine owners.
There is no parallel capability or versioned API layer.

```python
from core_pdf import PdfDocument

with PdfDocument.open("document.pdf") as document:
    print(document.structured_document.text)
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

An experimental PyMuPDF facade exists under `core_pdf.api.compat._unsupported.pymupdf`; it is
not part of the supported compatibility surface.

The pdfminer facade can be checked against pdfminer.six's own high-level extraction tests. The
runner executes the unchanged upstream test module in isolated interpreters: once with the
vendored pdfminer.six checkout and once with its imports redirected to the core-pdf facade.

```sh
git submodule update --init --recursive
uv run python scripts/run_pdfminer_compat_tests.py -- -q
```

A nonzero core-pdf result identifies a compatibility gap; the upstream result distinguishes those
gaps from fixture or test-environment failures. Use `--implementation upstream` or
`--implementation core-pdf` to run only one side.

The remaining supported facades have direct differential tests against their installed reference
libraries. By default, each facade runs against its own upstream fixture corpus, plus focused
cross-corpus redaction cases for x-ray:

```sh
uv run --group vendor-test pytest tests/src/core_pdf/compat/differential \
  -m compat_differential -n auto
```

Run the exhaustive every-facade/every-fixture matrix explicitly when compatibility work calls for
it:

```sh
CORE_PDF_COMPAT_DIFFERENTIAL_FULL=1 \
  uv run --group vendor-test pytest tests/src/core_pdf/compat/differential \
  -m compat_differential -n auto
```

The x-ray facade performs its redaction inspection directly from engine drawing, glyph, and
raster evidence:

```python
from core_pdf.api.compat import inspect_xray

findings = inspect_xray("document.pdf")
```

The vendored x-ray behavior suite can be run with:

```sh
PYTHONPATH=src uv run --with requests --with PyMuPDF \
  --with numpy --with imagecodecs \
  python scripts/run_xray_compat_tests.py -q
```
