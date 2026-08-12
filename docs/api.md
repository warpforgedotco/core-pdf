# Public API and compatibility facades

The canonical public API is exported from `core_pdf`. `PdfDocument`, `PdfPage`, structured
records, writers, runtime controls, and errors are lazy exports backed by their engine owners.
There is no parallel capability or versioned API layer.

```python
from core_pdf import PdfDocument

with PdfDocument.open("document.pdf") as document:
    print(document.structured_document.text)
```

## Compatibility facades

`core_pdf.api.compat` provides local, dependency-free projections for common third-party PDF
interfaces. Each facade imports engine owners directly and can be imported independently; the
package root resolves convenience exports lazily.

```python
from core_pdf.api.compat.pdfminer import extract_text
from core_pdf.api.compat.pdfplumber import open as open_pdf
from core_pdf.api.compat.pymupdf import open as open_fitz
from core_pdf.api.compat.pypdf import PdfReader

print(extract_text("document.pdf"))
with open_pdf("document.pdf") as pdf:
    print(pdf.pages[0].extract_words())
```

Additional local facades are available for `pikepdf`, `unstructured`, `llamaindex`, and x-ray.
They reproduce useful high-level behavior, not every implementation detail or private API of the
upstream libraries.

The pdfminer facade can be checked against pdfminer.six's own high-level extraction tests. The
runner executes the unchanged upstream test module in isolated interpreters: once with the
vendored pdfminer.six checkout and once with its imports redirected to the core-pdf facade.

```sh
uv run python scripts/run_pdfminer_compat_tests.py -- -q
```

A nonzero core-pdf result identifies a compatibility gap; the upstream result distinguishes those
gaps from fixture or test-environment failures. Use `--implementation upstream` or
`--implementation core-pdf` to run only one side.

The x-ray facade performs its redaction inspection directly from engine drawing, glyph, and
raster evidence:

```python
from core_pdf.api.compat import inspect_xray

findings = inspect_xray("document.pdf")
```

The vendored x-ray behavior suite can be run with:

```sh
PYTHONPATH=src uv run --with requests --with PyMuPDF \
  --with numpy --with tesserocr --with imagecodecs \
  python scripts/run_xray_compat_tests.py -q
```
