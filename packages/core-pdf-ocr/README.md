# core-pdf-ocr

OCR and vector text recognition for [core-pdf](../../README.md), including scanned pages,
hybrid native/recognized extraction, and selection-local font recovery.

```sh
uv add core-pdf-ocr
core-pdf-ocr document.pdf --print
python -m core_pdf_ocr document.pdf --print
```

Install Tesseract's English language data with your system package manager, or point
`TESSDATA_PREFIX` at a directory containing `eng.traineddata`.

```python
from core_pdf_ocr import PdfDocument

with PdfDocument("document.pdf") as document:
    result = document.extract()
    print(result.to_markdown())
```

The document/page interfaces, page selection, adapters, structured output, rendering, and
operation lifecycle match core-pdf. Import shared records and errors from `core_pdf`.
This package pins the exact compatible core-pdf release because its extraction pipeline
uses internal core stages.

For development from the repository root:

```sh
uv sync --all-packages --all-groups
uv run --all-packages pytest packages/core-pdf-ocr/tests -n auto
```

Licensed under AGPL-3.0-only; see [LICENSE.txt](LICENSE.txt). The vendored Newstroke
templates retain their original copyright and license notices.
