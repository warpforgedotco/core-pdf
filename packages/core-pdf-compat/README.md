# core-pdf-compat

Local projections of common third-party PDF interfaces for
[core-pdf](../../README.md): `pdfminer`, `pdfplumber`, `pypdf`, `pikepdf`, `unstructured`,
`llamaindex`, and x-ray. Each facade reproduces the useful high-level behavior of its
reference library on top of the core engine, not every upstream implementation detail or
private API. The facades never import the reference libraries themselves.

```sh
uv add core-pdf-compat
```

```python
from core_pdf_compat import inspect_xray
from core_pdf_compat.pdfminer import extract_text
from core_pdf_compat.pdfplumber import open as open_pdf

print(extract_text("document.pdf"))
with open_pdf("document.pdf") as pdf:
    print(pdf.pages[0].extract_words())
findings = inspect_xray("document.pdf")  # redaction inspection from engine evidence
```

Each facade imports independently; the package root resolves its convenience exports
lazily. Import shared records and errors from `core_pdf`. This package pins the exact
compatible core-pdf release because the facades use internal core stages.

## Optional extras

The Unstructured facade requires spaCy and the pinned `en_core_web_sm` 3.8.0 model. The
model wheel is not on PyPI, so install it from its release URL alongside the extra:

```sh
pip install "core-pdf-compat[unstructured]" \
  "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
```

Importing `core_pdf_compat.unstructured` raises `ImportError` if the model cannot load.
The facade never downloads models at runtime. Native core and other facades do not require
the extra.

The pdfplumber facade needs the `pdfplumber` extra only for `PageImage.show()`, which
displays a page through Pillow's platform viewer:

```sh
pip install "core-pdf-compat[pdfplumber]"
```

Rendering, drawing, `save()`, and `_repr_png_()` work without Pillow.

## Development

From the repository root:

```sh
git submodule update --init --recursive
uv sync --all-packages --all-groups --extra unstructured
uv run --locked --all-packages --extra unstructured --group test --group vendor-test \
  pytest packages/core-pdf-compat/tests -n auto
```

The differential tests under `tests/differential` compare each facade with its reference
library over that library's own fixture corpus. Set `CORE_PDF_COMPAT_DIFFERENTIAL_FULL=1`
to run every facade against every fixture.
