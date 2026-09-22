# core-pdf

High-Performance PDF Engine

![core-pdf — High-Performance PDF Engine](.github/assets/core-pdf-social-preview.jpg)

`core-pdf` parses native PDF text, images, graphics, and structure. OCR and vector text
recognition are available separately in [`core-pdf-ocr`](packages/core-pdf-ocr/README.md),
and compatibility facades for pdfminer, pdfplumber, pypdf, pikepdf, unstructured,
llamaindex, and x-ray in [`core-pdf-compat`](packages/core-pdf-compat/README.md).
Installing a companion does not change the behavior of `core_pdf`.

```python
from core_pdf import PdfDocument

with PdfDocument("document.pdf") as document:
    print(document.extract().to_markdown())
```

For scanned or hybrid documents, install `core-pdf-ocr` and change the import to
`from core_pdf_ocr import PdfDocument`. Its `core-pdf-ocr` command accepts the same arguments
as `core-pdf`; both packages support `python -m` invocation.

The Unstructured compatibility facade requires `core-pdf-compat[unstructured]` and its pinned
English spaCy model. Supply the official model wheel when installing the published package:

```sh
pip install "core-pdf-compat[unstructured]" \
  "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
```

Importing `core_pdf_compat.unstructured` raises `ImportError` if the model cannot load.
The facade never downloads models at runtime. Native core and other facades do not require
this extra. Workspace uv commands resolve the model from the configured source:

```sh
uv sync --all-packages --all-groups --extra unstructured
uv run --locked --all-packages --extra unstructured --group test --group vendor-test pytest -n auto
```

The pdfplumber compatibility facade needs `core-pdf-compat[pdfplumber]` only for
`PageImage.show()`, which opens the rendered page in the platform image viewer:

```sh
pip install "core-pdf-compat[pdfplumber]"
```

Without it `show()` raises `ImportError` naming the extra. Rendering, drawing, `save()`,
and `_repr_png_()` work without Pillow, so the rest of the facade needs no extra.

See [coverage and unused-code review](docs/coverage.md) for parallel coverage.py runs
and guidance on distinguishing dead code from missing tests.

Both `core-pdf` and `core-pdf-ocr` emit Markdown when `--print`, `--write`, or
`--output-dir` is selected; without those options they parse without emitting content.
The redundant `--mode markdown` and unused `--plain` options have been removed.
Omit them from existing invocations. The Python CLI helper `process_pdf` likewise
no longer accepts an `output_format` argument.

## License

core-pdf uses [Core License version 0.1.0](https://github.com/core-experiments/core-pdf/blob/main/docs/license/VERSION.md).

Unless a separate signed evaluation license, commercial license, or philanthropy waiver applies, core-pdf is licensed under the GNU Affero General Public License version 3 only. See [LICENSE.txt](LICENSE.txt).

Evaluation licenses, commercial licenses, and philanthropy waivers are available separately. See the [license documentation](https://github.com/core-experiments/core-pdf/blob/main/docs/license/README.md), [notice](https://github.com/core-experiments/core-pdf/blob/main/docs/license/NOTICE), [evaluation terms](https://github.com/core-experiments/core-pdf/blob/main/docs/license/LICENSE-EVALUATION.txt), [commercial terms](https://github.com/core-experiments/core-pdf/blob/main/docs/license/LICENSE-COMMERCIAL.txt), and [philanthropy waiver terms](https://github.com/core-experiments/core-pdf/blob/main/docs/license/LICENSE-PHILANTHROPY-WAIVER.txt). Those alternatives are effective only when signed by the project licensor.

Contact: <turcioskevinr@gmail.com>


The low-level algorithms are also available as the independently versioned `core-pdf-spec`
package (`core_pdf_spec`). It contains strict PDF semantics and wraps the referenced
standards, which are themselves separate packages (`core-predictors`, `core-postscript`,
`core-jbig2`, `core-pdf-crypto`, `core-adobe-fonts`) usable without PDF; `core-pdf`
composes these with reader recovery, font backends, extraction, and rendering.
See [the spec package](packages/core-pdf-spec/README.md) for the package boundary and
low-level usage.

`PdfDocument.standards` exposes document versions, extensions, and unverified profile
claims. The optional [`core-pdf-validate`](packages/core-pdf-validate/README.md) companion
checks original PDF bytes with a configured external validator. See
[PDF versions and standards validation](docs/standards.md) for supported interfaces and coverage.

The [`core-pdf-compat`](packages/core-pdf-compat/README.md) companion holds the
compatibility facades listed above; like `core-pdf-ocr` it pins the exact core release.
