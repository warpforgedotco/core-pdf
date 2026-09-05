# core-pdf

High-Performance PDF Engine

![core-pdf — High-Performance PDF Engine](.github/assets/core-pdf-social-preview.jpg)

`core-pdf` parses native PDF text, images, graphics, and structure. OCR and vector text
recognition are available separately in [`core-pdf-ocr`](packages/core-pdf-ocr/README.md).
Installing the companion does not change the behavior of `core_pdf` or its compatibility facades.

```python
from core_pdf import PdfDocument

with PdfDocument("document.pdf") as document:
    print(document.extract().to_markdown())
```

For scanned or hybrid documents, install `core-pdf-ocr` and change the import to
`from core_pdf_ocr import PdfDocument`. Its `core-pdf-ocr` command accepts the same arguments
as `core-pdf`; both packages support `python -m` invocation.

```sh
uv sync --all-packages --all-groups
uv run --all-packages pytest tests/ packages/core-pdf-ocr/tests/ -n auto
```

## License

core-pdf uses [Core License version 0.1.0](https://github.com/core-experiments/core-pdf/blob/main/docs/license/VERSION.md).

Unless a separate signed evaluation license, commercial license, or philanthropy waiver applies, core-pdf is licensed under the GNU Affero General Public License version 3 only. See [LICENSE.txt](LICENSE.txt).

Evaluation licenses, commercial licenses, and philanthropy waivers are available separately. See the [license documentation](https://github.com/core-experiments/core-pdf/blob/main/docs/license/README.md), [notice](https://github.com/core-experiments/core-pdf/blob/main/docs/license/NOTICE), [evaluation terms](https://github.com/core-experiments/core-pdf/blob/main/docs/license/LICENSE-EVALUATION.txt), [commercial terms](https://github.com/core-experiments/core-pdf/blob/main/docs/license/LICENSE-COMMERCIAL.txt), and [philanthropy waiver terms](https://github.com/core-experiments/core-pdf/blob/main/docs/license/LICENSE-PHILANTHROPY-WAIVER.txt). Those alternatives are effective only when signed by the project licensor.

Contact: <turcioskevinr@gmail.com>
