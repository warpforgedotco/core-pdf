# core-pdf

**High-Performance PDF Engine**

## Structured extraction

`PdfDocument.extract()` is the canonical extraction API. It returns immutable,
layout-aware page results containing ordered text blocks and resolved lines, including
their geometry, column index, rotation, source, and confidence metadata. Use
`PdfDocument.to_markdown()` when a Markdown view is needed.

The records are provided by the workspace-level `core-document` package and can also
be serialized as versioned JSON or semantic HTML.

Optional enrichers implement `core_document.DocumentAdapter` and can be passed to
`PdfDocument.extract(adapters=...)`; OCR is not a dependency of this package.

`core_pdf.serialize_document_to_pdf(document)` writes a new PDF from the IR using
standard Type1 fonts by default. Pass a `PdfFontProvider` for alternate font
resources; Unicode outside the provider’s encoding is rejected explicitly.

Pass `StandardPdfEncryption(user_password=...)` to enable PDF Standard Security
Revision 3 encryption. Incremental saves preserve that encryption for new and
replaced objects when the authenticated input uses Standard Security Revision 3.

`PdfSignaturePlan` provides a detached PDF signature container backed by an
external CMS/PKCS#7 signer. The signer receives the exact `/ByteRange` bytes;
core-pdf reserves and fills `/Contents` but does not manage keys or generate CMS.

`core-pdf` currently extracts native PDF text objects only. Image-only and scanned pages
return no text blocks; OCR is intentionally provided separately and is not invoked by
this package.

Native extraction snapshots are checked with:

```sh
uv run python scripts/native_snapshots.py
```

After an intentional extraction change, regenerate them with
`uv run python scripts/native_snapshots.py --update` and review the Markdown diff.

![core-pdf — High-Performance PDF Engine](.github/assets/core-pdf-social-preview.jpg)

## License

core-pdf uses [Core License version 0.1.0](https://github.com/core-experiments/core-pdf/blob/main/docs/license/VERSION.md).

Unless a separate signed evaluation license, commercial license, or philanthropy waiver applies, core-pdf is licensed under the GNU Affero General Public License version 3 only. See [LICENSE.txt](LICENSE.txt).

Evaluation licenses, commercial licenses, and philanthropy waivers are available separately. See the [license documentation](https://github.com/core-experiments/core-pdf/blob/main/docs/license/README.md), [notice](https://github.com/core-experiments/core-pdf/blob/main/docs/license/NOTICE), [evaluation terms](https://github.com/core-experiments/core-pdf/blob/main/docs/license/LICENSE-EVALUATION.txt), [commercial terms](https://github.com/core-experiments/core-pdf/blob/main/docs/license/LICENSE-COMMERCIAL.txt), and [philanthropy waiver terms](https://github.com/core-experiments/core-pdf/blob/main/docs/license/LICENSE-PHILANTHROPY-WAIVER.txt). Those alternatives are effective only when signed by the project licensor.

Contact: <turcioskevinr@gmail.com>
