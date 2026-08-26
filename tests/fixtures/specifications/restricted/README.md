# Restricted specifications (not committed)

Normative references behind the image filters and colour spaces. Every document
here is free to download but **not licensed for redistribution**, so this
directory is gitignored. Populate it with:

```sh
./scripts/fetch_pdf_specs.sh restricted
```

## Contents

| Fixture | Standard | Used by |
| --- | --- | --- |
| `itu-t/T.81-JPEG.pdf` | ITU-T T.81 (JPEG) | `DCTDecode` |
| `itu-t/T.88-JBIG2.pdf` | ITU-T T.88 (JBIG2) | `JBIG2Decode` |
| `itu-t/T.4-Group3-Fax.pdf` | ITU-T T.4 (Group 3) | `CCITTFaxDecode` K < 0 |
| `itu-t/T.6-Group4-Fax.pdf` | ITU-T T.6 (Group 4) | `CCITTFaxDecode` |
| `itu-t/T.800-JPEG2000.pdf` | ITU-T T.800 (JPEG 2000) | `JPXDecode` |
| `icc/ICC.1-2010.pdf` | ICC.1:2010 (= ISO 15076-1) | `ICCBased`, `Lab`, `CalRGB` |
| `icc/ICC.1-2022.pdf` | ICC.1:2022 | current revision |

ITU-T publishes recommendations at no charge but retains copyright; ICC likewise
distributes its specification freely without granting redistribution rights.

`T.81` is fetched from the long-standing W3C mirror at `w3.org/Graphics/JPEG/`,
because `itu.int` serves a cookie-gated HTML page rather than the PDF for that
particular recommendation. The document is still ITU copyright, which is why it
sits in this tier rather than `reference/`.

## PDF Association technical notes — `pdfa/`

Fetched by `scripts/fetch_pdfa_docs.py restricted` (Playwright; pdfa.org is
behind Cloudflare and rejects curl).

| Fixture | Subject |
| --- | --- |
| `pdfa/TN0001-PDFA1-and-Namespaces.pdf` | XMP namespaces in PDF/A-1 |
| `pdfa/TN0003-Metadata-in-PDFA1.pdf` | metadata requirements |
| `pdfa/TN0008-Predefined-XMP-Properties.pdf` | predefined XMP properties |
| `pdfa/TN0009-XMP-Extension-Schemas.pdf` | XMP extension schemas |
| `pdfa/TN0010-Clarifications-ISO19005.pdf` | clarifications for implementers |
| `pdfa/example-WTPDF-apryse-itext.pdf` | tagged PDF 2.0 sample (Apryse/iText) |
| `pdfa/example-WTPDF-BFO-PDFUA2.pdf` | tagged PDF 2.0 sample (BFO) |

The four PDF/A Competence Center TechNotes carry an explicit restriction --
"redistributing this document is only allowed with written approval" -- so they
are gitignored despite the PDF Association's newer publications being CC BY 4.0.
TN0010 states a bare copyright with no grant, and the two WTPDF example files
carry no licence at all, so both are treated the same way by default.

The WTPDF examples are the only tagged **PDF 2.0** documents in the corpus, which
makes them unusually useful to `s_14_structure`; the committed
`tests/fixtures/pdf20examples` submodule is CC BY-SA and covers PDF 2.0 syntax
more generally.
