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
