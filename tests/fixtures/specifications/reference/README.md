# Normative reference specifications (committed)

ISO 32000 specifies PDF, but delegates whole subsystems to outside standards.
This directory holds those standards, for the parts of the engine that implement
them. Unlike `../PDF/` and `../restricted/`, everything here is under a licence
that permits verbatim redistribution, so it is committed — see `NOTICE.md` for
the per-source attribution each licence requires.

Re-fetch with:

```sh
./scripts/fetch_pdf_specs.sh reference
```

## What implements what

| Reference | Licence | Engine code |
| --- | --- | --- |
| `fonts/TN5176-CFF.pdf` | CC BY-ND 4.0 | `spec/s_09_fonts` — the CFF parser |
| `fonts/TN5177-Type2-Charstring.pdf` | CC BY-ND 4.0 | charstring interpretation |
| `fonts/Type1-Font-Format.pdf` | CC BY-ND 4.0 | `Type1`, `MMType1` |
| `fonts/TN5015-Type1-Supplement.pdf` | CC BY-ND 4.0 | Type 1 edge cases |
| `fonts/TN5014-CMap-CIDFont.pdf` | CC BY-ND 4.0 | `Type0`, `CIDFontType0/2`, CMaps |
| `fonts/TN5094-CJK-Collections.pdf` | CC BY-ND 4.0 | Adobe-Japan1 etc. ordering |
| `fonts/TN5180-sfnt.pdf` | CC BY-ND 4.0 | sfnt/OpenType container |
| `glyph-lists/glyphlist.txt` | BSD-3-Clause | glyph-name → Unicode fallback |
| `glyph-lists/aglfn.txt` | BSD-3-Clause | new-font glyph naming |
| `rfc/rfc1950-zlib.txt` | IETF Trust | `FlateDecode` container |
| `rfc/rfc1951-deflate.txt` | IETF Trust | `FlateDecode` |
| `rfc/rfc5652-cms.txt` | IETF Trust | `adbe.pkcs7` signatures |
| `rfc/rfc3161-timestamp.txt` | IETF Trust | signature timestamps |
| `unicode/UnicodeData.txt` | Unicode | `ToUnicode` normalisation |
| `unicode/LineBreak.txt` | Unicode | line/word segmentation |
| `png/png-3rd-edition.html` | W3C Document | `FlateDecode` PNG predictors |
| `pdfa/Matterhorn-Protocol-1.1.pdf` | CC BY 4.0 | PDF/UA failure conditions, `s_14_structure` |
| `pdfa/Tagged-PDF-Best-Practice-Guide-Syntax.pdf` | CC BY 4.0 | structure tree, `s_14_structure` |
| `pdfa/Well-Tagged-PDF-WTPDF-1.0.pdf` | CC BY 4.0 | tagged PDF 2.0 |
| `pdfa/Conforming-to-PDFA-and-PDFUA.pdf` | CC BY 4.0 | overlapping subset constraints |
| `pdfa/Custom-Metadata-Structures-in-PDF.pdf` | CC BY 4.0 | XMP and metadata extraction |
| `pdfa/PDF-Declarations-1.0.pdf` | CC BY 4.0 | declaration/attestation dictionaries |
| `pdfa/BPG-Math-in-PDF.pdf` | CC BY 4.0 | formula tagging, text extraction |
| `pdfa/PDF-Extension-Brotli.pdf` | CC BY 4.0 | proposed Brotli stream filter |
| `pdfa/EA-PDF-v1.pdf` | CC BY 4.0 | email-archiving profile (context only) |

The glyph lists are worth calling out: unmapped glyph names are a known source
of text-extraction divergence against other engines, and `glyphlist.txt` is the
authoritative mapping.

## Known gaps

- **OpenType / ISO 14496-22 (OFF)** — Microsoft publishes the OpenType spec only
  as a multi-page site, with no single downloadable document; the ISO form is
  paywalled. `fonts/TN5180-sfnt.pdf` covers the container format.
- **PostScript Language Reference** — needed for Type 4 (PostScript calculator)
  functions. Adobe no longer serves a stable copy.
- **UAX #9 / #14 / #29** — the algorithm reports themselves are HTML-only and
  versioned; the UCD data files above carry the property tables they operate on.
- **ISO 32000-2 (PDF 2.0)** — the PDF Association gives it away, but only through
  a cart/checkout on pdfa-inc.org rather than a direct link, so it cannot be
  scripted. `pdfa/Well-Tagged-PDF-WTPDF-1.0.pdf` covers the PDF 2.0 tagging
  rules; the base specification still has to be fetched by hand.
