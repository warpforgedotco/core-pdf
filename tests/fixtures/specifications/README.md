# Specification fixtures

Specifications serve two distinct purposes here: as **normative references** for
the subsystems the engine implements, and as **parser fixtures** — large,
real-world PDFs produced by the tooling of their era.

They are split into three tiers by redistribution rights. Fetch a tier with
`./scripts/fetch_pdf_specs.sh <tier>`, or all of them with no argument.

The PDF Association's documents sit behind Cloudflare, which rejects curl even
for direct PDF URLs, so they are fetched by `scripts/fetch_pdfa_docs.py` using
Playwright. `fetch_pdf_specs.sh` delegates to it; run
`uvx --from playwright playwright install chromium` first. Those documents land
in `reference/pdfa/` or `restricted/pdfa/` according to their own licences.

| Tier | Committed? | Why |
| --- | --- | --- |
| `reference/` | **yes** | CC BY-ND / BSD / IETF / Unicode / W3C — verbatim redistribution permitted |
| `PDF/` | no | Adobe & ISO reserve all reproduction rights |
| `restricted/` | no | ITU-T and ICC distribute freely but reserve copyright |

`core-pdf` is a public AGPL-3.0 repository, so committing a document means
redistributing it. Only the `reference/` tier is licensed for that. The other
two tiers are gitignored and rebuilt on demand from their canonical sources.

Before adding anything, read the actual copyright notice in the document. The
PDF References carry a section titled "Copyright Permission" that grants the
right to *implement* PDF and not the right to redistribute the specification —
see `PDF/README.md`.

## Which package implements each document

| Fixture | Implementing package |
| --- | --- |
| `PDF/PDFReference-*.pdf`, `PDF/ISO32000-1-*.pdf`, `PDF/ISO32000-2-*.pdf` | `core-pdf-spec` chapters (`s_07_*` to `s_14_*`) |
| `PDF/ISO-TS-32003-*.pdf`, `PDF/ISO-TS-32004-*.pdf` | `core-pdf-crypto` (`ciphers`, `pdf_mac`) with `core_pdf_spec.s_07_security` glue |
| `PDF/ISO-TS-32001-*.pdf`, `ISO-TS-32002-*.pdf`, `ISO-TS-32005-*.pdf` | not implemented (SHA-3 OIDs are accepted by `core-pdf-crypto`) |
| `reference/fonts/TN5176-CFF.pdf`, `TN5177-Type2-Charstring.pdf` | `core-adobe-fonts` (`cff.font`, `cff.charstrings`) |
| `reference/fonts/Type1-Font-Format.pdf`, `TN5015-Type1-Supplement.pdf` | `core-adobe-fonts` (`type1.program`) |
| `reference/fonts/TN5014-CMap-CIDFont.pdf`, `TN5094-CJK-Collections.pdf` | `core-adobe-fonts` (`cmap.*`, `cmap/data/cmaps`) |
| `reference/fonts/TN5180-sfnt.pdf` | `core-pdf` vendored fontTools and `core_pdf_spec.s_09_fonts.font_program_truetype` |
| `reference/glyph-lists/*.txt` | `core-adobe-fonts` (`agl.*`, `_vendor/font_data`) |
| `reference/rfc/rfc5652-cms.txt` | `core-pdf-crypto` (`pdf_mac`) |
| `reference/rfc/rfc1950-*.txt`, `rfc1951-*.txt` | `core_pdf_spec.s_07_filters.codecs` via `zlib` |
| `reference/rfc/rfc3161-timestamp.txt` | not implemented |
| `reference/png/png-3rd-edition.html` | `core-predictors` (`png`) |
| `reference/unicode/*.txt` | `core-pdf` layout and text |
| `reference/pdfa/*.pdf` | `core_pdf_spec.s_14_structure`, `core_pdf_spec.standards`, `core-pdf-validate` |
| `restricted/itu-t/T.88-JBIG2.pdf` | `core-jbig2` with `core_pdf_spec.s_07_filters.jbig2` glue |
| `restricted/itu-t/T.81`, `T.4`, `T.6`, `T.800` | decoders supplied by `core-pdf` (imagecodecs) through the spec filter registry |
| `restricted/icc/*.pdf` | `core_pdf_spec.s_08_graphics` structure only; profiles applied by `core-pdf` |
| `restricted/pdfa/*.pdf` | `core-pdf-validate` and `core_pdf_spec.s_14_structure` |

The PostScript Language Reference (PLRM 3rd ed. 8.2), which is not obtainable as a fixture,
is implemented by `core-postscript`.

## A note on producer diversity

All eight documents in `PDF/` are FrameMaker → Acrobat Distiller output spanning
1993–2008. That is a genuine axis of coverage (Distiller 3 through 8), and one
of them, ISO 32000-1, is RC4-encrypted and so exercises `s_07_security`. But it
is a single producer lineage. Fixtures from pdfTeX, Word, Ghostscript, Chrome
print-to-PDF and scanner output would each exercise materially different code
paths; the `restricted/` tier incidentally adds some non-Adobe producers.
