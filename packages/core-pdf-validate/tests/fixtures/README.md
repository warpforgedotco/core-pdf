# Recorded veraPDF reports

These reports were generated locally with veraPDF **1.30.2**, using its bundled
Greenfield parser and validation profiles, on 2026-09-11. Only the absolute input
filename was replaced with `source.pdf`. Rule references, counts, descriptions,
contexts, build information, statuses, and timings are the actual engine output.

From the repository root with reference corpora initialized:

```sh
verapdf --format xml --flavour 1b --maxfailures -1 \
  tests/fixtures/pypdf/sample-files/021-pdfa/crazyones-pdfa.pdf
verapdf --format xml --flavour ua2 --maxfailures -1 \
  'tests/fixtures/pdf20examples/Simple PDF 2.0 file.pdf'
```

The first exits 0 and passes; the second exits 1 and fails eight rules.
The generated rule descriptions are supplied by veraPDF's bundled validation
profiles, © 2015 veraPDF Consortium, licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
The [upstream attribution notice](https://github.com/veraPDF/veraPDF-validation-profiles/blob/68a7318ca9a5f185a10aaadbb595b4e8440070a6/README.md)
and [PDF/A-1b](https://github.com/veraPDF/veraPDF-validation-profiles/blob/68a7318ca9a5f185a10aaadbb595b4e8440070a6/PDF_A/PDFA-1B.xml)
and [PDF/UA-2 with Tagged PDF](https://github.com/veraPDF/veraPDF-validation-profiles/blob/68a7318ca9a5f185a10aaadbb595b4e8440070a6/PDF_UA/PDFUA-2-ISO32005.xml)
profile sources are linked at a fixed reference snapshot. The rule descriptions
in these recordings were not edited; the filename change is described above.

The Java engine is separately distributed under GPLv3+ or MPLv2+; no engine is
bundled here. Its license is distinct from the profiles' CC BY 4.0 license.

Contracts checked against the pinned upstream release:

- [CLI validation options](https://docs.verapdf.org/cli/validation/)
- [XML report implementation](https://github.com/veraPDF/veraPDF-library/blob/v1.30.2/core/src/main/java/org/verapdf/processor/reports/ValidationReportImpl.java)
- [Exit status integration tests](https://github.com/veraPDF/veraPDF-apps/blob/v1.30.2/tests/exit-status.sh)

## Positive conformance fixtures

The `positive/` directory provides an actual veraPDF **1.30.2** machine-pass case
for every advertised target. All fifteen targets were run explicitly on
2026-09-11. Their XML pass recordings are authentic engine output, with only the
temporary input pathname replaced by `source.pdf`, as for the recordings above.
The live CI job repeats every positive case and all fifteen negative cases,
and validates all three original claims of the PDF/UA-2/WTPDF file together
through `profiles="declared"`.
These are smoke cases for the adapter and profiles, not exhaustive standards
coverage or certification of requirements needing human review.

The nine PDFs total **92,510 bytes**. Eight are unchanged copies from the
[veraPDF corpus at commit `377728596647e40edf16003d68d620773ce96ec5`](https://github.com/veraPDF/veraPDF-corpus/tree/377728596647e40edf16003d68d620773ce96ec5).
They are attributed to the **veraPDF Consortium and veraPDF-corpus contributors**
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), which also covers
our derivative below. The original upstream attribution and license notice is
preserved verbatim in [positive/UPSTREAM-README.md](positive/UPSTREAM-README.md).
[positive/manifest.json](positive/manifest.json) records every original path,
original SHA256, delivered SHA256, supported test targets, and modification.
Upstream `pass` filenames identify individual rule tests; we independently
verified complete machine-profile passes before selecting them.

| PDF | Explicit positive targets | Exercised content |
| --- | --- | --- |
| `pdfa-1a-unicode.pdf` | PDF/A-1a, PDF/A-1b | Tagged text and Unicode character mapping |
| `pdfa-2a-actualtext.pdf` | PDF/A-2a, PDF/A-2b, PDF/A-2u | Private-use character with structure-element ActualText |
| `pdfa-3a-actualtext.pdf` | PDF/A-3a, PDF/A-3u | Same tagged ActualText under PDF/A-3 identification |
| `pdfa-3b-associated-file.pdf` | PDF/A-3b | Associated text file and `AFRelationship /Source` |
| `pdfa-4-device-color.pdf` | PDF/A-4 | PDF 2.0 device-color requirements |
| `pdfa-4e-optional-content.pdf` | PDF/A-4e | Permitted `SetOCGState` action |
| `pdfa-4f-embedded-files.pdf` | PDF/A-4f | Embedded PDF and text files |
| `pdfua-1-text.pdf` | PDF/UA-1 | Tagged text |
| `pdfua-2-wtpdf-unicode.pdf` | PDF/UA-2, WTPDF 1.0 Reuse, WTPDF 1.0 Accessibility | PDF 2.0 tagging, Unicode mapping, and all three original declarations |

`pdfa-3a-actualtext.pdf` is derived from `pdfa-2a-actualtext.pdf` by replacing
the single XMP attribute `pdfaid:part="2"` with `pdfaid:part="3"`. This changes
exactly one byte. Every other byte, including stream lengths, xref offsets,
content, outlines, and original metadata dates, is preserved. The outlines
describe the original PDF/A-2 rule test; the derivative targets are documented
here and in the manifest. Both PDF/A-3a and PDF/A-3u pass the real engine without
repairs. An ordinary test checks the exact derivation and all fixture hashes.

For example, reproduce the three-target accessible fixture with separate runs:

```sh
verapdf --format xml --flavour ua2 --maxfailures -1 \
  packages/core-pdf-validate/tests/fixtures/positive/pdfua-2-wtpdf-unicode.pdf
verapdf --format xml --flavour wt1r --maxfailures -1 \
  packages/core-pdf-validate/tests/fixtures/positive/pdfua-2-wtpdf-unicode.pdf
verapdf --format xml --flavour wt1a --maxfailures -1 \
  packages/core-pdf-validate/tests/fixtures/positive/pdfua-2-wtpdf-unicode.pdf
```
