# Standard font data

These static tables were extracted from the fontTools source vendored by core-pdf.
`cff_tables.py` retains the CFF standard strings and predefined charsets from
`fontTools/cffLib/__init__.py`. `encoding_names.py` retains StandardEncoding and
MacRoman names from `fontTools/encodings`, and WinAnsi names resolved from the Adobe
Glyph List in `fontTools/agl.py`. MacRoman follows the PDF Annex D character-code
assignments (undefined controls, space at CA, currency at DB, undefined F0).

Only literal standard data is included; fontTools execution and font repair stay in
core-pdf. The original fontTools and Adobe license notices are in `LICENSE.txt` and
`LICENSE.external.txt`.
