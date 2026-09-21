# core-adobe-fonts

Adobe font format parsers and tables: CFF (TN 5176), Type 2 charstrings (TN 5177), Type 1 font programs, CMaps (TN 5014) with the Adobe cmap-resources data, the Adobe Glyph List, and Core 14 AFM metrics. Independent of PDF.

This distribution is a workspace floor package of `core-pdf`: it imports nothing from
`core-pdf`, `core-pdf-spec`, `core-pdf-ocr`, or `core-pdf-validate`, and none of the other
floor packages. PDF-specific glue (stream parameters, PDF objects, PDF exceptions) lives
in the `core-pdf-spec` chapter that ISO 32000 assigns.

Run its tests without the rest of the workspace installed:

```sh
uv sync --locked --package core-adobe-fonts --group test
uv run --locked --no-sync pytest packages/core-adobe-fonts/tests
```
