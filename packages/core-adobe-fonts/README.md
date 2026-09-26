# core-adobe-fonts

Adobe font format parsers and tables: CFF (TN 5176), Type 2 charstrings (TN 5177), Type 1 font programs, CMaps (TN 5014) with the Adobe cmap-resources data, the Adobe Glyph List, and Core 14 AFM metrics. Independent of PDF.

0.2.0 removes `cmap.tokenizer.iter_blocks`; iterate
`CMapProgram.parse(data).blocks(begin, end)` and read each block's `data` instead.
`type1.program.parse_type1_font_program_encoding` gains `skip_out_of_range`, and
`cmap.decoder` exports `MIN_CID` and `MAX_CID`.

This distribution is a workspace floor package of `core-pdf`: it imports nothing from
`core-pdf`, `core-pdf-spec`, `core-pdf-ocr`, or `core-pdf-validate`. It may import another floor package it declares as a
dependency, such as `core-records` for its value classes. PDF-specific glue (stream parameters, PDF objects, PDF exceptions) lives
in the `core-pdf-spec` chapter that ISO 32000 assigns.

Run its tests without the rest of the workspace installed:

```sh
uv sync --locked --package core-adobe-fonts --group test
uv run --locked --no-sync pytest packages/core-adobe-fonts/tests
```
