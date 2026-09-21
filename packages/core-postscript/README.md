# core-postscript

The PostScript calculator operator subset (PLRM 3rd ed. 8.2) as restricted by ISO 32000-1 7.10.5, compiled into Python callables without any PDF object model.

This distribution is a workspace floor package of `core-pdf`: it imports nothing from
`core-pdf`, `core-pdf-spec`, `core-pdf-ocr`, or `core-pdf-validate`, and none of the other
floor packages. PDF-specific glue (stream parameters, PDF objects, PDF exceptions) lives
in the `core-pdf-spec` chapter that ISO 32000 assigns.

Run its tests without the rest of the workspace installed:

```sh
uv sync --locked --package core-postscript --group test
uv run --locked --no-sync pytest packages/core-postscript/tests
```
