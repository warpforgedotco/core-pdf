# core-jbig2

ITU-T T.88 (JBIG2) embedded-stream decoding: segment headers, generic region headers, and page composition. Output uses T.88 polarity (1 = black).

The MQ arithmetic decoder is not here. Since 0.2.0 it lives in `core-pdf-cythonized`,
compiled, and `JBIG2PageDecoder.decode_generic_region` raises `Jbig2UnsupportedError`
for arithmetic-coded regions, as it does for MMR-coded ones. A subclass supplies the
decoding: `core-pdf` does, through its recovery decoder.

This distribution is a workspace floor package of `core-pdf`: it imports nothing from
`core-pdf`, `core-pdf-spec`, `core-pdf-ocr`, or `core-pdf-validate`, and none of the other
floor packages. PDF-specific glue (stream parameters, PDF objects, PDF exceptions) lives
in the `core-pdf-spec` chapter that ISO 32000 assigns.

Run its tests without the rest of the workspace installed:

```sh
uv sync --locked --package core-jbig2 --group test
uv run --locked --no-sync pytest packages/core-jbig2/tests
```
