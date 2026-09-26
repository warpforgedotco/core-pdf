# core-jbig2

ITU-T T.88 (JBIG2) embedded-stream decoding: segment headers, generic region headers, and page composition. Output uses T.88 polarity (1 = black).

The MQ arithmetic decoder is not here. Since 0.2.0 it lives in `core-pdf-cythonized`,
compiled, and `JBIG2PageDecoder.decode_generic_region` raises `Jbig2UnsupportedError`
for arithmetic-coded regions, as it does for MMR-coded ones. A subclass supplies the
decoding: `core-pdf` does, through its recovery decoder.

0.3.0 removes from `core_jbig2.codec`'s exports the byte readers (`read_be_*`,
`read_u*`), `parse_page_association`, `parse_referred_to_segments`, the
`jbig2_page_*` page-flag helpers, and `region_operator`, which nothing outside the
module used.

This distribution is a workspace floor package of `core-pdf`: it imports nothing from
`core-pdf`, `core-pdf-spec`, `core-pdf-ocr`, or `core-pdf-validate`. It may import another floor package it declares as a
dependency, such as `core-records` for its value classes. PDF-specific glue (stream parameters, PDF objects, PDF exceptions) lives
in the `core-pdf-spec` chapter that ISO 32000 assigns.

Run its tests without the rest of the workspace installed:

```sh
uv sync --locked --package core-jbig2 --group test
uv run --locked --no-sync pytest packages/core-jbig2/tests
```
