# core-predictors

PNG (3rd edition, section 9) filter reconstruction and TIFF 6.0 (section 14) horizontal differencing kernels, independent of PDF.

This distribution is a workspace floor package of `core-pdf`: it imports nothing from
`core-pdf`, `core-pdf-spec`, `core-pdf-ocr`, or `core-pdf-validate`, and none of the other
floor packages. PDF-specific glue (stream parameters, PDF objects, PDF exceptions) lives
in the `core-pdf-spec` chapter that ISO 32000 assigns.

The package is a single module: `core_predictors` exports `png_predict`,
`tiff_predict`, `tiff_predict_bits`, `unpack_subbyte_rows`, `uint8_view`,
`PredictorError`, and `UnsupportedPngFilterError`.

Run its tests without the rest of the workspace installed:

```sh
uv sync --locked --package core-predictors --group test
uv run --locked --no-sync pytest packages/core-predictors/tests
```
