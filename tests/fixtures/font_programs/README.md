# Font-program fixtures

`cff2-a.otf.zlib.hex` is a zlib-compressed, hex-encoded subset of
`hintordertest.otf` from Adobe's
[`cff2-hint-ordering-test`](https://github.com/adobe-fonts/cff2-hint-ordering-test)
repository. The upstream font is Copyright 2014-2023 Adobe with Reserved Font
Name `Source` and is distributed under the SIL Open Font License 1.1. The full
license is recorded at
`src/core_pdf/impl/engine/spec/s_09_fonts/data/raster_fonts/LICENSE-Noto-Symbols.txt`.

- Upstream SHA-256: `69349a374bd5cdcdba5d13a8ce87a37acd967bb714e7030a3fef271a443f207e`
- Decoded subset SHA-256: `9c5c093c83c461f39e01e00d0ad1647d2165b0e5d4754260a225a7ba788c5594`
- Generated with: `fonttools subset hintordertest.otf --glyphs=A --no-hinting`

The textual encoding keeps the small binary fixture reviewable without Git LFS.
