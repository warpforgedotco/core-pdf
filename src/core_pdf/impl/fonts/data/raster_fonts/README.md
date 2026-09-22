# Bundled raster fallback fonts

The twelve `Liberation*.ttf` files are Liberation Fonts 2.1.5, downloaded from the
upstream binary release:

https://github.com/liberationfonts/liberation-fonts/releases/tag/2.1.5

They are metrically compatible substitutes for the PDF standard Helvetica, Times,
and Courier families. They are used only when rasterizing a font without an embedded
program; text extraction and PDF font metrics remain unchanged.

Copyright and license terms are recorded in `LICENSE-Liberation.txt` (SIL Open Font
License 1.1).

`NotoSansSymbols-Regular.ttf` 2.003 and `NotoSansSymbols2-Regular.ttf` 2.008 are
from the [Noto Symbols releases](https://github.com/notofonts/symbols/releases).
They provide deterministic Symbol and ZapfDingbats coverage under the SIL Open
Font License 1.1 recorded in `LICENSE-Noto-Symbols.txt`.

## SHA-256 checksums

```text
bd62a0672d0b9b6710b01df434c80ad54fa5f0835207eb7b17b7a761463067bb  LiberationMono-Bold.ttf
79451f3c09fe25116098853b7a2ca6e2436220ccc11af022979adbcf195be130  LiberationMono-BoldItalic.ttf
605c01c711b44480a7508d349dfbf3264e81fa43d69e61cfa7d10b86e764c4d1  LiberationMono-Italic.ttf
f2b83c763e8afd21709333370bed4774337fae82267937e2b5aea7e2fbd922c1  LiberationMono-Regular.ttf
788abee4c806d660e8aee46689dd8540cd4bb98da03dcc9d171ce3efd99a9173  LiberationSans-Bold.ttf
698da70fc191cc5f33ad4d6d3fe830fe4624b898ea2e3169955928b7c491f1ee  LiberationSans-BoldItalic.ttf
e5bae5c4cde31f22142753855f4f8fb86da6ff39955ed3c0a11248b0d16948b0  LiberationSans-Italic.ttf
76d04c18ea243f426b7de1f3ad208e927008f961dc5945e5aad352d0dfde8ee8  LiberationSans-Regular.ttf
d754ba427cfe0bca54ae052384baa8f842da5bd6550ad4da024ac441e7a7d5ce  LiberationSerif-Bold.ttf
f17db8af71e24d2066b587546021d4f0b296be389512b658dec3c09affeb11a7  LiberationSerif-BoldItalic.ttf
0e3dea9f8d613e006ccfa62201f33e265d19167bd0907725c3e145368b04fc2e  LiberationSerif-Italic.ttf
058ea80864aef09a23f45cbec2bb5400bc3dfbdea01c3f10538a21fcb497fb74  LiberationSerif-Regular.ttf
d0e98e9a2c046594c5021437273943be7e79e0fd980fde125279e22302212595  NotoSansSymbols-Regular.ttf
c4a0a80f0041ce4be81e2478faad22776d23edb98ae3f0d19bd37044820ecf9d  NotoSansSymbols2-Regular.ttf
```
