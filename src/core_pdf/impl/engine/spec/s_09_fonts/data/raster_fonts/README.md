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
