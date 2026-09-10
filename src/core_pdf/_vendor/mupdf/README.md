# MuPDF shading raster kernels

`shade_raster.py` is a Python adaptation of the numeric axial/radial mesh generation
and triangle scan conversion in MuPDF **1.28.2**, Copyright (C) 2004–2026 Artifex
Software, Inc., licensed under GNU AGPL version 3 or later (see `LICENSE.txt`).

Pinned source files:

- [`source/fitz/shade.c`](https://github.com/ArtifexSoftware/mupdf/blob/1.28.2/source/fitz/shade.c)
  SHA-256: `3135b85d574f07a3557569175ef517afec1f357b376b62ec62fb4e380d023aaf`
- [`source/fitz/draw-mesh.c`](https://github.com/ArtifexSoftware/mupdf/blob/1.28.2/source/fitz/draw-mesh.c)
  SHA-256: `9eb4ccbb28029b3018f7f6e15159431b45a1dc8496dddd6e5f9fcf48d5c04c72`

Adaptations: retain only type 2/3 shading geometry, one-component lookup indices,
and coverage; replace MuPDF context/pixmap/callback machinery with a small array
owner; vectorize each horizontal sample span; explicitly round floating-point
operations to binary32. The triangle ordering, edge stepping, and 16.16 sample
interpolation are retained because they determine exported PNG bytes. PDF object
resolution, function evaluation, color conversion, clipping and encoding remain
in core-pdf's own modules. No MuPDF/PyMuPDF runtime is imported.
