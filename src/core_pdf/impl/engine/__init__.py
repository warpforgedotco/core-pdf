# SPDX-License-Identifier: AGPL-3.0-only
"""Internal PDF engine packages.

Spec-aligned packages live under ``engine.spec`` and are prefixed by their
PDF 32000 section:

- ``s_07_*``: syntax, objects, filters, security, document structure, content streams
- ``s_08_graphics``: graphics state, matrices, colour, images
- ``s_09_fonts``: text/font data structures and text decoding
- ``s_14_structure``: marked content, logical structure, tagged PDF

Root-level packages are shared or derived processing layers:
``extraction``, ``rendering``, and ``writing``. Reusable layout primitives
live in the workspace-level ``core-layout`` package.
"""

__all__: tuple[str, ...] = ()
