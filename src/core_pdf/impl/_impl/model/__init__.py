# SPDX-License-Identifier: AGPL-3.0-only
"""Foundational models and rules shared by the interpreter and downstream processing.

Geometry, text runs, glyph records, text primitives, and page selections live here.
Nothing here may import from ``spec/``, ``layout/``, ``extract/``, ``render/`` or
structured ``output/``. This keeps shared representations and normalization below
the processing stages that consume them.
"""
