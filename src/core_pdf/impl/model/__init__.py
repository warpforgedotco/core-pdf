# SPDX-License-Identifier: AGPL-3.0-only
"""The capture data model: geometry, text runs, and glyph records.

These are the records the content-stream interpreter produces and everything
downstream consumes. They sit beneath both ``spec/`` and ``layout/``: nothing here
may import from ``layout/``, ``parse/``, ``render/`` or ``spec/``, which is what lets
the interpreter depend on the model without depending on layout heuristics.
"""
