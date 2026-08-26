# SPDX-License-Identifier: AGPL-3.0-only
"""Engine-independent runtime infrastructure: buffers, caches, and bounded execution.

Nothing here may import from ``core_pdf.impl.engine``; these modules sit beneath the
whole engine so that ``spec/``, ``parse/`` and ``render/`` can depend on them downward.
"""
